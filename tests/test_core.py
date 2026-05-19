"""tests/test_core.py — Core process, state, and goals tests"""

import pytest
import time
import asyncio
from core.state import MotivationalState
from core.goals import GoalStack, GoalPriority, GoalStatus


class TestMotivationalState:
    def test_initial_values_in_range(self):
        s = MotivationalState()
        assert 0.0 <= s.arousal <= 1.0
        assert 0.0 <= s.focus <= 1.0
        assert 0.0 <= s.curiosity <= 1.0
        assert 0.0 <= s.urgency <= 1.0

    def test_tick_does_not_crash(self):
        s = MotivationalState()
        for _ in range(100):
            time.sleep(0.001)
            s.tick(idle_seconds=0.1, active_goals=1, salience_peak=0.5)
        assert 0.0 <= s.arousal <= 1.0

    def test_flag_urgent_raises_urgency(self):
        s = MotivationalState()
        initial_urgency = s.urgency

        class MockGoal:
            description = "test goal"
        s.flag_urgent(MockGoal())
        assert s.urgency > initial_urgency

    def test_idle_decays_arousal(self):
        s = MotivationalState()
        s.arousal = 0.8
        # Simulate many ticks of idleness
        for i in range(200):
            time.sleep(0.001)
            s.tick(idle_seconds=300, active_goals=0, salience_peak=0.0)
        assert s.arousal < 0.8
        assert s.arousal >= s.IDLE_AROUSAL_FLOOR

    def test_novel_input_boosts_curiosity(self):
        s = MotivationalState()
        s.curiosity = 0.3
        s.on_novel_input()
        assert s.curiosity > 0.3

    def test_to_dict_has_mode(self):
        s = MotivationalState()
        d = s.to_dict()
        assert "mode" in d
        assert d["mode"] in {"consolidating", "urgent", "deep_focus", "exploratory", "nominal"}

    def test_consolidating_mode_when_very_idle(self):
        s = MotivationalState()
        s.arousal = 0.1  # Force very low arousal
        d = s.to_dict()
        assert d["mode"] == "consolidating"


class TestGoalStack:
    def test_add_goal(self):
        gs = GoalStack()
        goal = gs.add("Complete the memory system", priority=GoalPriority.HIGH)
        assert goal.id is not None
        assert goal.description == "Complete the memory system"
        assert goal.priority == GoalPriority.HIGH
        assert gs.active_count == 1

    def test_complete_goal(self):
        gs = GoalStack()
        goal = gs.add("Test goal")
        gs.complete(goal.id, notes="Done")
        assert gs.active_count == 0
        assert gs._goals[goal.id].status == GoalStatus.COMPLETED

    def test_update_progress(self):
        gs = GoalStack()
        goal = gs.add("Progress goal")
        gs.update_progress(goal.id, 0.5, "halfway")
        assert gs._goals[goal.id].progress == pytest.approx(0.5)

    def test_check_urgency_returns_none_when_low_priority(self):
        gs = GoalStack()
        gs.add("Low priority goal", priority=GoalPriority.LOW)
        result = gs.check_urgency()
        assert result is None  # Low priority never reaches urgency threshold

    def test_check_urgency_with_overdue_deadline(self):
        gs = GoalStack()
        goal = gs.add(
            "Overdue goal",
            priority=GoalPriority.HIGH,
            deadline=time.time() - 1,  # Already past deadline
        )
        result = gs.check_urgency()
        assert result is not None
        assert result.id == goal.id

    def test_decay_stale_goals(self):
        gs = GoalStack()
        goal = gs.add("Stale goal")
        # Manually age the goal
        gs._goals[goal.id].created_at = time.time() - (gs.DECAY_THRESHOLD_HOURS * 3600 + 1)
        gs.run_decay()
        assert gs._goals[goal.id].status == GoalStatus.DECAYED

    def test_to_list_excludes_completed(self):
        gs = GoalStack()
        g1 = gs.add("Active goal")
        g2 = gs.add("To complete")
        gs.complete(g2.id)
        result = gs.to_list()
        ids = [g["id"] for g in result]
        assert g1.id in ids
        assert g2.id not in ids

    def test_urgency_score_range(self):
        gs = GoalStack()
        goal = gs.add("Test")
        assert 0.0 <= goal.urgency_score <= 1.0

    def test_multiple_goals_ordering(self):
        gs = GoalStack()
        gs.add("Low", priority=GoalPriority.LOW)
        gs.add("High", priority=GoalPriority.HIGH)
        gs.add("Medium", priority=GoalPriority.MEDIUM)
        result = gs.to_list()
        # Should be sorted by urgency descending
        scores = [g["urgency_score"] for g in result]
        assert scores == sorted(scores, reverse=True)


class TestPersistentCore:
    def test_init_and_state_snapshot(self, tmp_path):
        config = {
            "episodic_db_path": str(tmp_path / "ep.db"),
            "semantic_db_path": str(tmp_path / "sem.db"),
            "embed_path": str(tmp_path / "embed.pkl"),
            "adapter_path": str(tmp_path / "adapter"),
            "working_memory_capacity": 4096,
        }
        from core.process import PersistentCore
        core = PersistentCore(config)
        snapshot = core.get_state_snapshot()

        assert "uptime_seconds" in snapshot
        assert "heartbeat_count" in snapshot
        assert "motivational_state" in snapshot
        assert "active_goals" in snapshot
        assert snapshot["heartbeat_count"] == 0  # Not started yet

    def test_on_interaction_updates_state(self, tmp_path):
        config = {
            "episodic_db_path": str(tmp_path / "ep.db"),
            "semantic_db_path": str(tmp_path / "sem.db"),
            "embed_path": str(tmp_path / "embed.pkl"),
            "adapter_path": str(tmp_path / "adapter"),
            "working_memory_capacity": 4096,
        }
        from core.process import PersistentCore
        core = PersistentCore(config)
        assert core.metrics.last_interaction is None

        core.on_interaction("hello world", {"role": "user", "concepts": ["greeting"]})
        assert core.metrics.last_interaction is not None
        assert core.metrics.total_interactions == 1
        assert "greeting" in core.salience

    def test_on_interaction_boosts_salience(self, tmp_path):
        config = {
            "episodic_db_path": str(tmp_path / "ep.db"),
            "semantic_db_path": str(tmp_path / "sem.db"),
            "embed_path": str(tmp_path / "embed.pkl"),
            "adapter_path": str(tmp_path / "adapter"),
            "working_memory_capacity": 4096,
        }
        from core.process import PersistentCore
        core = PersistentCore(config)
        core.on_interaction("test", {"role": "user", "concepts": ["memory", "goals"]})
        assert core.salience.get("memory", 0) > 0
        assert core.salience.get("goals", 0) > 0

    def test_heartbeat_loop_increments_count(self, tmp_path):
        config = {
            "episodic_db_path": str(tmp_path / "ep.db"),
            "semantic_db_path": str(tmp_path / "sem.db"),
            "embed_path": str(tmp_path / "embed.pkl"),
            "adapter_path": str(tmp_path / "adapter"),
            "working_memory_capacity": 4096,
        }
        from core.process import PersistentCore
        core = PersistentCore(config)

        async def run_briefly():
            task = asyncio.create_task(core.start())
            await asyncio.sleep(0.5)  # Run for 500ms — should get ~5 heartbeats
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_briefly())
        assert core.metrics.heartbeat_count > 0
        assert core.metrics.uptime_seconds > 0
