import asyncio
from pathlib import Path

from adapters.lora import ExperienceAdapter, ExperienceDelta
from core.goals import GoalPriority
from core.process import PersistentCore


def core_config(tmp_path: Path) -> dict:
    data_dir = tmp_path / "runtime"
    return {
        "data_dir": str(data_dir),
        "working_memory_capacity": 2048,
        "episodic_db_path": str(data_dir / "episodic.db"),
        "semantic_db_path": str(data_dir / "semantic.db"),
        "adapter_path": str(data_dir / "adapter"),
        "core_state_path": str(data_dir / "core_state.json"),
        "journal_path": str(data_dir / "events.jsonl"),
    }


def test_core_restores_snapshot_and_replays_later_journal_events(tmp_path):
    config = core_config(tmp_path)
    first = PersistentCore(config)
    first.on_interaction(
        "remember the tea preference",
        {"role": "user", "concepts": ["tea"]},
    )
    first_goal = first.add_goal("snapshot goal", GoalPriority.HIGH)
    asyncio.run(first.run_consolidation_cycle())
    asyncio.run(first._save_state())

    replay_goal = first.add_goal("journal replay goal", GoalPriority.MEDIUM)

    restored = PersistentCore(config)
    restored_goal_ids = {goal["id"] for goal in restored.get_state_snapshot()["active_goals"]}
    assert first_goal.id in restored_goal_ids
    assert replay_goal.id in restored_goal_ids
    assert restored.replay_summary["restored_from_snapshot"] is True
    assert restored.replay_summary["events_replayed"] >= 1
    assert "tea" in restored.get_state_snapshot()["salience_map"]

    event_text = (tmp_path / "runtime" / "events.jsonl").read_text(encoding="utf-8")
    assert "consolidation_ran" in event_text


def test_adapter_deltas_survive_restart(tmp_path):
    config = core_config(tmp_path)
    adapter = ExperienceAdapter("mock-model", config)
    assert adapter.apply_delta(
        ExperienceDelta(
            content="Use concise implementation notes for continuity work.",
            feedback=0.9,
            domain="user_preferences",
            confidence=0.8,
        )
    )

    restarted = ExperienceAdapter("mock-model", config)
    assert restarted.stats()["update_count"] == 1
    assert restarted.stats()["deltas_in_memory"] == 1
    assert "concise implementation notes" in restarted.get_adaptation_context("concise")
