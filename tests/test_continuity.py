import asyncio
from pathlib import Path
import sys

from adapters.lora import ExperienceAdapter, ExperienceDelta
from daemon.supervisor import ProcessSupervisor, SupervisorConfig
from core.goals import GoalPriority
from core.process import PersistentCore
from memory.cold import SemanticMemory
from memory.warm import EpisodicMemory


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
        "embedding_dimensions": 64,
        "adapter_training_epochs": 10,
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
    assert restarted.stats()["low_rank_adapter"]["train_steps"] > 0
    assert (tmp_path / "runtime" / "adapter" / "low_rank_adapter.json").exists()


def test_vector_memory_retrieval_without_keyword_substring(tmp_path):
    semantic = SemanticMemory(str(tmp_path / "semantic.db"), embedding_dimensions=64)
    semantic.store_fact(
        content="Jae prefers concise implementation notes.",
        source_episode_ids=[],
        domain="user_preferences",
        confidence=0.9,
    )
    facts = semantic.retrieve("concise notes", limit=1)
    assert facts
    assert facts[0].domain == "user_preferences"
    assert facts[0].retrieval_score > 0

    episodic = EpisodicMemory(str(tmp_path / "episodic.db"), embedding_dimensions=64)
    episodic.store(
        content="The user asked for exact verification commands.",
        summary="Exact verification commands matter.",
        tags=["verification"],
        salience=0.8,
    )
    episodes = episodic.recall("verification commands", limit=1)
    assert episodes
    assert episodes[0].retrieval_score > 0


def test_supervisor_restarts_exited_child(tmp_path):
    supervisor = ProcessSupervisor(
        SupervisorConfig(
            command=[sys.executable, "-c", "import sys; sys.exit(7)"],
            cwd=str(tmp_path),
            check_interval_seconds=0.05,
            restart_backoff_seconds=0.01,
            max_restarts=2,
        )
    )
    stats = supervisor.supervise(max_cycles=20)
    assert stats.restarts == 2
    assert stats.last_exit_code == 7
    assert stats.last_restart_reason == "process_exited"
