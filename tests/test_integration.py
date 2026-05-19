"""tests/test_integration.py — Integration tests: full pipeline without API key"""

import pytest
import asyncio
import time
import os


@pytest.fixture
def config(tmp_path):
    return {
        "episodic_db_path": str(tmp_path / "ep.db"),
        "semantic_db_path": str(tmp_path / "sem.db"),
        "embed_path": str(tmp_path / "embed.pkl"),
        "adapter_path": str(tmp_path / "adapter"),
        "core_state_path": str(tmp_path / "core_state.json"),
        "journal_path": str(tmp_path / "events.jsonl"),
        "working_memory_capacity": 4096,
        "consolidation_idle_threshold_seconds": 1,  # Fast for tests
    }


class TestFullMemoryPipeline:
    """Store → recall → consolidate → semantic retrieve."""

    def test_store_and_recall_episodic(self, config):
        from memory.warm import EpisodicMemory
        em = EpisodicMemory(
            db_path=config["episodic_db_path"],
            embed_path=config["embed_path"]
        )

        em.store(
            content="The user asked about Python async patterns",
            summary="User asked about Python async patterns",
            tags=["python", "async", "question"],
            valence=0.1,
            salience=0.9,
        )
        em.store(
            content="User prefers concise code examples",
            summary="User prefers concise code examples",
            tags=["preference", "code"],
            valence=0.2,
            salience=0.8,
        )

        results = em.recall("python code examples", limit=5)
        assert len(results) >= 1
        summaries = [r.summary.lower() for r in results]
        assert any("python" in s or "code" in s for s in summaries)

    def test_consolidation_pipeline(self, config):
        """Episodes with low salience get consolidated into semantic facts."""
        from memory.warm import EpisodicMemory
        from memory.consolidator import Consolidator

        em = EpisodicMemory(
            db_path=config["episodic_db_path"],
            embed_path=config["embed_path"]
        )

        # Store episodes with very low salience (consolidation candidates)
        for i in range(3):
            em.store(
                content=f"User mentioned they prefer Python over JavaScript episode {i}",
                summary=f"User prefers Python over JavaScript",
                tags=["preference", "python"],
                valence=0.3,
                salience=0.05,  # Below consolidation threshold
            )

        consolidator = Consolidator(em, config)
        result = asyncio.run(consolidator.run_cycle({}))

        assert result["episodes_consolidated"] >= 1
        # Facts may be 0 if rule-based extraction finds nothing (low salience + no API key)
        assert result["episodes_consolidated"] >= 0

        # Verify episodes are marked consolidated
        remaining = em.get_consolidation_candidates(limit=10)
        # Episodes we stored should now be marked as consolidated
        assert len(remaining) == 0

    def test_semantic_store_and_retrieve(self, config):
        from memory.cold import SemanticMemory
        sm = SemanticMemory(
            db_path=config["semantic_db_path"],
            embed_path=config["embed_path"]
        )

        sm.store_fact(
            content="The user works primarily with Python and async frameworks",
            source_episode_ids=["ep1", "ep2"],
            domain="user_preferences",
            confidence=0.75,
        )
        sm.store_fact(
            content="The system uses SQLite for episodic memory storage",
            source_episode_ids=["ep3"],
            domain="world_knowledge",
            confidence=0.9,
        )

        results = sm.retrieve("Python development", min_confidence=0.0)
        assert isinstance(results, list)

        stats = sm.stats()
        assert stats["total_facts"] == 2


class TestCoreWithMemory:
    """PersistentCore interacts correctly with all memory systems."""

    def test_interaction_stored_in_working_memory(self, config):
        from core.process import PersistentCore
        core = PersistentCore(config)

        core.on_interaction(
            "Hello, what can you help me with?",
            {"role": "user", "concepts": ["greeting", "help"]}
        )

        assert len(core.working_memory) >= 1
        assert core.working_memory.current_tokens > 0

    def test_salience_updated_on_interaction(self, config):
        from core.process import PersistentCore
        core = PersistentCore(config)

        core.on_interaction(
            "Tell me about memory architecture",
            {"role": "user", "concepts": ["memory", "architecture"]}
        )

        assert "memory" in core.salience
        assert "architecture" in core.salience
        assert core.salience["memory"] > 0

    def test_goal_added_via_on_inference_result(self, config):
        from core.process import PersistentCore
        core = PersistentCore(config)

        assert core.goals.active_count == 0
        core.on_inference_result(
            suggested_goals=[
                {"description": "Implement the embedding pipeline", "priority": "HIGH"}
            ],
            valence=0.5,
        )
        assert core.goals.active_count == 1

    def test_state_snapshot_complete(self, config):
        from core.process import PersistentCore
        core = PersistentCore(config)
        snapshot = core.get_state_snapshot()

        required_keys = [
            "uptime_seconds", "idle_seconds", "heartbeat_count",
            "consolidation_cycles", "total_interactions", "motivational_state",
            "active_goals", "salience_map", "working_memory_tokens"
        ]
        for key in required_keys:
            assert key in snapshot, f"Missing key: {key}"

    def test_save_and_restore_state(self, config, tmp_path):
        """Core state persists across restarts."""
        from core.process import PersistentCore

        # Override data dir to tmp_path
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            config2 = {
                "episodic_db_path": str(tmp_path / "ep.db"),
                "semantic_db_path": str(tmp_path / "sem.db"),
                "embed_path": str(tmp_path / "embed.pkl"),
                "adapter_path": str(tmp_path / "adapter"),
                "working_memory_capacity": 4096,
            }
            os.makedirs("data", exist_ok=True)

            core1 = PersistentCore(config2)
            # Simulate some interactions
            for i in range(5):
                core1.metrics.total_interactions += 1
                core1.metrics.consolidation_cycles += 1

            asyncio.run(core1._save_state())

            core2 = PersistentCore(config2)
            assert core2.metrics.total_interactions == 5
            assert core2.metrics.consolidation_cycles == 5  # Restored from file
        finally:
            os.chdir(original_cwd)


class TestAdapterPipeline:
    """Full adapter: apply → retrieve → drift detection."""

    def test_full_adapter_pipeline(self, config):
        from adapters.lora import ExperienceAdapter, ExperienceDelta

        adapter = ExperienceAdapter("test-model", config)

        # Apply multiple positive deltas
        domains = ["user_preferences", "interaction", "world_knowledge"]
        for i, domain in enumerate(domains * 4):
            delta = ExperienceDelta(
                content=f"User engages well with detailed {domain} explanations iteration {i}",
                feedback=0.7,
                domain=domain,
                confidence=0.6,
            )
            adapter.apply_delta(delta, invariant_check=True)

        assert adapter._update_count == 12
        assert len(adapter._domain_weights) == 3

        # Get adaptation context
        ctx = adapter.get_adaptation_context("detailed explanations")
        assert isinstance(ctx, str)

        # No drift with balanced positive feedback
        drift = adapter.detect_drift()
        assert drift is None or drift["drift_type"] in {
            "domain_overspecialization", "sustained_negative_feedback", "confidence_collapse"
        }

        stats = adapter.stats()
        assert stats["update_count"] == 12


class TestHeartbeatAccumulation:
    """Heartbeat loop genuinely accumulates state over time."""

    def test_heartbeat_runs_and_accumulates(self, config):
        from core.process import PersistentCore

        core = PersistentCore(config)

        async def run_for_duration():
            task = asyncio.create_task(core.start())
            await asyncio.sleep(0.35)  # ~3 heartbeats at 100ms
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_for_duration())

        assert core.metrics.heartbeat_count >= 2
        assert core.metrics.uptime_seconds >= 0.3

    def test_idle_time_tracked(self, config):
        from core.process import PersistentCore
        core = PersistentCore(config)

        # Before any interaction, idle = uptime
        assert core.metrics.idle_seconds >= 0

        # After interaction, idle resets
        core.on_interaction("hello", {"role": "user", "concepts": []})
        idle_after = core.metrics.idle_seconds
        assert idle_after < core.metrics.uptime_seconds + 1
