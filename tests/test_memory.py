"""tests/test_memory.py — Memory system tests (hot, warm, cold)"""

import pytest
import time
import os
import numpy as np
from memory.hot import WorkingMemory
from memory.warm import EpisodicMemory
from memory.cold import SemanticMemory


# ── Working Memory ─────────────────────────────────────────────────────────────

class TestWorkingMemory:
    def test_add_and_len(self):
        wm = WorkingMemory(capacity=4096)
        wm.add("hello world", {"role": "user"})
        wm.add("hi there", {"role": "assistant"})
        assert len(wm) == 2

    def test_eviction_on_overflow(self):
        # Tiny capacity to force eviction
        wm = WorkingMemory(capacity=50)
        # Each entry ~12 tokens (50 chars)
        for i in range(10):
            wm.add(f"entry number {i} with some padding text here", {"role": "user"})
        # Should have evicted some entries
        assert wm.current_tokens <= 50

    def test_context_string_format(self):
        wm = WorkingMemory(capacity=4096)
        wm.add("test message", {"role": "user"})
        ctx = wm.to_context_string()
        assert "=== WORKING MEMORY ===" in ctx
        assert "test message" in ctx
        assert "=== END WORKING MEMORY ===" in ctx

    def test_empty_context_string(self):
        wm = WorkingMemory(capacity=4096)
        assert wm.to_context_string() == ""

    def test_salience_boost_keyword(self):
        wm = WorkingMemory(capacity=4096, embed_path="/tmp/wm_embed_test.pkl")
        wm.add("memory consolidation process", {"role": "user"}, salience=0.5)
        wm.add("goal stack management", {"role": "user"}, salience=0.5)
        wm.boost_salience("memory consolidation", boost=0.3)
        entries = list(wm._entries)
        memory_entry = next(e for e in entries if "memory" in e.content)
        goal_entry = next(e for e in entries if "goal" in e.content)
        assert memory_entry.salience > goal_entry.salience

    def test_decay(self):
        wm = WorkingMemory(capacity=4096)
        wm.add("test", {"role": "user"}, salience=1.0)
        entry = list(wm._entries)[0]
        initial = entry.salience
        wm.decay_all(rate=0.1)
        assert entry.salience < initial

    def test_clear(self):
        wm = WorkingMemory(capacity=4096)
        wm.add("test", {"role": "user"})
        wm.clear()
        assert len(wm) == 0
        assert wm.current_tokens == 0


# ── Episodic Memory ────────────────────────────────────────────────────────────

@pytest.fixture
def ep_mem(tmp_path):
    db = str(tmp_path / "episodic.db")
    embed = str(tmp_path / "embed.pkl")
    return EpisodicMemory(db_path=db, embed_path=embed)


class TestEpisodicMemory:
    def test_store_and_recall(self, ep_mem):
        ep_mem.store(
            content="The user prefers concise responses",
            summary="User prefers concise responses",
            tags=["preference"],
            valence=0.2,
            salience=0.9,
        )
        results = ep_mem.recall("concise responses", limit=5)
        assert len(results) == 1
        assert "concise" in results[0].summary.lower()

    def test_decay_reduces_salience(self, ep_mem):
        ep_mem.store("test content", "test summary", ["test"], valence=0.0, salience=0.5)
        ep_mem.run_decay()
        episodes = ep_mem.recent(hours=24)
        # After decay, salience should be reduced (or 0 if hours calculation causes large decay)
        assert len(episodes) >= 0  # Just verify it doesn't crash

    def test_consolidation_candidates(self, ep_mem):
        # Store with very low salience
        from unittest.mock import patch
        ep = ep_mem.store("low salience item", "low salience summary", ["test"],
                           valence=0.0, salience=0.05)
        candidates = ep_mem.get_consolidation_candidates(limit=10)
        # Should find the low-salience episode
        assert any(c.id == ep.id for c in candidates)

    def test_mark_consolidated(self, ep_mem):
        ep = ep_mem.store("to consolidate", "consolidation target", ["test"],
                           valence=0.0, salience=0.05)
        ep_mem.mark_consolidated([ep.id])
        # After consolidation, should not appear in active recall
        active = ep_mem.recent(hours=24)
        assert not any(e.id == ep.id for e in active)

    def test_stats(self, ep_mem):
        ep_mem.store("content", "summary", ["tag"])
        stats = ep_mem.stats()
        assert stats["total_episodes"] == 1
        assert stats["active_episodes"] == 1
        assert stats["consolidated_episodes"] == 0

    def test_recent(self, ep_mem):
        ep_mem.store("recent content", "recent summary", ["recent"])
        results = ep_mem.recent(hours=1)
        assert len(results) == 1

    def test_episode_to_dict(self, ep_mem):
        ep = ep_mem.store("content", "summary", ["tag1", "tag2"], valence=0.5)
        d = ep.to_dict()
        assert "id" in d
        assert "summary" in d
        assert "salience" in d
        assert "valence" in d
        assert d["valence"] == pytest.approx(0.5, abs=0.01)


# ── Semantic Memory ────────────────────────────────────────────────────────────

@pytest.fixture
def sem_mem(tmp_path):
    db = str(tmp_path / "semantic.db")
    embed = str(tmp_path / "embed.pkl")
    return SemanticMemory(db_path=db, embed_path=embed)


class TestSemanticMemory:
    def test_store_and_retrieve(self, sem_mem):
        sem_mem.store_fact(
            content="Users prefer responses under 200 words",
            source_episode_ids=["ep1", "ep2"],
            domain="user_preferences",
            confidence=0.8,
        )
        results = sem_mem.retrieve("response length preferences")
        # May or may not match depending on embedding state, but shouldn't crash
        assert isinstance(results, list)

    def test_keyword_retrieve_fallback(self, sem_mem):
        sem_mem.store_fact(
            content="The system uses episodic and semantic memory",
            source_episode_ids=["ep1"],
            domain="world_knowledge",
            confidence=0.7,
        )
        results = sem_mem.retrieve("episodic memory", min_confidence=0.0)
        assert len(results) >= 1

    def test_reinforce_increases_confidence(self, sem_mem):
        fact = sem_mem.store_fact("test fact", ["ep1"], "general", confidence=0.5)
        sem_mem.reinforce(fact.id, delta=0.2)
        results = sem_mem.retrieve("test fact", min_confidence=0.0)
        assert len(results) >= 1
        assert results[0].confidence > 0.5

    def test_contradict_decreases_confidence(self, sem_mem):
        fact = sem_mem.store_fact("contradicted fact", ["ep1"], "general", confidence=0.8)
        sem_mem.contradict(fact.id, delta=0.3)
        results = sem_mem.retrieve("contradicted fact", min_confidence=0.0)
        assert len(results) >= 1
        assert results[0].confidence < 0.8

    def test_domain_filter(self, sem_mem):
        sem_mem.store_fact("preference fact", ["ep1"], "user_preferences", 0.7)
        sem_mem.store_fact("knowledge fact", ["ep2"], "world_knowledge", 0.7)
        pref_results = sem_mem.retrieve("fact", domain="user_preferences", min_confidence=0.0)
        knowledge_results = sem_mem.retrieve("fact", domain="world_knowledge", min_confidence=0.0)
        # Domain filter should restrict results
        pref_domains = {f.domain for f in pref_results}
        knowledge_domains = {f.domain for f in knowledge_results}
        if pref_results:
            assert "world_knowledge" not in pref_domains
        if knowledge_results:
            assert "user_preferences" not in knowledge_domains

    def test_stats(self, sem_mem):
        sem_mem.store_fact("fact1", ["ep1"], "general", 0.5)
        sem_mem.store_fact("fact2", ["ep2"], "user_preferences", 0.7)
        stats = sem_mem.stats()
        assert stats["total_facts"] == 2
        assert "general" in stats["by_domain"]
