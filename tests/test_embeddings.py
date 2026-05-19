"""tests/test_embeddings.py — Real embedding engine tests"""

import pytest
import numpy as np
import os
import tempfile
from memory.embeddings import EmbeddingEngine


@pytest.fixture
def engine():
    path = "/tmp/pnp_test_embed.pkl"
    if os.path.exists(path):
        os.remove(path)
    eng = EmbeddingEngine(persist_path=path)
    # Seed with enough docs to fit
    docs = [
        "the persistent neural process runs continuously without interruption",
        "memory consolidation happens during idle periods similar to sleep",
        "the inference engine handles heavy computation tasks on demand",
        "episodic memory stores biographical events with timestamps and salience scores",
        "semantic memory contains consolidated facts extracted from raw experience",
        "the constitutional invariant layer anchors identity during self modification",
        "goals persist across multiple conversations and user interactions",
        "arousal and curiosity drive the motivational state of the core process",
        "the heartbeat loop maintains continuous operation at low compute cost",
        "experience adapter accumulates learning without catastrophic forgetting",
    ]
    for d in docs:
        eng.add_document(d)
    yield eng
    if os.path.exists(path):
        os.remove(path)


def test_engine_fits(engine):
    assert engine.is_fitted
    assert engine.corpus_size == 10


def test_encode_returns_unit_vector(engine):
    vec = engine.encode("memory consolidation")
    assert vec is not None
    assert vec.shape == (128,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_similarity_ordering(engine):
    """Memory-related texts should be more similar to each other than to unrelated texts."""
    sim_mem = engine.similarity(
        "memory consolidation happens during idle periods similar to sleep",
        "episodic memory stores biographical events with timestamps and salience scores"
    )
    sim_cross = engine.similarity(
        "memory consolidation happens during idle periods similar to sleep",
        "goals persist across multiple conversations and user interactions"
    )
    assert sim_mem > sim_cross, (
        f"Memory-to-memory ({sim_mem:.4f}) should exceed memory-to-goals ({sim_cross:.4f})"
    )


def test_most_similar_returns_results(engine):
    docs = [
        "the persistent neural process runs continuously without interruption",
        "memory consolidation happens during idle periods similar to sleep",
        "the inference engine handles heavy computation tasks on demand",
    ]
    results = engine.most_similar(docs[1], docs, top_k=3, threshold=0.0)
    assert len(results) > 0
    # Top result should be self-match
    assert results[0][0] == 1


def test_oov_query_uses_jaccard(engine):
    """Out-of-vocabulary queries should fall back to Jaccard and still return results."""
    docs = ["heartbeat loop", "memory consolidation", "goal stack management"]
    results = engine.most_similar("heartbeat continuous loop", docs, top_k=3, threshold=0.0)
    assert len(results) > 0
    # Heartbeat doc should rank first via Jaccard
    assert results[0][0] == 0


def test_persistence(engine):
    path = "/tmp/pnp_persist_test.pkl"
    if os.path.exists(path):
        os.remove(path)

    eng1 = EmbeddingEngine(persist_path=path)
    docs = [
        "the persistent neural process runs continuously",
        "memory consolidation happens during idle periods",
        "the inference engine handles computation",
        "episodic memory stores events with timestamps",
        "semantic memory contains consolidated facts",
    ]
    for d in docs:
        eng1.add_document(d)
    eng1._save()
    assert os.path.getsize(path) > 500

    eng2 = EmbeddingEngine(persist_path=path)
    assert eng2.is_fitted
    vec = eng2.encode("memory consolidation")
    assert vec is not None and vec.shape == (128,)

    if os.path.exists(path):
        os.remove(path)


def test_stats(engine):
    stats = engine.stats()
    assert stats["fitted"] is True
    assert stats["corpus_size"] == 10
    assert stats["vector_dim"] == 128
