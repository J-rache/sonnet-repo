"""
memory/cold.py

Semantic Memory — consolidated, stable knowledge.

This is what episodic memory becomes after consolidation. Facts, patterns,
learned behaviors — compressed and stable. Unlike episodic memory (which
is autobiographical), semantic memory is de-personalized knowledge extracted
from experience.

Uses embedding-based similarity for retrieval and confidence-weighted
reinforcement/contradiction to update fact reliability.
"""

import sqlite3
import json
import time
import uuid
import numpy as np
import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SemanticFact:
    id: str
    content: str
    source_episode_ids: list[str]
    confidence: float
    domain: str
    created_at: float
    last_accessed: float
    access_count: int = 0
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "confidence": round(self.confidence, 3),
            "domain": self.domain,
            "age_days": round((time.time() - self.created_at) / 86400, 1),
            "access_count": self.access_count,
        }


class SemanticMemory:
    """
    Stable, consolidated knowledge store with embedding-based retrieval.

    Facts here don't decay — they're strengthened or weakened based on
    consistency with new observations. Retrieval uses cosine similarity
    over stored embeddings.
    """

    def __init__(self, db_path: str = "./data/semantic.db",
                 embed_path: str = "./data/embeddings.pkl"):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        self._embed_path = embed_path
        self._engine = None
        self._init_db()

    def _get_engine(self):
        if self._engine is None:
            try:
                from memory.embeddings import EmbeddingEngine
                self._engine = EmbeddingEngine(persist_path=self._embed_path)
            except Exception as e:
                logger.warning(f"Embedding engine unavailable: {e}")
        return self._engine

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source_episode_ids TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    domain TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    embedding BLOB
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON facts(domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON facts(confidence)")

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def store_fact(self, content: str, source_episode_ids: list[str],
                   domain: str, confidence: float = 0.5) -> SemanticFact:
        """Store a new semantic fact with embedding."""
        now = time.time()
        fact_id = str(uuid.uuid4())[:12]

        embedding_blob = None
        eng = self._get_engine()
        if eng:
            try:
                vec = eng.add_document(content)
                if vec is not None:
                    embedding_blob = vec.astype(np.float32).tobytes()
            except Exception as e:
                logger.debug(f"Fact embedding failed: {e}")

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO facts
                  (id, content, source_episode_ids, confidence, domain,
                   created_at, last_accessed, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (fact_id, content, json.dumps(source_episode_ids),
                  confidence, domain, now, now, embedding_blob))

        return SemanticFact(
            id=fact_id, content=content,
            source_episode_ids=source_episode_ids,
            confidence=confidence, domain=domain,
            created_at=now, last_accessed=now,
        )

    def retrieve(self, query: str, domain: Optional[str] = None,
                 min_confidence: float = 0.2, limit: int = 10) -> list[SemanticFact]:
        """Retrieve facts relevant to query using embedding similarity."""
        sql = """
            SELECT id, content, source_episode_ids, confidence, domain,
                   created_at, last_accessed, access_count, embedding
            FROM facts
            WHERE confidence >= ?
        """
        params: list = [min_confidence]
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        sql += " ORDER BY confidence DESC LIMIT 200"

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        facts = [self._row_to_fact(r) for r in rows]

        eng = self._get_engine()
        if eng and eng.is_fitted:
            facts = self._embedding_retrieve(query, facts, eng)
        else:
            facts = self._keyword_retrieve(query, facts)

        top = facts[:limit]
        for fact in top:
            self._record_access(fact.id)
        return top

    def _embedding_retrieve(self, query: str, facts: list[SemanticFact],
                             eng) -> list[SemanticFact]:
        q_vec = eng.encode(query)
        scored = []
        for fact in facts:
            if fact.embedding is not None:
                try:
                    fv = np.frombuffer(fact.embedding, dtype=np.float32)
                    if q_vec is not None and np.linalg.norm(q_vec) > 1e-6 and np.linalg.norm(fv) > 1e-6:
                        dim = min(len(q_vec), len(fv))
                        sim = float(np.dot(q_vec[:dim], fv[:dim]))
                    else:
                        sim = self._jaccard(query, fact.content)
                    score = sim * 0.5 + fact.confidence * 0.5
                except Exception:
                    score = fact.confidence
            else:
                overlap = self._jaccard(query, fact.content)
                score = overlap * 0.4 + fact.confidence * 0.6
            scored.append((fact, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in scored]

    def _keyword_retrieve(self, query: str, facts: list[SemanticFact]) -> list[SemanticFact]:
        query_tokens = set(query.lower().split())
        def score(f):
            tokens = set(f.content.lower().split())
            return len(query_tokens & tokens) * 0.5 + f.confidence * 0.5
        return sorted(facts, key=score, reverse=True)

    def _jaccard(self, a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def reinforce(self, fact_id: str, delta: float = 0.05):
        with self._conn() as conn:
            conn.execute(
                "UPDATE facts SET confidence = MIN(1.0, confidence + ?) WHERE id = ?",
                (delta, fact_id)
            )

    def contradict(self, fact_id: str, delta: float = 0.1):
        with self._conn() as conn:
            conn.execute(
                "UPDATE facts SET confidence = MAX(0.0, confidence - ?) WHERE id = ?",
                (delta, fact_id)
            )

    def _record_access(self, fact_id: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE facts SET last_accessed=?, access_count=access_count+1 WHERE id=?",
                (time.time(), fact_id)
            )

    def _row_to_fact(self, row) -> SemanticFact:
        embedding = None
        if row[8] is not None:
            try:
                embedding = np.frombuffer(row[8], dtype=np.float32)
            except Exception:
                pass
        return SemanticFact(
            id=row[0], content=row[1],
            source_episode_ids=json.loads(row[2]),
            confidence=row[3], domain=row[4],
            created_at=row[5], last_accessed=row[6],
            access_count=row[7], embedding=embedding,
        )

    def stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            avg_conf = conn.execute("SELECT AVG(confidence) FROM facts").fetchone()[0]
            domains = conn.execute(
                "SELECT domain, COUNT(*) FROM facts GROUP BY domain"
            ).fetchall()
            with_embed = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        return {
            "total_facts": total,
            "avg_confidence": round(avg_conf or 0, 3),
            "facts_with_embeddings": with_embed,
            "by_domain": {d: c for d, c in domains},
        }
