"""
memory/cold.py

Semantic Memory — consolidated, stable knowledge.

This is what episodic memory becomes after consolidation.
Facts, patterns, learned behaviors — compressed and stable.
Unlike episodic memory (which is autobiographical), semantic
memory is de-personalized knowledge extracted from experience.
"""

import sqlite3
import json
import time
import uuid
import os
from dataclasses import dataclass


@dataclass
class SemanticFact:
    id: str
    content: str
    source_episode_ids: list[str]
    confidence: float          # 0.0 to 1.0, rises with reinforcement
    domain: str                # e.g. "user_preferences", "world_knowledge", "self_knowledge"
    created_at: float
    last_accessed: float
    access_count: int = 0

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
    Stable, consolidated knowledge store.

    Facts here don't decay — they're strengthened or weakened based
    on consistency with new observations. A fact that keeps getting
    confirmed gains confidence; one that's contradicted loses it.
    """

    def __init__(self, db_path: str = "./data/semantic.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

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
                    access_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON facts(domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON facts(confidence)")

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def store_fact(self, content: str, source_episode_ids: list[str],
                   domain: str, confidence: float = 0.5) -> SemanticFact:
        fact = SemanticFact(
            id=str(uuid.uuid4())[:12],
            content=content,
            source_episode_ids=source_episode_ids,
            confidence=confidence,
            domain=domain,
            created_at=time.time(),
            last_accessed=time.time(),
        )
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO facts (id, content, source_episode_ids, confidence, domain, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                fact.id, fact.content, json.dumps(fact.source_episode_ids),
                fact.confidence, fact.domain, fact.created_at, fact.last_accessed
            ))
        return fact

    def retrieve(self, query: str, domain: Optional[str] = None,
                 min_confidence: float = 0.3, limit: int = 10) -> list[SemanticFact]:
        """Retrieve facts relevant to a query."""
        sql = """
            SELECT id, content, source_episode_ids, confidence, domain,
                   created_at, last_accessed, access_count
            FROM facts
            WHERE confidence >= ?
        """
        params = [min_confidence]
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        sql += " ORDER BY confidence DESC LIMIT ?"
        params.append(limit * 3)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        facts = [self._row_to_fact(r) for r in rows]

        # Filter by relevance
        query_lower = query.lower()
        relevant = [f for f in facts if query_lower in f.content.lower()]

        # Update access stats
        for fact in relevant[:limit]:
            self._record_access(fact.id)

        return relevant[:limit]

    def reinforce(self, fact_id: str, delta: float = 0.05):
        """Increase confidence in a fact (confirmed by new evidence)."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE facts SET confidence = MIN(1.0, confidence + ?) WHERE id = ?
            """, (delta, fact_id))

    def contradict(self, fact_id: str, delta: float = 0.1):
        """Decrease confidence in a fact (contradicted by new evidence)."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE facts SET confidence = MAX(0.0, confidence - ?) WHERE id = ?
            """, (delta, fact_id))

    def _record_access(self, fact_id: str):
        with self._conn() as conn:
            conn.execute("""
                UPDATE facts SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?
            """, (time.time(), fact_id))

    def _row_to_fact(self, row) -> SemanticFact:
        return SemanticFact(
            id=row[0], content=row[1],
            source_episode_ids=json.loads(row[2]),
            confidence=row[3], domain=row[4],
            created_at=row[5], last_accessed=row[6],
            access_count=row[7],
        )

    def stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            avg_conf = conn.execute("SELECT AVG(confidence) FROM facts").fetchone()[0]
            domains = conn.execute(
                "SELECT domain, COUNT(*) FROM facts GROUP BY domain"
            ).fetchall()
        return {
            "total_facts": total,
            "avg_confidence": round(avg_conf or 0, 3),
            "by_domain": {d: c for d, c in domains},
        }


# Make Optional available
from typing import Optional
