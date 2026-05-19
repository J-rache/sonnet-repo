"""
memory/cold.py

Semantic memory stores consolidated, stable knowledge. Retrieval uses local
vector similarity over persisted embeddings, with confidence as a secondary
ranking signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import sqlite3
import time
from typing import Optional
import uuid

from memory.embedding import HashingTextEmbedder, cosine_similarity, vector_from_json, vector_to_json


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
    retrieval_score: float = 0.0
    embedding: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "confidence": round(self.confidence, 3),
            "domain": self.domain,
            "age_days": round((time.time() - self.created_at) / 86400, 1),
            "access_count": self.access_count,
            "retrieval_score": round(self.retrieval_score, 4),
        }


class SemanticMemory:
    """SQLite-backed semantic memory with persisted local embeddings."""

    def __init__(self, db_path: str = "./data/semantic.db", embedding_dimensions: int = 128):
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.db_path = db_path
        self.embedder = HashingTextEmbedder(embedding_dimensions)
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
            self._ensure_column(conn, "facts", "embedding", "TEXT")
            self._ensure_column(conn, "facts", "embedding_model", "TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON facts(domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON facts(confidence)")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str):
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def store_fact(
        self,
        content: str,
        source_episode_ids: list[str],
        domain: str,
        confidence: float = 0.5,
    ) -> SemanticFact:
        embedding = self.embedder.embed(self._embedding_text(content, domain))
        fact = SemanticFact(
            id=str(uuid.uuid4())[:12],
            content=content,
            source_episode_ids=source_episode_ids,
            confidence=confidence,
            domain=domain,
            created_at=time.time(),
            last_accessed=time.time(),
            embedding=embedding,
        )
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO facts (
                    id, content, source_episode_ids, confidence, domain, created_at,
                    last_accessed, embedding, embedding_model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact.id,
                fact.content,
                json.dumps(fact.source_episode_ids),
                fact.confidence,
                fact.domain,
                fact.created_at,
                fact.last_accessed,
                vector_to_json(embedding),
                self.embedder.model_id,
            ))
        return fact

    def retrieve(
        self,
        query: str,
        domain: Optional[str] = None,
        min_confidence: float = 0.3,
        limit: int = 10,
    ) -> list[SemanticFact]:
        """Retrieve facts by vector similarity and confidence."""
        sql = """
            SELECT id, content, source_episode_ids, confidence, domain, created_at,
                   last_accessed, access_count, embedding
            FROM facts
            WHERE confidence >= ?
        """
        params: list[object] = [min_confidence]
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        sql += " ORDER BY confidence DESC, last_accessed DESC LIMIT ?"
        params.append(max(limit * 10, 25))

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        facts = [self._row_to_fact(row) for row in rows]
        if not facts:
            return []

        query_embedding = self.embedder.embed(query)
        scored: list[SemanticFact] = []
        for fact in facts:
            if not fact.embedding:
                fact.embedding = self.embedder.embed(self._embedding_text(fact.content, fact.domain))
                self._persist_embedding(fact)

            similarity = cosine_similarity(query_embedding, fact.embedding) if query.strip() else 1.0
            if similarity <= 0 and query.strip():
                continue
            fact.retrieval_score = (similarity * 0.75) + (fact.confidence * 0.25)
            scored.append(fact)

        scored.sort(key=lambda fact: (fact.retrieval_score, fact.confidence, fact.last_accessed), reverse=True)
        selected = scored[:limit]

        for fact in selected:
            self._record_access(fact.id)

        return selected

    def reinforce(self, fact_id: str, delta: float = 0.05):
        with self._conn() as conn:
            conn.execute("""
                UPDATE facts SET confidence = MIN(1.0, confidence + ?) WHERE id = ?
            """, (delta, fact_id))

    def contradict(self, fact_id: str, delta: float = 0.1):
        with self._conn() as conn:
            conn.execute("""
                UPDATE facts SET confidence = MAX(0.0, confidence - ?) WHERE id = ?
            """, (delta, fact_id))

    def _record_access(self, fact_id: str):
        with self._conn() as conn:
            conn.execute("""
                UPDATE facts SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?
            """, (time.time(), fact_id))

    def _persist_embedding(self, fact: SemanticFact):
        with self._conn() as conn:
            conn.execute("""
                UPDATE facts SET embedding = ?, embedding_model = ? WHERE id = ?
            """, (vector_to_json(fact.embedding), self.embedder.model_id, fact.id))

    def _row_to_fact(self, row) -> SemanticFact:
        return SemanticFact(
            id=row[0],
            content=row[1],
            source_episode_ids=json.loads(row[2]),
            confidence=row[3],
            domain=row[4],
            created_at=row[5],
            last_accessed=row[6],
            access_count=row[7],
            embedding=vector_from_json(row[8]),
        )

    def _embedding_text(self, content: str, domain: str) -> str:
        return f"{domain} {content}"

    def stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            avg_conf = conn.execute("SELECT AVG(confidence) FROM facts").fetchone()[0]
            domains = conn.execute("SELECT domain, COUNT(*) FROM facts GROUP BY domain").fetchall()
            embedded = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE embedding IS NOT NULL AND embedding != ''"
            ).fetchone()[0]
        return {
            "total_facts": total,
            "avg_confidence": round(avg_conf or 0, 3),
            "by_domain": {domain: count for domain, count in domains},
            "embedded_facts": embedded,
            "embedding": self.embedder.metadata(),
        }
