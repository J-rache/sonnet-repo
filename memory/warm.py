"""
memory/warm.py

Episodic memory stores the autobiographical event record. Episodes are
timestamped, tagged, salience-ranked, and retrievable through local vector
similarity.
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
class Episode:
    content: str
    summary: str
    tags: list[str]
    valence: float
    salience: float
    timestamp: float
    id: str
    consolidated: bool = False
    reinforcement_count: int = 0
    retrieval_score: float = 0.0
    embedding: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "summary": self.summary,
            "tags": self.tags,
            "valence": self.valence,
            "salience": self.salience,
            "timestamp": self.timestamp,
            "age_hours": round((time.time() - self.timestamp) / 3600, 1),
            "reinforcement_count": self.reinforcement_count,
            "consolidated": self.consolidated,
            "retrieval_score": round(self.retrieval_score, 4),
        }


class EpisodicMemory:
    """SQLite-backed episodic memory store with persisted embeddings."""

    DECAY_RATE_PER_HOUR = 0.02
    CONSOLIDATION_THRESHOLD = 0.1

    def __init__(self, db_path: str = "./data/episodic.db", embedding_dimensions: int = 128):
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.db_path = db_path
        self.embedder = HashingTextEmbedder(embedding_dimensions)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    valence REAL DEFAULT 0.0,
                    salience REAL DEFAULT 1.0,
                    timestamp REAL NOT NULL,
                    consolidated INTEGER DEFAULT 0,
                    reinforcement_count INTEGER DEFAULT 0,
                    created_at REAL DEFAULT (unixepoch())
                )
            """)
            self._ensure_column(conn, "episodes", "embedding", "TEXT")
            self._ensure_column(conn, "episodes", "embedding_model", "TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_salience ON episodes(salience)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_consolidated ON episodes(consolidated)")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str):
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def store(
        self,
        content: str,
        summary: str,
        tags: list[str],
        valence: float = 0.0,
        salience: float = 1.0,
    ) -> Episode:
        embedding = self.embedder.embed(self._embedding_text(content, summary, tags))
        episode = Episode(
            id=str(uuid.uuid4())[:12],
            content=content,
            summary=summary,
            tags=tags,
            valence=valence,
            salience=salience,
            timestamp=time.time(),
            embedding=embedding,
        )
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO episodes (
                    id, content, summary, tags, valence, salience, timestamp,
                    embedding, embedding_model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.id,
                episode.content,
                episode.summary,
                json.dumps(episode.tags),
                episode.valence,
                episode.salience,
                episode.timestamp,
                vector_to_json(embedding),
                self.embedder.model_id,
            ))
        return episode

    def recall(self, query: str, limit: int = 5, min_salience: float = 0.1) -> list[Episode]:
        """Retrieve episodes by vector similarity, salience, and recency."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, content, summary, tags, valence, salience, timestamp,
                       consolidated, reinforcement_count, embedding
                FROM episodes
                WHERE salience >= ? AND consolidated = 0
                ORDER BY salience DESC, timestamp DESC
                LIMIT ?
            """, (min_salience, max(limit * 10, 25))).fetchall()

        episodes = [self._row_to_episode(row) for row in rows]
        if not episodes:
            return []

        query_embedding = self.embedder.embed(query)
        now = time.time()
        scored: list[Episode] = []
        for episode in episodes:
            if not episode.embedding:
                episode.embedding = self.embedder.embed(
                    self._embedding_text(episode.content, episode.summary, episode.tags)
                )
                self._persist_embedding(episode)

            similarity = cosine_similarity(query_embedding, episode.embedding) if query.strip() else 1.0
            if similarity <= 0 and query.strip():
                continue

            age_hours = max((now - episode.timestamp) / 3600, 0.0)
            recency = 1 / (1 + age_hours)
            episode.retrieval_score = (similarity * 0.65) + (episode.salience * 0.25) + (recency * 0.10)
            scored.append(episode)

        scored.sort(key=lambda episode: (episode.retrieval_score, episode.salience, episode.timestamp), reverse=True)
        selected = scored[:limit]

        for episode in selected:
            self._reinforce(episode.id)

        return selected

    def _reinforce(self, episode_id: str):
        with self._conn() as conn:
            conn.execute("""
                UPDATE episodes
                SET salience = MIN(1.0, salience + 0.1),
                    reinforcement_count = reinforcement_count + 1
                WHERE id = ?
            """, (episode_id,))

    def run_decay(self):
        with self._conn() as conn:
            conn.execute("""
                UPDATE episodes
                SET salience = MAX(0.0, salience - ? * ((unixepoch() - timestamp) / 3600.0 / 24.0))
                WHERE consolidated = 0
            """, (self.DECAY_RATE_PER_HOUR,))

    def get_consolidation_candidates(self, limit: int = 20) -> list[Episode]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, content, summary, tags, valence, salience, timestamp,
                       consolidated, reinforcement_count, embedding
                FROM episodes
                WHERE salience < ? AND consolidated = 0
                ORDER BY salience ASC
                LIMIT ?
            """, (self.CONSOLIDATION_THRESHOLD, limit)).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def mark_consolidated(self, episode_ids: list[str]):
        with self._conn() as conn:
            conn.executemany(
                "UPDATE episodes SET consolidated = 1 WHERE id = ?",
                [(episode_id,) for episode_id in episode_ids],
            )

    def recent(self, hours: float = 24, limit: int = 20) -> list[Episode]:
        since = time.time() - (hours * 3600)
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, content, summary, tags, valence, salience, timestamp,
                       consolidated, reinforcement_count, embedding
                FROM episodes
                WHERE timestamp >= ? AND consolidated = 0
                ORDER BY timestamp DESC
                LIMIT ?
            """, (since, limit)).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def _persist_embedding(self, episode: Episode):
        with self._conn() as conn:
            conn.execute("""
                UPDATE episodes SET embedding = ?, embedding_model = ? WHERE id = ?
            """, (vector_to_json(episode.embedding), self.embedder.model_id, episode.id))

    def _row_to_episode(self, row) -> Episode:
        return Episode(
            id=row[0],
            content=row[1],
            summary=row[2],
            tags=json.loads(row[3]),
            valence=row[4],
            salience=row[5],
            timestamp=row[6],
            consolidated=bool(row[7]),
            reinforcement_count=row[8],
            embedding=vector_from_json(row[9]),
        )

    def _embedding_text(self, content: str, summary: str, tags: list[str]) -> str:
        return f"{' '.join(tags)} {summary} {content}"

    def stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            active = conn.execute("SELECT COUNT(*) FROM episodes WHERE consolidated = 0").fetchone()[0]
            avg_sal = conn.execute("SELECT AVG(salience) FROM episodes WHERE consolidated = 0").fetchone()[0]
            embedded = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE embedding IS NOT NULL AND embedding != ''"
            ).fetchone()[0]
        return {
            "total_episodes": total,
            "active_episodes": active,
            "consolidated_episodes": total - active,
            "avg_salience": round(avg_sal or 0, 3),
            "embedded_episodes": embedded,
            "embedding": self.embedder.metadata(),
        }
