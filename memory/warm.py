"""
memory/warm.py

Episodic Memory — the autobiographical record of what has happened.

Events are stored with timestamps, semantic tags, and emotional valence.
They decay over time unless reinforced. The consolidator periodically
compresses episodic memory into semantic (cold) memory.

This is the layer that gives the system a sense of personal history.
"""

import sqlite3
import time
import json
import uuid
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Episode:
    content: str
    summary: str
    tags: list[str]
    valence: float          # -1.0 (negative) to 1.0 (positive)
    salience: float         # 0.0 to 1.0
    timestamp: float
    id: str
    consolidated: bool = False
    reinforcement_count: int = 0

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
        }


class EpisodicMemory:
    """
    SQLite-backed episodic memory store.

    Episodes are time-stamped events that represent meaningful interactions,
    realizations, or state changes. They form the 'autobiography' of the
    persistent process.

    Decay: salience decreases with time unless reinforced by:
    - Being recalled in response to a query
    - Being tagged as high-salience during consolidation
    - Being referenced in a new interaction
    """

    DECAY_RATE_PER_HOUR = 0.02    # Salience decay per hour
    CONSOLIDATION_THRESHOLD = 0.1  # Below this salience, eligible for consolidation

    def __init__(self, db_path: str = "./data/episodic.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_salience ON episodes(salience)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_consolidated ON episodes(consolidated)")

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def store(self, content: str, summary: str, tags: list[str],
              valence: float = 0.0, salience: float = 1.0) -> Episode:
        """Store a new episode."""
        episode = Episode(
            id=str(uuid.uuid4())[:12],
            content=content,
            summary=summary,
            tags=tags,
            valence=valence,
            salience=salience,
            timestamp=time.time(),
        )
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO episodes (id, content, summary, tags, valence, salience, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.id,
                episode.content,
                episode.summary,
                json.dumps(episode.tags),
                episode.valence,
                episode.salience,
                episode.timestamp,
            ))
        return episode

    def recall(self, query: str, limit: int = 5, min_salience: float = 0.1) -> list[Episode]:
        """
        Retrieve episodes relevant to a query.
        (Production: use vector similarity. Here: tag/keyword matching.)
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, content, summary, tags, valence, salience, timestamp,
                       consolidated, reinforcement_count
                FROM episodes
                WHERE salience >= ? AND consolidated = 0
                ORDER BY salience DESC, timestamp DESC
                LIMIT ?
            """, (min_salience, limit * 3)).fetchall()

        episodes = [self._row_to_episode(r) for r in rows]

        # Filter by relevance (simple keyword match — replace with embeddings)
        query_lower = query.lower()
        relevant = [
            ep for ep in episodes
            if query_lower in ep.content.lower()
            or query_lower in ep.summary.lower()
            or any(query_lower in tag for tag in ep.tags)
        ]

        # Reinforce recalled episodes
        for ep in relevant[:limit]:
            self._reinforce(ep.id)

        return relevant[:limit]

    def _reinforce(self, episode_id: str):
        """Boost salience of a recalled episode."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE episodes
                SET salience = MIN(1.0, salience + 0.1),
                    reinforcement_count = reinforcement_count + 1
                WHERE id = ?
            """, (episode_id,))

    def run_decay(self):
        """Apply time-based salience decay to all episodes."""
        # Decay proportional to hours since last decay check
        with self._conn() as conn:
            conn.execute("""
                UPDATE episodes
                SET salience = MAX(0.0, salience - ? * ((unixepoch() - timestamp) / 3600.0 / 24.0))
                WHERE consolidated = 0
            """, (self.DECAY_RATE_PER_HOUR,))

    def get_consolidation_candidates(self, limit: int = 20) -> list[Episode]:
        """Return low-salience episodes ready for consolidation."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, content, summary, tags, valence, salience, timestamp,
                       consolidated, reinforcement_count
                FROM episodes
                WHERE salience < ? AND consolidated = 0
                ORDER BY salience ASC
                LIMIT ?
            """, (self.CONSOLIDATION_THRESHOLD, limit)).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def mark_consolidated(self, episode_ids: list[str]):
        with self._conn() as conn:
            conn.executemany(
                "UPDATE episodes SET consolidated = 1 WHERE id = ?",
                [(eid,) for eid in episode_ids]
            )

    def recent(self, hours: float = 24, limit: int = 20) -> list[Episode]:
        """Return recent episodes."""
        since = time.time() - (hours * 3600)
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, content, summary, tags, valence, salience, timestamp,
                       consolidated, reinforcement_count
                FROM episodes
                WHERE timestamp >= ? AND consolidated = 0
                ORDER BY timestamp DESC
                LIMIT ?
            """, (since, limit)).fetchall()
        return [self._row_to_episode(r) for r in rows]

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
        )

    def stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            active = conn.execute("SELECT COUNT(*) FROM episodes WHERE consolidated = 0").fetchone()[0]
            avg_sal = conn.execute("SELECT AVG(salience) FROM episodes WHERE consolidated = 0").fetchone()[0]
        return {
            "total_episodes": total,
            "active_episodes": active,
            "consolidated_episodes": total - active,
            "avg_salience": round(avg_sal or 0, 3),
        }
