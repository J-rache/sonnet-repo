"""
memory/warm.py

Episodic Memory — the autobiographical record of what has happened.

Events are stored with timestamps, semantic tags, emotional valence,
and vector embeddings for real similarity-based recall. They decay
over time unless reinforced. The consolidator periodically compresses
episodic memory into semantic (cold) memory.
"""

import sqlite3
import time
import json
import uuid
import numpy as np
import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


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
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "summary": self.summary,
            "tags": self.tags,
            "valence": round(self.valence, 3),
            "salience": round(self.salience, 3),
            "timestamp": self.timestamp,
            "age_hours": round((time.time() - self.timestamp) / 3600, 1),
            "reinforcement_count": self.reinforcement_count,
            "consolidated": self.consolidated,
        }


class EpisodicMemory:
    """
    SQLite-backed episodic memory store with real embedding-based recall.

    Episodes are time-stamped events representing meaningful interactions,
    realizations, or state changes. They form the autobiography of the
    persistent process.

    Recall uses cosine similarity over stored embeddings when available,
    falling back to tag/keyword matching for episodes without embeddings.
    """

    DECAY_RATE_PER_HOUR = 0.02
    CONSOLIDATION_THRESHOLD = 0.15

    def __init__(self, db_path: str = "./data/episodic.db",
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
                    embedding BLOB
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_salience ON episodes(salience)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_consolidated ON episodes(consolidated)")

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def store(self, content: str, summary: str, tags: list[str],
              valence: float = 0.0, salience: float = 1.0) -> Episode:
        """Store a new episode with optional embedding."""
        episode_id = str(uuid.uuid4())[:12]
        now = time.time()

        # Compute embedding
        embedding_blob = None
        eng = self._get_engine()
        if eng:
            try:
                vec = eng.add_document(summary)
                if vec is not None:
                    embedding_blob = vec.astype(np.float32).tobytes()
            except Exception as e:
                logger.debug(f"Episode embedding failed: {e}")

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO episodes
                  (id, content, summary, tags, valence, salience, timestamp, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (episode_id, content, summary, json.dumps(tags),
                  valence, salience, now, embedding_blob))

        return Episode(
            id=episode_id, content=content, summary=summary,
            tags=tags, valence=valence, salience=salience,
            timestamp=now,
        )

    def recall(self, query: str, limit: int = 5, min_salience: float = 0.05) -> list[Episode]:
        """
        Retrieve episodes most relevant to query using embedding similarity.
        Falls back to tag/keyword matching when embeddings unavailable.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, content, summary, tags, valence, salience, timestamp,
                       consolidated, reinforcement_count, embedding
                FROM episodes
                WHERE salience >= ? AND consolidated = 0
                ORDER BY salience DESC, timestamp DESC
                LIMIT 200
            """, (min_salience,)).fetchall()

        if not rows:
            return []

        episodes = [self._row_to_episode(r) for r in rows]

        # Try embedding-based recall
        eng = self._get_engine()
        if eng and eng.is_fitted:
            scored = self._embedding_recall(query, episodes, eng)
        else:
            scored = self._keyword_recall(query, episodes)

        top = scored[:limit]

        # Reinforce recalled episodes
        ids = [ep.id for ep in top]
        if ids:
            self._reinforce_batch(ids)

        return top

    def _embedding_recall(self, query: str, episodes: list[Episode],
                           eng) -> list[Episode]:
        """Score episodes by embedding similarity to query."""
        # Encode query
        q_vec = eng.encode(query)

        scored = []
        for ep in episodes:
            if ep.embedding is not None:
                try:
                    ev = np.frombuffer(ep.embedding, dtype=np.float32)
                    if q_vec is not None and np.linalg.norm(q_vec) > 1e-6 and np.linalg.norm(ev) > 1e-6:
                        # Ensure same dimension
                        dim = min(len(q_vec), len(ev))
                        sim = float(np.dot(q_vec[:dim], ev[:dim]))
                    else:
                        sim = self._jaccard(query, ep.summary)
                    score = sim * 0.6 + ep.salience * 0.4
                except Exception:
                    score = ep.salience
            else:
                # No embedding stored — use keyword match + salience
                overlap = self._jaccard(query, ep.summary + " " + " ".join(ep.tags))
                score = overlap * 0.6 + ep.salience * 0.4

            scored.append((ep, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored]

    def _keyword_recall(self, query: str, episodes: list[Episode]) -> list[Episode]:
        """Keyword-based recall fallback."""
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        def score(ep):
            text = (ep.content + " " + ep.summary + " " + " ".join(ep.tags)).lower()
            text_tokens = set(text.split())
            overlap = len(query_tokens & text_tokens)
            return overlap * 0.5 + ep.salience * 0.5

        return sorted(episodes, key=score, reverse=True)

    def _jaccard(self, a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _reinforce_batch(self, episode_ids: list[str]):
        with self._conn() as conn:
            conn.executemany("""
                UPDATE episodes
                SET salience = MIN(1.0, salience + 0.08),
                    reinforcement_count = reinforcement_count + 1
                WHERE id = ?
            """, [(eid,) for eid in episode_ids])

    def run_decay(self):
        """Apply time-based salience decay to all active episodes."""
        with self._conn() as conn:
            # Decay based on hours since stored (gentle linear decay)
            conn.execute("""
                UPDATE episodes
                SET salience = MAX(0.0,
                    salience - ? * ((unixepoch() - timestamp) / 3600.0))
                WHERE consolidated = 0
            """, (self.DECAY_RATE_PER_HOUR / 24.0,))  # per-hour rate applied once

    def get_consolidation_candidates(self, limit: int = 20) -> list[Episode]:
        """Return low-salience episodes ready for consolidation."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, content, summary, tags, valence, salience, timestamp,
                       consolidated, reinforcement_count, embedding
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
        return [self._row_to_episode(r) for r in rows]

    def _row_to_episode(self, row) -> Episode:
        embedding = None
        if row[9] is not None:
            try:
                embedding = np.frombuffer(row[9], dtype=np.float32)
            except Exception:
                pass
        return Episode(
            id=row[0], content=row[1], summary=row[2],
            tags=json.loads(row[3]), valence=row[4], salience=row[5],
            timestamp=row[6], consolidated=bool(row[7]),
            reinforcement_count=row[8], embedding=embedding,
        )

    def stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            active = conn.execute("SELECT COUNT(*) FROM episodes WHERE consolidated=0").fetchone()[0]
            avg_sal = conn.execute("SELECT AVG(salience) FROM episodes WHERE consolidated=0").fetchone()[0]
            with_embed = conn.execute("SELECT COUNT(*) FROM episodes WHERE embedding IS NOT NULL").fetchone()[0]
        return {
            "total_episodes": total,
            "active_episodes": active,
            "consolidated_episodes": total - active,
            "avg_salience": round(avg_sal or 0, 3),
            "episodes_with_embeddings": with_embed,
        }
