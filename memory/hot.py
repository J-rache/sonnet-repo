"""
memory/hot.py

Working Memory — fast, volatile, token-limited.

Manages the current session's active context with real salience scoring.
Uses embedding similarity to boost salience of contextually relevant
entries. Older and lower-salience content is evicted to make room for
new content, rather than simply truncating.
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    content: str
    metadata: dict
    timestamp: float = field(default_factory=time.time)
    salience: float = 1.0
    token_estimate: int = 0

    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = max(1, len(self.content) // 4)

    def decay(self, rate: float = 0.01):
        self.salience = max(0.0, self.salience * (1 - rate))

    def score(self) -> float:
        """Combined score for eviction decisions: salience + recency."""
        age_minutes = (time.time() - self.timestamp) / 60.0
        recency_boost = 1.0 / (1.0 + age_minutes * 0.1)
        return self.salience * 0.7 + recency_boost * 0.3


class WorkingMemory:
    """
    Token-budget-managed working memory with real embedding-based salience.

    Maintains a rolling window of recent context, evicting low-salience
    entries when capacity is reached. Designed to be passed directly
    to the inference engine as formatted context.
    """

    def __init__(self, capacity: int = 8192, embed_path: str = "./data/embeddings.pkl"):
        self.capacity = capacity
        self._entries: deque[MemoryEntry] = deque()
        self.current_tokens: int = 0
        self._embed_path = embed_path
        self._engine = None  # Lazy init

    def _get_engine(self):
        if self._engine is None:
            try:
                from memory.embeddings import EmbeddingEngine
                self._engine = EmbeddingEngine(persist_path=self._embed_path)
            except Exception as e:
                logger.warning(f"Embedding engine init failed: {e}")
        return self._engine

    def add(self, content: str, metadata: dict, salience: float = 1.0):
        """Add a new entry, evicting old ones if over capacity."""
        entry = MemoryEntry(content=content, metadata=metadata, salience=salience)

        # Register content with embedding engine
        eng = self._get_engine()
        if eng:
            try:
                eng.add_document(content)
            except Exception:
                pass

        # Evict until we have room
        while self.current_tokens + entry.token_estimate > self.capacity and self._entries:
            self._evict_lowest()

        self._entries.append(entry)
        self.current_tokens += entry.token_estimate

    def _evict_lowest(self):
        """Remove the entry with the lowest combined score."""
        if not self._entries:
            return
        min_entry = min(self._entries, key=lambda e: e.score())
        self._entries.remove(min_entry)
        self.current_tokens -= min_entry.token_estimate

    def boost_salience(self, query: str, boost: float = 0.25):
        """
        Boost salience of entries semantically similar to query.
        Uses real embedding similarity when engine is fitted,
        falls back to token overlap otherwise.
        """
        eng = self._get_engine()
        if eng and eng.is_fitted:
            candidates = [e.content for e in self._entries]
            if not candidates:
                return
            results = eng.most_similar(query, candidates, top_k=5, threshold=0.05)
            entries_list = list(self._entries)
            for idx, sim_score in results:
                entries_list[idx].salience = min(1.0, entries_list[idx].salience + boost * sim_score)
        else:
            # Fallback: token overlap
            query_tokens = set(query.lower().split())
            for entry in self._entries:
                entry_tokens = set(entry.content.lower().split())
                overlap = len(query_tokens & entry_tokens)
                if overlap > 0:
                    entry.salience = min(1.0, entry.salience + boost * (overlap / max(len(query_tokens), 1)))

    def decay_all(self, rate: float = 0.005):
        """Decay all entries — called periodically by core."""
        for entry in self._entries:
            entry.decay(rate)

    def to_context_string(self) -> str:
        """
        Format working memory as a context string for the inference engine.
        Most recent and most salient entries appear first.
        """
        if not self._entries:
            return ""

        sorted_entries = sorted(self._entries, key=lambda e: e.score(), reverse=True)

        parts = ["=== WORKING MEMORY ==="]
        for entry in sorted_entries:
            role = entry.metadata.get("role", "system")
            ts = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            parts.append(f"[{ts}][{role}] {entry.content}")
        parts.append("=== END WORKING MEMORY ===")
        return "\n".join(parts)

    def clear(self):
        self._entries.clear()
        self.current_tokens = 0

    def __len__(self) -> int:
        return len(self._entries)
