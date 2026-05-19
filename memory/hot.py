"""
memory/hot.py

Working memory is the fast, volatile context buffer passed to inference. It is
token-budgeted and salience-ranked.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time

from memory.embedding import HashingTextEmbedder, cosine_similarity


@dataclass
class MemoryEntry:
    content: str
    metadata: dict
    timestamp: float = field(default_factory=time.time)
    salience: float = 1.0
    token_estimate: int = 0
    embedding: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = max(1, len(self.content) // 4)

    def decay(self, rate: float = 0.01):
        self.salience = max(0.0, self.salience * (1 - rate))


class WorkingMemory:
    """Token-budget-managed working memory."""

    def __init__(self, capacity: int = 8192, embedding_dimensions: int = 128):
        self.capacity = capacity
        self.embedder = HashingTextEmbedder(embedding_dimensions)
        self._entries: deque[MemoryEntry] = deque()
        self.current_tokens: int = 0

    def add(self, content: str, metadata: dict, salience: float = 1.0):
        entry = MemoryEntry(
            content=content,
            metadata=metadata,
            salience=salience,
            embedding=self.embedder.embed(content),
        )

        while self.current_tokens + entry.token_estimate > self.capacity and self._entries:
            self._evict_lowest_salience()

        self._entries.append(entry)
        self.current_tokens += entry.token_estimate

    def _evict_lowest_salience(self):
        if not self._entries:
            return
        min_entry = min(self._entries, key=lambda entry: entry.salience)
        self._entries.remove(min_entry)
        self.current_tokens -= min_entry.token_estimate

    def boost_salience(self, query: str, boost: float = 0.2):
        """Boost entries whose vectors are close to the query vector."""
        query_embedding = self.embedder.embed(query)
        for entry in self._entries:
            if not entry.embedding:
                entry.embedding = self.embedder.embed(entry.content)
            similarity = cosine_similarity(query_embedding, entry.embedding)
            if similarity > 0.15:
                entry.salience = min(1.0, entry.salience + (boost * similarity))

    def decay_all(self, rate: float = 0.005):
        for entry in self._entries:
            entry.decay(rate)

    def to_context_string(self) -> str:
        sorted_entries = sorted(
            self._entries,
            key=lambda entry: (
                entry.salience * 0.6 + (1 / (1 + time.time() - entry.timestamp)) * 0.4
            ),
            reverse=True,
        )

        parts = ["=== WORKING MEMORY ==="]
        for entry in sorted_entries:
            role = entry.metadata.get("role", "unknown")
            ts = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            parts.append(f"[{ts}] [{role}] {entry.content}")
        parts.append("=== END WORKING MEMORY ===")

        return "\n".join(parts)

    def clear(self):
        self._entries.clear()
        self.current_tokens = 0

    def to_snapshot(self) -> list[dict]:
        return [
            {
                "content": entry.content,
                "metadata": entry.metadata,
                "timestamp": entry.timestamp,
                "salience": entry.salience,
                "token_estimate": entry.token_estimate,
            }
            for entry in self._entries
        ]

    def load_snapshot(self, entries: list[dict]):
        self.clear()
        for raw in entries:
            entry = MemoryEntry(
                content=str(raw.get("content", "")),
                metadata=dict(raw.get("metadata", {})),
                timestamp=float(raw.get("timestamp", time.time())),
                salience=float(raw.get("salience", 1.0)),
                token_estimate=int(raw.get("token_estimate", 0)),
            )
            entry.embedding = self.embedder.embed(entry.content)
            self._entries.append(entry)
            self.current_tokens += entry.token_estimate

    def __len__(self) -> int:
        return len(self._entries)
