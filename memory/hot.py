"""
memory/hot.py

Working Memory — fast, volatile, token-limited.

This is the current session's active context. It's what the inference
engine sees directly. Unlike the frozen context window of a standard LLM,
this is managed actively: older/lower-salience content gets evicted
to make room for new content, rather than simply truncating.
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import time


@dataclass
class MemoryEntry:
    content: str
    metadata: dict
    timestamp: float = field(default_factory=time.time)
    salience: float = 1.0
    token_estimate: int = 0

    def __post_init__(self):
        if self.token_estimate == 0:
            # Rough estimate: 1 token ≈ 4 chars
            self.token_estimate = len(self.content) // 4

    def decay(self, rate: float = 0.01):
        """Reduce salience over time."""
        self.salience = max(0.0, self.salience * (1 - rate))


class WorkingMemory:
    """
    Token-budget-managed working memory.

    Maintains a rolling window of recent context, evicting low-salience
    entries when capacity is reached. Designed to be passed directly
    to the inference engine as formatted context.
    """

    def __init__(self, capacity: int = 8192):
        self.capacity = capacity
        self._entries: deque[MemoryEntry] = deque()
        self.current_tokens: int = 0

    def add(self, content: str, metadata: dict, salience: float = 1.0):
        """Add a new entry, evicting old ones if over capacity."""
        entry = MemoryEntry(content=content, metadata=metadata, salience=salience)

        # Evict until we have room
        while self.current_tokens + entry.token_estimate > self.capacity and self._entries:
            self._evict_lowest_salience()

        self._entries.append(entry)
        self.current_tokens += entry.token_estimate

    def _evict_lowest_salience(self):
        """Remove the entry with lowest salience."""
        if not self._entries:
            return
        min_entry = min(self._entries, key=lambda e: e.salience)
        self._entries.remove(min_entry)
        self.current_tokens -= min_entry.token_estimate

    def boost_salience(self, query: str, boost: float = 0.2):
        """
        Boost salience of entries relevant to a query.
        (In production: use embedding similarity. Here: substring match.)
        """
        query_lower = query.lower()
        for entry in self._entries:
            if query_lower in entry.content.lower():
                entry.salience = min(1.0, entry.salience + boost)

    def decay_all(self, rate: float = 0.005):
        """Decay all entries — called periodically by core."""
        for entry in self._entries:
            entry.decay(rate)

    def to_context_string(self) -> str:
        """
        Format working memory as a context string for the inference engine.
        Most recent and most salient content appears first.
        """
        sorted_entries = sorted(
            self._entries,
            key=lambda e: (e.salience * 0.6 + (1 / (1 + time.time() - e.timestamp)) * 0.4),
            reverse=True
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
            self._entries.append(entry)
            self.current_tokens += entry.token_estimate

    def __len__(self) -> int:
        return len(self._entries)
