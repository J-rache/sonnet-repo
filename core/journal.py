"""
Append-only continuity journal for replayable PNP events.

The journal is deliberately small and boring: one JSON object per line, with a
monotonic sequence number. Runtime state can be checkpointed, then any later
events can be replayed on restart.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any


@dataclass(frozen=True)
class JournalEvent:
    sequence: int
    timestamp: float
    type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "type": self.type,
            "payload": self.payload,
        }


class EventJournal:
    """Durable JSONL event journal with simple replay support."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_sequence = self._scan_last_sequence()

    @property
    def last_sequence(self) -> int:
        return self._last_sequence

    def append(self, event_type: str, payload: dict[str, Any] | None = None) -> JournalEvent:
        with self._lock:
            event = JournalEvent(
                sequence=self._last_sequence + 1,
                timestamp=time.time(),
                type=event_type,
                payload=payload or {},
            )
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
                f.flush()
            self._last_sequence = event.sequence
            return event

    def read(self, after_sequence: int = 0) -> list[JournalEvent]:
        if not self.path.exists():
            return []

        events: list[JournalEvent] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    sequence = int(raw["sequence"])
                    if sequence <= after_sequence:
                        continue
                    events.append(
                        JournalEvent(
                            sequence=sequence,
                            timestamp=float(raw.get("timestamp", 0.0)),
                            type=str(raw["type"]),
                            payload=dict(raw.get("payload", {})),
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return events

    def _scan_last_sequence(self) -> int:
        last = 0
        if not self.path.exists():
            return last
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    raw = json.loads(line)
                    last = max(last, int(raw.get("sequence", 0)))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
        return last
