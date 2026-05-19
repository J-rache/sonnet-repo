"""
core/process.py

The Persistent Core is the always-running low-compute process that keeps local
PNP state alive between inference calls.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import logging
import os
import time
from typing import Optional

from core.goals import GoalPriority
from core.journal import EventJournal, JournalEvent
from core.state import MotivationalState
from core.goals import GoalStack
from memory.hot import WorkingMemory
from memory.warm import EpisodicMemory
from memory.consolidator import Consolidator

logger = logging.getLogger(__name__)


@dataclass
class CoreMetrics:
    """Lightweight telemetry for the persistent core."""

    uptime_start: float = field(default_factory=time.time)
    heartbeat_count: int = 0
    last_interaction: Optional[float] = None
    consolidation_cycles: int = 0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.uptime_start

    @property
    def idle_seconds(self) -> float:
        if self.last_interaction is None:
            return self.uptime_seconds
        return time.time() - self.last_interaction


class PersistentCore:
    """
    The low-compute process that owns local continuity state.

    State changes are written to an append-only event journal. Shutdown writes a
    checkpoint snapshot, and startup restores the snapshot plus any newer
    journal events.
    """

    HEARTBEAT_INTERVAL_MS = 100
    CONSOLIDATION_IDLE_THRESHOLD = 30
    SALIENCE_DECAY_RATE = 0.001

    def __init__(self, config: dict):
        self.config = config
        self.HEARTBEAT_INTERVAL_MS = int(config.get("heartbeat_interval_ms", self.HEARTBEAT_INTERVAL_MS))
        self.CONSOLIDATION_IDLE_THRESHOLD = int(
            config.get("consolidation_idle_threshold_seconds", self.CONSOLIDATION_IDLE_THRESHOLD)
        )
        self.metrics = CoreMetrics()
        self.running = False

        self.data_dir = config.get("data_dir", "./data")
        self.core_state_path = config.get("core_state_path", os.path.join(self.data_dir, "core_state.json"))
        self.journal = EventJournal(config.get("journal_path", os.path.join(self.data_dir, "events.jsonl")))

        self.state = MotivationalState()
        self.goals = GoalStack()
        self.working_memory = WorkingMemory(capacity=config.get("working_memory_capacity", 8192))
        self.episodic_memory = EpisodicMemory(config.get("episodic_db_path", os.path.join(self.data_dir, "episodic.db")))
        self.consolidator = Consolidator(self.episodic_memory, config)
        self.salience: dict[str, float] = {}

        self.replay_summary = {
            "restored_from_snapshot": False,
            "snapshot_sequence": 0,
            "events_replayed": 0,
            "replayed_types": {},
        }
        self._restore_state()

        logger.info("PersistentCore initialized at %s", datetime.now(UTC).isoformat())

    async def start(self):
        """Start the persistent core loops."""
        self.running = True
        logger.info("Persistent core starting.")
        await asyncio.gather(
            self._heartbeat_loop(),
            self._consolidation_loop(),
        )

    async def stop(self):
        """Graceful shutdown: stop loops and checkpoint continuity state."""
        logger.info("Persistent core stopping; saving state.")
        self.running = False
        await self._save_state()

    async def _heartbeat_loop(self):
        while self.running:
            tick_start = time.monotonic()

            self.metrics.heartbeat_count += 1
            self._decay_salience()
            self.state.tick(
                idle_seconds=self.metrics.idle_seconds,
                active_goals=self.goals.active_count,
                salience_peak=max(self.salience.values()) if self.salience else 0.0,
            )

            urgent = self.goals.check_urgency()
            if urgent:
                logger.debug("Goal urgency detected: %s", urgent.description)
                self.state.flag_urgent(urgent)

            tick_duration = time.monotonic() - tick_start
            sleep_s = max(0, (self.HEARTBEAT_INTERVAL_MS / 1000) - tick_duration)
            await asyncio.sleep(sleep_s)

    async def _consolidation_loop(self):
        while self.running:
            if self.metrics.idle_seconds > self.CONSOLIDATION_IDLE_THRESHOLD:
                logger.debug("Idle threshold reached; running consolidation cycle.")
                await self.run_consolidation_cycle()

            await asyncio.sleep(5)

    async def run_consolidation_cycle(self) -> dict:
        result = await self.consolidator.run_cycle(self.salience)
        self.metrics.consolidation_cycles += 1
        self.record_event("consolidation_ran", result)
        return result

    def _decay_salience(self):
        to_remove = []
        for key in self.salience:
            self.salience[key] *= 1 - self.SALIENCE_DECAY_RATE
            if self.salience[key] < 0.01:
                to_remove.append(key)
        for key in to_remove:
            del self.salience[key]

    def on_interaction(self, content: str, metadata: dict) -> dict:
        """
        Record an external interaction, update salience, and write replayable
        journal events for the interaction and working-memory write.
        """
        self.metrics.last_interaction = time.time()

        concepts = metadata.get("concepts", [])
        for concept in concepts:
            self.salience[concept] = min(1.0, self.salience.get(concept, 0.0) + 0.3)

        self.record_event("interaction_received", {
            "content": content,
            "metadata": metadata,
            "last_interaction": self.metrics.last_interaction,
        })
        self.working_memory.add(content, metadata)
        self.record_memory_written("working", content, metadata, salience=1.0)

        return self.get_state_snapshot()

    def add_goal(self, description: str, priority: GoalPriority, deadline: Optional[float] = None):
        goal = self.goals.add(description=description, priority=priority, deadline=deadline)
        self.record_event("goal_added", {"goal": goal.to_dict()})
        return goal

    def complete_goal(self, goal_id: str, notes: str = "") -> bool:
        completed = self.goals.complete(goal_id, notes)
        if completed:
            self.record_event("goal_completed", {"goal_id": goal_id, "notes": notes})
        return completed

    def record_memory_written(self, layer: str, content: str, metadata: dict, salience: float = 1.0):
        self.record_event("memory_written", {
            "layer": layer,
            "content": content,
            "metadata": metadata,
            "salience": salience,
        })

    def record_adapter_delta(self, delta: dict, applied: bool):
        self.record_event("adapter_delta_applied", {
            "applied": applied,
            "delta": delta,
        })

    def record_event(self, event_type: str, payload: dict) -> JournalEvent:
        return self.journal.append(event_type, payload)

    def get_state_snapshot(self) -> dict:
        """Return the current observable core state."""
        return {
            "uptime_seconds": self.metrics.uptime_seconds,
            "idle_seconds": self.metrics.idle_seconds,
            "heartbeat_count": self.metrics.heartbeat_count,
            "consolidation_cycles": self.metrics.consolidation_cycles,
            "motivational_state": self.state.to_dict(),
            "active_goals": self.goals.to_list(),
            "salience_map": dict(sorted(
                self.salience.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]),
            "working_memory_tokens": self.working_memory.current_tokens,
            "continuity": {
                "journal_path": str(self.journal.path),
                "journal_last_sequence": self.journal.last_sequence,
                "restored_from_snapshot": self.replay_summary["restored_from_snapshot"],
                "events_replayed": self.replay_summary["events_replayed"],
            },
        }

    async def _save_state(self):
        os.makedirs(os.path.dirname(self.core_state_path), exist_ok=True)
        with open(self.core_state_path, "w", encoding="utf-8") as f:
            json.dump(self.get_persistence_snapshot(), f, indent=2)
        logger.info("Core state saved.")

    def get_persistence_snapshot(self) -> dict:
        return {
            "schema_version": 1,
            "saved_at": time.time(),
            "journal_last_sequence": self.journal.last_sequence,
            "metrics": {
                "heartbeat_count": self.metrics.heartbeat_count,
                "last_interaction": self.metrics.last_interaction,
                "consolidation_cycles": self.metrics.consolidation_cycles,
            },
            "motivational_state": self.state.to_snapshot(),
            "goals": self.goals.to_snapshot(),
            "working_memory": self.working_memory.to_snapshot(),
            "salience": self.salience,
        }

    def _restore_state(self):
        after_sequence = 0
        if os.path.exists(self.core_state_path):
            with open(self.core_state_path, encoding="utf-8") as f:
                snapshot = json.load(f)

            metrics = snapshot.get("metrics", {})
            self.metrics.heartbeat_count = int(metrics.get("heartbeat_count", 0))
            self.metrics.last_interaction = metrics.get("last_interaction")
            self.metrics.consolidation_cycles = int(metrics.get("consolidation_cycles", 0))
            self.state.load_snapshot(snapshot.get("motivational_state", {}))
            self.goals.load_snapshot(snapshot.get("goals", []))
            self.working_memory.load_snapshot(snapshot.get("working_memory", []))
            self.salience = {
                str(k): float(v)
                for k, v in snapshot.get("salience", {}).items()
            }
            after_sequence = int(snapshot.get("journal_last_sequence", 0))
            self.replay_summary["restored_from_snapshot"] = True
            self.replay_summary["snapshot_sequence"] = after_sequence

        events = self.journal.read(after_sequence=after_sequence)
        replayed_types: dict[str, int] = {}
        for event in events:
            if self._apply_replay_event(event):
                replayed_types[event.type] = replayed_types.get(event.type, 0) + 1

        self.replay_summary["events_replayed"] = sum(replayed_types.values())
        self.replay_summary["replayed_types"] = replayed_types

    def _apply_replay_event(self, event: JournalEvent) -> bool:
        payload = event.payload

        if event.type == "interaction_received":
            self.metrics.last_interaction = payload.get("last_interaction", event.timestamp)
            concepts = payload.get("metadata", {}).get("concepts", [])
            for concept in concepts:
                self.salience[concept] = min(1.0, self.salience.get(concept, 0.0) + 0.3)
            return True

        if event.type == "memory_written" and payload.get("layer") == "working":
            self.working_memory.add(
                content=str(payload.get("content", "")),
                metadata=dict(payload.get("metadata", {})),
                salience=float(payload.get("salience", 1.0)),
            )
            return True

        if event.type == "goal_added":
            self.goals.upsert(payload.get("goal", {}))
            return True

        if event.type == "goal_completed":
            return self.goals.complete(
                goal_id=str(payload.get("goal_id", "")),
                notes=str(payload.get("notes", "")),
            )

        if event.type == "consolidation_ran":
            self.metrics.consolidation_cycles += 1
            return True

        return event.type in {"adapter_delta_applied", "memory_written"}
