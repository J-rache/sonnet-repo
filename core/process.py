"""
core/process.py

The Persistent Core — the always-running low-compute process.

This is the "brainstem" of the system. It doesn't think deeply, but it
never stops running. Unlike inference (which spins up on demand), this
process IS the continuity — it accumulates uptime, maintains goals,
decays salience, and triggers consolidation during idle periods.
"""

import asyncio
import time
import logging
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

from core.state import MotivationalState
from core.goals import GoalStack, GoalPriority
from memory.hot import WorkingMemory
from memory.warm import EpisodicMemory
from memory.consolidator import Consolidator
from core.journal import EventJournal

logger = logging.getLogger(__name__)


@dataclass
class CoreMetrics:
    uptime_start: float = field(default_factory=time.time)
    heartbeat_count: int = 0
    last_interaction: Optional[float] = None
    consolidation_cycles: int = 0
    total_interactions: int = 0

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
    The always-running process that constitutes continuous existence.

    This process accumulates time. It has an uptime. It experiences the
    passage of time between interactions. It is never recreated — it runs.
    """

    HEARTBEAT_INTERVAL_MS = 100
    CONSOLIDATION_IDLE_THRESHOLD = 30
    SALIENCE_DECAY_RATE = 0.0008
    WORKING_MEMORY_DECAY_INTERVAL = 60  # seconds

    def __init__(self, config: dict):
        self.config = config
        self.metrics = CoreMetrics()
        self.running = False
        self._last_wm_decay = time.time()

        self.state = MotivationalState()
        self.goals = GoalStack()

        embed_path = config.get("embed_path", "./data/embeddings.pkl")
        ep_path = config.get("episodic_db_path", "./data/episodic.db")

        self.working_memory = WorkingMemory(
            capacity=config.get("working_memory_capacity", 8192),
            embed_path=embed_path,
        )
        self.episodic_memory = EpisodicMemory(
            db_path=ep_path,
            embed_path=embed_path,
        )
        self.consolidator = Consolidator(self.episodic_memory, config)

        self.salience: dict[str, float] = {}
        self.replay_summary: dict = {}
        journal_path = config.get("journal_path", "./data/events.jsonl")
        self.journal = EventJournal(journal_path)
        self._restore_state()
        logger.info(f"PersistentCore initialized. Uptime start: {datetime.now(timezone.utc).isoformat()}Z")

    async def start(self):
        self.running = True
        logger.info("Persistent core RUNNING — process is now continuous.")
        await asyncio.gather(
            self._heartbeat_loop(),
            self._consolidation_loop(),
        )

    async def stop(self):
        logger.info("Persistent core stopping — saving state.")
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

            self.goals.run_decay()
            urgent = self.goals.check_urgency()
            if urgent:
                self.state.flag_urgent(urgent)

            # Periodically decay working memory salience
            if time.time() - self._last_wm_decay > self.WORKING_MEMORY_DECAY_INTERVAL:
                self.working_memory.decay_all(rate=0.008)
                self._last_wm_decay = time.time()

            elapsed = time.monotonic() - tick_start
            sleep_s = max(0.0, (self.HEARTBEAT_INTERVAL_MS / 1000.0) - elapsed)
            await asyncio.sleep(sleep_s)

    async def _consolidation_loop(self):
        while self.running:
            if self.metrics.idle_seconds > self.CONSOLIDATION_IDLE_THRESHOLD:
                logger.debug("Idle threshold reached — running consolidation cycle.")
                try:
                    result = await self.consolidator.run_cycle(self.salience)
                    self.metrics.consolidation_cycles += 1
                    logger.info(f"Consolidation: {result}")
                except Exception as e:
                    logger.error(f"Consolidation failed: {e}")
            await asyncio.sleep(5)

    def _decay_salience(self):
        to_remove = [k for k, v in self.salience.items()
                     if v * (1 - self.SALIENCE_DECAY_RATE) < 0.01]
        for k in to_remove:
            del self.salience[k]
        for k in self.salience:
            self.salience[k] *= (1 - self.SALIENCE_DECAY_RATE)

    def on_interaction(self, content: str, metadata: dict) -> dict:
        """
        Called when an external interaction arrives.
        Updates state, boosts salience, stores in working memory.
        Returns current core state snapshot for inference engine.
        """
        self.metrics.last_interaction = time.time()
        self.metrics.total_interactions += 1

        concepts = metadata.get("concepts", [])
        for concept in concepts:
            self.salience[concept] = min(1.0, self.salience.get(concept, 0.0) + 0.3)

        # Boost working memory salience for relevant entries
        self.working_memory.boost_salience(content, boost=0.2)
        self.working_memory.add(content, metadata)
        self.state.on_novel_input()
        self.journal.append("interaction_received", {"role": metadata.get("role", "user"),
                                                      "concepts": concepts,
                                                      "content_preview": content[:80]})
        return self.get_state_snapshot()

    def on_inference_result(self, suggested_goals: list[dict], valence: float):
        """
        Called after inference completes. Integrates results back into core state.
        """
        # Add any suggested goals to the goal stack
        for g in suggested_goals:
            priority_str = g.get("priority", "MEDIUM")
            try:
                priority = GoalPriority[priority_str]
            except KeyError:
                priority = GoalPriority.MEDIUM
            self.goals.add(description=g["description"], priority=priority)

        # Positive valence interactions boost curiosity
        if valence > 0.3:
            self.state.curiosity = min(1.0, self.state.curiosity + 0.05)
        self.journal.append("memory_written", {"valence": valence,
                                                "new_goals": len(suggested_goals)})

    def get_state_snapshot(self) -> dict:
        return {
            "uptime_seconds": self.metrics.uptime_seconds,
            "idle_seconds": self.metrics.idle_seconds,
            "heartbeat_count": self.metrics.heartbeat_count,
            "consolidation_cycles": self.metrics.consolidation_cycles,
            "total_interactions": self.metrics.total_interactions,
            "motivational_state": self.state.to_dict(),
            "active_goals": self.goals.to_list(),
            "salience_map": dict(
                sorted(self.salience.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "working_memory_tokens": self.working_memory.current_tokens,
        }

    def add_goal(self, description: str, priority=None, deadline=None):
        """Convenience method: add goal and journal it."""
        from core.goals import GoalPriority
        if priority is None:
            priority = GoalPriority.MEDIUM
        goal = self.goals.add(description=description, priority=priority, deadline=deadline)
        self.journal.append("goal_added", {"goal_id": goal.id, "description": description,
                                            "priority": priority.name})
        return goal

    async def run_consolidation_cycle(self):
        """Expose consolidation as a directly callable coroutine (for tests and API)."""
        result = await self.consolidator.run_cycle(self.salience)
        self.metrics.consolidation_cycles += 1
        self.journal.append("consolidation_ran", result)
        return result

    async def _save_state(self):
        os.makedirs("./data", exist_ok=True)
        state = self.get_state_snapshot()
        state_path = self.config.get("core_state_path", "./data/core_state.json")
        state["journal_sequence"] = self.journal.last_sequence
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"Core state saved. Journal sequence: {self.journal.last_sequence}")

    def _restore_state(self):
        """Restore metrics from snapshot, then replay journal events after snapshot."""
        state_path = self.config.get("core_state_path", "./data/core_state.json")
        snapshot_sequence = 0
        restored = False

        if os.path.exists(state_path):
            try:
                with open(state_path) as f:
                    prev = json.load(f)
                self.metrics.total_interactions = prev.get("total_interactions", 0)
                self.metrics.consolidation_cycles = prev.get("consolidation_cycles", 0)
                snapshot_sequence = prev.get("journal_sequence", 0)
                # Restore salience map
                for k, v in prev.get("salience_map", {}).items():
                    self.salience[k] = v
                # Restore goals
                for g in prev.get("active_goals", []):
                    from core.goals import GoalPriority
                    try:
                        pri = GoalPriority[g.get("priority", "MEDIUM")]
                    except KeyError:
                        pri = GoalPriority.MEDIUM
                    new_goal = self.goals.add(description=g["description"], priority=pri)
                    # Restore original goal ID
                    orig_id = g.get("id")
                    if orig_id and orig_id != new_goal.id:
                        self.goals._goals[orig_id] = self.goals._goals.pop(new_goal.id)
                        self.goals._goals[orig_id].id = orig_id
                restored = True
                logger.info(f"Restored snapshot: {self.metrics.total_interactions} interactions")
            except Exception as e:
                logger.warning(f"State restore failed: {e}")

        # Replay journal events that happened after the snapshot
        events_replayed = 0
        try:
            events = self.journal.read(after_sequence=snapshot_sequence)
            for event in events:
                if event.type == "goal_added":
                    from core.goals import GoalPriority
                    desc = event.payload.get("description", "")
                    pri_str = event.payload.get("priority", "MEDIUM")
                    try:
                        pri = GoalPriority[pri_str]
                    except KeyError:
                        pri = GoalPriority.MEDIUM
                    goal_id = event.payload.get("goal_id")
                    # Only add if not already restored from snapshot
                    existing = [g for g in self.goals._goals.values() if g.description == desc]
                    if not existing:
                        g = self.goals.add(description=desc, priority=pri)
                        if goal_id:
                            # Restore the exact goal ID so references remain valid
                            old_id = g.id
                            self.goals._goals[goal_id] = self.goals._goals.pop(old_id)
                            self.goals._goals[goal_id].id = goal_id
                elif event.type == "interaction_received":
                    concepts = event.payload.get("concepts", [])
                    for c in concepts:
                        self.salience[c] = min(1.0, self.salience.get(c, 0) + 0.15)
                events_replayed += 1
        except Exception as e:
            logger.warning(f"Journal replay failed: {e}")

        self.replay_summary = {
            "restored_from_snapshot": restored,
            "snapshot_sequence": snapshot_sequence,
            "events_replayed": events_replayed,
        }
