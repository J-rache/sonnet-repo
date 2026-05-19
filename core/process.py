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

    async def _save_state(self):
        os.makedirs("./data", exist_ok=True)
        state = self.get_state_snapshot()
        with open("./data/core_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info("Core state saved to ./data/core_state.json")

    def _restore_state(self):
        """Restore metrics from previous run if available."""
        path = "./data/core_state.json"
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                prev = json.load(f)
            self.metrics.total_interactions = prev.get("total_interactions", 0)
            self.metrics.consolidation_cycles = prev.get("consolidation_cycles", 0)
            logger.info(
                f"Restored prior state: {self.metrics.total_interactions} interactions, "
                f"{self.metrics.consolidation_cycles} consolidation cycles"
            )
        except Exception as e:
            logger.warning(f"State restore failed: {e}")
