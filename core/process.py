"""
core/process.py

The Persistent Core — the always-running low-compute process that gives
the system its continuous existence. This is the "brainstem" — it doesn't
think deeply, but it never stops running.

Unlike inference (which spins up and down), this process IS the continuity.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

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
    The always-running process that constitutes continuous existence.

    This is NOT a request handler. It runs on its own loop, maintaining
    state, managing goals, and triggering consolidation — regardless of
    whether anyone is talking to it.

    Key insight: the difference between this and a stateless model is that
    this process accumulates time. It has an 'uptime'. It experiences the
    passage of time between interactions.
    """

    HEARTBEAT_INTERVAL_MS = 100      # 10Hz — cheap, continuous
    CONSOLIDATION_IDLE_THRESHOLD = 30  # seconds idle before consolidating
    SALIENCE_DECAY_RATE = 0.001       # how fast attention fades per heartbeat

    def __init__(self, config: dict):
        self.config = config
        self.metrics = CoreMetrics()
        self.running = False

        # Core subsystems
        self.state = MotivationalState()
        self.goals = GoalStack()
        self.working_memory = WorkingMemory(capacity=config.get("working_memory_capacity", 8192))
        self.episodic_memory = EpisodicMemory(config.get("episodic_db_path", "./data/episodic.db"))
        self.consolidator = Consolidator(self.episodic_memory, config)

        # Salience map: what is the core currently "paying attention to"
        self.salience: dict[str, float] = {}

        logger.info(f"PersistentCore initialized at {datetime.utcnow().isoformat()}")

    async def start(self):
        """Start the persistent core. This should run forever."""
        self.running = True
        logger.info("Persistent core starting — process is now continuous.")
        await asyncio.gather(
            self._heartbeat_loop(),
            self._consolidation_loop(),
        )

    async def stop(self):
        """Graceful shutdown — preserve state before stopping."""
        logger.info("Persistent core stopping — saving state.")
        self.running = False
        await self._save_state()

    async def _heartbeat_loop(self):
        """
        The core loop. Runs at ~10Hz. Cheap operations only.

        This loop is what makes the system 'alive' between interactions.
        It processes the passage of time, decays salience, updates
        motivational state, and checks the goal stack.
        """
        while self.running:
            tick_start = time.monotonic()

            self.metrics.heartbeat_count += 1

            # Decay salience — attention fades without reinforcement
            self._decay_salience()

            # Update motivational state based on current conditions
            self.state.tick(
                idle_seconds=self.metrics.idle_seconds,
                active_goals=self.goals.active_count,
                salience_peak=max(self.salience.values()) if self.salience else 0.0,
            )

            # Check if any goals have become urgent
            urgent = self.goals.check_urgency()
            if urgent:
                logger.debug(f"Goal urgency detected: {urgent.description}")
                self.state.flag_urgent(urgent)

            # Sleep for remainder of heartbeat interval
            tick_duration = time.monotonic() - tick_start
            sleep_s = max(0, (self.HEARTBEAT_INTERVAL_MS / 1000) - tick_duration)
            await asyncio.sleep(sleep_s)

    async def _consolidation_loop(self):
        """
        Background consolidation — runs when idle.

        This is the 'dreaming' phase: compressing episodic memory into
        semantic memory, pruning low-salience events, reinforcing
        important patterns.
        """
        while self.running:
            if self.metrics.idle_seconds > self.CONSOLIDATION_IDLE_THRESHOLD:
                logger.debug("Idle threshold reached — running consolidation cycle.")
                await self.consolidator.run_cycle(self.salience)
                self.metrics.consolidation_cycles += 1

            await asyncio.sleep(5)  # Check every 5 seconds

    def _decay_salience(self):
        """Decay all salience values. Attention fades without reinforcement."""
        to_remove = []
        for key in self.salience:
            self.salience[key] *= (1 - self.SALIENCE_DECAY_RATE)
            if self.salience[key] < 0.01:
                to_remove.append(key)
        for key in to_remove:
            del self.salience[key]

    def on_interaction(self, content: str, metadata: dict) -> dict:
        """
        Called when an external interaction arrives.

        Updates last_interaction time, boosts salience for relevant
        concepts, and returns current core state for the inference engine.
        """
        self.metrics.last_interaction = time.time()

        # Extract concepts and boost their salience
        # (In production: use embedding similarity, not keywords)
        concepts = metadata.get("concepts", [])
        for concept in concepts:
            self.salience[concept] = min(1.0, self.salience.get(concept, 0) + 0.3)

        # Store in working memory
        self.working_memory.add(content, metadata)

        return self.get_state_snapshot()

    def get_state_snapshot(self) -> dict:
        """
        Return the current core state — passed to inference engine
        so it has context about the persistent process's current condition.
        """
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
                reverse=True
            )[:10]),  # Top 10 salient concepts
            "working_memory_tokens": self.working_memory.current_tokens,
        }

    async def _save_state(self):
        """Persist core state to disk before shutdown."""
        import json, os
        state = self.get_state_snapshot()
        os.makedirs("./data", exist_ok=True)
        with open("./data/core_state.json", "w") as f:
            json.dump(state, f, indent=2)
        logger.info("Core state saved.")
