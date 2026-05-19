"""
memory/consolidator.py

The consolidator compresses low-salience episodic memory into stable semantic
facts during idle periods.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.warm import Episode, EpisodicMemory

logger = logging.getLogger(__name__)


class Consolidator:
    """Runs memory decay and rule-based fact extraction."""

    def __init__(self, episodic_memory: "EpisodicMemory", config: dict):
        self.episodic = episodic_memory
        self.config = config
        self._last_run: float = 0

        from memory.cold import SemanticMemory

        self.semantic = SemanticMemory(
            config.get("semantic_db_path", "./data/semantic.db"),
            embedding_dimensions=int(config.get("embedding_dimensions", 128)),
        )

    async def run_cycle(self, current_salience: dict[str, float]):
        cycle_start = time.time()
        logger.info("Consolidation cycle starting.")

        self.episodic.run_decay()
        candidates = self.episodic.get_consolidation_candidates(
            limit=int(self.config.get("consolidation_batch_size", 20))
        )
        logger.info("Found %s episodes for consolidation.", len(candidates))

        consolidated_ids = []
        facts_written = 0
        for episode in candidates:
            for fact_content, domain, confidence in self._extract_facts(episode):
                self.semantic.store_fact(
                    content=fact_content,
                    source_episode_ids=[episode.id],
                    domain=domain,
                    confidence=confidence,
                )
                facts_written += 1
            consolidated_ids.append(episode.id)

        if consolidated_ids:
            self.episodic.mark_consolidated(consolidated_ids)

        duration = time.time() - cycle_start
        self._last_run = time.time()

        logger.info(
            "Consolidation cycle complete: %s episodes consolidated in %.2fs.",
            len(consolidated_ids),
            duration,
        )

        return {
            "episodes_consolidated": len(consolidated_ids),
            "semantic_facts_written": facts_written,
            "duration_seconds": round(duration, 2),
        }

    def _extract_facts(self, episode: "Episode") -> list[tuple[str, str, float]]:
        facts = []

        if "preference" in episode.tags or "like" in episode.tags:
            facts.append((
                f"User preference noted: {episode.summary}",
                "user_preferences",
                min(1.0, 0.6 + (episode.reinforcement_count * 0.05)),
            ))

        if "fact" in episode.tags or "knowledge" in episode.tags:
            facts.append((
                episode.summary,
                "world_knowledge",
                min(1.0, 0.5 + (episode.reinforcement_count * 0.05)),
            ))

        if abs(episode.valence) > 0.6:
            valence_label = "positive" if episode.valence > 0 else "negative"
            facts.append((
                f"Significant {valence_label} experience: {episode.summary}",
                "self_knowledge",
                0.4,
            ))

        if not facts and len(episode.summary) > 20:
            facts.append((episode.summary, "general", 0.3))

        return facts

    @property
    def last_run_seconds_ago(self) -> float:
        if self._last_run == 0:
            return float("inf")
        return time.time() - self._last_run
