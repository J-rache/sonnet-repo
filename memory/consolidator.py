"""
memory/consolidator.py

The Consolidator — background process that compresses episodic memory
into semantic memory during idle periods.

This is the system's equivalent of sleep consolidation in biological
brains: taking the raw events of experience and extracting durable
knowledge from them.
"""

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.warm import EpisodicMemory, Episode

logger = logging.getLogger(__name__)


class Consolidator:
    """
    Runs during idle periods to:
    1. Decay old episodic memories
    2. Extract semantic facts from low-salience episodes
    3. Mark consolidated episodes
    4. Prune fully-decayed episodes

    In production: this would call an LLM to extract facts from episodes.
    Here: rule-based extraction as a placeholder.
    """

    def __init__(self, episodic_memory: "EpisodicMemory", config: dict):
        self.episodic = episodic_memory
        self.config = config
        self._last_run: float = 0

        # Lazy import to avoid circular dependency
        from memory.cold import SemanticMemory
        self.semantic = SemanticMemory(config.get("semantic_db_path", "./data/semantic.db"))

    async def run_cycle(self, current_salience: dict[str, float]):
        """
        Run one consolidation cycle.
        Called when the system has been idle long enough.
        """
        cycle_start = time.time()
        logger.info("Consolidation cycle starting.")

        # Step 1: Decay episodic salience
        self.episodic.run_decay()

        # Step 2: Get consolidation candidates
        candidates = self.episodic.get_consolidation_candidates(limit=20)
        logger.info(f"Found {len(candidates)} episodes for consolidation.")

        # Step 3: Extract semantic facts
        consolidated_ids = []
        for episode in candidates:
            facts = self._extract_facts(episode)
            for fact_content, domain, confidence in facts:
                self.semantic.store_fact(
                    content=fact_content,
                    source_episode_ids=[episode.id],
                    domain=domain,
                    confidence=confidence,
                )
            consolidated_ids.append(episode.id)

        # Step 4: Mark as consolidated
        if consolidated_ids:
            self.episodic.mark_consolidated(consolidated_ids)

        duration = time.time() - cycle_start
        self._last_run = time.time()

        logger.info(
            f"Consolidation cycle complete: {len(consolidated_ids)} episodes consolidated "
            f"in {duration:.2f}s."
        )

        return {
            "episodes_consolidated": len(consolidated_ids),
            "duration_seconds": round(duration, 2),
        }

    def _extract_facts(self, episode: "Episode") -> list[tuple[str, str, float]]:
        """
        Extract semantic facts from an episode.

        Returns list of (fact_content, domain, confidence) tuples.

        Production: call LLM with episode content, ask it to extract
        durable facts. Here: simple heuristic extraction.
        """
        facts = []

        # Heuristic: episodes tagged with "preference" → user_preferences domain
        if "preference" in episode.tags or "like" in episode.tags:
            facts.append((
                f"User preference noted: {episode.summary}",
                "user_preferences",
                0.6 + (episode.reinforcement_count * 0.05),
            ))

        # Heuristic: episodes tagged with "fact" or "knowledge"
        if "fact" in episode.tags or "knowledge" in episode.tags:
            facts.append((
                episode.summary,
                "world_knowledge",
                0.5 + (episode.reinforcement_count * 0.05),
            ))

        # Heuristic: high-valence episodes → self_knowledge
        if abs(episode.valence) > 0.6:
            valence_label = "positive" if episode.valence > 0 else "negative"
            facts.append((
                f"Significant {valence_label} experience: {episode.summary}",
                "self_knowledge",
                0.4,
            ))

        # Default: store summary as general knowledge if no other tags match
        if not facts and len(episode.summary) > 20:
            facts.append((
                episode.summary,
                "general",
                0.3,
            ))

        return facts

    @property
    def last_run_seconds_ago(self) -> float:
        if self._last_run == 0:
            return float("inf")
        return time.time() - self._last_run
