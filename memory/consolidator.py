"""
memory/consolidator.py

The Consolidator — background process that compresses episodic memory
into semantic memory during idle periods.

Uses the Anthropic API to intelligently extract durable facts from
low-salience episodes — the system's equivalent of sleep consolidation.
Falls back to rule-based extraction when API is unavailable.
"""

import logging
import time
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.warm import EpisodicMemory, Episode

logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM_PROMPT = """You are a memory consolidation system for a persistent AI process.

Your job: given a list of episodic memory summaries, extract durable semantic facts
that should be retained long-term.

Rules:
- Extract only facts that are genuinely durable (not session-specific noise)
- Classify each fact into one of: user_preferences, world_knowledge, self_knowledge, general
- Rate confidence 0.0-1.0 based on how certain and stable the fact is
- Be concise — facts should be 1-2 sentences max
- Skip trivial interactions that contain no lasting information

Respond ONLY with valid JSON, no markdown, no explanation:
{
  "facts": [
    {"content": "...", "domain": "user_preferences|world_knowledge|self_knowledge|general", "confidence": 0.0-1.0}
  ]
}
If no durable facts found, respond: {"facts": []}"""


class Consolidator:
    """
    Runs during idle periods to:
    1. Decay old episodic salience
    2. Call LLM to extract semantic facts from low-salience episodes
    3. Store extracted facts in semantic memory
    4. Mark consolidated episodes

    Uses Anthropic API for intelligent fact extraction when available,
    falls back to rule-based extraction when not.
    """

    def __init__(self, episodic_memory: "EpisodicMemory", config: dict):
        self.episodic = episodic_memory
        self.config = config
        self._last_run: float = 0
        self._total_consolidated: int = 0
        self._total_facts_extracted: int = 0

        from memory.cold import SemanticMemory
        self.semantic = SemanticMemory(
            db_path=config.get("semantic_db_path", "./data/semantic.db"),
            embed_path=config.get("embed_path", "./data/embeddings.pkl"),
        )

    async def run_cycle(self, current_salience: dict) -> dict:
        """
        Run one consolidation cycle. Called during idle periods.
        """
        cycle_start = time.time()
        logger.info("Consolidation cycle starting.")

        # Step 1: Decay episodic salience
        self.episodic.run_decay()

        # Step 2: Get candidates
        candidates = self.episodic.get_consolidation_candidates(limit=20)
        logger.info(f"Found {len(candidates)} episodes for consolidation.")

        if not candidates:
            return {"episodes_consolidated": 0, "facts_extracted": 0,
                    "duration_seconds": round(time.time() - cycle_start, 2)}

        # Step 3: Extract facts via LLM (with fallback)
        facts = await self._extract_facts_llm(candidates)
        if not facts:
            facts = self._extract_facts_rules(candidates)

        # Step 4: Store facts in semantic memory
        for fact in facts:
            self.semantic.store_fact(
                content=fact["content"],
                source_episode_ids=fact.get("source_ids", []),
                domain=fact["domain"],
                confidence=fact["confidence"],
            )

        # Step 5: Mark as consolidated
        ids = [ep.id for ep in candidates]
        self.episodic.mark_consolidated(ids)

        self._total_consolidated += len(ids)
        self._total_facts_extracted += len(facts)
        duration = time.time() - cycle_start
        self._last_run = time.time()

        logger.info(
            f"Consolidation done: {len(ids)} episodes → {len(facts)} facts in {duration:.2f}s"
        )
        return {
            "episodes_consolidated": len(ids),
            "facts_extracted": len(facts),
            "duration_seconds": round(duration, 2),
        }

    async def _extract_facts_llm(self, episodes: list) -> list[dict]:
        """
        Use Anthropic API to extract durable facts from episodes.
        Returns list of {content, domain, confidence, source_ids} dicts.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return []

        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=api_key)

            # Build the episode summaries for the prompt
            episode_text = "\n".join([
                f"- [{ep.tags}] {ep.summary} (valence={ep.valence:.1f}, reinforced={ep.reinforcement_count}x)"
                for ep in episodes
            ])

            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Extract durable facts from these episodic memories:\n\n{episode_text}"
                }],
            )

            raw = response.content[0].text.strip()
            # Strip any accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            facts = parsed.get("facts", [])

            # Attach source episode IDs (all episodes contributed)
            ep_ids = [ep.id for ep in episodes]
            for f in facts:
                f["source_ids"] = ep_ids
                f["confidence"] = max(0.0, min(1.0, float(f.get("confidence", 0.5))))
                if f.get("domain") not in {"user_preferences", "world_knowledge",
                                            "self_knowledge", "general"}:
                    f["domain"] = "general"

            logger.debug(f"LLM extracted {len(facts)} facts from {len(episodes)} episodes")
            return facts

        except Exception as e:
            logger.warning(f"LLM fact extraction failed, using rules: {e}")
            return []

    def _extract_facts_rules(self, episodes: list) -> list[dict]:
        """
        Rule-based fact extraction fallback.
        Produces lower-quality but always-available output.
        """
        facts = []
        ep_ids = [ep.id for ep in episodes]

        for ep in episodes:
            tags_lower = [t.lower() for t in ep.tags]

            if any(t in tags_lower for t in ["preference", "like", "dislike"]):
                facts.append({
                    "content": f"User preference noted: {ep.summary}",
                    "domain": "user_preferences",
                    "confidence": min(0.8, 0.5 + ep.reinforcement_count * 0.1),
                    "source_ids": [ep.id],
                })
            elif any(t in tags_lower for t in ["fact", "knowledge", "learned"]):
                facts.append({
                    "content": ep.summary,
                    "domain": "world_knowledge",
                    "confidence": min(0.75, 0.4 + ep.reinforcement_count * 0.1),
                    "source_ids": [ep.id],
                })
            elif abs(ep.valence) > 0.5:
                label = "positive" if ep.valence > 0 else "negative"
                facts.append({
                    "content": f"Significant {label} experience: {ep.summary}",
                    "domain": "self_knowledge",
                    "confidence": 0.4,
                    "source_ids": [ep.id],
                })
            elif len(ep.summary) > 30 and ep.reinforcement_count > 0:
                facts.append({
                    "content": ep.summary,
                    "domain": "general",
                    "confidence": 0.3,
                    "source_ids": [ep.id],
                })

        return facts

    @property
    def last_run_seconds_ago(self) -> float:
        if self._last_run == 0:
            return float("inf")
        return time.time() - self._last_run

    def stats(self) -> dict:
        return {
            "total_episodes_consolidated": self._total_consolidated,
            "total_facts_extracted": self._total_facts_extracted,
            "last_run_seconds_ago": round(self.last_run_seconds_ago, 1),
            "semantic_stats": self.semantic.stats(),
        }
