"""
memory/consolidator.py

Background process that compresses episodic memory into semantic memory during
idle periods. It can ask the configured inference provider to extract durable
facts, then falls back to rules when no live provider is available.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.warm import EpisodicMemory

logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM_PROMPT = """You are a memory consolidation system for a persistent AI process.

Your job: given a list of episodic memory summaries, extract durable semantic facts
that should be retained long-term.

Rules:
- Extract only facts that are genuinely durable, not session-specific noise
- Classify each fact into one of: user_preferences, world_knowledge, self_knowledge, general
- Rate confidence 0.0-1.0 based on how certain and stable the fact is
- Be concise; facts should be 1-2 sentences max
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
    2. Extract semantic facts from low-salience episodes
    3. Store extracted facts in semantic memory
    4. Mark consolidated episodes
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
        """Run one consolidation cycle."""
        cycle_start = time.time()
        logger.info("Consolidation cycle starting.")

        self.episodic.run_decay()
        candidates = self.episodic.get_consolidation_candidates(limit=20)
        logger.info("Found %s episodes for consolidation.", len(candidates))

        if not candidates:
            return {
                "episodes_consolidated": 0,
                "facts_extracted": 0,
                "duration_seconds": round(time.time() - cycle_start, 2),
            }

        facts = await self._extract_facts_provider(candidates)
        if not facts:
            facts = self._extract_facts_rules(candidates)

        for fact in facts:
            self.semantic.store_fact(
                content=fact["content"],
                source_episode_ids=fact.get("source_ids", []),
                domain=fact["domain"],
                confidence=fact["confidence"],
            )

        ids = [ep.id for ep in candidates]
        self.episodic.mark_consolidated(ids)

        self._total_consolidated += len(ids)
        self._total_facts_extracted += len(facts)
        duration = time.time() - cycle_start
        self._last_run = time.time()

        logger.info("Consolidation done: %s episodes -> %s facts in %.2fs", len(ids), len(facts), duration)
        return {
            "episodes_consolidated": len(ids),
            "facts_extracted": len(facts),
            "duration_seconds": round(duration, 2),
        }

    async def _extract_facts_provider(self, episodes: list) -> list[dict]:
        """Use the configured inference provider to extract durable facts."""
        provider_name = os.environ.get("PNP_INFERENCE_PROVIDER") or self.config.get("inference_provider", "mock")
        if provider_name.lower() == "mock":
            return []

        try:
            from inference.engine import InferenceRequest
            from inference.providers import provider_from_config

            episode_text = "\n".join(
                f"- [{ep.tags}] {ep.summary} (valence={ep.valence:.1f}, reinforced={ep.reinforcement_count}x)"
                for ep in episodes
            )
            user_input = f"Extract durable facts from these episodic memories:\n\n{episode_text}"
            model = (
                os.environ.get("PNP_CONSOLIDATION_MODEL_ID")
                or os.environ.get("PNP_MODEL_ID")
                or os.environ.get("PNP_MODEL")
                or self.config.get("consolidation_model_id")
                or self.config.get("model_id")
                or self.config.get("base_model")
                or "local-model"
            )
            request = InferenceRequest(
                user_input=user_input,
                working_memory_context="",
                episodic_context="",
                semantic_context="",
                adaptation_context="",
                core_state={},
                model=model,
                max_tokens=int(self.config.get("consolidation_max_tokens", 1000)),
            )
            provider = provider_from_config(self.config)
            provider_response = await provider.generate(
                request,
                EXTRACTION_SYSTEM_PROMPT,
                [{"role": "user", "content": user_input}],
            )
            return self._parse_provider_facts(provider_response.content, episodes)
        except Exception as e:
            logger.warning("Provider fact extraction failed, using rules: %s", e)
            return []

    def _parse_provider_facts(self, raw_text: str, episodes: list) -> list[dict]:
        raw = raw_text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        facts = parsed.get("facts", [])

        ep_ids = [ep.id for ep in episodes]
        normalized = []
        for fact in facts:
            if not isinstance(fact, dict) or not fact.get("content"):
                continue
            domain = fact.get("domain", "general")
            if domain not in {"user_preferences", "world_knowledge", "self_knowledge", "general"}:
                domain = "general"
            normalized.append(
                {
                    "content": str(fact["content"]),
                    "domain": domain,
                    "confidence": max(0.0, min(1.0, float(fact.get("confidence", 0.5)))),
                    "source_ids": ep_ids,
                }
            )

        logger.debug("Provider extracted %s facts from %s episodes", len(normalized), len(episodes))
        return normalized

    def _extract_facts_rules(self, episodes: list) -> list[dict]:
        """Rule-based fact extraction fallback."""
        facts = []

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
