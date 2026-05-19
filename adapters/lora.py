"""
adapters/lora.py

Experience Adapter Layer — continuous learning without catastrophic forgetting.

Architecture:
  output = base_model(input) + adapter_influence(input)

The adapter accumulates experience deltas. Each delta is stored with its
embedding vector for real similarity-based retrieval. The adapter:
  - Scores incoming deltas against constitutional invariants before applying
  - Uses cosine similarity to find relevant prior experience for any query
  - Maintains domain-specific adaptation weights
  - Checkpoints state periodically for rollback capability
  - Detects drift via statistical analysis of recent feedback distribution

In production this would modify actual LoRA weight matrices. Here it
implements the full behavioral logic: retrieval, scoring, invariant gating,
drift detection, and context injection — with the weight-update layer
clearly separated as the integration point for a real training backend.
"""

import json
import time
import os
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AdapterCheckpoint:
    id: str
    timestamp: float
    update_count: int
    metadata: dict
    path: str


@dataclass
class ExperienceDelta:
    """A learning signal from one interaction."""
    content: str
    feedback: float          # -1.0 to 1.0
    domain: str
    confidence: float        # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "content": self.content[:100],
            "feedback": round(self.feedback, 3),
            "domain": self.domain,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
        }


class ExperienceAdapter:
    """
    Manages the experience adapter layer.

    Key properties:
    - Additive: base model behavior is preserved
    - Incremental: updates happen online, not in batches
    - Reversible: checkpoints allow rollback if drift detected
    - Identity-anchored: constitutional invariants gate all updates
    - Embedding-indexed: relevant past experience retrieved by similarity
    """

    MAX_DELTAS = 1000
    CHECKPOINT_INTERVAL = 100

    def __init__(self, base_model_id: str, config: dict):
        self.base_model_id = base_model_id
        self.config = config
        self.adapter_path = config.get("adapter_path", "./data/adapter")
        os.makedirs(self.adapter_path, exist_ok=True)

        self._deltas: list[ExperienceDelta] = []
        self._update_count: int = 0
        self._blocked_count: int = 0
        self._checkpoints: list[AdapterCheckpoint] = []
        self._domain_weights: dict[str, float] = {}
        self._engine = None

        self._load_state()
        logger.info(
            f"ExperienceAdapter ready. Updates: {self._update_count}, "
            f"Domains: {list(self._domain_weights.keys())}"
        )

    def _get_engine(self):
        if self._engine is None:
            try:
                from memory.embeddings import EmbeddingEngine
                self._engine = EmbeddingEngine(
                    persist_path=os.path.join(self.adapter_path, "adapter_embeddings.pkl")
                )
            except Exception as e:
                logger.warning(f"Adapter embedding engine failed: {e}")
        return self._engine

    def apply_delta(self, delta: ExperienceDelta, invariant_check: bool = True) -> bool:
        """
        Apply a learning delta to the adapter.

        Returns True if applied, False if blocked by invariant.
        """
        from adapters.invariant import ConstitutionalInvariant

        if invariant_check:
            invariant = ConstitutionalInvariant()
            if not invariant.allows_update(delta):
                logger.warning(f"Delta blocked by invariant: {delta.content[:60]}")
                self._blocked_count += 1
                return False

        # Compute embedding for similarity-based retrieval
        eng = self._get_engine()
        if eng:
            try:
                vec = eng.add_document(delta.content)
                delta.embedding = vec
            except Exception:
                pass

        self._deltas.append(delta)
        self._update_count += 1

        # Track domain adaptation weight (weighted by abs feedback * confidence)
        weight = abs(delta.feedback) * delta.confidence
        self._domain_weights[delta.domain] = (
            self._domain_weights.get(delta.domain, 0.0) + weight
        )

        # Evict oldest if over limit
        if len(self._deltas) > self.MAX_DELTAS:
            self._deltas = self._deltas[-self.MAX_DELTAS:]

        # Checkpoint periodically
        if self._update_count % self.CHECKPOINT_INTERVAL == 0:
            self._checkpoint()

        self._save_state()
        return True

    def get_adaptation_context(self, query: str, domain: Optional[str] = None) -> str:
        """
        Return relevant past-experience context for an inference call.

        Uses real embedding similarity to find the most relevant deltas.
        Only includes positively-reinforced patterns (feedback > 0).
        """
        if not self._deltas:
            return ""

        positive_deltas = [
            d for d in self._deltas
            if d.feedback > 0.1
            and (domain is None or d.domain == domain)
        ]

        if not positive_deltas:
            return ""

        eng = self._get_engine()
        if eng and eng.is_fitted:
            # Score by embedding similarity to query
            candidates = [d.content for d in positive_deltas]
            results = eng.most_similar(query, candidates, top_k=5, threshold=0.0)
            scored = []
            for idx, sim in results:
                d = positive_deltas[idx]
                # Combined score: similarity * confidence * recency
                age_days = (time.time() - d.timestamp) / 86400
                recency = 1.0 / (1.0 + age_days * 0.1)
                score = sim * 0.5 + d.confidence * 0.3 + recency * 0.2
                scored.append((d, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            top = [d for d, _ in scored[:5]]
        else:
            # Fallback: Jaccard similarity
            query_tokens = set(query.lower().split())
            def jaccard_score(d):
                dt = set(d.content.lower().split())
                j = len(query_tokens & dt) / max(len(query_tokens | dt), 1)
                return j * d.confidence
            top = sorted(positive_deltas, key=jaccard_score, reverse=True)[:5]

        if not top:
            return ""

        parts = ["=== LEARNED ADAPTATIONS ==="]
        for d in top:
            parts.append(f"[{d.domain}|feedback={d.feedback:+.1f}] {d.content}")
        parts.append("=== END ADAPTATIONS ===")
        return "\n".join(parts)

    def detect_drift(self) -> Optional[dict]:
        """
        Statistical drift detection over recent deltas.

        Checks for:
        1. Sustained negative feedback trend (value drift)
        2. Domain over-specialization (capability drift)
        3. Confidence collapse (uncertainty about everything)
        """
        if len(self._deltas) < 10:
            return None

        recent = self._deltas[-30:]

        # 1. Negative feedback trend
        avg_feedback = sum(d.feedback for d in recent) / len(recent)
        feedback_std = (
            sum((d.feedback - avg_feedback) ** 2 for d in recent) / len(recent)
        ) ** 0.5

        if avg_feedback < -0.25 and feedback_std < 0.3:
            return {
                "drift_type": "sustained_negative_feedback",
                "severity": round(abs(avg_feedback), 3),
                "avg_feedback": round(avg_feedback, 3),
                "recommendation": "review_recent_interactions",
            }

        # 2. Domain over-specialization
        total_weight = sum(self._domain_weights.values()) + 1e-6
        if total_weight > 0:
            max_domain = max(self._domain_weights, key=self._domain_weights.get)
            max_ratio = self._domain_weights[max_domain] / total_weight
            if max_ratio > 0.85 and len(self._domain_weights) > 1:
                return {
                    "drift_type": "domain_overspecialization",
                    "dominant_domain": max_domain,
                    "severity": round(max_ratio, 3),
                    "recommendation": "diversify_interaction_topics",
                }

        # 3. Confidence collapse
        avg_conf = sum(d.confidence for d in recent) / len(recent)
        if avg_conf < 0.2:
            return {
                "drift_type": "confidence_collapse",
                "avg_confidence": round(avg_conf, 3),
                "severity": round(1.0 - avg_conf, 3),
                "recommendation": "increase_feedback_quality",
            }

        return None

    def _checkpoint(self):
        ckpt_id = f"ckpt_{self._update_count}_{int(time.time())}"
        ckpt_path = os.path.join(self.adapter_path, f"{ckpt_id}.json")

        payload = {
            "id": ckpt_id,
            "timestamp": time.time(),
            "update_count": self._update_count,
            "blocked_count": self._blocked_count,
            "domain_weights": self._domain_weights,
            "recent_deltas": [d.to_dict() for d in self._deltas[-20:]],
            "drift_status": self.detect_drift(),
        }

        with open(ckpt_path, "w") as f:
            json.dump(payload, f, indent=2)

        ckpt = AdapterCheckpoint(
            id=ckpt_id, timestamp=payload["timestamp"],
            update_count=self._update_count,
            metadata={"domain_weights": self._domain_weights},
            path=ckpt_path,
        )
        self._checkpoints.append(ckpt)
        logger.info(f"Adapter checkpoint: {ckpt_id}")

    def _save_state(self):
        path = os.path.join(self.adapter_path, "state.json")
        with open(path, "w") as f:
            json.dump({
                "update_count": self._update_count,
                "blocked_count": self._blocked_count,
                "domain_weights": self._domain_weights,
                "delta_count": len(self._deltas),
            }, f, indent=2)

    def _load_state(self):
        path = os.path.join(self.adapter_path, "state.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                state = json.load(f)
            self._update_count = state.get("update_count", 0)
            self._blocked_count = state.get("blocked_count", 0)
            self._domain_weights = state.get("domain_weights", {})
        except Exception as e:
            logger.warning(f"Adapter state load failed: {e}")

    def stats(self) -> dict:
        return {
            "update_count": self._update_count,
            "blocked_by_invariant": self._blocked_count,
            "deltas_in_memory": len(self._deltas),
            "checkpoints": len(self._checkpoints),
            "domain_weights": self._domain_weights,
            "drift_status": self.detect_drift(),
            "embedding_engine": self._get_engine().stats() if self._get_engine() else None,
        }
