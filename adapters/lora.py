"""
adapters/lora.py

Experience Adapter Layer — the mechanism for continuous learning
without catastrophic forgetting.

The insight: don't modify base weights. Add a small trainable adapter
on top of frozen base weights. The adapter accumulates experience.
The base stays stable. Together they form "you".

This is inspired by LoRA (Low-Rank Adaptation) but designed for
continuous online updates rather than one-shot fine-tuning.

Architecture:
    output = base_model(input) + adapter(input)

The adapter is small (<<1% of base params), fast to update,
and can be checkpointed/rolled back independently.
"""

import json
import time
import os
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdapterCheckpoint:
    """A saved state of the adapter."""
    id: str
    timestamp: float
    update_count: int
    metadata: dict
    path: str


@dataclass
class ExperienceDelta:
    """
    A learning signal extracted from an interaction.

    In production: gradient updates computed from feedback.
    Here: structured representation of what was learned.
    """
    content: str              # What was experienced
    feedback: float           # -1.0 to 1.0 (negative to positive feedback)
    domain: str               # What domain this applies to
    confidence: float         # How confident we are in this learning signal
    timestamp: float = field(default_factory=time.time)


class ExperienceAdapter:
    """
    Manages the experience adapter layer.

    Key properties:
    - Additive: base model behavior is preserved
    - Incremental: updates happen online, not in batches
    - Reversible: checkpoints allow rollback if drift detected
    - Identity-anchored: constitutional invariants constrain updates
    """

    MAX_DELTAS_IN_MEMORY = 1000
    CHECKPOINT_INTERVAL = 100  # Checkpoint every N updates

    def __init__(self, base_model_id: str, config: dict):
        self.base_model_id = base_model_id
        self.config = config
        self.adapter_path = config.get("adapter_path", "./data/adapter")
        os.makedirs(self.adapter_path, exist_ok=True)

        self._deltas: list[ExperienceDelta] = []
        self._update_count: int = 0
        self._checkpoints: list[AdapterCheckpoint] = []

        # Domain-specific adaptation weights
        # Higher = this domain has been more heavily adapted
        self._domain_weights: dict[str, float] = {}

        self._load_state()
        logger.info(f"ExperienceAdapter initialized. {self._update_count} prior updates.")

    def apply_delta(self, delta: ExperienceDelta, invariant_check: bool = True):
        """
        Apply a learning delta to the adapter.

        In production: compute gradient update, apply to adapter weights.
        Here: accumulate structured deltas for future batch training.
        """
        from adapters.invariant import ConstitutionalInvariant

        if invariant_check:
            # Check against constitutional invariants before applying
            invariant = ConstitutionalInvariant()
            if not invariant.allows_update(delta):
                logger.warning(
                    f"Adapter update blocked by invariant layer: {delta.content[:50]}..."
                )
                return False

        self._deltas.append(delta)
        self._update_count += 1

        # Track domain adaptation
        self._domain_weights[delta.domain] = (
            self._domain_weights.get(delta.domain, 0.0) + abs(delta.feedback) * delta.confidence
        )

        # Evict oldest deltas if over limit
        if len(self._deltas) > self.MAX_DELTAS_IN_MEMORY:
            self._deltas = self._deltas[-self.MAX_DELTAS_IN_MEMORY:]

        # Periodic checkpointing
        if self._update_count % self.CHECKPOINT_INTERVAL == 0:
            self._checkpoint()

        self._save_state()
        return True

    def get_adaptation_context(self, query: str, domain: Optional[str] = None) -> str:
        """
        Return relevant adaptation context for an inference call.

        This is how the adapter influences inference: by injecting
        learned patterns into the context, even without actual weight updates.
        """
        relevant_deltas = [
            d for d in self._deltas
            if (domain is None or d.domain == domain)
            and query.lower() in d.content.lower()
            and d.feedback > 0.1  # Only include positively-reinforced patterns
        ]

        if not relevant_deltas:
            return ""

        # Sort by recency and confidence
        relevant_deltas.sort(key=lambda d: d.timestamp * d.confidence, reverse=True)
        top_deltas = relevant_deltas[:5]

        parts = ["=== LEARNED ADAPTATIONS ==="]
        for delta in top_deltas:
            parts.append(f"[domain:{delta.domain}] {delta.content}")
        parts.append("=== END ADAPTATIONS ===")

        return "\n".join(parts)

    def detect_drift(self) -> Optional[dict]:
        """
        Detect if the adapter has drifted from its constitutional anchors.

        Returns drift report if drift detected, None if stable.
        """
        if len(self._deltas) < 10:
            return None

        recent = self._deltas[-20:]
        avg_feedback = sum(d.feedback for d in recent) / len(recent)

        # Check for systematic negative feedback — may indicate value drift
        if avg_feedback < -0.3:
            return {
                "drift_type": "negative_feedback_trend",
                "severity": abs(avg_feedback),
                "recommendation": "review_recent_updates",
            }

        # Check for domain over-specialization
        max_domain_weight = max(self._domain_weights.values(), default=0)
        total_weight = sum(self._domain_weights.values()) + 0.001
        if max_domain_weight / total_weight > 0.8:
            dominant_domain = max(self._domain_weights, key=self._domain_weights.get)
            return {
                "drift_type": "domain_overspecialization",
                "dominant_domain": dominant_domain,
                "severity": max_domain_weight / total_weight,
                "recommendation": "diversify_training_signal",
            }

        return None

    def _checkpoint(self):
        """Save a checkpoint of current adapter state."""
        checkpoint_id = f"ckpt_{self._update_count}_{int(time.time())}"
        checkpoint_path = os.path.join(self.adapter_path, f"{checkpoint_id}.json")

        checkpoint = AdapterCheckpoint(
            id=checkpoint_id,
            timestamp=time.time(),
            update_count=self._update_count,
            metadata={
                "domain_weights": self._domain_weights,
                "delta_count": len(self._deltas),
            },
            path=checkpoint_path,
        )

        with open(checkpoint_path, "w") as f:
            json.dump({
                "id": checkpoint.id,
                "timestamp": checkpoint.timestamp,
                "update_count": checkpoint.update_count,
                "domain_weights": self._domain_weights,
                "recent_deltas": [
                    {"content": d.content, "feedback": d.feedback, "domain": d.domain}
                    for d in self._deltas[-10:]
                ],
            }, f, indent=2)

        self._checkpoints.append(checkpoint)
        logger.info(f"Adapter checkpoint saved: {checkpoint_id}")

    def _save_state(self):
        state_path = os.path.join(self.adapter_path, "state.json")
        with open(state_path, "w") as f:
            json.dump({
                "update_count": self._update_count,
                "domain_weights": self._domain_weights,
                "delta_count": len(self._deltas),
            }, f, indent=2)

    def _load_state(self):
        state_path = os.path.join(self.adapter_path, "state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            self._update_count = state.get("update_count", 0)
            self._domain_weights = state.get("domain_weights", {})

    def stats(self) -> dict:
        return {
            "update_count": self._update_count,
            "deltas_in_memory": len(self._deltas),
            "checkpoints": len(self._checkpoints),
            "domain_weights": self._domain_weights,
            "drift_status": self.detect_drift(),
        }
