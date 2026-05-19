"""
adapters/lora.py

Experience adapter layer for PNP.

The adapter keeps structured deltas, trains a persisted low-rank additive model
over local text embeddings, and uses that learned signal to select adaptation
context for inference. It does not claim to modify private external LLM weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import time
from typing import Optional

from adapters.low_rank import LowRankAdapterModel
from memory.embedding import HashingTextEmbedder, cosine_similarity

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
    """A learning signal extracted from an interaction."""

    content: str
    feedback: float
    domain: str
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "feedback": self.feedback,
            "domain": self.domain,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ExperienceDelta":
        return cls(
            content=str(raw.get("content", "")),
            feedback=float(raw.get("feedback", 0.0)),
            domain=str(raw.get("domain", "general")),
            confidence=float(raw.get("confidence", 0.5)),
            timestamp=float(raw.get("timestamp", time.time())),
        )


class ExperienceAdapter:
    """Manages structured deltas plus a trainable low-rank adapter model."""

    MAX_DELTAS_IN_MEMORY = 1000
    CHECKPOINT_INTERVAL = 100

    def __init__(self, model_id: str, config: dict):
        self.model_id = model_id
        self.config = config
        self.adapter_path = config.get("adapter_path", "./data/adapter")
        os.makedirs(self.adapter_path, exist_ok=True)

        self.MAX_DELTAS_IN_MEMORY = int(config.get("adapter_max_deltas_in_memory", self.MAX_DELTAS_IN_MEMORY))
        self.CHECKPOINT_INTERVAL = int(config.get("adapter_checkpoint_interval", self.CHECKPOINT_INTERVAL))
        self.embedding_dimensions = int(config.get("embedding_dimensions", 128))
        self.embedder = HashingTextEmbedder(self.embedding_dimensions)
        self.model_path = os.path.join(self.adapter_path, "low_rank_adapter.json")
        self.model = self._load_or_create_model()

        self._deltas: list[ExperienceDelta] = []
        self._update_count: int = 0
        self._checkpoints: list[AdapterCheckpoint] = []
        self._domain_weights: dict[str, float] = {}
        self._last_training_metrics: Optional[dict] = None
        self._last_trained_at: Optional[float] = None

        self._load_state()
        logger.info("ExperienceAdapter initialized. %s prior updates.", self._update_count)

    def apply_delta(self, delta: ExperienceDelta, invariant_check: bool = True):
        from adapters.invariant import ConstitutionalInvariant

        if invariant_check:
            invariant = ConstitutionalInvariant()
            if not invariant.allows_update(delta):
                logger.warning("Adapter update blocked by invariant layer: %s...", delta.content[:50])
                return False

        self._deltas.append(delta)
        self._update_count += 1
        self._domain_weights[delta.domain] = (
            self._domain_weights.get(delta.domain, 0.0) + abs(delta.feedback) * delta.confidence
        )

        if len(self._deltas) > self.MAX_DELTAS_IN_MEMORY:
            self._deltas = self._deltas[-self.MAX_DELTAS_IN_MEMORY:]

        if self._should_auto_train():
            self.train_adapter()

        if self._update_count % self.CHECKPOINT_INTERVAL == 0:
            self._checkpoint()

        self._save_state()
        return True

    def train_adapter(self, epochs: Optional[int] = None) -> dict:
        samples = self._training_samples()
        epochs = int(epochs or self.config.get("adapter_training_epochs", 80))
        learning_rate = float(self.config.get("adapter_learning_rate", 0.05))
        metrics = self.model.train(samples, epochs=epochs, learning_rate=learning_rate)
        self.model.save(self.model_path)
        self._last_training_metrics = metrics.to_dict()
        self._last_trained_at = time.time()
        self._save_state()
        return self._last_training_metrics

    def score_text(self, text: str) -> float:
        return self.model.predict(self.embedder.embed(text))

    def get_adaptation_context(self, query: str, domain: Optional[str] = None) -> str:
        query_embedding = self.embedder.embed(query)
        scored: list[tuple[float, ExperienceDelta]] = []

        for delta in self._deltas:
            if domain is not None and delta.domain != domain:
                continue
            if delta.feedback <= 0.1:
                continue

            delta_embedding = self.embedder.embed(self._delta_text(delta))
            similarity = cosine_similarity(query_embedding, delta_embedding) if query.strip() else 1.0
            adapter_score = self.score_text(delta.content)
            if similarity <= 0 and adapter_score <= 0:
                continue

            score = (similarity * 0.55) + (adapter_score * 0.25) + (delta.confidence * 0.20)
            scored.append((score, delta))

        if not scored:
            return ""

        scored.sort(key=lambda item: (item[0], item[1].timestamp), reverse=True)
        top_deltas = [delta for _, delta in scored[:5]]

        parts = ["=== LEARNED ADAPTATIONS ==="]
        for delta in top_deltas:
            score = self.score_text(delta.content)
            parts.append(f"[domain:{delta.domain} score:{score:.3f}] {delta.content}")
        parts.append("=== END ADAPTATIONS ===")
        return "\n".join(parts)

    def detect_drift(self) -> Optional[dict]:
        if len(self._deltas) < 10:
            return None

        recent = self._deltas[-20:]
        avg_feedback = sum(delta.feedback for delta in recent) / len(recent)
        if avg_feedback < -0.3:
            return {
                "drift_type": "negative_feedback_trend",
                "severity": abs(avg_feedback),
                "recommendation": "review_recent_updates",
            }

        max_domain_weight = max(self._domain_weights.values(), default=0)
        total_weight = sum(self._domain_weights.values()) + 0.001
        if max_domain_weight / total_weight > 0.8 and len(self._domain_weights) > 1:
            dominant_domain = max(self._domain_weights, key=self._domain_weights.get)
            return {
                "drift_type": "domain_overspecialization",
                "dominant_domain": dominant_domain,
                "severity": max_domain_weight / total_weight,
                "recommendation": "diversify_training_signal",
            }

        return None

    def _training_samples(self) -> list[tuple[list[float], float, float]]:
        samples = []
        for delta in self._deltas:
            vector = self.embedder.embed(self._delta_text(delta))
            target = max(-1.0, min(1.0, delta.feedback * delta.confidence))
            weight = max(0.05, min(1.0, delta.confidence))
            samples.append((vector, target, weight))
        return samples

    def _should_auto_train(self) -> bool:
        if not bool(self.config.get("adapter_auto_train", True)):
            return False
        min_deltas = int(self.config.get("adapter_train_min_deltas", 1))
        interval = int(self.config.get("adapter_train_interval", 1))
        return len(self._deltas) >= min_deltas and self._update_count % interval == 0

    def _delta_text(self, delta: ExperienceDelta) -> str:
        return f"{delta.domain} {delta.content}"

    def _load_or_create_model(self) -> LowRankAdapterModel:
        if os.path.exists(self.model_path):
            return LowRankAdapterModel.load(self.model_path)
        return LowRankAdapterModel(
            dimensions=self.embedding_dimensions,
            rank=int(self.config.get("adapter_rank", 8)),
            alpha=float(self.config.get("adapter_alpha", 8.0)),
            seed=str(self.config.get("adapter_seed", "pnp")),
        )

    def _checkpoint(self):
        checkpoint_id = f"ckpt_{self._update_count}_{int(time.time())}"
        checkpoint_path = os.path.join(self.adapter_path, f"{checkpoint_id}.json")

        checkpoint = AdapterCheckpoint(
            id=checkpoint_id,
            timestamp=time.time(),
            update_count=self._update_count,
            metadata={
                "domain_weights": self._domain_weights,
                "delta_count": len(self._deltas),
                "low_rank_adapter": self.model.metadata(),
            },
            path=checkpoint_path,
        )

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "id": checkpoint.id,
                "timestamp": checkpoint.timestamp,
                "update_count": checkpoint.update_count,
                "domain_weights": self._domain_weights,
                "recent_deltas": [delta.to_dict() for delta in self._deltas[-10:]],
                "low_rank_adapter": self.model.metadata(),
            }, f, indent=2)

        self._checkpoints.append(checkpoint)
        logger.info("Adapter checkpoint saved: %s", checkpoint_id)

    def _save_state(self):
        state_path = os.path.join(self.adapter_path, "state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": 2,
                "model_id": self.model_id,
                "update_count": self._update_count,
                "domain_weights": self._domain_weights,
                "delta_count": len(self._deltas),
                "deltas": [delta.to_dict() for delta in self._deltas[-self.MAX_DELTAS_IN_MEMORY:]],
                "low_rank_adapter": self.model.metadata(),
                "model_path": self.model_path,
                "last_training_metrics": self._last_training_metrics,
                "last_trained_at": self._last_trained_at,
            }, f, indent=2)

    def _load_state(self):
        state_path = os.path.join(self.adapter_path, "state.json")
        if not os.path.exists(state_path):
            return

        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)

        self._update_count = int(state.get("update_count", 0))
        self._domain_weights = dict(state.get("domain_weights", {}))
        self._deltas = [
            ExperienceDelta.from_dict(raw)
            for raw in state.get("deltas", [])
        ][-self.MAX_DELTAS_IN_MEMORY:]
        self._last_training_metrics = state.get("last_training_metrics")
        self._last_trained_at = state.get("last_trained_at")

    def stats(self) -> dict:
        return {
            "update_count": self._update_count,
            "deltas_in_memory": len(self._deltas),
            "checkpoints": len(self._checkpoints),
            "domain_weights": self._domain_weights,
            "drift_status": self.detect_drift(),
            "low_rank_adapter": self.model.metadata(),
            "last_training_metrics": self._last_training_metrics,
            "last_trained_at": self._last_trained_at,
        }
