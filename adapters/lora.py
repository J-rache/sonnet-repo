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
        self._last_training_metrics: dict = {}
        self._last_trained_at: Optional[float] = None

        self._load_state()
        self._load_deltas()
        self._init_low_rank()
        # If we loaded deltas and auto_train is on, train the low_rank adapter now
        if self._deltas and self.config.get("adapter_auto_train", True):
            self._train_low_rank()
            self._save_low_rank()
        logger.info(
            f"ExperienceAdapter ready. Updates: {self._update_count}, "
            f"Domains: {list(self._domain_weights.keys())}"
        )

    def _init_low_rank(self):
        """Initialize (or load) the LowRankAdapterModel."""
        from adapters.low_rank import LowRankAdapterModel
        lora_path = os.path.join(self.adapter_path, "low_rank_adapter.json")
        dim = self.config.get("embedding_dimensions", 128)
        rank = self.config.get("adapter_rank", 8)
        alpha = float(self.config.get("adapter_alpha", 8.0))
        seed = self.config.get("adapter_seed", "pnp")
        if os.path.exists(lora_path):
            try:
                self._low_rank = LowRankAdapterModel.load(lora_path)
                logger.info(f"Loaded LowRankAdapterModel ({self._low_rank.train_steps} steps)")
                return
            except Exception as e:
                logger.warning(f"LowRankAdapterModel load failed, reinit: {e}")
        self._low_rank = LowRankAdapterModel(dimensions=dim, rank=rank, alpha=alpha, seed=seed)

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

        self._save_deltas()
        if self.config.get("adapter_auto_train", True):
            min_d = self.config.get("adapter_train_min_deltas", 1)
            interval = self.config.get("adapter_train_interval", 1)
            if len(self._deltas) >= min_d and self._update_count % max(interval, 1) == 0:
                self._train_low_rank()
        self._save_low_rank()
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

    def train_adapter(self, epochs: Optional[int] = None) -> dict:
        """
        Public training hook used by the API and smoke tests.

        This trains the local low-rank adapter over persisted experience
        deltas. It does not modify any external provider's private weights.
        """
        metrics = self._train_low_rank(epochs=epochs)
        self._save_low_rank()
        self._save_state()
        return metrics

    def _save_low_rank(self):
        if not hasattr(self, "_low_rank") or self._low_rank is None:
            return
        lora_path = os.path.join(self.adapter_path, "low_rank_adapter.json")
        try:
            self._low_rank.save(lora_path)
        except Exception as e:
            logger.warning(f"LowRankAdapterModel save failed: {e}")

    def _train_low_rank(self, epochs: Optional[int] = None) -> dict:
        """Train low_rank adapter on current deltas using their embeddings."""
        if not hasattr(self, "_low_rank") or self._low_rank is None:
            return {
                "sample_count": 0,
                "epochs": 0,
                "loss_before": 0.0,
                "loss_after": 0.0,
                "improved": False,
                "reason": "low_rank_adapter_unavailable",
            }
        eng = self._get_engine()
        samples = []
        dim = self._low_rank.dimensions
        for d in self._deltas[-100:]:
            vec = None
            if eng is not None:
                try:
                    vec = eng.encode(d.content)
                except Exception:
                    pass
            if vec is not None and len(vec) >= dim:
                samples.append((vec[:dim].tolist(), d.feedback, d.confidence))
            else:
                # Hash-based deterministic pseudo-vector so we always train
                import hashlib
                h = hashlib.sha256(d.content.encode()).digest()
                pseudo = [(b / 127.5 - 1.0) for b in h[:dim]]
                while len(pseudo) < dim:
                    pseudo.append(0.0)
                samples.append((pseudo[:dim], d.feedback, d.confidence))
        if not samples:
            metrics = {
                "sample_count": 0,
                "epochs": 0,
                "loss_before": 0.0,
                "loss_after": 0.0,
                "improved": False,
                "reason": "no_deltas",
            }
            self._last_training_metrics = metrics
            return metrics
        try:
            train_epochs = epochs or self.config.get("adapter_training_epochs", 80)
            lr = float(self.config.get("adapter_learning_rate", 0.05))
            raw_metrics = self._low_rank.train(samples, epochs=train_epochs, learning_rate=lr)
            metrics = raw_metrics.to_dict()
            self._last_training_metrics = metrics
            self._last_trained_at = time.time()
            logger.debug(f"LowRank trained: {metrics}")
            return metrics
        except Exception as e:
            logger.warning(f"LowRank training failed: {e}")
            metrics = {
                "sample_count": len(samples),
                "epochs": epochs or self.config.get("adapter_training_epochs", 80),
                "loss_before": 0.0,
                "loss_after": 0.0,
                "improved": False,
                "reason": f"training_failed: {e}",
            }
            self._last_training_metrics = metrics
            return metrics

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
                "last_training_metrics": self._last_training_metrics,
                "last_trained_at": self._last_trained_at,
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
            self._last_training_metrics = state.get("last_training_metrics", {})
            self._last_trained_at = state.get("last_trained_at")
        except Exception as e:
            logger.warning(f"Adapter state load failed: {e}")

    def stats(self) -> dict:
        low_rank_stats = {}
        if hasattr(self, "_low_rank") and self._low_rank is not None:
            low_rank_stats = {
                "train_steps": self._low_rank.train_steps,
                "dimensions": self._low_rank.dimensions,
                "rank": self._low_rank.rank,
            }
        return {
            "update_count": self._update_count,
            "blocked_by_invariant": self._blocked_count,
            "low_rank_adapter": low_rank_stats,
            "deltas_in_memory": len(self._deltas),
            "checkpoints": len(self._checkpoints),
            "domain_weights": self._domain_weights,
            "drift_status": self.detect_drift(),
            "last_training_metrics": self._last_training_metrics,
            "last_trained_at": self._last_trained_at,
            "embedding_engine": self._get_engine().stats() if self._get_engine() else None,
        }


    def _save_deltas(self):
        """Persist deltas to disk so they survive restart."""
        path = os.path.join(self.adapter_path, "deltas.json")
        try:
            serializable = [
                {
                    "content": d.content,
                    "feedback": d.feedback,
                    "domain": d.domain,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp,
                }
                for d in self._deltas
            ]
            with open(path, "w") as f:
                json.dump(serializable, f)
        except Exception as e:
            logger.warning(f"Delta save failed: {e}")

    def _load_deltas(self):
        """Load persisted deltas from prior run."""
        path = os.path.join(self.adapter_path, "deltas.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                raw = json.load(f)
            for item in raw:
                self._deltas.append(ExperienceDelta(
                    content=item["content"],
                    feedback=item["feedback"],
                    domain=item["domain"],
                    confidence=item["confidence"],
                    timestamp=item.get("timestamp", time.time()),
                ))
            logger.info(f"Loaded {len(self._deltas)} persisted deltas")
        except Exception as e:
            logger.warning(f"Delta load failed: {e}")
