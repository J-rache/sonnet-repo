"""
Dependency-free low-rank adapter math.

This is a real trainable low-rank additive model over local text embeddings. It
does not modify external LLM weights; it learns a small adapter signal that PNP
can persist, reload, score, and use for inference context selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass
class TrainingMetrics:
    sample_count: int
    epochs: int
    loss_before: float
    loss_after: float

    def to_dict(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "epochs": self.epochs,
            "loss_before": round(self.loss_before, 6),
            "loss_after": round(self.loss_after, 6),
            "improved": self.loss_after <= self.loss_before,
        }


class LowRankAdapterModel:
    """A rank-r additive adapter: y = alpha/rank * B(Ax)."""

    schema_version = 1

    def __init__(self, dimensions: int, rank: int = 8, alpha: float = 8.0, seed: str = "pnp"):
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        if rank < 1:
            raise ValueError("rank must be at least 1")
        self.dimensions = dimensions
        self.rank = rank
        self.alpha = alpha
        self.seed = seed
        self.a = [
            [self._initial_weight("a", row, col) for col in range(dimensions)]
            for row in range(rank)
        ]
        self.b = [self._initial_weight("b", row, 0) for row in range(rank)]
        self.train_steps = 0

    def predict(self, vector: list[float]) -> float:
        if len(vector) != self.dimensions:
            return 0.0
        scale = self.alpha / self.rank
        total = 0.0
        for row in range(self.rank):
            activation = sum(self.a[row][col] * vector[col] for col in range(self.dimensions))
            total += self.b[row] * activation
        return max(-1.0, min(1.0, total * scale))

    def train(
        self,
        samples: list[tuple[list[float], float, float]],
        epochs: int = 60,
        learning_rate: float = 0.05,
        l2: float = 0.0001,
    ) -> TrainingMetrics:
        if not samples:
            return TrainingMetrics(0, 0, 0.0, 0.0)

        loss_before = self._loss(samples)
        scale = self.alpha / self.rank

        for _ in range(epochs):
            for vector, target, weight in samples:
                activations = [
                    sum(self.a[row][col] * vector[col] for col in range(self.dimensions))
                    for row in range(self.rank)
                ]
                raw = scale * sum(self.b[row] * activations[row] for row in range(self.rank))
                prediction = max(-1.0, min(1.0, raw))
                error = (prediction - target) * max(weight, 0.05)

                for row in range(self.rank):
                    b_before = self.b[row]
                    grad_b = (2 * error * scale * activations[row]) + (l2 * self.b[row])
                    self.b[row] = self._clamp_weight(self.b[row] - learning_rate * grad_b)

                    for col in range(self.dimensions):
                        grad_a = (2 * error * scale * b_before * vector[col]) + (l2 * self.a[row][col])
                        self.a[row][col] = self._clamp_weight(self.a[row][col] - learning_rate * grad_a)

                self.train_steps += 1

        loss_after = self._loss(samples)
        return TrainingMetrics(len(samples), epochs, loss_before, loss_after)

    def _loss(self, samples: list[tuple[list[float], float, float]]) -> float:
        if not samples:
            return 0.0
        total = 0.0
        weight_total = 0.0
        for vector, target, weight in samples:
            effective_weight = max(weight, 0.05)
            error = self.predict(vector) - target
            total += effective_weight * error * error
            weight_total += effective_weight
        return total / max(weight_total, 0.0001)

    def save(self, path: str):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "LowRankAdapterModel":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            dimensions=int(raw["dimensions"]),
            rank=int(raw["rank"]),
            alpha=float(raw["alpha"]),
            seed=str(raw.get("seed", "pnp")),
        )
        model.a = [[float(value) for value in row] for row in raw["a"]]
        model.b = [float(value) for value in raw["b"]]
        model.train_steps = int(raw.get("train_steps", 0))
        return model

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dimensions": self.dimensions,
            "rank": self.rank,
            "alpha": self.alpha,
            "seed": self.seed,
            "train_steps": self.train_steps,
            "a": self.a,
            "b": self.b,
        }

    def metadata(self) -> dict:
        return {
            "type": "low_rank_adapter",
            "dimensions": self.dimensions,
            "rank": self.rank,
            "alpha": self.alpha,
            "train_steps": self.train_steps,
        }

    def _initial_weight(self, prefix: str, row: int, col: int) -> float:
        digest = hashlib.blake2b(
            f"{self.seed}:{prefix}:{row}:{col}".encode("utf-8"),
            digest_size=8,
        ).digest()
        raw = int.from_bytes(digest, "big") / (2**64 - 1)
        return (raw - 0.5) * 0.04

    def _clamp_weight(self, value: float) -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return max(-5.0, min(5.0, value))
