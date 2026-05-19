"""
Small local text embeddings used by PNP memory and adapter retrieval.

The default embedder is deterministic and dependency-free. It is not a
transformer embedding model, but it does produce normalized dense vectors and
uses cosine similarity for retrieval instead of substring matching.
"""

from __future__ import annotations

import hashlib
import json
import math
import re


TOKEN_RE = re.compile(r"[a-zA-Z0-9_']+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class HashingTextEmbedder:
    """Deterministic hashed bag-of-words embedder."""

    model_id = "hashing-text-embedding-v1"

    def __init__(self, dimensions: int = 128):
        if dimensions < 8:
            raise ValueError("Embedding dimensions must be at least 8")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % self.dimensions
            sign = 1.0 if (raw >> 63) == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def metadata(self) -> dict:
        return {"model": self.model_id, "dimensions": self.dimensions}


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def vector_to_json(vector: list[float]) -> str:
    return json.dumps([round(value, 8) for value in vector], separators=(",", ":"))


def vector_from_json(raw: str | None) -> list[float]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [float(value) for value in parsed]
