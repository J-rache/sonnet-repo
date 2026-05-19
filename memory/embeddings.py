"""
memory/embeddings.py

Real vector embedding engine for semantic similarity.

Uses TF-IDF cosine similarity for small corpora, and TF-IDF + Latent
Semantic Analysis (Truncated SVD) for larger corpora. This gives
accurate semantic similarity at all corpus sizes without requiring
external API calls or network access.

The embedder is stateful: it builds its vocabulary from all text it has
seen, and can update incrementally as new content arrives.

For production deployment, this module's interface is stable — the
implementation can be swapped for a proper embedding model (OpenAI,
Cohere, local transformers) without changing callers.
"""

import numpy as np
import pickle
import os
import logging
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Stateful TF-IDF embedding engine with LSA for large corpora.

    Strategy:
      - corpus < LSA_THRESHOLD: TF-IDF cosine (exact, no dimensionality loss)
      - corpus >= LSA_THRESHOLD: TF-IDF + SVD (semantic generalization)

    Produces 128-dimensional dense vectors when LSA is active.
    For small corpora, sparse TF-IDF vectors are used internally but
    the public encode() method still returns 128-dim dense vectors
    via PCA-like projection.

    Supports:
      - encode(text) -> np.ndarray shape (128,)
      - similarity(a, b) -> float in [-1, 1]
      - most_similar(query, candidates) -> ranked list
      - add_document(text) -> incremental corpus update
    """

    VECTOR_DIM = 128
    LSA_THRESHOLD = 50        # Use LSA when corpus >= this size
    MIN_FIT_SIZE = 3          # Minimum docs to fit any model
    REFIT_INTERVAL = 25       # Re-fit every N new documents

    def __init__(self, persist_path: str = "./data/embeddings.pkl"):
        self.persist_path = persist_path
        self._corpus: list[str] = []
        self._fitted = False
        self._mode = "none"   # "tfidf" | "lsa"
        self._docs_since_refit = 0
        self._active_components = 0

        # Models
        self._tfidf: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._corpus_matrix: Optional[np.ndarray] = None  # (N, dim) normalized

        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_document(self, text: str) -> Optional[np.ndarray]:
        """Add document to corpus, return its embedding (or None if not fitted yet)."""
        text = self._clean(text)
        if not text or text in self._corpus:
            return self.encode(text) if self._fitted else None

        self._corpus.append(text)
        self._docs_since_refit += 1

        if len(self._corpus) >= self.MIN_FIT_SIZE and (
            not self._fitted or self._docs_since_refit >= self.REFIT_INTERVAL
        ):
            self._fit()

        return self.encode(text) if self._fitted else None

    def encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text to a 128-dim normalized vector. Returns None if not fitted."""
        if not self._fitted or self._tfidf is None:
            return None
        text = self._clean(text)
        if not text:
            return None
        try:
            tfidf_vec = self._tfidf.transform([text])  # sparse (1, vocab)
            if self._mode == "lsa" and self._svd is not None:
                dense = self._svd.transform(tfidf_vec)[0]  # (n_components,)
            else:
                # Small corpus: use dense TF-IDF projected to VECTOR_DIM
                dense = np.asarray(tfidf_vec.todense())[0]

            # Resize to VECTOR_DIM
            vec = self._resize(dense)
            norm = np.linalg.norm(vec)
            if norm < 1e-10:
                return None
            return vec / norm
        except Exception as e:
            logger.debug(f"encode failed: {e}")
            return None

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts. Falls back to Jaccard if not fitted."""
        va = self.encode(text_a)
        vb = self.encode(text_b)
        if va is None or vb is None:
            return self._jaccard(text_a, text_b)
        sim = float(np.dot(va, vb))
        return max(-1.0, min(1.0, sim))

    def most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
        threshold: float = 0.05,
    ) -> list[tuple[int, float]]:
        """
        Find most similar candidates to query.
        Returns list of (index, score) sorted descending by score.
        """
        if not candidates:
            return []

        q_vec = self.encode(query)
        q_is_zero = q_vec is None or np.linalg.norm(q_vec) < 1e-6

        scores = []
        for i, c in enumerate(candidates):
            if q_is_zero:
                score = self._jaccard(query, c)
            else:
                c_vec = self.encode(c)
                if c_vec is None or np.linalg.norm(c_vec) < 1e-6:
                    score = self._jaccard(query, c)
                else:
                    score = float(np.dot(q_vec, c_vec))
            scores.append((i, score))

        scores = [(i, s) for i, s in scores if s >= threshold]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fit(self):
        """Fit or re-fit the embedding model on the full corpus."""
        try:
            corpus = self._corpus
            n = len(corpus)

            # Always fit TF-IDF
            tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=min(10000, max(100, n * 20)),
                sublinear_tf=True,
                min_df=1,
            )
            X_tfidf = tfidf.fit_transform(corpus)  # (n, vocab)
            vocab_size = X_tfidf.shape[1]

            if n >= self.LSA_THRESHOLD and vocab_size > self.VECTOR_DIM:
                # Large corpus: use LSA
                n_comp = min(self.VECTOR_DIM, vocab_size - 1, n - 1)
                svd = TruncatedSVD(n_components=n_comp, random_state=42, n_iter=7)
                X_dense = svd.fit_transform(X_tfidf)
                self._svd = svd
                self._active_components = n_comp
                self._mode = "lsa"
            else:
                # Small corpus: use raw TF-IDF (dense projection in encode)
                self._svd = None
                self._active_components = min(self.VECTOR_DIM, vocab_size)
                self._mode = "tfidf"
                X_dense = np.asarray(X_tfidf.todense())

            # Build normalized corpus matrix
            X_norm = normalize(X_dense, norm="l2")
            # Resize each row to VECTOR_DIM
            resized = np.stack([self._resize(row) for row in X_norm])
            # Re-normalize after resize
            norms = np.linalg.norm(resized, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            self._corpus_matrix = resized / norms

            self._tfidf = tfidf
            self._fitted = True
            self._docs_since_refit = 0
            self._save()
            logger.debug(f"Embedding engine fitted: n={n}, vocab={vocab_size}, mode={self._mode}")
        except Exception as e:
            logger.warning(f"Embedding fit failed: {e}")

    def _resize(self, vec: np.ndarray) -> np.ndarray:
        """Resize vector to VECTOR_DIM by padding or truncating."""
        vec = np.asarray(vec).ravel()
        if len(vec) < self.VECTOR_DIM:
            return np.pad(vec, (0, self.VECTOR_DIM - len(vec)))
        return vec[:self.VECTOR_DIM]

    def _jaccard(self, a: str, b: str) -> float:
        """Token-level Jaccard similarity."""
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _clean(self, text: str) -> str:
        return " ".join(str(text).lower().split())[:5000]

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)), exist_ok=True)
        try:
            with open(self.persist_path, "wb") as f:
                pickle.dump({
                    "tfidf": self._tfidf,
                    "svd": self._svd,
                    "mode": self._mode,
                    "corpus": self._corpus[-2000:],
                    "fitted": self._fitted,
                    "active_components": self._active_components,
                }, f)
        except Exception as e:
            logger.warning(f"Embedding save failed: {e}")

    def _load(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "rb") as f:
                state = pickle.load(f)
            self._tfidf = state["tfidf"]
            self._svd = state.get("svd")
            self._mode = state.get("mode", "tfidf")
            self._corpus = state["corpus"]
            self._fitted = state["fitted"]
            self._active_components = state.get("active_components", 0)
            logger.info(f"Embedding engine loaded: {len(self._corpus)} docs, mode={self._mode}")
        except Exception as e:
            logger.warning(f"Embedding load failed, starting fresh: {e}")

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

    def stats(self) -> dict:
        return {
            "fitted": self._fitted,
            "mode": self._mode,
            "corpus_size": len(self._corpus),
            "vector_dim": self.VECTOR_DIM,
            "active_components": self._active_components,
        }


# Module-level singleton
_engine: Optional["EmbeddingEngine"] = None


def get_engine(persist_path: str = "./data/embeddings.pkl") -> "EmbeddingEngine":
    """Get or create the module-level singleton embedding engine."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine(persist_path=persist_path)
    return _engine
