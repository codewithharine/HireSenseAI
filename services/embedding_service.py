"""Embedding generation, caching, and vector search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class VectorIndex(Protocol):
    """Search interface shared by FAISS and fallback indexes."""

    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        ...


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize vectors so inner product equals cosine similarity."""

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (embeddings / norms).astype("float32")


class EmbeddingService:
    """Lazy sentence-transformer wrapper with candidate embedding cache."""

    def __init__(self, model_name: str, cache_path: Path) -> None:
        self.model_name = model_name
        self.cache_path = cache_path
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: {}", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return normalize_embeddings(np.asarray(embeddings))

    def dataset_fingerprint(self, ids: list[str], texts: list[str]) -> str:
        payload = json.dumps({"ids": ids, "texts": texts, "model": self.model_name}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_candidate_embeddings(self, ids: list[str], texts: list[str]) -> np.ndarray:
        """Load cached embeddings when the candidate dataset has not changed."""

        fingerprint = self.dataset_fingerprint(ids, texts)
        if self.cache_path.exists():
            cached = np.load(self.cache_path, allow_pickle=False)
            if str(cached["fingerprint"]) == fingerprint:
                logger.info("Loaded cached candidate embeddings from {}", self.cache_path)
                return np.asarray(cached["embeddings"], dtype="float32")

        logger.info("Generating candidate embeddings for {} profiles", len(texts))
        embeddings = self.encode(texts)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cache_path, fingerprint=fingerprint, embeddings=embeddings)
        return embeddings


class FaissIndex:
    """FAISS cosine-similarity index over normalized vectors."""

    def __init__(self, embeddings: np.ndarray) -> None:
        if faiss is None:
            raise RuntimeError("faiss-cpu is not installed.")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype("float32"))

    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(query_embedding.astype("float32"), top_k)


class SklearnIndex:
    """Fallback cosine index for environments without FAISS."""

    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings
        self.index = NearestNeighbors(metric="cosine", algorithm="brute")
        self.index.fit(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = self.index.kneighbors(query_embedding, n_neighbors=min(top_k, len(self.embeddings)))
        similarities = 1 - distances
        return similarities.astype("float32"), indices


def build_index(embeddings: np.ndarray) -> VectorIndex:
    """Build the best available vector index."""

    if len(embeddings) == 0:
        raise ValueError("Cannot build an index with zero embeddings.")
    if faiss is not None:
        logger.info("Building FAISS index with {} vectors", len(embeddings))
        return FaissIndex(embeddings)
    logger.warning("FAISS is unavailable; using sklearn cosine search fallback.")
    return SklearnIndex(embeddings)
