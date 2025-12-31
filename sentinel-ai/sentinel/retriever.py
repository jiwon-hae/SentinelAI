from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / denom


class InMemoryRetriever:
    """
    Simple cosine similarity retriever over pre-embedded chunk vectors.
    Hackathon-friendly (no vector DB required).
    """
    def __init__(self, chunk_ids: List[str], chunk_texts: List[str], chunk_vectors: List[List[float]]):
        if len(chunk_ids) != len(chunk_texts) or len(chunk_ids) != len(chunk_vectors):
            raise ValueError("chunk_ids, chunk_texts, chunk_vectors must have same length.")

        self.chunk_ids = chunk_ids
        self.chunk_texts = chunk_texts
        V = np.array(chunk_vectors, dtype=np.float32)
        self.V = _normalize_rows(V)

    def top_k(self, query_vec: List[float], k: int = 4) -> List[RetrievedChunk]:
        if not self.chunk_ids:
            return []

        q = np.array([query_vec], dtype=np.float32)
        q = _normalize_rows(q)
        sims = (self.V @ q.T).reshape(-1)

        k = min(k, sims.shape[0])
        idx = np.argsort(-sims)[:k]

        return [
            RetrievedChunk(
                chunk_id=self.chunk_ids[int(i)],
                score=float(sims[int(i)]),
                text=self.chunk_texts[int(i)],
            )
            for i in idx
        ]