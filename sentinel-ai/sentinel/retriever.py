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

        import time
        t0 = time.time()

        q = np.array([query_vec], dtype=np.float32)
        q = _normalize_rows(q)
        sims = (self.V @ q.T).reshape(-1)

        k = min(k, sims.shape[0])
        idx = np.argsort(-sims)[:k]

        results = [
            RetrievedChunk(
                chunk_id=self.chunk_ids[int(i)],
                score=float(sims[int(i)]),
                text=self.chunk_texts[int(i)],
            )
            for i in idx
        ]

        # Emit SLO metrics
        latency_ms = int((time.time() - t0) * 1000)
        avg_score = float(np.mean([r.score for r in results])) if results else 0.0
        self._emit_metrics(latency_ms, k, len(results), avg_score)

        return results

    def _emit_metrics(self, latency_ms: int, requested_k: int, retrieved_count: int, avg_score: float) -> None:
        """Emit component-level SLO metrics for retrieval"""
        try:
            from sentinel.telemetry import emit_component_metric

            tags = [f"top_k:{requested_k}"]

            # Latency metrics
            emit_component_metric("retriever", "latency_ms", latency_ms, "histogram", tags)

            # Quality metrics
            emit_component_metric("retriever", "avg_score", avg_score, "gauge", tags)
            emit_component_metric("retriever", "retrieved_count", retrieved_count, "gauge", tags)

            # Request count
            emit_component_metric("retriever", "requests", 1, "count", tags)
        except Exception:
            # Don't fail if metrics emission fails
            pass