from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import re
import numpy as np


def split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class GroundingResult:
    hallucination_rate: float
    total_sentences: int
    flagged: List[Dict[str, Any]]  # {sentence, max_similarity}
    threshold: float


def grounding_check(
    answer: str,
    retrieved_chunk_vectors: List[List[float]],
    embedder,
    *,
    threshold: float = 0.75,
) -> GroundingResult:
    sents = split_sentences(answer)
    if not sents or not retrieved_chunk_vectors:
        return GroundingResult(
            hallucination_rate=0.0,
            total_sentences=len(sents),
            flagged=[],
            threshold=threshold,
        )

    # Embed answer sentences with RETRIEVAL_DOCUMENT task type
    # (comparing against document embeddings)
    sent_vecs = embedder.embed(sents, task_type="RETRIEVAL_DOCUMENT")

    V = np.array(retrieved_chunk_vectors, dtype=np.float32)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    flagged: List[Dict[str, Any]] = []
    for sent, vec in zip(sents, sent_vecs):
        q = np.array(vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = V @ q
        best = float(np.max(sims)) if sims.size else 0.0

        if best < threshold:
            flagged.append({"sentence": sent, "max_similarity": best})

    rate = len(flagged) / max(len(sents), 1)
    return GroundingResult(
        hallucination_rate=rate,
        total_sentences=len(sents),
        flagged=flagged,
        threshold=threshold,
    )