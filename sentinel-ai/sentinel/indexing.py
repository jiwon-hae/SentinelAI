from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    text: str


def simple_chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """
    Hackathon-friendly chunker:
      - splits by character length
      - overlaps to preserve context
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_chunks_from_file(doc_path: str) -> List[Chunk]:
    raw = load_text_file(doc_path)
    texts = simple_chunk_text(raw)
    return [Chunk(chunk_id=f"chunk_{i}", text=t) for i, t in enumerate(texts)]