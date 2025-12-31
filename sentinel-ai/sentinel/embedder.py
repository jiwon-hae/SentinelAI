from __future__ import annotations

from dataclasses import dataclass
from typing import List
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel


@dataclass
class EmbeddingConfig:
    project_id: str
    location: str = "us-central1"
    model_name: str = "textembedding-gecko@003"


class VertexEmbedder:
    """
    Vertex AI text embedding wrapper (batch).
    Used for:
      - embedding corpus chunks (index build)
      - embedding queries (retrieval)
      - embedding answer sentences (grounding check)
    """
    def __init__(self, cfg: EmbeddingConfig):
        vertexai.init(project=cfg.project_id, location=cfg.location)
        self._model = TextEmbeddingModel.from_pretrained(cfg.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embs = self._model.get_embeddings(texts)
        return [e.values for e in embs]