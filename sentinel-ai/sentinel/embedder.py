from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import os


@dataclass
class EmbeddingConfig:
    project_id: str
    location: str = "us-central1"
    model_name: str = "text-embedding-005"  # Latest stable model
    api_key: Optional[str] = None  # Optional API key for direct API


class VertexEmbedder:
    """
    Text embedding wrapper supporting both Vertex AI and direct API with API keys.

    Authentication modes:
      1. Vertex AI (Application Default Credentials) - no api_key needed
      2. Direct Gemini API (API Key) - provide api_key parameter

    Vertex AI embedding models (per official docs):
      - gemini-embedding-001 (recommended, up to 3072 dimensions, 2048 token max)
      - text-embedding-005 (latest, up to 768 dimensions, 2048 token max)
      - text-multilingual-embedding-002 (multilingual, up to 768 dimensions, 2048 token max)
      - multilingual-e5-small (open model, 384 dimensions)
      - multilingual-e5-large (open model, 1024 dimensions)

    Direct API embedding models (with API key):
      - text-embedding-004 (latest)
      - embedding-001 (legacy)
    """
    def __init__(self, cfg: EmbeddingConfig):
        self.model_name = cfg.model_name
        self.api_key = cfg.api_key
        self.project_id = cfg.project_id
        self.location = cfg.location

        if cfg.api_key:
            # Use direct Gemini API with API key
            try:
                import google.generativeai as genai
                genai.configure(api_key=cfg.api_key)
                self.use_direct_api = True
                self.genai = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package required for API key authentication. "
                    "Install with: pip install google-generativeai"
                )
        else:
            # Use Vertex AI with Application Default Credentials
            import vertexai
            from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
            vertexai.init(project=cfg.project_id, location=cfg.location)
            self._model = TextEmbeddingModel.from_pretrained(cfg.model_name)
            self.TextEmbeddingInput = TextEmbeddingInput
            self.use_direct_api = False

    def embed(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        dimensionality: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            task_type: Task type for embedding optimization:
                - "RETRIEVAL_DOCUMENT" - For embedding corpus documents (default)
                - "RETRIEVAL_QUERY" - For embedding search queries
                - "CLASSIFICATION" - For classification tasks
                - "CLUSTERING" - For clustering tasks
                - "SEMANTIC_SIMILARITY" - For semantic similarity
            dimensionality: Optional output dimensions (e.g., 768, 256)
                Only supported for certain models like text-embedding-005

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if self.use_direct_api:
            # Direct API approach
            # Map task types to Direct API format
            task_type_map = {
                "RETRIEVAL_DOCUMENT": "retrieval_document",
                "RETRIEVAL_QUERY": "retrieval_query",
                "CLASSIFICATION": "classification",
                "CLUSTERING": "clustering",
                "SEMANTIC_SIMILARITY": "semantic_similarity",
            }
            api_task_type = task_type_map.get(task_type, "retrieval_document")

            embeddings = []
            for text in texts:
                result = self.genai.embed_content(
                    model=f"models/{self.model_name}",
                    content=text,
                    task_type=api_task_type
                )
                embeddings.append(result['embedding'])
            return embeddings
        else:
            # Vertex AI approach
            kwargs = {}
            if dimensionality is not None:
                kwargs['output_dimensionality'] = dimensionality

            if (self.model_name.startswith("text-embedding") or
                self.model_name.startswith("text-multilingual") or
                self.model_name.startswith("gemini-embedding")):
                # Create TextEmbeddingInput objects for the newer API
                inputs = [self.TextEmbeddingInput(text=text, task_type=task_type) for text in texts]
                embs = self._model.get_embeddings(inputs, **kwargs)
            else:
                # Legacy API or open models (e5-small, e5-large)
                embs = self._model.get_embeddings(texts)

            return [e.values for e in embs]