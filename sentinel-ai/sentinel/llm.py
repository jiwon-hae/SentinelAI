from __future__ import annotations

from dataclasses import dataclass
from typing import List
import time
import vertexai
from vertexai.preview.generative_models import GenerativeModel


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: int


class VertexGeminiClient:
    """
    Wrapper around Gemini on Vertex AI.
    """
    def __init__(self, project_id: str, location: str = "us-central1", model_name: str = "gemini-1.5-pro"):
        vertexai.init(project=project_id, location=location)
        self.model_name = model_name
        self.model = GenerativeModel(model_name)

    def generate(self, question: str, sources: List[str], *, temperature: float = 0.2, max_output_tokens: int = 600) -> LLMResult:
        system = (
            "You are a helpful assistant.\n"
            "Use ONLY the provided SOURCES to answer.\n"
            "If the SOURCES are insufficient, say you are unsure.\n"
            "Do not invent facts, numbers, dates, or citations.\n"
        )

        sources_block = "\n\n".join([f"[SOURCE {i+1}]\n{t}" for i, t in enumerate(sources)])
        prompt = f"{system}\nSOURCES:\n{sources_block}\n\nQUESTION:\n{question}\n\nANSWER:"

        t0 = time.time()
        resp = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )
        latency_ms = int((time.time() - t0) * 1000)

        return LLMResult(text=str(getattr(resp, "text", "") or "").strip(), model=self.model_name, latency_ms=latency_ms)