from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import time
import os


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: int


class VertexGeminiClient:
    """
    Wrapper around Gemini supporting both Vertex AI and direct API with API keys.

    Authentication modes:
      1. Vertex AI (Application Default Credentials) - no api_key needed
      2. Direct Gemini API (API Key) - provide api_key parameter

    Vertex AI models:
      - gemini-1.5-pro (latest, recommended)
      - gemini-1.5-flash (fast, cost-effective)
      - gemini-1.0-pro (legacy)

    Direct API models:
      - gemini-2.0-flash-exp (experimental)
      - gemini-1.5-pro (stable)
      - gemini-1.5-flash (fast)
    """
    def __init__(
        self,
        project_id: str,
        location: str = "global",
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.project_id = project_id
        self.location = location

        if api_key:
            # Use direct Gemini API with API key
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
                self.use_vertex = False
            except ImportError:
                raise ImportError(
                    "google-generativeai package required for API key authentication. "
                    "Install with: pip install google-generativeai"
                )
        else:
            # Use Vertex AI with Application Default Credentials
            import vertexai
            from vertexai.generative_models import GenerativeModel
            vertexai.init(project=project_id, location=location)
            self.model = GenerativeModel(model_name)
            self.use_vertex = True

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
        try:
            if self.use_vertex:
                # Vertex AI approach
                from vertexai.generative_models import GenerationConfig
                config = GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                resp = self.model.generate_content(prompt, generation_config=config)
            else:
                # Direct API approach
                config = {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
                resp = self.model.generate_content(prompt, generation_config=config)

            latency_ms = int((time.time() - t0) * 1000)

            # Extract text from response
            text = ""
            if hasattr(resp, 'text'):
                text = str(resp.text).strip()
            elif hasattr(resp, 'candidates') and resp.candidates:
                # Handle response structure
                candidate = resp.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text')]).strip()

            return LLMResult(text=text, model=self.model_name, latency_ms=latency_ms)
        except Exception as e:
            # Return error details for debugging
            latency_ms = int((time.time() - t0) * 1000)
            raise Exception(f"Gemini API error: {str(e)}") from e