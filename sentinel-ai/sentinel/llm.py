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
    # LLM Evaluation Metrics
    ttft_ms: int = 0                    # Time to first token (streaming only)
    output_tokens: int = 0              # Estimated output token count
    generation_time_ms: int = 0         # Time spent generating (after TTFT)


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

            # Emit SLO metrics for LLM generation
            self._emit_metrics(latency_ms, len(prompt), len(text), error=False)

            return LLMResult(text=text, model=self.model_name, latency_ms=latency_ms)
        except Exception as e:
            # Return error details for debugging
            latency_ms = int((time.time() - t0) * 1000)

            # Emit error metrics
            self._emit_metrics(latency_ms, len(prompt), 0, error=True)

            raise Exception(f"Gemini API error: {str(e)}") from e

    def generate_streaming(self, question: str, sources: List[str], *, temperature: float = 0.2, max_output_tokens: int = 600) -> LLMResult:
        """
        Generate answer using streaming mode to capture TTFT (Time to First Token).

        Returns LLMResult with:
        - ttft_ms: Time to first token
        - generation_time_ms: Time spent generating after first token
        - output_tokens: Estimated output token count
        """
        system = (
            "You are a helpful assistant.\n"
            "Use ONLY the provided SOURCES to answer.\n"
            "If the SOURCES are insufficient, say you are unsure.\n"
            "Do not invent facts, numbers, dates, or citations.\n"
        )

        sources_block = "\n\n".join([f"[SOURCE {i+1}]\n{t}" for i, t in enumerate(sources)])
        prompt = f"{system}\nSOURCES:\n{sources_block}\n\nQUESTION:\n{question}\n\nANSWER:"

        t0 = time.time()
        ttft_ms = 0
        full_text = ""

        try:
            if self.use_vertex:
                from vertexai.generative_models import GenerationConfig
                config = GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                response_iter = self.model.generate_content(prompt, generation_config=config, stream=True)
            else:
                config = {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
                response_iter = self.model.generate_content(prompt, generation_config=config, stream=True)

            # Iterate through streaming chunks
            for i, chunk in enumerate(response_iter):
                # Capture TTFT on first chunk
                if i == 0:
                    ttft_ms = int((time.time() - t0) * 1000)

                # Accumulate text from chunk
                if hasattr(chunk, 'text') and chunk.text:
                    full_text += chunk.text
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                full_text += part.text

            total_ms = int((time.time() - t0) * 1000)
            generation_time_ms = total_ms - ttft_ms

            # Estimate output tokens (~4 chars per token)
            output_tokens = max(1, len(full_text) // 4)

            # Emit metrics
            self._emit_metrics(total_ms, len(prompt), len(full_text), error=False)

            return LLMResult(
                text=full_text.strip(),
                model=self.model_name,
                latency_ms=total_ms,
                ttft_ms=ttft_ms,
                output_tokens=output_tokens,
                generation_time_ms=generation_time_ms,
            )

        except Exception as e:
            total_ms = int((time.time() - t0) * 1000)
            self._emit_metrics(total_ms, len(prompt), 0, error=True)
            raise Exception(f"Gemini streaming API error: {str(e)}") from e

    def _emit_metrics(self, latency_ms: int, prompt_len: int, output_len: int, error: bool = False) -> None:
        """Emit component-level SLO metrics for LLM generation"""
        try:
            from sentinel.telemetry import emit_component_metric

            tags = [f"model:{self.model_name}", f"error:{error}"]

            # Latency metrics
            emit_component_metric("llm", "latency_ms", latency_ms, "histogram", tags)

            # Token/character metrics (approximate)
            emit_component_metric("llm", "prompt_chars", prompt_len, "gauge", tags)
            emit_component_metric("llm", "output_chars", output_len, "gauge", tags)

            # Success/error counts
            emit_component_metric("llm", "requests", 1, "count", tags)
            if error:
                emit_component_metric("llm", "errors", 1, "count", tags)
        except Exception:
            # Don't fail if metrics emission fails
            pass