"""
vLLM Client for LLM Sentinel.

Provides an OpenAI-compatible client for vLLM server with streaming support
and TTFT (Time to First Token) measurement.

Usage:
    # Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-chat-hf \
        --port 8000

    # Then use this client:
    client = VLLMClient(api_url="http://localhost:8000", model_name="meta-llama/Llama-2-7b-chat-hf")
    result = client.generate_streaming(question, sources)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Iterator

from sentinel.llm import LLMResult


class VLLMClient:
    """
    vLLM client using OpenAI-compatible API.

    vLLM exposes an OpenAI-compatible endpoint, so we use the openai library
    to communicate with it. This allows easy switching between vLLM and other
    OpenAI-compatible backends.

    Supported models (examples):
      - meta-llama/Llama-2-7b-chat-hf
      - meta-llama/Llama-2-13b-chat-hf
      - mistralai/Mistral-7B-Instruct-v0.1
      - mistralai/Mixtral-8x7B-Instruct-v0.1
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        api_key: str = "EMPTY",  # vLLM doesn't require API key by default
    ):
        self.api_url = api_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key

        # Initialize OpenAI client pointing to vLLM server
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=f"{self.api_url}/v1",
                api_key=api_key,
            )
        except ImportError:
            raise ImportError(
                "openai package required for vLLM client. "
                "Install with: pip install openai>=1.0.0"
            )

    def generate(
        self,
        question: str,
        sources: List[str],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 600,
    ) -> LLMResult:
        """
        Generate answer (non-streaming).

        For metrics like TTFT, use generate_streaming() instead.
        """
        prompt = self._build_prompt(question, sources)

        t0 = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_output_tokens,
                stream=False,
            )

            latency_ms = int((time.time() - t0) * 1000)
            text = response.choices[0].message.content.strip()
            output_tokens = max(1, len(text) // 4)

            return LLMResult(
                text=text,
                model=self.model_name,
                latency_ms=latency_ms,
                ttft_ms=0,
                output_tokens=output_tokens,
                generation_time_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            raise Exception(f"vLLM API error: {str(e)}") from e

    def generate_streaming(
        self,
        question: str,
        sources: List[str],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 600,
    ) -> LLMResult:
        """
        Generate answer with streaming to capture TTFT (Time to First Token).

        Returns LLMResult with:
        - ttft_ms: Time to first token
        - generation_time_ms: Time spent generating after first token
        - output_tokens: Estimated output token count
        """
        prompt = self._build_prompt(question, sources)

        t0 = time.time()
        ttft_ms = 0
        full_text = ""

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_output_tokens,
                stream=True,
            )

            for i, chunk in enumerate(stream):
                # Capture TTFT on first chunk
                if i == 0:
                    ttft_ms = int((time.time() - t0) * 1000)

                # Accumulate text from chunk
                if chunk.choices and chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content

            total_ms = int((time.time() - t0) * 1000)
            generation_time_ms = total_ms - ttft_ms

            # Estimate output tokens (~4 chars per token)
            output_tokens = max(1, len(full_text) // 4)

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
            raise Exception(f"vLLM streaming API error: {str(e)}") from e

    def generate_streaming_iter(
        self,
        question: str,
        sources: List[str],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 600,
    ) -> Iterator[str]:
        """
        Generate answer with streaming, yielding chunks as they arrive.

        This is useful for real-time display in a dashboard.
        Use this for UI streaming, then call generate_streaming() for final metrics.
        """
        prompt = self._build_prompt(question, sources)

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_output_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _build_prompt(self, question: str, sources: List[str]) -> str:
        """Build prompt with sources context."""
        system = (
            "You are a helpful assistant.\n"
            "Use ONLY the provided SOURCES to answer.\n"
            "If the SOURCES are insufficient, say you are unsure.\n"
            "Do not invent facts, numbers, dates, or citations.\n"
        )

        sources_block = "\n\n".join(
            [f"[SOURCE {i+1}]\n{t}" for i, t in enumerate(sources)]
        )

        return f"{system}\nSOURCES:\n{sources_block}\n\nQUESTION:\n{question}\n\nANSWER:"

    def health_check(self) -> bool:
        """Check if vLLM server is available."""
        try:
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception:
            return False
