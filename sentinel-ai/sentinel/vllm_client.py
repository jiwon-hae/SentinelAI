"""
vLLM Client for LLM Sentinel.

Provides an OpenAI-compatible client for vLLM server with streaming support
and TTFT (Time to First Token) measurement.

Usage:
    # Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-chat-hf \
        --port 8000

    # For NVIDIA Nemotron with reasoning:
    vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --served-model-name nemotron \
        --max-num-seqs 8 \
        --tensor-parallel-size 1 \
        --max-model-len 262144 \
        --port 8000 \
        --trust-remote-code

    # Then use this client:
    client = VLLMClient(api_url="http://localhost:8000", model_name="meta-llama/Llama-2-7b-chat-hf")
    result = client.generate_streaming(question, sources)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Iterator, Dict, Any

from sentinel.llm import LLMResult


# Pre-configured model presets
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "nemotron-nano": {
        "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "display_name": "NVIDIA Nemotron-3-Nano (30B/3.5B active)",
        "default_temperature": 0.6,  # Recommended for tool calling / RAG
        "reasoning_temperature": 1.0,  # Recommended for reasoning tasks
        "supports_reasoning": True,
        "context_length": 262144,
        "vllm_args": [
            "--trust-remote-code",
            "--max-num-seqs", "8",
        ],
    },
    "llama-2-7b": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "display_name": "Llama 2 7B Chat",
        "default_temperature": 0.2,
        "supports_reasoning": False,
        "context_length": 4096,
        "vllm_args": [],
    },
    "llama-2-13b": {
        "model_id": "meta-llama/Llama-2-13b-chat-hf",
        "display_name": "Llama 2 13B Chat",
        "default_temperature": 0.2,
        "supports_reasoning": False,
        "context_length": 4096,
        "vllm_args": [],
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
        "display_name": "Mistral 7B Instruct",
        "default_temperature": 0.2,
        "supports_reasoning": False,
        "context_length": 8192,
        "vllm_args": [],
    },
    "mixtral-8x7b": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "display_name": "Mixtral 8x7B Instruct",
        "default_temperature": 0.2,
        "supports_reasoning": False,
        "context_length": 32768,
        "vllm_args": [],
    },
}


class VLLMClient:
    """
    vLLM client using OpenAI-compatible API.

    vLLM exposes an OpenAI-compatible endpoint, so we use the openai library
    to communicate with it. This allows easy switching between vLLM and other
    OpenAI-compatible backends.

    Supported models (examples):
      - nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (with reasoning support)
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
        enable_reasoning: bool = False,  # For Nemotron reasoning mode
    ):
        self.api_url = api_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.enable_reasoning = enable_reasoning

        # Check if this is a Nemotron model
        self.is_nemotron = "nemotron" in model_name.lower()

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
        temperature: Optional[float] = None,
        max_output_tokens: int = 600,
    ) -> LLMResult:
        """
        Generate answer (non-streaming).

        For metrics like TTFT, use generate_streaming() instead.

        Args:
            temperature: Generation temperature. If None, uses model-appropriate default
                        (0.6 for Nemotron in RAG mode, 0.2 for other models)
        """
        prompt = self._build_prompt(question, sources)

        # Use appropriate temperature for model
        if temperature is None:
            temperature = 0.6 if self.is_nemotron else 0.2

        t0 = time.time()
        try:
            # Build request kwargs
            request_kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "stream": False,
            }

            # For Nemotron, control reasoning mode via extra_body
            if self.is_nemotron and not self.enable_reasoning:
                request_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }

            response = self.client.chat.completions.create(**request_kwargs)

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
        temperature: Optional[float] = None,
        max_output_tokens: int = 600,
    ) -> LLMResult:
        """
        Generate answer with streaming to capture TTFT (Time to First Token).

        Returns LLMResult with:
        - ttft_ms: Time to first token
        - generation_time_ms: Time spent generating after first token
        - output_tokens: Estimated output token count

        Args:
            temperature: Generation temperature. If None, uses model-appropriate default
                        (0.6 for Nemotron in RAG mode, 0.2 for other models)
        """
        prompt = self._build_prompt(question, sources)

        # Use appropriate temperature for model
        if temperature is None:
            temperature = 0.6 if self.is_nemotron else 0.2

        t0 = time.time()
        ttft_ms = 0
        full_text = ""

        try:
            # Build request kwargs
            request_kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "stream": True,
            }

            # For Nemotron, control reasoning mode via extra_body
            if self.is_nemotron and not self.enable_reasoning:
                request_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }

            stream = self.client.chat.completions.create(**request_kwargs)

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


def get_vllm_launch_command(
    preset: str = "nemotron-nano",
    port: int = 8000,
    tensor_parallel: int = 1,
    max_model_len: Optional[int] = None,
) -> str:
    """
    Generate vLLM server launch command for a model preset.

    Args:
        preset: Model preset name (see MODEL_PRESETS)
        port: Port to serve on
        tensor_parallel: Number of GPUs for tensor parallelism
        max_model_len: Maximum context length (uses model default if None)

    Returns:
        Shell command to launch vLLM server

    Example:
        >>> print(get_vllm_launch_command("nemotron-nano"))
        vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
          --served-model-name nemotron-nano \\
          --port 8000 \\
          --tensor-parallel-size 1 \\
          --max-model-len 262144 \\
          --trust-remote-code \\
          --max-num-seqs 8
    """
    if preset not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset}. Available: {available}")

    config = MODEL_PRESETS[preset]
    model_id = config["model_id"]
    context_len = max_model_len or config["context_length"]

    cmd_parts = [
        f"vllm serve {model_id}",
        f"  --served-model-name {preset}",
        f"  --port {port}",
        f"  --tensor-parallel-size {tensor_parallel}",
        f"  --max-model-len {context_len}",
    ]

    # Add model-specific args
    for arg in config.get("vllm_args", []):
        cmd_parts.append(f"  {arg}")

    return " \\\n".join(cmd_parts)


def list_model_presets() -> Dict[str, str]:
    """
    List available model presets with their display names.

    Returns:
        Dict mapping preset name to display name
    """
    return {k: v["display_name"] for k, v in MODEL_PRESETS.items()}
