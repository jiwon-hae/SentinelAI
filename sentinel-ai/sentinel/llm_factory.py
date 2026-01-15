"""
LLM Factory for LLM Sentinel.

Provides a unified factory to create LLM clients based on backend selection.
Supports both Vertex AI Gemini and vLLM backends.
"""

from __future__ import annotations

import os
from typing import Optional, Union

from sentinel.llm import VertexGeminiClient, LLMResult
from sentinel.vllm_client import VLLMClient


# Type alias for LLM clients
LLMClient = Union[VertexGeminiClient, VLLMClient]


def create_llm_client(
    backend: str = "gemini",
    *,
    # Gemini options
    project_id: Optional[str] = None,
    location: str = "global",
    gemini_model: str = "gemini-2.0-flash-exp",
    api_key: Optional[str] = None,
    # vLLM options
    vllm_url: str = "http://localhost:8000",
    vllm_model: str = "meta-llama/Llama-2-7b-chat-hf",
) -> LLMClient:
    """
    Factory to create LLM client based on backend choice.

    Args:
        backend: "gemini" | "vllm"

        # Gemini options
        project_id: Google Cloud project ID (required for Gemini)
        location: Vertex AI location
        gemini_model: Gemini model name
        api_key: Optional API key for direct Gemini API

        # vLLM options
        vllm_url: vLLM server URL
        vllm_model: Model name served by vLLM

    Returns:
        LLM client instance (VertexGeminiClient or VLLMClient)

    Example:
        # Create Gemini client
        llm = create_llm_client("gemini", project_id="my-project")

        # Create vLLM client
        llm = create_llm_client("vllm", vllm_url="http://localhost:8000")
    """
    backend = backend.lower()

    if backend == "vllm":
        return VLLMClient(
            api_url=vllm_url,
            model_name=vllm_model,
        )
    elif backend == "gemini":
        if not project_id:
            project_id = os.getenv("VERTEX_PROJECT_ID")
        if not project_id:
            raise ValueError(
                "project_id required for Gemini backend. "
                "Set VERTEX_PROJECT_ID env var or pass project_id parameter."
            )

        return VertexGeminiClient(
            project_id=project_id,
            location=location,
            model_name=gemini_model,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'gemini' or 'vllm'.")


def create_llm_client_from_env() -> LLMClient:
    """
    Create LLM client from environment variables.

    Environment variables:
        LLM_BACKEND: "gemini" | "vllm" (default: "gemini")

        # Gemini
        VERTEX_PROJECT_ID: Google Cloud project ID
        VERTEX_LOCATION: Vertex AI location (default: "global")
        VERTEX_GEMINI_MODEL: Model name (default: "gemini-2.0-flash-exp")
        VERTEX_API_KEY: Optional API key for direct API

        # vLLM
        VLLM_API_URL: vLLM server URL (default: "http://localhost:8000")
        VLLM_MODEL: Model name (default: "meta-llama/Llama-2-7b-chat-hf")
    """
    backend = os.getenv("LLM_BACKEND", "gemini")

    return create_llm_client(
        backend=backend,
        project_id=os.getenv("VERTEX_PROJECT_ID"),
        location=os.getenv("VERTEX_LOCATION", "global"),
        gemini_model=os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.0-flash-exp"),
        api_key=os.getenv("VERTEX_API_KEY"),
        vllm_url=os.getenv("VLLM_API_URL", "http://localhost:8000"),
        vllm_model=os.getenv("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
    )
