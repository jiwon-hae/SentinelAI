"""
APM (Application Performance Monitoring) instrumentation for LLM Sentinel.

Provides utilities for distributed tracing with Datadog APM, including:
- LLM-specific span tags following Datadog's LLM observability conventions
- Automatic instrumentation helpers
- Span context management
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from contextlib import contextmanager

try:
    from ddtrace import tracer
    from ddtrace import patch
    DDTRACE_AVAILABLE = True
except ImportError:
    DDTRACE_AVAILABLE = False
    tracer = None


class APMConfig:
    """Configuration for APM tracing"""
    def __init__(
        self,
        service_name: str = "sentinel-ai",
        env: str = "production",
        enabled: bool = True,
    ):
        self.service_name = service_name
        self.env = env
        self.enabled = enabled and DDTRACE_AVAILABLE


def initialize_apm(config: APMConfig) -> bool:
    """
    Initialize Datadog APM tracing.

    Args:
        config: APM configuration

    Returns:
        True if APM was successfully initialized

    Note:
        Configure Datadog Agent connection via environment variables:
        - DD_AGENT_HOST (default: localhost)
        - DD_TRACE_AGENT_PORT (default: 8126)
        - DD_TRACE_ENABLED (default: true)
    """
    if not config.enabled:
        return False

    if not DDTRACE_AVAILABLE:
        print("⚠️  ddtrace not available - APM disabled. Install with: pip install ddtrace")
        return False

    # Configure tracer settings
    # Note: Agent hostname/port are configured via DD_AGENT_HOST and DD_TRACE_AGENT_PORT env vars
    # tracer.configure(
    #     service="my-custom-service",
    #     env="staging",
    #     url="http://localhost:8126"
    # )

    # Auto-instrument HTTP requests (for Datadog API calls)
    try:
        patch(requests=True)
    except Exception:
        # Patching may fail if already patched or not available
        pass

    return True


@contextmanager
def llm_span(
    operation_name: str,
    service: str = "sentinel-ai",
    resource: Optional[str] = None,
    span_type: str = "llm",
):
    """
    Context manager for creating LLM operation spans.

    Args:
        operation_name: Name of the operation (e.g., "llm.completion", "llm.embedding")
        service: Service name
        resource: Resource name (e.g., model name)
        span_type: Span type for categorization

    Example:
        with llm_span("llm.completion", resource="gemini-1.5-pro") as span:
            span.set_tag("llm.request.model", "gemini-1.5-pro")
            result = llm.generate(prompt)
            span.set_tag("llm.response.finish_reason", "stop")
    """
    if not DDTRACE_AVAILABLE or not tracer.enabled:
        # No-op context manager if tracing is disabled
        class NoOpSpan:
            def set_tag(self, key, value): pass
            def set_tags(self, tags): pass
            def set_metric(self, key, value): pass

        yield NoOpSpan()
        return

    with tracer.trace(
        operation_name,
        service=service,
        resource=resource,
        span_type=span_type,
    ) as span:
        yield span


def set_llm_completion_tags(
    span,
    model: str,
    prompt: str,
    completion: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    """
    Set standard LLM completion tags on a span following Datadog conventions.

    See: https://docs.datadoghq.com/tracing/llm_observability/

    Args:
        span: The ddtrace span to tag
        model: Model name
        prompt: Input prompt
        completion: Generated completion
        input_tokens: Number of input tokens (optional)
        output_tokens: Number of output tokens (optional)
        temperature: Temperature parameter (optional)
        max_tokens: Max tokens parameter (optional)
    """
    if not span:
        return

    # Core LLM tags
    span.set_tag("llm.request.model", model)
    span.set_tag("llm.request.type", "completion")

    # Request parameters
    if temperature is not None:
        span.set_tag("llm.request.temperature", temperature)
    if max_tokens is not None:
        span.set_tag("llm.request.max_tokens", max_tokens)

    # Prompt and completion (truncate if too long)
    max_tag_length = 5000  # Datadog tag value limit
    span.set_tag("llm.request.prompt", prompt[:max_tag_length])
    span.set_tag("llm.response.completion", completion[:max_tag_length])

    # Token usage metrics
    if input_tokens is not None:
        span.set_metric("llm.usage.prompt_tokens", input_tokens)
    if output_tokens is not None:
        span.set_metric("llm.usage.completion_tokens", output_tokens)
    if input_tokens is not None and output_tokens is not None:
        span.set_metric("llm.usage.total_tokens", input_tokens + output_tokens)


def set_llm_embedding_tags(
    span,
    model: str,
    input_text: str,
    input_count: int = 1,
    dimensions: Optional[int] = None,
    task_type: Optional[str] = None,
):
    """
    Set standard LLM embedding tags on a span.

    Args:
        span: The ddtrace span to tag
        model: Embedding model name
        input_text: Input text (or preview if multiple)
        input_count: Number of inputs being embedded
        dimensions: Embedding dimensions
        task_type: Task type (e.g., "RETRIEVAL_DOCUMENT")
    """
    if not span:
        return

    span.set_tag("llm.request.model", model)
    span.set_tag("llm.request.type", "embedding")

    # Truncate input preview
    max_tag_length = 5000
    span.set_tag("llm.request.input", input_text[:max_tag_length])
    span.set_metric("llm.request.input_count", input_count)

    if dimensions is not None:
        span.set_metric("llm.embedding.dimensions", dimensions)
    if task_type:
        span.set_tag("llm.embedding.task_type", task_type)


def set_rag_retrieval_tags(
    span,
    query: str,
    top_k: int,
    retrieved_count: int,
    avg_score: Optional[float] = None,
    chunk_ids: Optional[List[str]] = None,
):
    """
    Set RAG retrieval-specific tags on a span.

    Args:
        span: The ddtrace span to tag
        query: Search query
        top_k: Number of documents requested
        retrieved_count: Number of documents actually retrieved
        avg_score: Average similarity score
        chunk_ids: IDs of retrieved chunks
    """
    if not span:
        return

    span.set_tag("rag.query", query[:5000])
    span.set_metric("rag.top_k", top_k)
    span.set_metric("rag.retrieved_count", retrieved_count)

    if avg_score is not None:
        span.set_metric("rag.avg_similarity", avg_score)

    if chunk_ids:
        span.set_tag("rag.chunk_ids", ",".join(chunk_ids[:10]))  # Limit to first 10


def set_hallucination_tags(
    span,
    hallucination_rate: float,
    flagged_count: int,
    total_sentences: int,
    threshold: float,
    severity: str,
):
    """
    Set hallucination detection tags on a span.

    Args:
        span: The ddtrace span to tag
        hallucination_rate: Hallucination rate (0-1)
        flagged_count: Number of flagged sentences
        total_sentences: Total sentences analyzed
        threshold: Grounding threshold used
        severity: Severity level
    """
    if not span:
        return

    span.set_metric("llm.hallucination.rate", hallucination_rate)
    span.set_metric("llm.hallucination.flagged_sentences", flagged_count)
    span.set_metric("llm.hallucination.total_sentences", total_sentences)
    span.set_metric("llm.hallucination.threshold", threshold)
    span.set_tag("llm.hallucination.severity", severity)

    # Set error if high severity
    if severity in ("high", "critical"):
        span.set_tag("error", "true")
        span.set_tag("error.type", "hallucination")
        span.set_tag("error.message", f"High hallucination rate: {hallucination_rate:.2f}")
