from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os


# SLO Thresholds (configurable via environment variables)
@dataclass
class SLOConfig:
    """Service Level Objective thresholds for quality gates"""
    # Latency SLO: requests should complete within this threshold
    latency_threshold_ms: int = 3000  # 3 seconds

    # Quality SLO: hallucination rate should be below this threshold
    hallucination_threshold: float = 0.3  # 30%

    # Token Budget SLO: total tokens per request should be below this
    token_budget: int = 10000  # Combined input + output tokens

    # Retrieval Quality SLO: average similarity should be above this
    retrieval_quality_threshold: float = 0.5

    @classmethod
    def from_env(cls) -> "SLOConfig":
        """Load SLO configuration from environment variables"""
        return cls(
            latency_threshold_ms=int(os.getenv("SLO_LATENCY_MS", "3000")),
            hallucination_threshold=float(os.getenv("SLO_HALLUCINATION_MAX", "0.3")),
            token_budget=int(os.getenv("SLO_TOKEN_BUDGET", "10000")),
            retrieval_quality_threshold=float(os.getenv("SLO_RETRIEVAL_MIN", "0.5")),
        )


def estimate_tokens(text: str) -> int:
    # rough heuristic (~4 chars/token). Good enough for hackathon “tokens/cost proxy”
    t = (text or "")
    return max(1, len(t) // 4)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate = float(os.getenv("COST_PER_1K_INPUT_TOKENS", "0.0"))
    out_rate = float(os.getenv("COST_PER_1K_OUTPUT_TOKENS", "0.0"))
    return (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate


def severity_from_rate(rate: float) -> str:
    if rate > 0.50:
        return "high"
    if rate > 0.20:
        return "medium"
    return "low"


def avg(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


@dataclass
class RequestTelemetry:
    request_id: str
    model: str
    prompt: str
    answer: str
    latency_ms: int
    error: bool
    error_type: Optional[str]

    hallucination_rate: float
    hallucinated_sentences: int
    grounding_threshold: float

    # Quality Issue Flags (mutually exclusive root causes):
    is_hallucination: bool  # LLM unfaithfulness: Good context found, but LLM ignored it
    missing_reference: bool  # Knowledge base gap: No relevant context found for query

    retrieved_context: List[Dict[str, Any]]  # [{chunk_id, score, text_preview}]
    topk_avg_similarity: float

    input_tokens_est: int
    output_tokens_est: int
    cost_usd_est: float

    severity: str

    # SLO Status (whether this request meets each SLO)
    slo_availability: bool  # No errors
    slo_latency: bool  # Latency under threshold
    slo_quality: bool  # Hallucination rate under threshold
    slo_tokens: bool  # Token usage under budget
    slo_retrieval: bool  # Retrieval quality above threshold


def build_request_telemetry(
    *,
    request_id: str,
    model: str,
    prompt: str,
    answer: str,
    latency_ms: int,
    error: bool,
    error_type: Optional[str],
    hallucination_rate: float,
    hallucinated_sentences: int,
    grounding_threshold: float,
    retrieved: List[Dict[str, Any]],
    topk_scores: List[float],
    slo_config: Optional[SLOConfig] = None,
) -> RequestTelemetry:
    """
    Build telemetry object with SLO status indicators.

    Args:
        slo_config: SLO thresholds. If None, uses defaults from environment.
    """
    if slo_config is None:
        slo_config = SLOConfig.from_env()

    input_tokens_est = estimate_tokens(prompt + "\n" + "\n".join([r.get("text_preview", "") for r in retrieved]))
    output_tokens_est = estimate_tokens(answer)
    total_tokens = input_tokens_est + output_tokens_est
    cost = estimate_cost_usd(model, input_tokens_est, output_tokens_est)
    sev = severity_from_rate(hallucination_rate)
    topk_avg = avg(topk_scores) if topk_scores else 0.0

    # Calculate SLO status
    slo_availability = not error
    slo_latency = latency_ms <= slo_config.latency_threshold_ms
    slo_quality = hallucination_rate <= slo_config.hallucination_threshold
    slo_tokens = total_tokens <= slo_config.token_budget
    slo_retrieval = topk_avg >= slo_config.retrieval_quality_threshold

    # Determine root cause of quality issues:
    #
    # Missing Reference = Knowledge base gap (retrieval failed)
    #   - Cause: Corpus doesn't have information to answer the query
    #   - Action: Update knowledge base, add more documents
    #
    # Hallucination = LLM unfaithfulness (good retrieval, but LLM ignored context)
    #   - Cause: LLM generated content not grounded in retrieved docs
    #   - Action: Adjust temperature, improve prompting, or use different model
    #
    # These are mutually exclusive - if retrieval fails, we can't blame the LLM

    missing_reference = topk_avg < slo_config.retrieval_quality_threshold

    # Only flag hallucination if we HAD good context (retrieval succeeded)
    # If retrieval failed, the issue is missing_reference, not hallucination
    is_hallucination = (
        hallucination_rate > slo_config.hallucination_threshold and
        topk_avg >= slo_config.retrieval_quality_threshold  # Had good context
    )

    return RequestTelemetry(
        request_id=request_id,
        model=model,
        prompt=prompt,
        answer=answer,
        latency_ms=latency_ms,
        error=error,
        error_type=error_type,
        hallucination_rate=hallucination_rate,
        hallucinated_sentences=hallucinated_sentences,
        grounding_threshold=grounding_threshold,
        is_hallucination=is_hallucination,
        retrieved_context=retrieved,
        topk_avg_similarity=topk_avg,
        missing_reference=missing_reference,
        input_tokens_est=input_tokens_est,
        output_tokens_est=output_tokens_est,
        cost_usd_est=cost,
        severity=sev,
        slo_availability=slo_availability,
        slo_latency=slo_latency,
        slo_quality=slo_quality,
        slo_tokens=slo_tokens,
        slo_retrieval=slo_retrieval,
    )


def emit_component_metric(
    component: str,
    operation: str,
    value: float,
    metric_type: str = "gauge",
    tags: Optional[List[str]] = None
) -> None:
    """
    Emit a component-level metric via ddtrace StatsD.

    Args:
        component: Component name (e.g., "llm", "embedder", "retriever", "grounding")
        operation: Operation name (e.g., "generate", "embed", "retrieve", "check")
        value: Metric value
        metric_type: "gauge", "count", or "histogram"
        tags: Optional list of tags
    """
    try:
        from ddtrace import tracer
        if not tracer.enabled:
            return
    except ImportError:
        return

    try:
        from ddtrace.internal.dogstatsd import get_dogstatsd_client
        statsd = get_dogstatsd_client(url="udp://localhost:8125")
    except (ImportError, AttributeError):
        return

    if not statsd:
        return

    metric_name = f"llm.component.{component}.{operation}"
    tag_list = tags or []

    if metric_type == "gauge":
        statsd.gauge(metric_name, value, tags=tag_list)
    elif metric_type == "count":
        statsd.increment(metric_name, value=int(value), tags=tag_list)
    elif metric_type == "histogram":
        statsd.histogram(metric_name, value, tags=tag_list)


def emit_slo_metrics(telem: RequestTelemetry, tags: List[str]) -> None:
    """
    Emit SLO metrics using ddtrace StatsD client.

    For each SLO (availability, latency, quality, tokens, retrieval):
    - Emit total event count
    - Emit good event count (1 if SLO met, 0 if not)

    This allows creating metric-based SLOs in Datadog:
    SLO % = sum(slo.*.good) / sum(slo.*.total) * 100

    Args:
        telem: Request telemetry with SLO status
        tags: List of tags (e.g., ["service:sentinel-ai", "env:prod"])
    """
    try:
        from ddtrace import tracer
        if not tracer.enabled:
            return
    except ImportError:
        return

    # Use ddtrace's statsd client for metrics
    # Note: ddtrace automatically adds service/env tags if configured
    try:
        # Try to get the statsd client from ddtrace
        from ddtrace.internal.dogstatsd import get_dogstatsd_client
        statsd = get_dogstatsd_client(url="udp://localhost:8125")
    except (ImportError, AttributeError):
        # Fallback: metrics won't be sent if ddtrace isn't properly configured
        return

    if not statsd:
        return

    # Convert tags list to format expected by statsd
    tag_str = tags if isinstance(tags, list) else []

    # === SLO 1: Availability (no errors) ===
    statsd.increment("llm.slo.availability.total", tags=tag_str)
    statsd.increment("llm.slo.availability.good", tags=tag_str, value=1 if telem.slo_availability else 0)

    # === SLO 2: Latency (under threshold) ===
    statsd.increment("llm.slo.latency.total", tags=tag_str)
    statsd.increment("llm.slo.latency.good", tags=tag_str, value=1 if telem.slo_latency else 0)
    statsd.gauge("llm.slo.latency.ms", telem.latency_ms, tags=tag_str)

    # === SLO 3: Quality (low hallucination) ===
    statsd.increment("llm.slo.quality.total", tags=tag_str)
    statsd.increment("llm.slo.quality.good", tags=tag_str, value=1 if telem.slo_quality else 0)
    statsd.gauge("llm.slo.quality.hallucination_rate", telem.hallucination_rate, tags=tag_str)

    # === SLO 4: Token Budget ===
    statsd.increment("llm.slo.tokens.total", tags=tag_str)
    statsd.increment("llm.slo.tokens.good", tags=tag_str, value=1 if telem.slo_tokens else 0)
    statsd.gauge("llm.slo.tokens.used", telem.input_tokens_est + telem.output_tokens_est, tags=tag_str)

    # === SLO 5: Retrieval Quality ===
    statsd.increment("llm.slo.retrieval.total", tags=tag_str)
    statsd.increment("llm.slo.retrieval.good", tags=tag_str, value=1 if telem.slo_retrieval else 0)
    statsd.gauge("llm.slo.retrieval.avg_similarity", telem.topk_avg_similarity, tags=tag_str)

    # === Overall SLO Compliance ===
    # Check if ALL SLOs are met
    all_slos_met = (
        telem.slo_availability and
        telem.slo_latency and
        telem.slo_quality and
        telem.slo_tokens and
        telem.slo_retrieval
    )
    statsd.increment("llm.slo.overall.total", tags=tag_str)
    statsd.increment("llm.slo.overall.good", tags=tag_str, value=1 if all_slos_met else 0)

    # === Hallucination Event Counter ===
    # Track hallucination events separately for easy alerting
    if telem.is_hallucination:
        statsd.increment("llm.hallucination.events", tags=tag_str)
        # Also emit as a gauge for rate calculations
        statsd.gauge("llm.hallucination.detected", 1, tags=tag_str)
    else:
        statsd.gauge("llm.hallucination.detected", 0, tags=tag_str)

    # === Missing Reference Event Counter ===
    # Track when retrieval fails to find relevant context
    if telem.missing_reference:
        statsd.increment("llm.retrieval.missing_reference", tags=tag_str)
        statsd.gauge("llm.retrieval.reference_found", 0, tags=tag_str)
    else:
        statsd.gauge("llm.retrieval.reference_found", 1, tags=tag_str)


def to_datadog_log(
    telem: RequestTelemetry,
    *,
    service: str,
    env: str,
) -> Dict[str, Any]:
    return {
        "service": service,
        "env": env,
        "event": "llm_request",
        "log_version": "v1",

        "status": "error" if telem.error else "ok",
        "request_id": telem.request_id,
        "model": telem.model,

        "latency_ms": telem.latency_ms,
        "hallucination_rate": telem.hallucination_rate,
        "hallucination_severity": telem.severity,
        "hallucinated_sentences": telem.hallucinated_sentences,
        "grounding_threshold": telem.grounding_threshold,
        "is_hallucination": telem.is_hallucination,  # Boolean flag for easy filtering
        "topk_avg_similarity": telem.topk_avg_similarity,
        "missing_reference": telem.missing_reference,  # Boolean flag for retrieval failure

        "tokens_input_est": telem.input_tokens_est,
        "tokens_output_est": telem.output_tokens_est,
        "cost_usd_est": telem.cost_usd_est,

        # SLO Status
        "slo_availability": telem.slo_availability,
        "slo_latency": telem.slo_latency,
        "slo_quality": telem.slo_quality,
        "slo_tokens": telem.slo_tokens,
        "slo_retrieval": telem.slo_retrieval,

        # Truncate in logs if you want; keep full for hackathon demos if acceptable
        "prompt": telem.prompt,
        "answer": telem.answer,

        "retrieved_context": telem.retrieved_context,

        "error_type": telem.error_type,

        # "Runbook hint" can be pasted into monitor messages / incidents
        "runbook_hint": (
            "Triage: open dashboard; filter logs by request_id. "
            "Diagnose: check retrieved_context scores + flagged_sentences. "
            "Mitigate: adjust chunking/top-k/prompt grounding; reduce temperature."
        ),
    }