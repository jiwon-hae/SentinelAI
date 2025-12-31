from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os


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

    retrieved_context: List[Dict[str, Any]]  # [{chunk_id, score, text_preview}]
    topk_avg_similarity: float

    input_tokens_est: int
    output_tokens_est: int
    cost_usd_est: float

    severity: str


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
) -> RequestTelemetry:
    input_tokens_est = estimate_tokens(prompt + "\n" + "\n".join([r.get("text_preview", "") for r in retrieved]))
    output_tokens_est = estimate_tokens(answer)
    cost = estimate_cost_usd(model, input_tokens_est, output_tokens_est)
    sev = severity_from_rate(hallucination_rate)

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
        retrieved_context=retrieved,
        topk_avg_similarity=avg(topk_scores) if topk_scores else 0.0,
        input_tokens_est=input_tokens_est,
        output_tokens_est=output_tokens_est,
        cost_usd_est=cost,
        severity=sev,
    )


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
        "topk_avg_similarity": telem.topk_avg_similarity,

        "tokens_input_est": telem.input_tokens_est,
        "tokens_output_est": telem.output_tokens_est,
        "cost_usd_est": telem.cost_usd_est,

        # Truncate in logs if you want; keep full for hackathon demos if acceptable
        "prompt": telem.prompt,
        "answer": telem.answer,

        "retrieved_context": telem.retrieved_context,

        "error_type": telem.error_type,

        # “Runbook hint” can be pasted into monitor messages / incidents
        "runbook_hint": (
            "Triage: open dashboard; filter logs by request_id. "
            "Diagnose: check retrieved_context scores + flagged_sentences. "
            "Mitigate: adjust chunking/top-k/prompt grounding; reduce temperature."
        ),
    }