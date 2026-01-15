import os
import uuid
from dotenv import load_dotenv

from sentinel.embedder import VertexEmbedder, EmbeddingConfig
from sentinel.retriever import InMemoryRetriever
from sentinel.llm import VertexGeminiClient
from sentinel.hallucination import grounding_check
from sentinel.indexing import build_chunks_from_file
from sentinel.datadog import DatadogClient, DatadogConfig
from sentinel.telemetry import build_request_telemetry, to_datadog_log, emit_slo_metrics, SLOConfig
from sentinel.apm import (
    APMConfig, initialize_apm, initialize_llmobs, llm_span,
    set_llm_completion_tags, set_llm_embedding_tags,
    set_rag_retrieval_tags, set_hallucination_tags, set_llm_performance_tags
)


def main():
    load_dotenv()

    project_id = os.getenv("VERTEX_PROJECT_ID")
    location = os.getenv("VERTEX_LOCATION", "global")
    gemini_model = os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.0-flash-exp")
    embed_model = os.getenv("VERTEX_EMBED_MODEL", "text-embedding-005")
    vertex_api_key = os.getenv("VERTEX_API_KEY")  # Optional: for direct API access

    dd_api_key = os.getenv("DATADOG_API_KEY")
    dd_site = os.getenv("DATADOG_SITE", "datadoghq.com")
    dd_service = os.getenv("DD_SERVICE", "sentinel-ai")
    dd_env = os.getenv("DD_ENV", "demo")
    dd_apm_enabled = os.getenv("DD_APM_ENABLED", "true").lower() == "true"
    dd_llmobs_enabled = os.getenv("DATADOG_LLMOBS_ENABLED", "false").lower() == "true"

    top_k = int(os.getenv("RAG_TOP_K", "4"))
    threshold = float(os.getenv("GROUNDING_THRESHOLD", "0.75"))
    max_out = int(os.getenv("ANSWER_MAX_TOKENS", "600"))
    temperature = float(os.getenv("TEMPERATURE", "0.2"))

    if not project_id:
        raise RuntimeError("Missing VERTEX_PROJECT_ID in .env")
    if not dd_api_key:
        raise RuntimeError("Missing DATADOG_API_KEY in .env")

    # Initialize APM (Application Performance Monitoring)
    apm_config = APMConfig(
        service_name=dd_service,
        env=dd_env,
        enabled=dd_apm_enabled,
        llmobs_enabled=dd_llmobs_enabled,
        llmobs_ml_app=dd_service,
        llmobs_api_key=dd_api_key,
        llmobs_site=dd_site,
        llmobs_agentless=True,
    )
    apm_initialized = initialize_apm(apm_config)
    if apm_initialized:
        print(f"✓ Datadog APM initialized (service={dd_service}, env={dd_env})")
    else:
        print("ℹ️  Datadog APM disabled (metrics and logs only)")

    # Initialize LLM Observability (LLMObs)
    llmobs_initialized = initialize_llmobs(apm_config)
    if llmobs_initialized:
        print(f"✓ Datadog LLMObs initialized (ml_app={dd_service})")
    else:
        print("ℹ️  Datadog LLMObs disabled")

    # Load SLO configuration
    slo_config = SLOConfig.from_env()
    print(f"✓ SLO thresholds: latency={slo_config.latency_threshold_ms}ms, "
          f"quality={slo_config.hallucination_threshold:.0%}, "
          f"tokens={slo_config.token_budget}")

    # 1) Build corpus chunks and embeddings once
    chunks = build_chunks_from_file("data/docs.txt")
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_texts = [c.text for c in chunks]

    # Note: Embeddings don't support API key auth - always use Vertex AI with OAuth2
    embedder = VertexEmbedder(EmbeddingConfig(
        project_id=project_id,
        location=location,
        model_name=embed_model,
        # Don't pass api_key - embeddings require OAuth2/ADC
    ))
    # Embed corpus chunks with RETRIEVAL_DOCUMENT task type
    chunk_vecs = embedder.embed(chunk_texts, task_type="RETRIEVAL_DOCUMENT")

    retriever = InMemoryRetriever(chunk_ids, chunk_texts, chunk_vecs)

    # LLM also uses Vertex AI with OAuth2 for consistency
    llm = VertexGeminiClient(
        project_id=project_id,
        location=location,
        model_name=gemini_model,
        # Don't use api_key - keep everything on Vertex AI for now
    )

    dd = DatadogClient(DatadogConfig(api_key=dd_api_key, site=dd_site, service=dd_service, env=dd_env))

    # 2) Ask a question (demo loop)
    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()
        if not question or question.lower() in ("exit", "quit"):
            break

        request_id = str(uuid.uuid4())
        tags = [f"service:{dd_service}", f"env:{dd_env}", f"model:{gemini_model}"]

        error = False
        error_type = None
        answer = ""
        latency_ms = 0

        retrieved = []
        topk_scores = []
        retrieved_vecs = []

        # LLM evaluation metrics (from streaming)
        ttft_ms = 0
        generation_time_ms = 0
        output_tokens = 0

        # Create root span for the entire LLM request
        with llm_span("llm.request", service=dd_service, resource=question[:100]) as request_span:
            request_span.set_tag("request_id", request_id)
            request_span.set_tag("llm.application", "rag-qa")

            try:
                # SPAN 1: Query Embedding
                with llm_span("llm.embedding", service=dd_service, resource=embed_model) as embed_span:
                    q_vec = embedder.embed([question], task_type="RETRIEVAL_QUERY")[0]
                    set_llm_embedding_tags(
                        embed_span,
                        model=embed_model,
                        input_text=question,
                        input_count=1,
                        task_type="RETRIEVAL_QUERY"
                    )

                # SPAN 2: RAG Retrieval
                with llm_span("rag.retrieval", service=dd_service, resource=f"top_{top_k}") as retrieval_span:
                    top = retriever.top_k(q_vec, k=top_k)

                    retrieved = [{
                        "chunk_id": r.chunk_id,
                        "score": r.score,
                        "text_preview": (r.text[:300] + "…") if len(r.text) > 300 else r.text,
                    } for r in top]
                    topk_scores = [r.score for r in top]

                    # Tag retrieval span
                    avg_score = sum(topk_scores) / len(topk_scores) if topk_scores else 0.0
                    set_rag_retrieval_tags(
                        retrieval_span,
                        query=question,
                        top_k=top_k,
                        retrieved_count=len(retrieved),
                        avg_score=avg_score,
                        chunk_ids=[r["chunk_id"] for r in retrieved]
                    )

                # Align vectors for grounding check (by chunk_id index)
                id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
                retrieved_vecs = [chunk_vecs[id_to_idx[r["chunk_id"]]] for r in retrieved]
                sources = [chunk_texts[id_to_idx[r["chunk_id"]]] for r in retrieved]

                # SPAN 3: LLM Generation (with streaming for TTFT measurement)
                with llm_span("llm.completion", service=dd_service, resource=gemini_model) as completion_span:
                    res = llm.generate_streaming(question, sources, temperature=temperature, max_output_tokens=max_out)
                    answer = res.text
                    latency_ms = res.latency_ms

                    # Store LLM evaluation metrics for telemetry
                    ttft_ms = res.ttft_ms
                    generation_time_ms = res.generation_time_ms
                    output_tokens = res.output_tokens

                    # Get actual output tokens from streaming result
                    output_tokens_actual = res.output_tokens
                    input_tokens_est = len(question.split()) * 2  # Rough estimate

                    # Tag completion span
                    set_llm_completion_tags(
                        completion_span,
                        model=gemini_model,
                        prompt=question,
                        completion=answer,
                        input_tokens=input_tokens_est,
                        output_tokens=output_tokens_actual,
                        temperature=temperature,
                        max_tokens=max_out
                    )

                    # Calculate and tag LLM performance metrics
                    tpot_ms = res.generation_time_ms / max(1, res.output_tokens) if res.output_tokens > 0 else 0.0
                    throughput_tps = res.output_tokens / max(0.001, res.generation_time_ms / 1000) if res.output_tokens > 0 else 0.0

                    set_llm_performance_tags(
                        completion_span,
                        ttft_ms=res.ttft_ms,
                        tpot_ms=tpot_ms,
                        throughput_tokens_per_sec=throughput_tps,
                        generation_time_ms=res.generation_time_ms,
                        output_tokens=res.output_tokens,
                    )

                # SPAN 4: Grounding Check (Hallucination Detection)
                with llm_span("llm.grounding", service=dd_service, resource="hallucination_check") as grounding_span:
                    gr = grounding_check(answer, retrieved_vecs, embedder, threshold=threshold)

                    # Tag grounding span
                    set_hallucination_tags(
                        grounding_span,
                        hallucination_rate=gr.hallucination_rate,
                        flagged_count=len(gr.flagged),
                        total_sentences=gr.total_sentences,
                        threshold=gr.threshold,
                        severity="high" if gr.hallucination_rate > 0.5 else "medium" if gr.hallucination_rate > 0.2 else "low"
                    )

            except Exception as e:
                error = True
                error_type = type(e).__name__
                answer = answer or ""
                latency_ms = latency_ms or 0
                gr = grounding_check(answer, retrieved_vecs, embedder, threshold=threshold)

                # Tag request span with error
                request_span.set_tag("error", "true")
                request_span.set_tag("error.type", error_type)
                request_span.set_tag("error.message", str(e))

                # Print error for debugging
                print(f"\n⚠️  ERROR DETAILS: {error_type}")
                print(f"Error message: {str(e)}")
                import traceback
                traceback.print_exc()

        # ---- Build telemetry object (includes tokens/cost proxy + severity + SLO status) ----
        telem = build_request_telemetry(
            request_id=request_id,
            model=gemini_model,
            prompt=question,
            answer=answer,
            latency_ms=latency_ms,
            error=error,
            error_type=error_type,
            hallucination_rate=gr.hallucination_rate,
            hallucinated_sentences=len(gr.flagged),
            grounding_threshold=gr.threshold,
            retrieved=retrieved,
            topk_scores=topk_scores,
            slo_config=slo_config,
            # LLM Evaluation Metrics (from streaming)
            ttft_ms=ttft_ms,
            generation_time_ms=generation_time_ms,
            output_tokens=output_tokens,
        )

        # ---- Emit SLO metrics via ddtrace ----
        emit_slo_metrics(telem, tags=tags)

        # ---- Emit legacy metrics for dashboards / monitors ----
        # SLO total events
        dd.send_metric("llm.request_count", 1, tags=tags, metric_type="count")
        # SLO good events (availability)
        dd.send_metric("llm.request_good_count", 0 if error else 1, tags=tags, metric_type="count")
        # Error events
        dd.send_metric("llm.error_count", 1 if error else 0, tags=tags, metric_type="count")

        # Latency health
        dd.send_metric("llm.latency_ms", telem.latency_ms, tags=tags, metric_type="gauge")
        # Latency SLO helper (optional; makes SLO setup trivial)
        dd.send_metric("llm.latency_good_count", 1 if telem.latency_ms < 2000 and not error else 0, tags=tags, metric_type="count")

        # LLM observability signals
        dd.send_metric("llm.sentinel.hallucination_rate", telem.hallucination_rate, tags=tags, metric_type="gauge")
        dd.send_metric("llm.sentinel.hallucinated_sentences", telem.hallucinated_sentences, tags=tags, metric_type="gauge")

        # Tokens/cost proxy (health view requirement)
        dd.send_metric("llm.tokens.input", telem.input_tokens_est, tags=tags, metric_type="gauge")
        dd.send_metric("llm.tokens.output", telem.output_tokens_est, tags=tags, metric_type="gauge")
        dd.send_metric("llm.cost_usd", telem.cost_usd_est, tags=tags, metric_type="gauge")

        # Retrieval quality (nice security/observability signal)
        dd.send_metric("llm.rag.topk_avg_similarity", telem.topk_avg_similarity, tags=tags, metric_type="gauge")

        # LLM Performance metrics (TTFT, TPOT, throughput)
        dd.send_metric("llm.performance.ttft_ms", telem.ttft_ms, tags=tags, metric_type="gauge")
        dd.send_metric("llm.performance.tpot_ms", telem.tpot_ms, tags=tags, metric_type="gauge")
        dd.send_metric("llm.performance.throughput_tps", telem.throughput_tokens_per_sec, tags=tags, metric_type="gauge")

        # ---- Emit structured log (references + context always included) ----
        dd.send_log(to_datadog_log(telem, service=dd_service, env=dd_env))

        # ---- Print local demo output ----
        print("\n--- Answer ---")
        print(answer if answer else "(no answer)")
        print("\n--- Summary ---")
        print(f"request_id={request_id}")
        print(f"error={error} ({error_type}) latency_ms={latency_ms}")
        print(f"hallucination_rate={telem.hallucination_rate:.2f} severity={telem.severity}")

        # LLM Evaluation metrics
        print(f"\n--- LLM Performance ---")
        print(f"TTFT (Time to First Token): {telem.ttft_ms}ms")
        print(f"TPOT (Time Per Output Token): {telem.tpot_ms:.2f}ms/token")
        print(f"Throughput: {telem.throughput_tokens_per_sec:.1f} tokens/sec")
        print(f"Generation time: {telem.generation_time_ms}ms")

        if telem.hallucinated_sentences:
            print("\nFlagged sentences:")
            for item in gr.flagged:
                print(f"- ({item['max_similarity']:.2f}) {item['sentence']}")

        if retrieved:
            print("\nTop references used:")
            for r in retrieved:
                print(f"- {r['chunk_id']} score={r['score']:.3f}")


if __name__ == "__main__":
    main()