import os
import uuid
from dotenv import load_dotenv

from sentinel.embedder import VertexEmbedder, EmbeddingConfig
from sentinel.retriever import InMemoryRetriever
from sentinel.llm import VertexGeminiClient
from sentinel.hallucination import grounding_check
from sentinel.indexing import build_chunks_from_file
from sentinel.datadog import DatadogClient, DatadogConfig
from sentinel.telemetry import build_request_telemetry, to_datadog_log


def main():
    load_dotenv()

    project_id = os.getenv("VERTEX_PROJECT_ID")
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    gemini_model = os.getenv("VERTEX_GEMINI_MODEL", "gemini-1.5-pro")
    embed_model = os.getenv("VERTEX_EMBED_MODEL", "textembedding-gecko@003")

    dd_api_key = os.getenv("DATADOG_API_KEY")
    dd_site = os.getenv("DATADOG_SITE", "datadoghq.com")
    dd_service = os.getenv("DD_SERVICE", "llm-sentinel")
    dd_env = os.getenv("DD_ENV", "demo")

    top_k = int(os.getenv("RAG_TOP_K", "4"))
    threshold = float(os.getenv("GROUNDING_THRESHOLD", "0.75"))
    max_out = int(os.getenv("ANSWER_MAX_TOKENS", "600"))
    temperature = float(os.getenv("TEMPERATURE", "0.2"))

    if not project_id:
        raise RuntimeError("Missing VERTEX_PROJECT_ID in .env")
    if not dd_api_key:
        raise RuntimeError("Missing DATADOG_API_KEY in .env")

    # 1) Build corpus chunks and embeddings once
    chunks = build_chunks_from_file("data/docs.txt")
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_texts = [c.text for c in chunks]

    embedder = VertexEmbedder(EmbeddingConfig(project_id=project_id, location=location, model_name=embed_model))
    chunk_vecs = embedder.embed(chunk_texts)

    retriever = InMemoryRetriever(chunk_ids, chunk_texts, chunk_vecs)
    llm = VertexGeminiClient(project_id=project_id, location=location, model_name=gemini_model)

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

        try:
            # Retrieve
            q_vec = embedder.embed([question])[0]
            top = retriever.top_k(q_vec, k=top_k)

            retrieved = [{
                "chunk_id": r.chunk_id,
                "score": r.score,
                "text_preview": (r.text[:300] + "…") if len(r.text) > 300 else r.text,
            } for r in top]
            topk_scores = [r.score for r in top]

            # Align vectors for grounding check (by chunk_id index)
            id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
            retrieved_vecs = [chunk_vecs[id_to_idx[r["chunk_id"]]] for r in retrieved]

            sources = [chunk_texts[id_to_idx[r["chunk_id"]]] for r in retrieved]

            # Generate
            res = llm.generate(question, sources, temperature=temperature, max_output_tokens=max_out)
            answer = res.text
            latency_ms = res.latency_ms

            # Grounding check
            gr = grounding_check(answer, retrieved_vecs, embedder, threshold=threshold)

        except Exception as e:
            error = True
            error_type = type(e).__name__
            answer = answer or ""
            latency_ms = latency_ms or 0
            gr = grounding_check(answer, retrieved_vecs, embedder, threshold=threshold)

        # ---- Build telemetry object (includes tokens/cost proxy + severity) ----
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
        )

        # ---- Emit metrics for dashboards / monitors / SLOs ----
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

        # ---- Emit structured log (references + context always included) ----
        dd.send_log(to_datadog_log(telem, service=dd_service, env=dd_env))

        # ---- Print local demo output ----
        print("\n--- Answer ---")
        print(answer if answer else "(no answer)")
        print("\n--- Summary ---")
        print(f"request_id={request_id}")
        print(f"error={error} ({error_type}) latency_ms={latency_ms}")
        print(f"hallucination_rate={telem.hallucination_rate:.2f} severity={telem.severity}")
        if telem.hallucinated_sentences:
            print("Flagged sentences:")
            for item in gr.flagged:
                print(f"- ({item['max_similarity']:.2f}) {item['sentence']}")

        if retrieved:
            print("\nTop references used:")
            for r in retrieved:
                print(f"- {r['chunk_id']} score={r['score']:.3f}")


if __name__ == "__main__":
    main()