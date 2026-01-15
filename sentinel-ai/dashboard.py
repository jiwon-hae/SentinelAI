"""
LLM Sentinel Dashboard

A Streamlit dashboard for demonstrating LLM observability with:
- Live chat interface with streaming responses
- Real-time metrics display (TTFT, TPOT, throughput, hallucination rate)
- Support for both vLLM and Gemini backends

Run with:
    cd sentinel-ai
    streamlit run dashboard.py
"""

import os
import uuid
import time
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import sentinel modules
from sentinel.llm_factory import create_llm_client, LLMClient
from sentinel.embedder import VertexEmbedder, EmbeddingConfig
from sentinel.retriever import InMemoryRetriever
from sentinel.hallucination import grounding_check
from sentinel.indexing import build_chunks_from_file
from sentinel.telemetry import build_request_telemetry, SLOConfig


# Page configuration
st.set_page_config(
    page_title="LLM Sentinel Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
    }
    .slo-pass {
        color: #28a745;
        font-weight: bold;
    }
    .slo-fail {
        color: #dc3545;
        font-weight: bold;
    }
    .hallucination-low {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
    }
    .hallucination-medium {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
    }
    .hallucination-high {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []
if "last_telemetry" not in st.session_state:
    st.session_state.last_telemetry = None
if "corpus_loaded" not in st.session_state:
    st.session_state.corpus_loaded = False
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chunk_vecs" not in st.session_state:
    st.session_state.chunk_vecs = None
if "chunk_ids" not in st.session_state:
    st.session_state.chunk_ids = None
if "chunk_texts" not in st.session_state:
    st.session_state.chunk_texts = None


@st.cache_resource
def load_corpus_and_embedder(project_id: str, location: str, embed_model: str):
    """Load document corpus and create embedder (cached)."""
    try:
        # Build corpus chunks
        chunks = build_chunks_from_file("data/docs.txt")
        chunk_ids = [c.chunk_id for c in chunks]
        chunk_texts = [c.text for c in chunks]

        # Initialize embedder
        embedder = VertexEmbedder(EmbeddingConfig(
            project_id=project_id,
            location=location,
            model_name=embed_model,
        ))

        # Embed corpus chunks
        chunk_vecs = embedder.embed(chunk_texts, task_type="RETRIEVAL_DOCUMENT")

        # Create retriever
        retriever = InMemoryRetriever(chunk_ids, chunk_texts, chunk_vecs)

        return embedder, retriever, chunk_vecs, chunk_ids, chunk_texts
    except Exception as e:
        st.error(f"Failed to load corpus: {e}")
        return None, None, None, None, None


def create_llm_from_config(backend: str, vllm_url: str, vllm_model: str,
                           project_id: str, location: str, gemini_model: str) -> Optional[LLMClient]:
    """Create LLM client from dashboard config."""
    try:
        if backend == "vLLM":
            return create_llm_client(
                "vllm",
                vllm_url=vllm_url,
                vllm_model=vllm_model,
            )
        else:  # Gemini
            return create_llm_client(
                "gemini",
                project_id=project_id,
                location=location,
                gemini_model=gemini_model,
            )
    except Exception as e:
        st.error(f"Failed to create LLM client: {e}")
        return None


# ============================================================================
# SIDEBAR: Configuration
# ============================================================================
with st.sidebar:
    st.header("🛡️ LLM Sentinel")
    st.markdown("---")

    # Backend selection
    st.subheader("LLM Backend")
    backend = st.selectbox(
        "Select Backend",
        ["vLLM", "Gemini"],
        help="Choose between local vLLM server or cloud Gemini"
    )

    if backend == "vLLM":
        st.markdown("##### vLLM Configuration")
        vllm_url = st.text_input("API URL", "http://localhost:8000")

        # Model preset selector
        from sentinel.vllm_client import MODEL_PRESETS, get_vllm_launch_command
        model_options = {
            "Custom": "custom",
            "NVIDIA Nemotron-3-Nano (30B)": "nemotron-nano",
            "Llama 2 7B": "llama-2-7b",
            "Llama 2 13B": "llama-2-13b",
            "Mistral 7B": "mistral-7b",
            "Mixtral 8x7B": "mixtral-8x7b",
        }
        selected_preset = st.selectbox(
            "Model Preset",
            list(model_options.keys()),
            help="Select a pre-configured model or choose Custom"
        )

        if model_options[selected_preset] == "custom":
            vllm_model = st.text_input("Model Name", "meta-llama/Llama-2-7b-chat-hf")
            enable_reasoning = False
        else:
            preset_key = model_options[selected_preset]
            preset_config = MODEL_PRESETS[preset_key]
            vllm_model = preset_config["model_id"]
            st.caption(f"Model: `{vllm_model}`")

            # Show reasoning toggle for Nemotron
            if preset_config.get("supports_reasoning", False):
                enable_reasoning = st.checkbox(
                    "Enable Reasoning Mode",
                    value=False,
                    help="Enable chain-of-thought reasoning (slower but more accurate)"
                )
            else:
                enable_reasoning = False

            # Show launch command in expander
            with st.expander("📋 vLLM Launch Command"):
                launch_cmd = get_vllm_launch_command(preset_key)
                st.code(launch_cmd, language="bash")
                st.caption("Run this command on a Linux machine with GPU to start the vLLM server")

        # Store reasoning mode in session state
        if "enable_reasoning" not in st.session_state:
            st.session_state.enable_reasoning = False
        st.session_state.enable_reasoning = enable_reasoning

        # Health check
        if st.button("Check vLLM Server"):
            try:
                from sentinel.vllm_client import VLLMClient
                client = VLLMClient(api_url=vllm_url, model_name=vllm_model)
                if client.health_check():
                    st.success("✓ vLLM server is running")
                else:
                    st.error("✗ vLLM server not responding")
            except Exception as e:
                st.error(f"✗ Error: {e}")
    else:
        vllm_url = ""
        vllm_model = ""

    st.markdown("---")

    # Vertex AI / Gemini configuration
    st.subheader("Vertex AI Config")
    project_id = st.text_input(
        "Project ID",
        os.getenv("VERTEX_PROJECT_ID", ""),
        help="Google Cloud project ID"
    )
    location = st.text_input(
        "Location",
        os.getenv("VERTEX_LOCATION", "global"),
    )
    gemini_model = st.text_input(
        "Gemini Model",
        os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.0-flash-exp"),
    )
    embed_model = st.text_input(
        "Embedding Model",
        os.getenv("VERTEX_EMBED_MODEL", "text-embedding-005"),
    )

    st.markdown("---")

    # RAG configuration
    st.subheader("RAG Settings")
    top_k = st.slider("Top-K Retrieval", 1, 10, 4)
    threshold = st.slider("Grounding Threshold", 0.5, 0.95, 0.75)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    max_tokens = st.slider("Max Output Tokens", 100, 1000, 600)

    st.markdown("---")

    # Load corpus button
    if st.button("Load Document Corpus", type="primary"):
        if not project_id:
            st.error("Please set Project ID first")
        else:
            with st.spinner("Loading corpus and creating embeddings..."):
                result = load_corpus_and_embedder(project_id, location, embed_model)
                if result[0] is not None:
                    st.session_state.embedder = result[0]
                    st.session_state.retriever = result[1]
                    st.session_state.chunk_vecs = result[2]
                    st.session_state.chunk_ids = result[3]
                    st.session_state.chunk_texts = result[4]
                    st.session_state.corpus_loaded = True
                    st.success(f"✓ Loaded {len(result[3])} document chunks")

    if st.session_state.corpus_loaded:
        st.success(f"✓ Corpus loaded ({len(st.session_state.chunk_ids)} chunks)")


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Title
st.title("🛡️ LLM Sentinel Dashboard")
st.markdown("Real-time LLM observability with hallucination detection and performance metrics")

# Check if corpus is loaded
if not st.session_state.corpus_loaded:
    st.warning("⚠️ Please load the document corpus using the sidebar button before chatting.")

# Create two columns: Chat (left) and Metrics (right)
col_chat, col_metrics = st.columns([2, 1])

# ============================================================================
# LEFT COLUMN: Chat Interface
# ============================================================================
with col_chat:
    st.header("💬 Chat")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question...", disabled=not st.session_state.corpus_loaded):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            metrics_placeholder = st.empty()

            try:
                # Create LLM client
                llm = create_llm_from_config(
                    backend, vllm_url, vllm_model,
                    project_id, location, gemini_model
                )

                if llm is None:
                    response_placeholder.error("Failed to create LLM client")
                else:
                    request_id = str(uuid.uuid4())

                    # Step 1: Embed query
                    with metrics_placeholder:
                        st.text("🔍 Embedding query...")
                    q_vec = st.session_state.embedder.embed([prompt], task_type="RETRIEVAL_QUERY")[0]

                    # Step 2: Retrieve context
                    with metrics_placeholder:
                        st.text("📚 Retrieving context...")
                    top = st.session_state.retriever.top_k(q_vec, k=top_k)
                    retrieved = [{
                        "chunk_id": r.chunk_id,
                        "score": r.score,
                        "text_preview": (r.text[:300] + "…") if len(r.text) > 300 else r.text,
                    } for r in top]
                    topk_scores = [r.score for r in top]

                    # Get vectors and sources for grounding
                    id_to_idx = {cid: i for i, cid in enumerate(st.session_state.chunk_ids)}
                    retrieved_vecs = [st.session_state.chunk_vecs[id_to_idx[r["chunk_id"]]] for r in retrieved]
                    sources = [st.session_state.chunk_texts[id_to_idx[r["chunk_id"]]] for r in retrieved]

                    # Step 3: Generate response with streaming
                    with metrics_placeholder:
                        st.text("🤖 Generating response...")

                    res = llm.generate_streaming(
                        prompt, sources,
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )

                    # Display response
                    response_placeholder.markdown(res.text)

                    # Step 4: Grounding check
                    with metrics_placeholder:
                        st.text("🔬 Checking hallucination...")
                    gr = grounding_check(res.text, retrieved_vecs, st.session_state.embedder, threshold=threshold)

                    # Build telemetry
                    slo_config = SLOConfig.from_env()
                    telem = build_request_telemetry(
                        request_id=request_id,
                        model=res.model,
                        prompt=prompt,
                        answer=res.text,
                        latency_ms=res.latency_ms,
                        error=False,
                        error_type=None,
                        hallucination_rate=gr.hallucination_rate,
                        hallucinated_sentences=len(gr.flagged),
                        grounding_threshold=gr.threshold,
                        retrieved=retrieved,
                        topk_scores=topk_scores,
                        slo_config=slo_config,
                        ttft_ms=res.ttft_ms,
                        generation_time_ms=res.generation_time_ms,
                        output_tokens=res.output_tokens,
                    )

                    # Store telemetry
                    st.session_state.last_telemetry = telem
                    st.session_state.metrics_history.append({
                        "timestamp": time.time(),
                        "ttft_ms": telem.ttft_ms,
                        "tpot_ms": telem.tpot_ms,
                        "throughput": telem.throughput_tokens_per_sec,
                        "hallucination_rate": telem.hallucination_rate,
                        "latency_ms": telem.latency_ms,
                    })

                    # Clear metrics placeholder
                    metrics_placeholder.empty()

                    # Add assistant message to history
                    st.session_state.messages.append({"role": "assistant", "content": res.text})

                    # Force rerun to update metrics display
                    st.rerun()

            except Exception as e:
                response_placeholder.error(f"Error: {e}")
                import traceback
                st.error(traceback.format_exc())


# ============================================================================
# RIGHT COLUMN: Metrics Display
# ============================================================================
with col_metrics:
    st.header("📊 Metrics")

    if st.session_state.last_telemetry:
        telem = st.session_state.last_telemetry

        # Performance Metrics
        st.subheader("⚡ Performance")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("TTFT", f"{telem.ttft_ms}ms")
            st.metric("Latency", f"{telem.latency_ms}ms")
        with m2:
            st.metric("TPOT", f"{telem.tpot_ms:.1f}ms/tok")
            st.metric("Throughput", f"{telem.throughput_tokens_per_sec:.1f} tok/s")

        st.markdown("---")

        # Hallucination Metrics
        st.subheader("🔬 Hallucination")

        rate = telem.hallucination_rate
        if rate > 0.3:
            st.error(f"⚠️ High: {rate:.0%}")
        elif rate > 0.1:
            st.warning(f"Moderate: {rate:.0%}")
        else:
            st.success(f"✓ Low: {rate:.0%}")

        st.caption(f"Severity: {telem.severity}")
        st.caption(f"Flagged sentences: {telem.hallucinated_sentences}")

        st.markdown("---")

        # SLO Status
        st.subheader("🎯 SLO Status")

        slos = [
            ("Availability", telem.slo_availability),
            ("Latency", telem.slo_latency),
            ("Quality", telem.slo_quality),
            ("Tokens", telem.slo_tokens),
            ("Retrieval", telem.slo_retrieval),
        ]

        for name, passed in slos:
            if passed:
                st.markdown(f"✅ **{name}**")
            else:
                st.markdown(f"❌ **{name}**")

        st.markdown("---")

        # Retrieved Context
        st.subheader("📚 Retrieved Context")

        with st.expander("View retrieved chunks", expanded=False):
            for chunk in telem.retrieved_context:
                st.markdown(f"**{chunk['chunk_id']}** (score: {chunk['score']:.3f})")
                st.caption(chunk['text_preview'][:200] + "...")
                st.markdown("---")

        st.markdown("---")

        # Token Usage
        st.subheader("🪙 Token Usage")
        st.metric("Input Tokens (est)", telem.input_tokens_est)
        st.metric("Output Tokens (est)", telem.output_tokens_est)
        st.metric("Cost (est)", f"${telem.cost_usd_est:.4f}")

    else:
        st.info("Ask a question to see metrics")


# ============================================================================
# FOOTER: History Chart
# ============================================================================
if st.session_state.metrics_history:
    st.markdown("---")
    st.subheader("📈 Metrics History")

    import pandas as pd

    df = pd.DataFrame(st.session_state.metrics_history)

    tab1, tab2, tab3 = st.tabs(["TTFT", "Throughput", "Hallucination"])

    with tab1:
        st.line_chart(df["ttft_ms"], use_container_width=True)

    with tab2:
        st.line_chart(df["throughput"], use_container_width=True)

    with tab3:
        st.line_chart(df["hallucination_rate"], use_container_width=True)
