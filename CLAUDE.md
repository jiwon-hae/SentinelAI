# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM Sentinel** is an end-to-end observability wrapper for LLM applications powered by Vertex AI Gemini. It monitors hallucination behavior, context usage, and runtime health in RAG-based systems, streaming actionable telemetry to Datadog.

Built for the Datadog Challenge of the AI Partner Catalyst Hackathon, it reframes hallucinations as an observability problem rather than anecdotal failures.

## Architecture

### Core Pipeline

```
User Query
  ↓
RAG Retrieval (Vertex AI Embeddings)
  ↓ (top-K document chunks + embeddings)
Gemini Generation (Vertex AI Gemini)
  ↓ (answer text)
Grounding Analysis (sentence-level semantic comparison)
  ↓ (hallucination signals)
Datadog Telemetry (metrics + structured logs)
  ↓
Dashboards → Detection Rules → Alerts
```

### Module Structure

The `sentinel/` package contains:

- **`embedder.py`** - `VertexEmbedder` wraps Vertex AI text embeddings API
- **`retriever.py`** - `InMemoryRetriever` performs cosine similarity search over pre-embedded chunks
- **`llm.py`** - `VertexGeminiClient` generates answers using Vertex AI Gemini
- **`hallucination.py`** - `grounding_check()` performs sentence-level semantic comparison against retrieved context
- **`indexing.py`** - `build_chunks_from_file()` chunks documents for RAG corpus
- **`datadog.py`** - `DatadogClient` sends metrics and logs to Datadog APIs
- **`telemetry.py`** - `build_request_telemetry()` and `to_datadog_log()` structure telemetry payloads
- **`apm.py`** - APM instrumentation with `llm_span()` context manager and tag helpers for distributed tracing

### How Hallucination Detection Works

1. **Retrieval**: Query is embedded, top-K most similar chunks are retrieved from the corpus
2. **Generation**: Gemini generates an answer using only the retrieved chunks as context
3. **Grounding Check**:
   - Response is split into individual sentences
   - Each sentence is embedded and compared semantically against the retrieved chunk embeddings
   - Sentences with max similarity below threshold (default 0.75) are flagged as **ungrounded**
4. **Scoring**: `hallucination_rate = flagged_sentences / total_sentences`, severity calculated from rate

## Development Commands

### Setup

**Prerequisites:**
- Google Cloud project with Vertex AI API enabled (or Google AI Studio API key)
- Datadog account with API key
- Python 3.8+

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Environment variables:**
Create `.env` file:
```env
VERTEX_PROJECT_ID=your_project_id
VERTEX_LOCATION=global
VERTEX_GEMINI_MODEL=gemini-1.5-pro
VERTEX_EMBED_MODEL=gemini-embedding-001
VERTEX_API_KEY=  # Optional: for Direct API authentication (get from https://aistudio.google.com/app/apikey)

DATADOG_API_KEY=your_datadog_api_key
DATADOG_SITE=datadoghq.com
DD_SERVICE=llm-sentinel
DD_ENV=demo
DD_APM_ENABLED=true  # Enable APM tracing (requires Datadog Agent)
DATADOG_LLMOBS_ENABLED=false  # Enable LLM Observability (optional)

# SLO Thresholds
SLO_LATENCY_MS=3000  # Max latency in milliseconds (default: 3000)
SLO_HALLUCINATION_MAX=0.3  # Max hallucination rate 0-1 (default: 0.3 = 30%)
SLO_TOKEN_BUDGET=10000  # Max total tokens per request (default: 10000)
SLO_RETRIEVAL_MIN=0.5  # Min avg retrieval similarity 0-1 (default: 0.5)

RAG_TOP_K=4
GROUNDING_THRESHOLD=0.75
ANSWER_MAX_TOKENS=600
TEMPERATURE=0.2
```

**Authentication Setup:**

Option 1 - Vertex AI (recommended for production):
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
# Leave VERTEX_API_KEY empty or unset in .env
```

Option 2 - Direct API (easier for local development):
```bash
# Get API key from https://aistudio.google.com/app/apikey
# Set VERTEX_API_KEY in .env
# No gcloud authentication needed
```

### Running the Application

**Main demo app:**
```bash
python sentinel-ai/app.py
```

This runs an interactive loop that:
1. Loads and embeds the document corpus from `data/docs.txt`
2. Accepts user questions
3. Performs RAG retrieval + Gemini generation + grounding analysis
4. Sends telemetry to Datadog
5. Prints answer and hallucination summary locally

### Running Tests

**Smoke tests** (no external API calls, all dependencies mocked):
```bash
cd test
pytest test_smoke.py -v
```

**Run specific test class:**
```bash
pytest test_smoke.py::TestRetriever -v
pytest test_smoke.py::TestHallucination -v
pytest test_smoke.py::TestIntegration -v
```

**With coverage report:**
```bash
pytest test_smoke.py -v --cov=sentinel --cov-report=term-missing
```

See `test/TESTING.md` for detailed testing documentation.

## Datadog Integration

LLM Sentinel provides comprehensive observability through **three complementary Datadog products**:

1. **Metrics** - Dashboards, monitors, and SLOs
2. **Logs** - Structured logs with full request context
3. **APM (Application Performance Monitoring)** - Distributed tracing with LLM-specific spans

### APM (Application Performance Monitoring)

**Enable APM** by setting `DD_APM_ENABLED=true` in `.env` and installing the Datadog Agent:

```bash
# Install ddtrace
pip install ddtrace>=2.0.0

# Run Datadog Agent (Docker example)
docker run -d --name dd-agent \
  -e DD_API_KEY=${DATADOG_API_KEY} \
  -e DD_SITE=${DATADOG_SITE} \
  -e DD_APM_ENABLED=true \
  -p 8126:8126 \
  datadog/agent:latest
```

**APM provides:**
- **Distributed Traces**: End-to-end visibility of each LLM request
- **Automatic Latency Tracking**: P50/P95/P99 percentiles per operation
- **Service Maps**: Visualize dependencies between RAG components
- **Error Tracking**: Automatic error capture with stack traces
- **LLM-Specific Tags**: Model, tokens, prompts, completions visible in trace UI

**Trace Structure** (4 spans per request):
1. `llm.request` (root span)
   - Tags: `request_id`, `llm.application=rag-qa`
2. `llm.embedding` (query embedding)
   - Tags: `llm.request.model`, `llm.request.type=embedding`, `llm.embedding.task_type`
3. `rag.retrieval` (semantic search)
   - Tags: `rag.query`, `rag.top_k`, `rag.avg_similarity`, `rag.chunk_ids`
4. `llm.completion` (Gemini generation)
   - Tags: `llm.request.model`, `llm.request.prompt`, `llm.response.completion`, `llm.usage.prompt_tokens`, `llm.usage.completion_tokens`
5. `llm.grounding` (hallucination detection)
   - Tags: `llm.hallucination.rate`, `llm.hallucination.severity`, `error=true` (if high severity)

**APM Configuration** (`.env`):
```env
DD_APM_ENABLED=true  # Enable APM tracing (default: true)
DD_SERVICE=llm-sentinel  # Service name in APM
DD_ENV=production  # Environment tag
DD_AGENT_HOST=localhost  # Datadog Agent hostname (optional, default: localhost)
DD_TRACE_AGENT_PORT=8126  # Trace agent port (optional, default: 8126)
```

**Viewing APM Data:**
- Navigate to **APM → Traces** in Datadog
- Filter by `service:llm-sentinel` and `env:production`
- Click traces to see flamegraphs and LLM-specific tags
- Create monitors on APM metrics (e.g., `trace.llm.completion.duration`)

### SLOs (Service Level Objectives)

**LLM Sentinel automatically tracks 5 SLOs** using ddtrace StatsD metrics:

| SLO | Description | Metric Formula | Default Threshold |
|-----|-------------|----------------|-------------------|
| **Availability** | Requests without errors | `sum(llm.slo.availability.good) / sum(llm.slo.availability.total)` | 99.9% |
| **Latency** | Requests completing within threshold | `sum(llm.slo.latency.good) / sum(llm.slo.latency.total)` | 95% < 3000ms |
| **Quality** | Requests with low hallucination rate | `sum(llm.slo.quality.good) / sum(llm.slo.quality.total)` | 95% < 30% hallucination |
| **Token Budget** | Requests within token limit | `sum(llm.slo.tokens.good) / sum(llm.slo.tokens.total)` | 90% < 10k tokens |
| **Retrieval Quality** | Requests with good retrieval scores | `sum(llm.slo.retrieval.good) / sum(llm.slo.retrieval.total)` | 90% > 0.5 similarity |
| **Overall** | All SLOs met simultaneously | `sum(llm.slo.overall.good) / sum(llm.slo.overall.total)` | 85% |

**SLO Configuration** (`.env`):
```env
SLO_LATENCY_MS=3000  # Max latency in milliseconds
SLO_HALLUCINATION_MAX=0.3  # Max hallucination rate (30%)
SLO_TOKEN_BUDGET=10000  # Max tokens per request
SLO_RETRIEVAL_MIN=0.5  # Min average retrieval similarity
```

**Creating SLOs in Datadog:**

1. Navigate to **Service Management → SLOs → New SLO**
2. Select **Metric Based**
3. Configure each SLO:

**Example: Latency SLO**
```
Name: LLM Latency
Description: 95% of requests complete within 3 seconds
Target: 95%
Time Window: 30 days

Good events: sum:llm.slo.latency.good{service:llm-sentinel,env:production}
Total events: sum:llm.slo.latency.total{service:llm-sentinel,env:production}
```

**Example: Quality SLO (Hallucination)**
```
Name: LLM Quality (Low Hallucination)
Description: 95% of requests have hallucination rate < 30%
Target: 95%
Time Window: 30 days

Good events: sum:llm.slo.quality.good{service:llm-sentinel,env:production}
Total events: sum:llm.slo.quality.total{service:llm-sentinel,env:production}
```

**SLO Alerting:**
- Each SLO can trigger alerts when error budget is consumed
- Set up **Error Budget Alerts** at 75% and 90% burn rate
- Configure **Incident Management** for SLO breaches

**SLO Dashboard Widgets:**
```json
{
  "viz": "timeseries",
  "requests": [{
    "q": "sum:llm.slo.latency.good{service:llm-sentinel}.as_count() / sum:llm.slo.latency.total{service:llm-sentinel}.as_count() * 100",
    "display_type": "line"
  }],
  "title": "Latency SLO Compliance %"
}
```

### Metrics Sent

**SLO Metrics:**
- `llm.request_count` - Total requests
- `llm.request_good_count` - Non-error requests
- `llm.latency_good_count` - Requests under 2s latency

**Hallucination Signals:**
- `llm.sentinel.hallucination_rate` - Ratio of ungrounded sentences (0-1)
- `llm.sentinel.hallucinated_sentences` - Count of flagged sentences

**Runtime Health:**
- `llm.latency_ms` - End-to-end latency
- `llm.error_count` - Error events

**Cost/Token Tracking:**
- `llm.tokens.input` - Estimated input tokens
- `llm.tokens.output` - Estimated output tokens
- `llm.cost_usd` - Estimated cost in USD

**RAG Quality:**
- `llm.rag.topk_avg_similarity` - Average similarity of top-K retrieved chunks

**LLM Performance (Evaluation Metrics):**
- `llm.performance.ttft_ms` - Time to First Token (streaming-based)
- `llm.performance.tpot_ms` - Time Per Output Token (ms/token)
- `llm.performance.throughput_tps` - Throughput (tokens/sec)
- `llm.performance.generation_time_ms` - Generation time after TTFT

### Structured Logs

Each request emits a structured log including:
- `request_id`, `model`, `prompt`, `answer`
- `hallucination_rate`, `hallucinated_sentences`, `severity`
- `retrieved[]` - Array of chunk IDs, scores, and text previews
- `grounding_threshold`
- `latency_ms`, `error`, `error_type`
- `input_tokens_est`, `output_tokens_est`, `cost_usd_est`
- `topk_avg_similarity`

This ensures every alert includes **actionable context**: not just "hallucination spiked" but which sentences were ungrounded, what chunks were retrieved, and their similarity scores.

### Dashboard & Detection Rules

**Dashboard should track:**
- Hallucination rate over time
- Hallucinated sentence count
- Latency percentiles (p50, p95)
- Error rate
- Recently flagged responses with context

**Detection rules trigger on:**
- Hallucination rate above threshold
- Latency regressions
- Error spikes

## Key Technical Considerations

### Grounding Threshold

The `GROUNDING_THRESHOLD` (default 0.75) determines sensitivity:
- **Higher threshold** (0.8-0.9): Stricter, more sentences flagged as hallucinations
- **Lower threshold** (0.6-0.7): More lenient, fewer false positives

Tune based on domain requirements and retrieved context quality.

### RAG Corpus

The system expects a document corpus at `data/docs.txt`. This is chunked and embedded at startup. For production:
- Pre-compute and cache embeddings to avoid startup latency
- Consider vector databases (Vertex AI Vector Search, Weaviate, Pinecone) for large corpora
- The current `InMemoryRetriever` is suitable for demos and small corpuses

### Token/Cost Estimation

Token counts and cost are estimated in `telemetry.py` based on character counts and Gemini pricing. For precise tracking, parse actual usage from Gemini API responses.

### Error Handling

The app catches exceptions during retrieval/generation and still attempts grounding check with partial data. Errors are tagged in telemetry with `error=True` and `error_type`.

## Google Cloud Services Used

The project supports **two authentication modes** for Vertex AI:

### Authentication Mode 1: Vertex AI with Application Default Credentials
- No API key required
- Uses `gcloud auth application-default login`
- Best for production deployments with service accounts

### Authentication Mode 2: Direct Gemini API with API Key
- Set `VERTEX_API_KEY` in `.env` file
- Get API key from Google AI Studio (https://aistudio.google.com/app/apikey)
- Easier for local development and testing
- Requires `google-generativeai` package

Both `VertexEmbedder` and `VertexGeminiClient` automatically detect which mode to use based on presence of `api_key` parameter.

### Gemini Model Options

**Vertex AI Models** (when using Application Default Credentials):
- `gemini-1.5-pro` - Latest Gemini 1.5 Pro (recommended)
- `gemini-1.5-flash` - Faster, more cost-effective variant
- `gemini-1.0-pro` - Legacy model

**Direct API Models** (when using API key):
- `gemini-2.0-flash-exp` - Latest experimental model
- `gemini-2.5-flash` - Experimental Flash 2.5 variant
- `gemini-1.5-pro` - Stable 1.5 Pro
- `gemini-1.5-flash` - Stable 1.5 Flash

**Important Notes:**
- Vertex AI uses model names WITHOUT version suffixes (e.g., `gemini-1.5-pro`, not `gemini-1.5-pro-002`)
- Direct API may use different model names (e.g., `gemini-2.0-flash-exp`)
- The code automatically adapts API calls based on authentication mode

### Embedding Model Options

Per official Vertex AI documentation, supported embedding models include:

**Recommended:**
- `gemini-embedding-001` - Recommended model, up to 3072 dimensions, 2048 token max
- `text-embedding-005` - Latest stable model, up to 768 dimensions, 2048 token max
- `text-multilingual-embedding-002` - Multilingual support, up to 768 dimensions, 2048 token max

**Open Models:**
- `multilingual-e5-small` - 384 dimensions
- `multilingual-e5-large` - 1024 dimensions

**Direct API Models** (when using API key):
- `text-embedding-004` - Latest for direct API
- `embedding-001` - Legacy for direct API

### Embedding Task Types (CRITICAL for Performance)

The embedder supports task-specific optimization via the `task_type` parameter. Using the correct task type significantly improves retrieval quality:

**Task Types:**
- `RETRIEVAL_DOCUMENT` - For embedding corpus documents (default)
- `RETRIEVAL_QUERY` - For embedding search queries
- `CLASSIFICATION` - For classification tasks
- `CLUSTERING` - For clustering tasks
- `SEMANTIC_SIMILARITY` - For semantic similarity tasks

**Best Practice Usage Pattern:**
```python
# Embed corpus chunks at startup
chunk_vecs = embedder.embed(chunk_texts, task_type="RETRIEVAL_DOCUMENT")

# Embed user query for search
query_vec = embedder.embed([question], task_type="RETRIEVAL_QUERY")[0]

# Embed generated sentences for grounding check (comparing against documents)
sent_vecs = embedder.embed(sentences, task_type="RETRIEVAL_DOCUMENT")
```

**Why This Matters:**
Using different task types for documents vs queries optimizes the embedding space for asymmetric search, where queries and documents have different characteristics.

### Dimensionality Control

For models that support it (e.g., `text-embedding-005`), you can specify output dimensions:
```python
embeddings = embedder.embed(texts, dimensionality=256)  # Reduce from 768 to 256
```

This reduces storage and computation costs while maintaining good performance for many applications.
