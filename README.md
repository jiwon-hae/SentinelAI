# 🛡️ LLM Sentinel  
**End-to-end LLM observability, reliability, and incident response on Vertex AI**

LLM Sentinel is an **end-to-end observability system** for LLM applications powered by **Vertex AI Gemini**.  
It streams **LLM behavior, runtime telemetry, and reliability signals** to **Datadog**, defines **detection rules and SLOs**, and creates **actionable incidents with full context** for AI engineers to act on.

This project demonstrates how LLM hallucinations and reliability issues can be treated as **first-class operational signals**, not post-hoc failures.

---

## 🎯 Challenge Alignment (Datadog Hard Requirements)

| Requirement | How LLM Sentinel Satisfies It |
|------------|-------------------------------|
| Vertex AI / Gemini required | Uses **Gemini on Vertex AI** for generation and **Vertex AI embeddings** |
| End-to-end observability | Metrics, logs, detection rules, SLOs, dashboards |
| ≥3 detection rules | Hallucination spike, latency regression, error-rate regression |
| Actionable record | Datadog **Incident Management** with runbook + context |
| In-Datadog health view | Dashboard showing latency, errors, tokens, cost, hallucinations, SLOs, incidents |
| Telemetry to Datadog | Custom metrics + structured logs via Datadog API |

---

## 🚩 Problem

LLM applications frequently fail in subtle ways:
- hallucinated or ungrounded responses  
- degraded retrieval quality  
- latency or error regressions after prompt or model changes  

In production, these failures are difficult to detect and usually surface only through user complaints, without enough context for engineers to debug quickly.

---

## 💡 Solution

**LLM Sentinel** reframes LLM reliability as an **observability and incident response problem**.

It wraps a **Vertex AI Gemini RAG application** with:
- context-aware hallucination detection  
- reference tracking for every query  
- end-to-end runtime telemetry  
- Datadog monitors, SLOs, dashboards, and incidents  

This allows AI engineers to **detect, investigate, and remediate** LLM issues using familiar operational workflows.

---

## 🧠 Architecture & Flow

```
User Query
   ↓
Vertex AI Embeddings (Query)
   ↓
Top-K Retrieval (RAG)
   ↓
Vertex AI Gemini (Generation)
   ↓
Sentence-level Grounding Analysis
   ↓
Datadog Metrics + Logs
   ↓
Detection Rules → Incident / Alert
```

---

## 🔍 What LLM Sentinel Reports (Every Request)

### LLM Reliability Signals
- Hallucination rate  
- Number of ungrounded sentences  
- Severity level (low / medium / high)  

### Context & References (Always Reported)
- Retrieved document chunk IDs  
- Similarity scores  
- Text previews of context used for generation  

### Application Health Telemetry
- End-to-end latency
- Error count / error rate
- Token usage (input / output)
- Estimated cost
- Model name

### LLM Performance Metrics (Evaluation)
- **TTFT** (Time to First Token) - User-perceived responsiveness
- **TPOT** (Time Per Output Token) - Generation efficiency (ms/token)
- **Throughput** - Token generation rate (tokens/sec)
- **Generation Time** - Time spent generating after first token

All signals are emitted to Datadog as **metrics and structured logs**.

---

## 📊 Datadog Observability Strategy

### Metrics

**Runtime Health:**
- `llm.latency_ms` - End-to-end latency
- `llm.error_count` - Error events
- `llm.request_count` - Total requests

**Token & Cost:**
- `llm.tokens.input` - Estimated input tokens
- `llm.tokens.output` - Estimated output tokens
- `llm.cost_usd` - Estimated cost in USD

**Hallucination Signals:**
- `llm.sentinel.hallucination_rate` - Ratio of ungrounded sentences (0-1)
- `llm.sentinel.hallucinated_sentences` - Count of flagged sentences

**LLM Performance (Evaluation):**
- `llm.performance.ttft_ms` - Time to First Token (streaming-based)
- `llm.performance.tpot_ms` - Time Per Output Token (ms/token)
- `llm.performance.throughput_tps` - Throughput (tokens/sec)
- `llm.performance.generation_time_ms` - Generation time after TTFT

### Logs
Each request emits a structured log containing:
- prompt and response
- retrieved references
- hallucination analysis
- LLM performance metrics (TTFT, TPOT, throughput)
- request_id for correlation
- model metadata

These logs provide the **context required to act** when an alert fires.

---

## 🚨 Detection Rules (Monitors)

LLM Sentinel defines **at least three Datadog monitors**:

1. **Hallucination Spike Monitor**  
   Triggers when hallucination rate exceeds a threshold over time.

2. **Latency Regression Monitor**  
   Triggers when p95 latency exceeds expected bounds.

3. **Error Rate Monitor**  
   Triggers when request failure rate increases.

Each monitor links directly to logs, dashboards, and remediation steps.

---

## 📉 Service Level Objectives (SLOs)

LLM Sentinel defines and visualizes SLOs inside Datadog:

- **Availability SLO**  
  % of requests without errors

- **Latency SLO**  
  % of requests completing under a latency threshold

SLO burn-down and status are displayed on the dashboard to provide a clear reliability signal.

---

## 🚑 Actionable Incident Management

When a detection rule is triggered, Datadog automatically creates an **Incident** containing:

- Triggering signal and metric  
- Affected service and model  
- Direct links to logs and dashboards  
- A runbook with next steps for AI engineers:
  - inspect retrieved context  
  - review hallucinated sentences  
  - adjust retrieval, prompt, or model parameters  

This ensures issues are **actionable**, not just observable.

---

## 📈 Datadog Dashboard (In-Datadog View)

The dashboard provides a single-pane view of:

- Latency (p50 / p95)
- Error rate
- Token usage and estimated cost
- Hallucination rate trends
- **LLM Performance**: TTFT, TPOT, throughput
- Monitor and SLO status
- Active incidents
- Recently flagged responses

This satisfies the requirement for a **clear, in-Datadog view of application health**.

---

## 🧰 Tech Stack

- **Vertex AI Gemini** (LLM generation with streaming)
- **Vertex AI Text Embeddings** (retrieval & grounding)
- Python
- Datadog Metrics, Logs, APM, Monitors, SLOs, Incident Management

---

## ⚙️ Setup (Local Demo)

### Prerequisites
- Google Cloud project with Vertex AI enabled  
- Datadog account and API key  
- Application Default Credentials configured  

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Environment Variables
```env
VERTEX_PROJECT_ID=your_project_id
VERTEX_LOCATION=us-central1
VERTEX_GEMINI_MODEL=gemini-1.5-pro

DATADOG_API_KEY=your_datadog_api_key
DATADOG_SITE=datadoghq.com
```

### Run
```bash
python sentinel-ai/app.py
```

Example output:
```
--- Answer ---
[Generated answer from Gemini]

--- Summary ---
request_id=abc123
error=False (None) latency_ms=2450
hallucination_rate=0.15 severity=low

--- LLM Performance ---
TTFT (Time to First Token): 245ms
TPOT (Time Per Output Token): 12.50ms/token
Throughput: 80.0 tokens/sec
Generation time: 1500ms
```

---

## 🎯 Why This Matters

LLM Sentinel shows how hallucinations and reliability issues can be handled using **production-grade observability practices**.

By integrating LLM behavior into Datadog’s monitoring, SLO, and incident workflows, teams gain:
- faster detection  
- clearer diagnosis  
- safer deployment of LLM systems  

---

## 🏁 Hackathon Context

Built for the **Datadog Challenge** of the **AI Partner Catalyst Hackathon**, this project demonstrates an innovative, production-oriented approach to LLM observability using **Vertex AI** and **Datadog**.
