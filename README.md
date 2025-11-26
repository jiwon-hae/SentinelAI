# 🛡️ SentinelAI
Real-Time Observability, Drift Detection, and Hallucination Intelligence for LLM Systems  
Built for the Google x Datadog Hackathon

---

## 🌟 Overview

**SentinelAI** is an end-to-end observability and safety monitoring system for Large Language Models.  
It provides **real-time telemetry**, **hallucination scoring**, **prompt injection detection**,  
**latency and performance monitoring**, and **incident generation** using **Datadog** and **Vertex AI/Gemini**.

This project turns LLM behavior into **structured, intelligent signals** and continuously analyzes model health  
as the system processes live requests. When risky or anomalous behavior occurs, Datadog automatically  
creates incidents with contextual insights, empowering AI engineers to respond immediately.

---

## 🎯 Features

### ✔ Real-Time LLM Telemetry Streaming  
- Token-by-token output streaming  
- Full prompt chain visibility  
- Latency, cost, and throughput metrics  
- Embedding drift and semantic deviation tracking  

### ✔ Hallucination & Risk Detection
- Hallucination probability score  
- Confidence mismatch detection  
- Inconsistency analysis using Vertex/Gemini  
- Prompt injection classifier  
- Toxicity and safety violations  

### ✔ Datadog Dashboards  
Visualize:
- Request volume  
- Token generation speeds  
- Latency spikes  
- Drift over time  
- Hallucination trends  
- Injection attempts  
- Error patterns  

### ✔ Datadog Detection Rules  
Triggers include:
- High hallucination score  
- Spike in prompt injection attempts  
- Latency degradation  
- Output toxicity increase  
- Semantic drift beyond threshold  

### ✔ Automated Incident Generation  
When detection rules fire, Datadog creates:
- Incident tickets  
- Slack or email alerts  
- Contextual summaries from Gemini  
- Suggested mitigation steps  

---

## 🧱 Architecture
