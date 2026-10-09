# Microsatellite Telemetry Anomaly Detection with RAG-based Explanation

> Shanghai Dianji University — Undergraduate Innovation and Entrepreneurship Training Program
> Duration: Apr 2026 – Mar 2027 | Supervisor: Lihua Lu

<p align="right"><b>English</b> | <a href="README.md">中文</a></p>

<p align="center"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"> <img alt="Backend" src="https://img.shields.io/badge/backend-FastAPI-009688"> <img alt="Frontend" src="https://img.shields.io/badge/frontend-React%2019%20%2B%20TypeScript-3178C6"> <img alt="RAG" src="https://img.shields.io/badge/RAG-BGE--M3%20%2B%20FAISS-orange"> <img alt="LLM" src="https://img.shields.io/badge/LLM-DeepSeek%20V3-4B6BFB"> <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey"></p>

Satellite telemetry is full of anomalies, but nobody knows **why** they happen. This project does two things: it detects anomalous segments with Isolation Forest, then uses RAG over a satellite handbook knowledge base to have an LLM explain each anomaly and cite traceable evidence.

**Scale**: built on the public ESA OPS-SAT dataset — 303,493 sampled points / 2,123 segments / 9 channels / 20.4% segment-level anomaly rate.

<p align="center">
  <img alt="Real-time alert center" src="docs/assets/dashboard.png" width="31%"> <img alt="Algorithm benchmark" src="docs/assets/detection.png" width="31%"> <img alt="In-depth diagnosis" src="docs/assets/rag_explain.png" width="31%">
</p>

## Key Results

| Metric | Result |
|---|---|
| Segment-level F1 | 0.5882 → **0.6281** (+6.8%) |
| Alert false positives | **20 → 0** |
| RAG citation traceability | **200 / 200** |
| Median retrieval latency | **25 ms** |
| Embedding speedup (GPU vs CPU) | 128 chunks 2.17 s → 0.26 s (**8.3×**) |
| Knowledge base size | **4,898** chunks |

## Quick Start

```bash
git clone -b main https://github.com/Jerry518520/microsat-anomaly-analysis.git
cd microsat-anomaly-analysis
python scripts/start_ui.py
```

Open http://localhost:5180. The detection dashboard works without an API key; RAG explanations require a key in `.env`.

## Architecture

```
┌───────────────────────────────────────────────────────┐
│           Frontend  React 19 + TypeScript             │
│   Dashboard · Detection · Explainer · Settings        │
└───────────────────────────────────────────────────────┘
                          │  REST + WebSocket
                          ▼
┌───────────────────────────────────────────────────────┐
│                 FastAPI backend                       │
│  /api/dashboard · /api/detection · /api/explanation   │
│  /api/stream                                          │
└───────────────────────────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────┐
│  Feature engineering → Detection → Gated fusion → RAG │
│  18-d segment features  Isolation Forest  Rule engine │
└───────────────────────────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────┐
│  FAISS vector store · BGE-M3 · DeepSeek V3 · SQLite   │
└───────────────────────────────────────────────────────┘
```

| Layer | Technology |
|------|------|
| Frontend | React 19 + TypeScript + Vite + Tailwind CSS |
| Backend | FastAPI + Uvicorn (REST + WebSocket) |
| Anomaly detection | 18-d segment features + per-channel adaptive Isolation Forest + rule-engine gated fusion |
| Retrieval | LlamaParse / BGE-M3 / FAISS / BM25 |
| LLM | DeepSeek V3 |
| Storage | SQLite (alerts and segment verdicts) |

## Two Things Worth Mentioning

**An explicit engineering trade-off**: to push median retrieval latency down to 25 ms, I deliberately dropped online incremental writes in favour of offline index rebuilds. The cost is that updating the knowledge base requires a rebuild; the benefit is a minimal retrieval path with predictable latency.

**Finding and fixing a non-obvious AI defect**: the anti-hallucination prompt was in place but never took effect — generated content drifted away from the retrieved evidence while the system raised no error at all. I located the root cause through assertion checks, then added a "refuse to answer when nothing is retrieved" fallback plus a regression test to prevent recurrence.

## Team and Roles

| Member | Responsibility | Deliverables | How code was merged |
|---|---|---|---|
| **Bingjie Li** (project lead) | Core algorithm + RAG + system architecture | 18-d feature engineering, gated fusion, full RAG pipeline, FastAPI backend, testing and engineering standards | Direct commits to main (123 commits) |
| Yijun Wang | Data processing + visualisation + frontend | `data/` cleaning scripts, frontend visualisation components | Developed locally, reviewed and merged by the lead |
| Shiyi Tao | Knowledge base + presentation + docs | Curation and proofreading of `docs/knowledge_base/` | Direct documentation commits |

> Note: all code was reviewed and merged by the project lead, so the GitHub contributor graph shows a single author. The actual division of work is as listed above.

## On AI-Assisted Development

Architecture design, prompt engineering, evaluation methodology and failure localisation were led by me. Boilerplate and repetitive implementations were generated with AI coding tools, then reviewed line by line, committed and covered with unit tests. The anti-hallucination failure described above is a typical issue surfaced by that manual review step.

## Repository Layout

```
microsat-anomaly-analysis/
├── src/            # Feature engineering / detection / gated fusion / RAG / FastAPI backend
├── frontend/       # React 19 + TypeScript frontend
├── configs/        # Global parameters and RAG configuration
├── data/           # Telemetry data, experiment results, vector index
├── models/         # BGE-M3 embedding model (not tracked in git)
├── scripts/        # Index building / experiments / one-command launcher
├── docs/           # Knowledge base, papers and project documentation
└── tests/          # pytest tests
```

| Document | Contents |
|---|---|
| [`DEPLOY.md`](DEPLOY.md) | Clone-to-deploy guide; live data ingestion (`ingest` / `replay`, WebSocket alerts) — *in Chinese* |
| [`API.md`](API.md) | Backend API contract (routes, fields, severity thresholds) — *in Chinese* |
| [`docs/指标口径说明.md`](docs/指标口径说明.md) | Definitions of and differences between segment-level and point-level anomaly rates — *in Chinese* |

---

> The Chinese README ([`README.md`](README.md)) additionally contains the full project description, dataset documentation and the internal deployment guide for team members.
