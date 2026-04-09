# backend/

Unified retrieval API that merges two search branches — **agentic** (intent-aware, multimodal LangGraph pipeline) and **heuristic** (dense hybrid RRF) — behind a single FastAPI endpoint.

## What this module is responsible for

| Concern | Detail |
|---|---|
| **Query preprocessing** | Unicode NFC normalization, whitespace collapsing, punctuation stripping, emoji removal, language detection (Vietnamese heuristics + `langdetect`), translation to English via pluggable LLM provider, deterministic rewrite generation (original + translated + keyword-only variant). Implemented in the search middleware. |
| **Agentic retrieval** | LangGraph pipeline: intent extraction → modality routing → multi-modal Qdrant search (5 named vectors) → weighted fusion → multi-signal reranking. |
| **Heuristic retrieval** | Full production implementation. 1 HTTP call to `/embed/query` → 2-tier Qdrant fallback search → True RRF fusion → Count Bonus multiplier. |
| **Cross-source reranking** | Reciprocal Rank Fusion (RRF, k=60) merging ranked lists from both branches. Frames appearing in both branches receive a natural agreement bonus. |
| **Response building** | Flattens internal `raw_payload` dicts into a stable `SearchResultItem` schema with `azure_url`, `youtube_link`, `timestamp_sec`, `caption`, `ocr_text`. |
| **Latency reporting** | Per-phase timing (`agentic_ms`, `heuristic_ms`, `rerank_ms`, `total_ms`) included in every response. |

## What this module is NOT responsible for

- **Data ingestion, keyframe extraction, embedding generation, OCR** — see `data-processing/`.
- **Qdrant collection creation / upsert** — see `data-processing/src/qdrant/qdrant_upsert.py`.
- **Azure blob storage migration** — see `data-processing/src/azure_migrator.py`.
- **Embedding model hosting** — the Azure VM Embedding API service is external (see `azure-ai-provider/`).
- **Frontend rendering** — handled by `streamlit/app.py`, which consumes this API.

## Structure

```
backend/
├── api.py                              # FastAPI entry point, CORS, /health
├── .env.example                        # Required env vars (4 total)
├── requirements.txt                    # Merged deps from agentic + heuristic
│
├── src/
│   ├── config.py                       # Env loading, constants (COLLECTION_NAME, dims)
│   ├── schemas.py                      # Pydantic API models + internal TypedDicts
│   │
│   ├── routes/
│   │   └── search_route.py             # POST /search — the only retrieval endpoint
│   │
│   ├── middlewares/
│   │   └── search_middleware.py         # Clean → detect lang → translate → rewrites
│   │
│   ├── controllers/
│   │   ├── search_controller.py        # Orchestrator: call both services → rerank → respond
│   │   ├── rerank.py                   # Cross-source RRF logic
│   │   └── response_builder.py         # Candidate dicts → SearchResponse
│   │
│   └── services/
│       ├── translator.py               # Shared: language detection + LLM translation
│       │
│       ├── llm/                         # Pluggable LLM provider layer
│       │   ├── __init__.py             # Re-exports LLMProvider, get_llm_provider
│       │   ├── provider.py             # Abstract base class
│       │   ├── gemini_provider.py       # Google Gemini (google-genai SDK)
│       │   ├── openai_compat_provider.py # Any OpenAI-compatible server
│       │   └── factory.py              # Env-based singleton factory
│       │
│       ├── agentic_retrieve/
│       │   ├── service.py              # AgenticRetrieveService (singleton wrapper)
│       │   ├── graph.py                # LangGraph: 5 nodes, no normalization
│       │   ├── state.py                # AgentState TypedDict
│       │   ├── llm_service.py          # LLM-backed intent extraction (uses llm/ provider)
│       │   ├── qdrant_search.py        # Multi-modal Qdrant search (4 vectors + metadata)
│       │   ├── scoring.py              # Weight normalization, minmax, dedup
│       │   ├── nodes/
│       │   │   ├── intent_extraction.py  # LLM-based structured intent parsing
│       │   │   ├── routing.py            # Modality weight computation from intent
│       │   │   ├── retrieval.py          # Parallel search across 5 named vectors
│       │   │   ├── fusion.py             # Weighted score fusion with dedup
│       │   │   └── rerank.py             # Multi-signal rerank (agreement + coverage)
│       │   │                             # See docs/rerank-design-rationale.md
│       │   └── utils/
│       │       └── json_utils.py       # JSON extraction from LLM output
│       │
│       └── heuristic_retrieve/
│           └── service.py              # 2-tier fallback search, True RRF, Count Bonus multiplier
│
└── test/
    └── run_agentic_demo.py             # Standalone CLI demo (not a test suite)
```

## Entry points

| Entry point | What it does |
|---|---|
| `api.py` | FastAPI app object. Include `search_router` and a `/health` endpoint. |
| `test/run_agentic_demo.py` | Standalone CLI to run the pipeline without starting the server. Accepts `--query`, `--top_k`, `--verbose`, `--all-samples`. |
| `test/bench_latency.py` | Standalone script to benchmark latency for BOTH `agentic` and `heuristic` strategies. Supports `--runs`, `--backend`, and CSV export. |

## API contract

### `GET /health`

Returns `200` immediately. Does not initialize models.

```json
{"status": "ok", "collection": "keyframes_v1", "device": "remote (Azure VM)"}
```

### `POST /search`

**Request body** (`SearchRequest`):

| Field | Type | Required | Constraints |
|---|---|---|---|
| `raw_query` | `str` | yes | `min_length=1` |
| `top_k` | `int` | no (default 10) | `1 ≤ top_k ≤ 50` |

**Response body** (`SearchResponse`):

```jsonc
{
  "query": "original raw query",
  "total_results": 5,
  "results": [
    {
      "rank": 1,
      "video_id": "abc123",
      "frame_id": 42,
      "score": 0.016393,         // RRF score (not raw similarity)
      "branch": "agentic",       // "agentic" | "heuristic" | "fused"
      "azure_url": "https://...",
      "youtube_link": "https://...",
      "timestamp_sec": 12.5,
      "caption": "a person speaking outdoors",
      "ocr_text": null,          // populated only if text was detected
      "evidence": ["keyframe", "caption", "object"]
    }
  ],
  "latency_ms": {
    "agentic_ms": 5898.44,
    "heuristic_ms": 0.34,
    "rerank_ms": 0.03,
    "total_ms": 5898.81
  }
}
```

> **Note on `score`:** The score is an RRF-derived value, not a raw cosine similarity. It is useful for relative ranking but not for absolute thresholding.

## External dependencies (runtime)

| Service | Env var | Purpose |
|---|---|---|
| **Qdrant Cloud** | `QDRANT_URL`, `QDRANT_API_KEY` | Vector search against `keyframes_v1` collection |
| **LLM Provider** | `LLM_BACKEND` | `"gemini"` (default), `"openai_compat"`, or `"llama_cpp"` |
| **Gemini API** | `GEMINI_API_KEY` | Required when `LLM_BACKEND=gemini` |
| **Self-hosted LLM** | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME` | Required when `LLM_BACKEND=openai_compat` or `llama_cpp` |
| **Azure Embedding API** | `EMBEDDING_API_BASE_URL` | text→vector encoding (SigLIP, BGE-M3, BM25 sparse). Must expose `/embed/visual`, `/embed/semantic`, `/embed/sparse` |

`QDRANT_URL`, `QDRANT_API_KEY`, `EMBEDDING_API_BASE_URL` are always required. LLM vars depend on `LLM_BACKEND`. Copy `.env.example` to `.env` and fill in values.

## Qdrant collection contract

This module queries (read-only) the `keyframes_v1` collection. It expects the following named vectors and payload fields:

**Named vectors** (provisioned by `data-processing/src/qdrant/qdrant_upsert.py`):

| Vector name | Type | Dim | Use |
|---|---|---|---|
| `keyframe-dense` | dense | 1152 (SigLIP) | Visual similarity search |
| `keyframe-caption-dense` | dense | 1024 (BGE-M3) | Semantic caption search |
| `keyframe-object-sparse` | sparse | — | BM25 object tag search |
| `keyframe-ocr-sparse` | sparse | — | BM25 OCR text search |

**Expected payload fields** (consumed by `response_builder.py`):

`video_id`, `frame_idx`, `azure_url`, `timestamp_sec`, `youtube_link`, `tags`, `caption`, `detailed_caption`, `object_counts`, `ocr_text`, `title`

> If payload fields are missing, the response will have `null` for those fields. No crash — graceful degradation.

## Commands

```bash
# ── Setup ──────────────────────────────────────────────────
cd backend
cp .env.example .env          # fill in all 4 values
pip install -r requirements.txt

# ── Run (development) ─────────────────────────────────────
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# ── Run (standalone demo & benchmark) ─────────────────────────
cd backend
python test/run_agentic_demo.py --query "người đang nói chuyện" -k 10
python test/run_agentic_demo.py --all-samples --verbose
python test/bench_latency.py --strategy both --runs 3    # Measures per-node latency

# ── Quick smoke test (inferred) ──────────────────────────
curl http://localhost:8000/health
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"raw_query": "a person speaking on stage", "top_k": 5}'
```

> Commands above are **inferred** from the codebase. There is no Makefile, CI config, or script runner present.

## Request flow

```
Client                          Backend
  │                               │
  │ POST /search                  │
  │ {"raw_query": "...", top_k}   │
  │─────────────────────────────►│
  │                               │
  │              search_middleware │  clean → detect_language → translate → rewrites
  │              (Depends)        │  → QueryBundle
  │                               │
  │              search_controller│  ┌─ AgenticRetrieveService.retrieve(qb, top_k*2)
  │              .execute_search()│  │   └─ LangGraph: intent → route → search → fuse → rerank
  │                               │  │
  │                               │  ├─ HeuristicRetrieveService.retrieve(qb, top_k*2)
  │                               │  │   └─ 2-tier search → True RRF → Count Bonus
  │                               │  │
  │                               │  ├─ cross_source_rerank (RRF k=60)
  │                               │  └─ build_response (flatten raw_payload → SearchResultItem)
  │                               │
  │ SearchResponse                │
  │◄─────────────────────────────│
```

## Singleton lifecycle

Both retrieval services are initialized **lazily on first `/search` request** and reused for all subsequent requests. This means:

- **First request is slow** (~6s) — Qdrant client connects, LLM client warms up.
- **Subsequent requests** skip initialization.
- There is **no graceful shutdown** hook to close Qdrant connections.

The singletons live in `search_controller.py` as module-level globals.

## What to test after changes

| If you change… | Then verify… |
|---|---|
| `schemas.py` (response fields) | Frontend `streamlit/app.py` still renders correctly. Check `render_result_card()` field access. |
| `search_middleware.py` | Translation works for Vietnamese input. English passthrough is unchanged. Rewrites list is non-empty. |
| `services/llm/*` | Set `LLM_BACKEND=gemini` → verify existing behaviour. Set `LLM_BACKEND=openai_compat` with a running llama.cpp server → verify translation + intent extraction. |
| `controllers/rerank.py` | Same (video_id, frame_id) from both branches → `branch: "fused"`. Scores are RRF-based, not raw. |
| `controllers/response_builder.py` | `azure_url` extracted from `raw_payload`. Missing payload fields → `null`, not crash. |
| `agentic_retrieve/nodes/*` | Run `test/run_agentic_demo.py` with `--verbose` to trace per-node output. |
| `agentic_retrieve/qdrant_search.py` | Embedding API calls succeed. All 5 search functions return results. |
| `heuristic_retrieve/service.py` | Ensure returned dictionary keys remain: `video_id`, `frame_id`, `score`, `branch`, `evidence`, `raw_payload`. |
| `config.py` / `.env` | Server starts without `EnvironmentError`. Health endpoint returns expected values. |
| `requirements.txt` | `pip install -r requirements.txt` completes. Server starts. |

## Technical debt and TODOs

| Item | Location | Severity |
|---|---|---|
| **No automated tests** | `test/` | 🔴 High — `run_agentic_demo.py` and `bench_latency.py` are manual CLI scripts, not a pytest suite. No unit tests for middleware, controller, or rerank. |
| **Singleton via module globals** | `controllers/search_controller.py:24-25` | 🟡 Medium — module-level `_agentic_service` / `_heuristic_service` globals are not thread-safe and not testable. Should migrate to FastAPI `Depends()` + `@lru_cache` or lifespan events. |
| **No graceful shutdown** | `api.py` | 🟡 Medium — Qdrant client and LLM client connections are never explicitly closed. |
| **CORS allows all origins** | `api.py:49` | 🟡 Medium — `allow_origins=["*"]` is acceptable for local dev but must be restricted for deployment. |
| **`.gitignore` is empty** | `.gitignore` | 🟡 Medium — should at minimum exclude `.env`, `__pycache__/`, `*.pyc`. |
| **Duplicate TypedDicts** | `schemas.py` + `agentic_retrieve/state.py` | 🟢 Low — `QueryBundle`, `QueryIntent`, `Candidate`, `TraceLog` are defined in both files. `schemas.py` is canonical; `state.py` copies are for LangGraph internal typing. |
| **Rewrite generation is rule-based** | `middlewares/search_middleware.py:98-134` | 🟢 Low — `_generate_safe_rewrites` produces up to 3 variants (translated + cleaned + keyword-only). Keyword extraction uses EN+VI stopword lists. Future improvement: LLM-based paraphrase rewrites. |
| **No request validation beyond Pydantic** | `routes/search_route.py` | 🟢 Low — no rate limiting, auth, or request size limiting. |
| **`src/` in import path** | All internal imports | 🟢 Low — `from src.xxx` is unconventional for Python packages. Works due to `sys.path.insert` in `api.py`. Functional but fragile for testing from repo root. |
