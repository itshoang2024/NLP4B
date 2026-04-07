# System Architecture

## Overview

NLP4B is a multimodal video retrieval system. Videos are preprocessed offline into searchable artifacts (keyframes, embeddings, annotations), indexed in Qdrant Cloud, and queried at runtime through a unified FastAPI backend that combines agentic and heuristic retrieval with cross-source reranking.

The system has four runtime boundaries:

| Boundary | Module | Runs when |
|---|---|---|
| **Offline processing** | `data-processing/` | During data ingestion (Colab/Kaggle/local) |
| **Online retrieval** | `backend/` | At query time (FastAPI on port 8000) |
| **Presentation** | `streamlit/` | User-facing demo (calls backend API) |
| **Service** | `azure-ai-provider/` | Always-on embedding API on Azure VM |

> **Note:** `retrieval/agentic_retrieval/` is the **legacy** standalone pipeline. Its logic has been migrated to `backend/src/services/agentic_retrieve/`. The old folder is kept for reference.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OFFLINE PROCESSING                              │
│                     (data-processing/)                              │
│                                                                     │
│  Excel manifest                                                    │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐ │
│  │ Download  │───▶│  Keyframe    │───▶│ Embedding │───▶│  Object  │ │
│  │ + ffprobe │    │  Extraction  │    │ (SigLIP)  │    │Detection │ │
│  │ main.py   │    │  LMSKE.py    │    │           │    │ + OCR    │ │
│  └──────────┘    └──────────────┘    └───────────┘    └──────────┘ │
│       │                │                   │               │        │
│       ▼                ▼                   ▼               ▼        │
│  normalized_      keyframes/          <vid>.npy       <vid>_object  │
│  manifest.csv     <vid>/              <vid>_frames    _detection    │
│  video_metadata   <vid>_XXXXX.jpg     .json           .json         │
│  .csv                                                  + OCR .json  │
│                                                                     │
│                    ┌─────────────────┐                              │
│                    │ Azure Migrator  │  ← uploads keyframes,       │
│                    │ azure_migrator  │    embeddings, OCR to       │
│                    │ .py             │    Azure Blob Storage       │
│                    └────────┬────────┘                              │
│                             │                                       │
│                    ┌────────▼────────┐                              │
│                    │ Qdrant Upsert   │  ← builds 4-vector points   │
│                    │ qdrant_upsert   │    and payload per frame    │
│                    │ .py             │                              │
│                    └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
                             │
                    Qdrant Cloud (keyframes_v1)
                             │
┌─────────────────────────────────────────────────────────────────────┐
│                     ONLINE RETRIEVAL                                │
│                     (backend/)                                      │
│                                                                     │
│  POST /search {"raw_query": "...", "top_k": 10}                   │
│       │                                                             │
│       ▼                                                             │
│  ┌────────────────┐                                                │
│  │ Middleware      │  Clean, detect language, translate → English   │
│  │ (search_       │  Generate rewrites → QueryBundle               │
│  │  middleware.py) │                                                │
│  └────────┬───────┘                                                │
│           │                                                         │
│     ┌─────┴──────────────────────────────────┐                     │
│     │                                         │                     │
│     ▼                                         ▼                     │
│  ┌──────────────────┐     ┌────────────────────────┐               │
│  │ Agentic Retrieve │     │ Heuristic Retrieve     │               │
│  │ (LangGraph)      │     │ (⚠ MOCK — WIP)        │               │
│  │                  │     │                        │               │
│  │ Intent Extraction│     │ Dense hybrid RRF with  │               │
│  │  → Routing       │     │ SigLIP + BGE-M3        │               │
│  │  → Parallel      │     │ (to be implemented)    │               │
│  │    Retrieval     │     │                        │               │
│  │  → Fusion        │     │                        │               │
│  │  → Rerank        │     │                        │               │
│  └────────┬─────────┘     └───────────┬────────────┘               │
│           │                            │                            │
│     ┌─────┴────────────────────────────┘                           │
│     ▼                                                               │
│  ┌──────────────────────┐                                          │
│  │ Cross-Source RRF     │  Reciprocal Rank Fusion (k=60)           │
│  │ Rerank               │  Merge both branches, dedup, re-score   │
│  │ (controllers/        │                                          │
│  │  rerank.py)          │                                          │
│  └──────────┬───────────┘                                          │
│             ▼                                                       │
│  ┌──────────────────────┐                                          │
│  │ Response Builder     │  Flatten raw_payload → SearchResultItem  │
│  │ (response_builder.py)│  azure_url, youtube_link, caption, etc.  │
│  └──────────────────────┘                                          │
│             │                                                       │
│        SearchResponse                                               │
└─────────────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────────────┐
│                     PRESENTATION                                    │
│                     (streamlit/)                                     │
│                                                                     │
│  Streamlit app.py → LookUp.ai UI                                   │
│  Calls POST localhost:8000/search → renders result cards            │
│  with Azure keyframe images, branch badges, and latency bars       │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Boundaries

### `data-processing/`

**Responsibility:** All offline artifact generation — from raw YouTube URLs to indexed vectors in Qdrant.

**Not responsible for:** Runtime query handling, UI rendering, live API serving.

**Key scripts:**

| Script | Input | Output |
|---|---|---|
| `src/download/main.py` | Excel manifest (URLs) | `normalized_manifest.csv`, `video_metadata.csv`, downloaded `.mp4` files |
| `src/keyframe_extraction/LMSKE.py` | Video file path/URL | `<output>/<video_id>/<video_id>_XXXXX.jpg`, `_scenes.txt` |
| `src/embedding/embedding.py` | Keyframe image directory | `<video_id>.npy` (N×1152), `<video_id>_frames.json` |
| `src/object_detection/object_detection.py` | Keyframe image directory | `<video_id>_object_detection.json` |
| `src/ocr/` | Keyframe images | OCR text extraction (CRAFT + PaddleOCR) |
| `src/azure_migrator.py` | Local artifact directories | Azure Blob containers: `keyframes`, `embeddings`, `ocr` |
| `src/qdrant/qdrant_upsert.py` | Azure Blob artifacts | Qdrant `keyframes_v1` collection (4-vector points) |
| `src/faiss/build_index.py` | Embedding `.npy` files | Local FAISS index |

---

### `backend/`

**Responsibility:** Unified retrieval API — query preprocessing, agentic + heuristic retrieval, cross-source reranking, response building.

**Not responsible for:** Data ingestion, model training, embedding model hosting, UI rendering.

**Architecture:** `Route → Middleware → Controller → Services`

| Layer | File(s) | Role |
|---|---|---|
| Route | `src/routes/search_route.py` | `POST /search` endpoint |
| Middleware | `src/middlewares/search_middleware.py` | Clean, detect language, translate, generate rewrites |
| Controller | `src/controllers/search_controller.py` | Orchestrate both services, RRF rerank, build response |
| Agentic service | `src/services/agentic_retrieve/` | LangGraph pipeline: intent → routing → retrieval → fusion → rerank |
| Heuristic service | `src/services/heuristic_retrieve/` | ⚠️ Mock stub — returns dummy data (WIP) |

**Agentic graph nodes** (5 nodes, executed sequentially via LangGraph):

| Node | File | Function |
|---|---|---|
| `query_intent_extraction` | `nodes/intent_extraction.py` | LLM-based structured intent (objects, actions, scene, text cues, etc.) |
| `modality_routing` | `nodes/routing.py` | Compute per-source retrieval weights |
| `parallel_retrieval` | `nodes/retrieval.py` | Query Qdrant across 5 sources: keyframe, caption, object, OCR, metadata |
| `candidate_fusion` | `nodes/fusion.py` | Weighted-sum merge: `Σ(routing_weight × normalized_score)` per frame |
| `frame_reranking` | `nodes/rerank.py` | Multi-signal reranking: cross-source agreement + intent coverage → `agent_topk` |

> **Note:** Query normalization is handled by the middleware (shared between both branches), not as a graph node. See [docs/rerank-design-rationale.md](rerank-design-rationale.md) for the rerank formula design.

See [backend/README.md](../backend/README.md) for full API contract, Qdrant collection expectations, and change-impact guide.

---

### `retrieval/agentic_retrieval/` (legacy)

> ⚠️ **Migrated to `backend/src/services/agentic_retrieve/`.** This folder is kept for reference only. See [backend/README.md](../backend/README.md) for the active codebase.

**Original responsibility:** Standalone LangGraph agentic retrieval pipeline with 6 nodes (including normalization).

---

### `azure-ai-provider/`

**Responsibility:** Hosted embedding inference service (FastAPI on Azure VM, CPU-only).

**API surface:**

| Endpoint | Model | Output |
|---|---|---|
| `POST /embed/semantic` | BAAI/bge-m3 | 1024d dense vector |
| `POST /embed/sparse` | Qdrant/bm25 | Sparse vector (indices + values) |
| `POST /embed/visual` | google/siglip-so400m-patch14-384 | 1152d dense vector |
| `GET /health` | — | Model status |

---

### `streamlit/`

**Responsibility:** User-facing demo interface ("LookUp.ai").

**Current state:** Fully functional — submits queries to `POST localhost:8000/search`, renders result cards with Azure images, branch badges (`agentic`/`heuristic`/`fused`), and per-phase latency breakdown.

See [streamlit/README.md](../streamlit/README.md) for details.

## Qdrant Collection Schema

The shared interface between offline processing and online retrieval is the Qdrant `keyframes_v1` collection. See [docs/contracts/qdrant-collection-schema.md](contracts/qdrant-collection-schema.md) for the full schema contract.

**4-vector architecture:** `keyframe-dense` (SigLIP 1152d), `keyframe-caption-dense` (BGE-M3 1024d), `keyframe-object-sparse` (BM25), `keyframe-ocr-sparse` (BM25). See the [full schema contract](contracts/qdrant-collection-schema.md) for vector population rules, payload fields, and compatibility risks.

## Model Alignment

Models used in indexing (qdrant_upsert.py) **must match** models hosted in the embedding service (app.py). At retrieval time, `qdrant_search.py` calls the Azure Embedding API (`EMBEDDING_API_BASE_URL`) — it **no longer loads models locally**.

| Model ID | Purpose | Dimension | Used in |
|---|---|---|---|
| `google/siglip-so400m-patch14-384` | Visual embeddings (images) & text→image search | 1152 | embedding.py, qdrant_upsert.py, embedding_service/app.py → consumed by qdrant_search.py via `/embed/visual` |
| `BAAI/bge-m3` | Multilingual semantic text embeddings | 1024 | qdrant_upsert.py, embedding_service/app.py → consumed by qdrant_search.py via `/embed/semantic` |
| `Qdrant/bm25` | Sparse lexical encoding | sparse | qdrant_upsert.py, embedding_service/app.py → consumed by qdrant_search.py via `/embed/sparse` |
| `openai/clip-vit-large-patch14` | Feature extraction for keyframe clustering | 768 | LMSKE.py (offline only, not indexed) |
| `microsoft/Florence-2-large-ft` | Rich captioning + object detection | — | object_detection.py |
| `yolov8m-worldv2` | Fast object detection | — | object_detection.py |

> **Important:** Changing a model in the embedding service without matching it in qdrant_upsert.py will produce mismatched vectors and silently break retrieval. The retrieval module (`qdrant_search.py`) is model-agnostic — it delegates all encoding to the Azure API.

## Current Limitations

1. **Heuristic retrieval is a mock** — `backend/src/services/heuristic_retrieve/service.py` returns dummy data. A team member is responsible for implementing the real logic.
2. **No metadata vector** — title/metadata search uses lexical fallback instead of semantic search; the Qdrant collection has no dedicated metadata vector
3. **No automated CI/CD** — no test runner, lint, or deployment automation
4. **Colab-first scripts** — several data-processing scripts auto-install dependencies via subprocess, which can cause issues in managed environments
5. **Detection JSON upload gap** — `azure_migrator.py` uploads keyframes, embeddings, and OCR, but detection JSONs are uploaded separately (not via `azure_migrator.py`). `qdrant_upsert.py` reads from an `object-detection` Azure container that must be populated through another mechanism.

## Change Impact Map

See [AGENTS.md](../AGENTS.md) for detailed per-subsystem change-impact checklists.

**Critical interfaces (changing these can silently break downstream):**

| Interface | Producer | Consumer(s) |
|---|---|---|
| Qdrant `keyframes_v1` schema | `qdrant_upsert.py` | `backend/src/services/agentic_retrieve/qdrant_search.py` |
| Azure Embedding API (`/embed/*`) | `embedding_service/app.py` | `backend/src/services/agentic_retrieve/qdrant_search.py` (via `EMBEDDING_API_BASE_URL`) |
| `POST /search` response schema | `backend/src/controllers/response_builder.py` | `streamlit/app.py` |
| Object detection JSON schema | `object_detection.py` | `qdrant_upsert.py` |
| Embedding `.npy` shape (N×1152) | `embedding.py` | `qdrant_upsert.py`, `azure_migrator.py` |
| `_frames.json` ordering | `embedding.py` | `qdrant_upsert.py` |
| Azure Blob container layout | `azure_migrator.py` | `qdrant_upsert.py` |
| `AgentState` TypedDict | `backend/src/services/agentic_retrieve/state.py` | All nodes in `nodes/`, `graph.py` |
