# System Architecture

## Overview

NLP4B is a multimodal video retrieval system. Videos are preprocessed offline into searchable artifacts (keyframes, embeddings, annotations), indexed in Qdrant Cloud, and queried at runtime through an agentic retrieval pipeline.

The system has three runtime boundaries:

| Boundary | Module | Runs when |
|---|---|---|
| **Offline processing** | `data-processing/` | During data ingestion (Colab/Kaggle/local) |
| **Online retrieval** | `retrieval/agentic_retrieval/` | At query time |
| **Presentation** | `streamlit/` | User-facing demo |
| **Service** | `azure-ai-provider/` | Always-on embedding API on Azure VM |

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
│                     (retrieval/agentic_retrieval/)                   │
│                                                                     │
│  User query                                                        │
│       │                                                             │
│       ▼                                                             │
│  ┌────────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────────┐│
│  │   Query    │─▶│  Intent   │─▶│ Modality │─▶│    Parallel      ││
│  │Normalization│  │Extraction │  │ Routing  │  │   Retrieval      ││
│  │            │  │  (Gemini) │  │          │  │ (Qdrant 4-vec)   ││
│  └────────────┘  └───────────┘  └──────────┘  └────────┬─────────┘│
│                                                         │          │
│                                              ┌──────────▼────────┐ │
│                                              │  Candidate Fusion │ │
│                                              │  (RRF merge)     │ │
│                                              └────────┬─────────┘ │
│                                                       │            │
│                                              ┌────────▼─────────┐ │
│                                              │ Frame Reranking  │ │
│                                              │ → agent_topk     │ │
│                                              └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                             │
                       agent_topk[]
                             │
┌─────────────────────────────────────────────────────────────────────┐
│                     PRESENTATION                                    │
│                     (streamlit/)                                     │
│                                                                     │
│  Streamlit app.py → LookUp.ai UI                                   │
│  (currently scaffold; consumes retrieval outputs)                  │
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

### `retrieval/agentic_retrieval/`

**Responsibility:** Query-time orchestration — normalize, understand, route, retrieve, fuse, rerank.

**Not responsible for:** Artifact generation, model training, UI rendering.

**Graph nodes** (executed sequentially via LangGraph):

| Node | File | Function |
|---|---|---|
| `query_normalization` | `nodes/normalization.py` | Clean, detect language, translate to English |
| `query_intent_extraction` | `nodes/intent_extraction.py` | LLM-based structured intent (objects, actions, scene, text cues, etc.) |
| `modality_routing` | `nodes/routing.py` | Compute per-source retrieval weights |
| `parallel_retrieval` | `nodes/retrieval.py` | Query Qdrant across 5 sources: keyframe, caption, object, OCR, metadata |
| `candidate_fusion` | `nodes/fusion.py` | RRF-based merge across sources |
| `frame_reranking` | `nodes/rerank.py` | Final scoring and truncation to `agent_topk` |

**Services:**

| Service | File | Purpose |
|---|---|---|
| `LLMService` | `services/llm_service.py` | Gemini API wrapper for intent extraction |
| `QdrantSearchService` | `services/qdrant_search.py` | Multi-vector Qdrant client with model encoders |
| `translator` | `services/translator.py` | Language translation utility |
| `scoring` | `services/scoring.py` | Score normalization |

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

**Current state:** Scaffold — shows search bar with loading animation. Does not yet integrate with retrieval pipeline programmatically.

## Qdrant Collection Schema

The shared interface between offline processing and online retrieval is the Qdrant `keyframes_v1` collection. See [docs/contracts/qdrant-collection-schema.md](contracts/qdrant-collection-schema.md) for the full schema contract.

**4-vector architecture:** `keyframe-dense` (SigLIP 1152d), `keyframe-caption-dense` (BGE-M3 1024d), `keyframe-object-sparse` (BM25), `keyframe-ocr-sparse` (BM25). See the [full schema contract](contracts/qdrant-collection-schema.md) for vector population rules, payload fields, and compatibility risks.

## Model Alignment

Models used in indexing (qdrant_upsert.py) **must match** models used in retrieval (qdrant_search.py) and the embedding service (app.py):

| Model ID | Purpose | Dimension | Used in |
|---|---|---|---|
| `google/siglip-so400m-patch14-384` | Visual embeddings (images) & text→image search | 1152 | embedding.py, qdrant_upsert.py, qdrant_search.py, embedding_service/app.py |
| `BAAI/bge-m3` | Multilingual semantic text embeddings | 1024 | qdrant_upsert.py, qdrant_search.py, embedding_service/app.py |
| `Qdrant/bm25` | Sparse lexical encoding | sparse | qdrant_upsert.py, embedding_service/app.py |
| `openai/clip-vit-large-patch14` | Feature extraction for keyframe clustering | 768 | LMSKE.py (offline only, not indexed) |
| `microsoft/Florence-2-large-ft` | Rich captioning + object detection | — | object_detection.py |
| `yolov8m-worldv2` | Fast object detection | — | object_detection.py |

> **Important:** Changing a model in one place without matching it in all others will produce mismatched vectors and silently break retrieval.

## Current Limitations

1. **Streamlit UI is a scaffold** — does not yet call the retrieval pipeline
2. **No metadata vector** — title/metadata search uses lexical fallback instead of semantic search; the Qdrant collection has no dedicated metadata vector
3. **No automated CI/CD** — no test runner, lint, or deployment automation
4. **Colab-first scripts** — several scripts auto-install dependencies via subprocess, which can cause issues in managed environments
5. **Detection JSON upload gap** — `azure_migrator.py` uploads keyframes, embeddings, and OCR, but detection JSONs are uploaded separately (not via `azure_migrator.py`). `qdrant_upsert.py` reads from an `object-detection` Azure container that must be populated through another mechanism.

## Change Impact Map

See [AGENTS.md](../AGENTS.md) for detailed per-subsystem change-impact checklists.

**Critical interfaces (changing these can silently break downstream):**

| Interface | Producer | Consumer(s) |
|---|---|---|
| Qdrant `keyframes_v1` schema | `qdrant_upsert.py` | `qdrant_search.py`, `run_agentic_demo.py` |
| Object detection JSON schema | `object_detection.py` | `qdrant_upsert.py` |
| Embedding `.npy` shape (N×1152) | `embedding.py` | `qdrant_upsert.py`, `azure_migrator.py` |
| `_frames.json` ordering | `embedding.py` | `qdrant_upsert.py` |
| Azure Blob container layout | `azure_migrator.py` | `qdrant_upsert.py` |
| `AgentState` TypedDict | `state.py` | All nodes in `nodes/`, `graph.py` |
