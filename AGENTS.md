# AGENTS.md

This file tells coding agents how to work safely in this repository.

## Purpose
This repository is an end-to-end multimodal video retrieval system with three major concerns:
- offline data processing for video ingestion and artifact generation
- retrieval-time orchestration and ranking
- demo / application surfaces for human interaction

The codebase appears to support a pipeline like:
1. ingest video URLs and metadata
2. download videos and extract technical metadata
3. extract semantic keyframes from videos
4. generate embeddings and visual annotations for keyframes
5. index or migrate generated artifacts to serving/storage systems
6. perform retrieval through an agentic pipeline
7. expose results through a Streamlit app or related demo surfaces

## Read these first
Before making non-trivial changes, inspect these locations first:
- root `README.md` if present
- `data-processing/src/download/main.py`
- `data-processing/src/embedding/embedding.py`
- `data-processing/src/keyframe_extraction/LMSKE.py`
- `data-processing/src/object_detection/object_detection.py`
- `data-processing/src/azure_migrator.py`
- `backend/api.py`
- `backend/src/controllers/search_controller.py`
- `backend/src/services/agentic_retrieve/graph.py`
- `backend/src/services/agentic_retrieve/nodes/`
- `streamlit/app.py`
- `requirements.txt` files in each major module

## Major module map

### `azure-ai-provider/`
Azure-facing service layer. The `embedding_service/` submodule appears to host an app and Dockerfile for embedding-related service deployment.

Treat this as a runtime service boundary. Avoid changing request/response behavior casually without checking all consumers.

### `data-processing/`
Offline preprocessing pipeline for creating artifacts consumed downstream.

Important subareas:
- `src/download/`: manifest loading, YouTube download, ffprobe metadata extraction
- `src/keyframe_extraction/`: shot segmentation, feature extraction, clustering, redundancy elimination, keyframe save logic
- `src/embedding/`: keyframe embedding generation
- `src/object_detection/`: object detection + captioning / region annotation pipeline
- `src/ocr/`: text detection (CRAFT) and recognition (PaddleOCR) from keyframe images
- `src/qdrant/` and `src/faiss/`: vector indexing / search support
- `src/scripts/`: auxiliary scripts
- `notebook/`: experimentation and staged processing notebooks
- `output/`: generated manifests, logs, metadata snapshots
- `templates/`: input templates such as Excel manifest skeletons

This area is the main artifact producer.

### `backend/`
Unified retrieval API (FastAPI). This is the **active** online serving module.

Important parts:
- `api.py`: FastAPI entry point (port 8000), `/health` and `/search` endpoints
- `src/middlewares/search_middleware.py`: query cleaning, language detection, translation
- `src/controllers/search_controller.py`: orchestrates agentic + heuristic services, cross-source RRF rerank
- `src/controllers/rerank.py`: Reciprocal Rank Fusion merging logic
- `src/controllers/response_builder.py`: flattens raw_payload → SearchResultItem
- `src/services/agentic_retrieve/`: LangGraph pipeline (5 nodes, no normalization node)
- `src/services/heuristic_retrieve/`: mock stub (WIP — to be implemented)
- `src/services/translator.py`: shared language detection + Gemini translation
- `src/config.py`: unified env loading
- `src/schemas.py`: shared Pydantic models + TypedDicts
- `test/run_agentic_demo.py`: standalone CLI demo

This area is the online decision-making path and the primary serving layer.

### `retrieval/agentic_retrieval/` (legacy)

> ⚠️ **Migrated to `backend/src/services/agentic_retrieve/`.** Kept for reference only.

Original standalone LangGraph pipeline with 6 nodes (including normalization).

### `streamlit/`
User-facing app / demo layer.

Important parts:
- `app.py`: top-level app entry
- `assets/styles/`: UI styling
- `components/`: reusable app components
- `.streamlit/config.toml`: Streamlit configuration

Assume this layer consumes upstream outputs rather than producing canonical artifacts.

## Key artifact map
Do not rename or silently change these artifact conventions without updating docs and downstream readers.

### Download + metadata artifacts
Produced by `data-processing/src/download/main.py`:
- `normalized_manifest.csv`
- `download_log.csv`
- `download_archive.txt`
- `video_metadata.csv`
- `metadata_log.csv`

### Keyframe artifacts
Produced by keyframe extraction flow in `LMSKE.py` and related save utilities:
- `<output_root>/<video_id>/...jpg`
- shot cache / scene text files such as `<video_id>_scenes.txt`
- intermediate feature cache files under temp or output locations

### Embedding artifacts
Produced by `data-processing/src/embedding/embedding.py`:
- `<video_id>.npy`
- `<video_id>_frames.json`

Interpretation:
- `.npy` stores stacked embeddings for the discovered keyframes
- `_frames.json` maps embedding row order to original frame indices

### Object detection artifacts
Produced by `data-processing/src/object_detection/object_detection.py`:
- `<video_id>_object_detection.json`

This JSON should be treated as a schema contract for downstream retrieval or UI consumers.

### Azure migration layout
Produced / consumed by `azure_migrator.py`:
- Azure container `keyframes`: `{video_id}/{filename.jpg}`
- Azure container `embeddings`: `{video_id}/{filename}`
- Azure container `ocr`: `{video_id}/{video_id}_ocr.json`

### OCR artifacts
Produced by `data-processing/src/ocr/` pipeline (CRAFT + PaddleOCR):
- `<video_id>_ocr.json` — array of `{"image": "<filename>", "ocr_text": "<text>"}` entries

Consumed by `qdrant_upsert.py` via `build_ocr_lookup()` to populate the `ocr_text` payload field and `keyframe-ocr-sparse` vector.

### Retrieval outputs
The exact retrieval response format should be confirmed from `graph.py`, node logic, and serving layer before changing any interface.

## Golden workflows

### Workflow A: video ingestion and metadata extraction
Entry point:
- `python -m src.download.main ...` run from `data-processing/`

Expected result:
- normalized manifest
- download logs
- ffprobe-derived metadata CSV/log outputs

### Workflow B: keyframe extraction
Entry point:
- `LMSKE.py`

Expected result:
- per-video keyframe image folder
- cached shot boundaries
- cached feature files as applicable

### Workflow C: embedding generation
Entry point:
- `embedding.py --input_dir ... --output_dir ...`

Expected result:
- one `.npy` per video folder
- one frame-index mapping JSON per video folder

### Workflow D: visual annotation / object detection
Entry point:
- `object_detection.py -i <keyframe_dir> -o <output_dir>`

Expected result:
- one combined JSON per video folder

### Workflow E: retrieval-time execution (backend API)
Entry points:
- `backend/api.py` — FastAPI server (`uvicorn api:app --port 8000`)
- `backend/test/run_agentic_demo.py` — standalone agentic demo (CLI)

Expected result:
- `/health` returns `{"status": "ok", "collection": "keyframes_v1"}`
- `/search` returns `SearchResponse` with ranked results from both branches

## Safe change rules

### 1. Preserve artifact contracts
Do not change:
- output filenames
- JSON top-level keys
- CSV column names
- folder layout conventions
without checking all known downstream readers.

### 2. Treat CLI signatures as public interfaces
If you change CLI arguments in scripts such as:
- `main.py`
- `LMSKE.py`
- `embedding.py`
- `object_detection.py`
- `azure_migrator.py`
then update:
- usage docs
- module README files
- example commands
- tests or smoke tests if present

### 3. Separate offline vs online concerns
Avoid mixing preprocessing assumptions into retrieval-time code unless explicitly intended.

### 4. Avoid silent schema drift
If you add fields to produced JSON/CSV files, document:
- field meaning
- whether downstream readers require them
- whether older outputs remain supported

### 5. Be cautious with environment assumptions
Some scripts are written to be Colab-friendly and may auto-install dependencies or assume a specific working directory. If you refactor these behaviors, keep local-dev, notebook, and CI ergonomics in mind.

## What to inspect after changing each subsystem

### If you modify `src/download/`
Check:
- manifest loading assumptions
- generated CSV column names
- downstream metadata readers
- any notebook examples that depend on the same outputs

### If you modify keyframe extraction
Check:
- keyframe folder naming
- frame filename pattern
- frame index semantics
- embedding step compatibility
- any Azure upload logic expecting the same folder layout

### If you modify embedding generation
Check:
- `.npy` shape expectations
- `_frames.json` ordering
- downstream retrieval indexers or migrators
- storage sync logic

### If you modify object detection output
Check:
- JSON schema stability
- field names under metadata / descriptions / objects
- any retrieval or UI components reading detection output

### If you modify retrieval graph, nodes, or services
Check:
- state schema compatibility (`backend/src/services/agentic_retrieve/state.py`)
- service adapter expectations
- prompt / routing assumptions (normalization is now in middleware, not a node)
- rerank / fusion ordering semantics
- demo script: `backend/test/run_agentic_demo.py`

### If you modify backend middleware, controller, or route
Check:
- `query_bundle` dict keys expected by both service branches
- `SearchResponse` / `SearchResultItem` schema consumed by `streamlit/app.py`
- cross-source RRF rerank logic in `controllers/rerank.py`
- response_builder payload field extraction
- env vars in `src/config.py`

### If you modify Streamlit app code
Check:
- upstream data loading assumptions
- local asset paths
- any module-specific config in `.streamlit/config.toml`

### If you modify OCR code (`src/ocr/`)
Check:
- OCR JSON schema: `{"image": str, "ocr_text": str}` array format
- `build_ocr_lookup()` in `qdrant_upsert.py` which parses OCR JSON
- `keyframe-ocr-sparse` vector encoding in Qdrant
- `ocr_text` payload field expectations in `qdrant_search.py`
- Azure `ocr` container blob naming convention: `{video_id}/{video_id}_ocr.json`

## Documentation expectations for future changes
When making meaningful code changes, also update the most relevant docs:
- root `README.md` for top-level workflows
- `docs/architecture.md` for system flow changes
- module README files for invocation or artifact changes
- `docs/contracts/*.md` when schema or IO contracts change
- runbooks when setup or troubleshooting steps change

## Questions to ask before major changes
Ask these before large refactors or interface changes:
1. Is this module's output consumed by a downstream step, notebook, or demo that assumes the current file or schema format?
2. Am I changing a CLI, output filename, or JSON/CSV field that should be treated as public?
3. Is the current behavior designed for Colab / Kaggle / local dev compatibility that I might accidentally break?
4. Does this change belong in offline preprocessing, retrieval-time logic, or the app layer?
5. Do I need a contract doc or migration note because older artifacts may still exist?

## Assumptions captured in this file
These points should be revalidated against code if you are about to depend on them heavily:
- the repository supports an end-to-end multimodal video retrieval workflow
- `data-processing/` is the canonical artifact-generation area
- `backend/` is the active online serving layer (unified FastAPI API)
- `retrieval/agentic_retrieval/` is legacy — migrated to `backend/src/services/agentic_retrieve/`
- `streamlit/` is the presentation / demo layer — calls `backend/` API
- some scripts were optimized for notebook-first environments before being hardened for repository-scale development
