# NLP4B - Multimodal Video Retrieval System

End-to-end pipeline for ingesting, indexing, and retrieving video content using multimodal AI. Built for the **CSC15012 - Applications of NLP in Industry** course at HCMUS.

## What This Project Does

Given a natural-language query (Vietnamese or English), the system retrieves the most relevant video keyframes from an indexed collection. The pipeline covers:

1. **Video ingestion** - download YouTube videos and extract technical metadata
2. **Keyframe extraction** - shot segmentation (TransNetV2) + adaptive clustering (CLIP + K-Means)
3. **Embedding generation** - encode keyframes with SigLIP (1152d visual embeddings)
4. **Visual annotation** - object detection (YOLO + Florence-2) and OCR
5. **Indexing** - upsert 4-vector schema into Qdrant Cloud
6. **Azure migration** - upload keyframes, embeddings, and OCR artifacts to Azure Blob Storage
7. **Unified retrieval API** - FastAPI backend combining agentic (LangGraph) and heuristic (RRF) retrieval with cross-source reranking
8. **Demo UI** - Streamlit app for interactive search

## Repository Structure

```
NLP4B/
├── data-processing/          # Offline artifact generation pipeline
│   ├── src/
│   │   ├── download/         # YouTube download + ffprobe metadata
│   │   ├── keyframe_extraction/  # TransNetV2 + CLIP clustering (LMSKE)
│   │   ├── embedding/        # SigLIP keyframe embeddings
│   │   ├── object_detection/ # YOLO + Florence-2 hybrid detection
│   │   ├── ocr/              # CRAFT + PaddleOCR text recognition
│   │   ├── qdrant/           # Qdrant upsert (4-vector schema)
│   │   ├── faiss/            # FAISS index builder
│   │   ├── scripts/          # Auxiliary scripts
│   │   └── azure_migrator.py # Upload artifacts to Azure Blob
│   ├── notebook/             # Step-by-step processing notebooks (01-04)
│   ├── output/               # Generated manifests and metadata
│   └── templates/            # Input Excel templates
│
├── backend/                  # ⭐ Unified retrieval API (FastAPI)
│   ├── api.py                # Entry point — /health + /search endpoints
│   ├── src/
│   │   ├── middlewares/      # Query cleaning + translation
│   │   ├── controllers/     # Orchestration, cross-source RRF rerank
│   │   └── services/
│   │       ├── agentic_retrieve/   # LangGraph intent-aware pipeline
│   │       └── heuristic_retrieve/ # 2-tier Qdrant fallback search + True RRF
│   └── test/                 # Standalone demo scripts
│
├── retrieval/                # ⚠️ Legacy — migrated to backend/
│   └── agentic_retrieval/    # Original standalone agentic retrieval
│
├── azure-ai-provider/        # Embedding-as-a-Service (FastAPI + Docker)
│   ├── embedding_service/    # FastAPI app with BGE-M3, BM25, SigLIP
│   ├── docker-compose.yml
│   └── DEPLOY.md             # Azure VM deployment guide
│
├── streamlit/                # Demo UI layer — calls backend API
│   ├── app.py                # Streamlit entry point ("LookUp.ai")
│   ├── components/           # Reusable UI components
│   └── assets/styles/        # CSS styling
│
├── docs/                     # Architecture, contracts, and reference docs
├── AGENTS.md                 # Coding-agent safety guide
└── README.md                 # ← You are here
```

## Quick Start

### Prerequisites

- Python 3.11+ (3.12 also works)
- FFmpeg (for video processing and ffprobe metadata)
- Git

### Setup

```bash
# Clone
git clone https://github.com/CallmeAndree/NLP4B.git
cd NLP4B

# Install dependencies for the module you need:
pip install -r data-processing/requirements.txt      # Offline pipeline
pip install -r backend/requirements.txt               # Backend API
```

### Run the Data Processing Pipeline

See [data-processing/README.md](data-processing/README.md) for step-by-step instructions.

```bash
cd data-processing

# Step 1: Download videos
python -m src.download.main \
    --input-excel templates/link_videos_template.xlsx \
    --output-root ./output

# Step 2: Extract keyframes
python src/keyframe_extraction/LMSKE.py \
    --video ./output/videos/<video_id>.mp4 \
    --output_dir ./output/keyframes

# Step 3: Generate embeddings
python src/embedding/embedding.py \
    --input_dir ./output/keyframes/<video_id> \
    --output_dir ./output/embeddings

# Step 4: Object detection
python src/object_detection/object_detection.py \
    -i ./output/keyframes/<video_id> \
    -o ./output/detections
```

### Run the Backend API

See [backend/README.md](backend/README.md) for full details.

```bash
cd backend
cp .env.example .env    # Fill in QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY, EMBEDDING_API_BASE_URL
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Run the Standalone Agentic Demo

```bash
cd backend
python test/run_agentic_demo.py --query "a person in red shirt cooking"
```

> **Note:** The original standalone agentic pipeline in `retrieval/agentic_retrieval/` has been migrated to `backend/src/services/agentic_retrieve/`. See [retrieval/agentic_retrieval/README.md](retrieval/agentic_retrieval/README.md) for legacy reference.

### Run the Streamlit App

```bash
cd streamlit
streamlit run app.py
# Requires the backend API to be running on localhost:8000
```

## Key Documentation

| Document | Purpose |
|---|---|
| [AGENTS.md](AGENTS.md) | Coding-agent safety rules and module map |
| [docs/architecture.md](docs/architecture.md) | System architecture, pipeline flow, and module boundaries |
| [backend/README.md](backend/README.md) | Backend API: endpoints, request flow, Qdrant contract, tech debt |
| [docs/rerank-design-rationale.md](docs/rerank-design-rationale.md) | Multi-signal rerank formula design rationale |
| [docs/contracts/qdrant-collection-schema.md](docs/contracts/qdrant-collection-schema.md) | Qdrant `keyframes_v1` collection schema contract |
| [docs/contracts/object-detection-output.md](docs/contracts/object-detection-output.md) | Object detection JSON output schema contract |
| [data-processing/README.md](data-processing/README.md) | Offline pipeline: submodules, CLI usage, artifacts |
| [retrieval/agentic_retrieval/README.md](retrieval/agentic_retrieval/README.md) | Legacy agentic retrieval reference (migrated to backend/) |
| [azure-ai-provider/DEPLOY.md](azure-ai-provider/DEPLOY.md) | Embedding service Azure VM deployment |

## Environment Variables

Different modules require different environment variables:

| Variable | Used by | Description |
|---|---|---|
| `QDRANT_URL` | backend, data-processing | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | backend, data-processing | Qdrant Cloud API key |
| `GEMINI_API_KEY` | backend | Google Gemini API key for LLM service |
| `EMBEDDING_API_BASE_URL` | backend | Azure Embedding API base URL (e.g. `http://<VM_IP>:8000`) |
| `AZURE_STORAGE_CONNECTION_STRING` | data-processing | Azure Blob Storage connection string |

## License

This project is part of coursework for CSC15012 at HCMUS.
