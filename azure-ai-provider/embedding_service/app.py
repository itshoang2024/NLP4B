"""
app.py — Embedding-as-a-Service (CPU-Only Production Server)
==============================================================

Hosts 3 embedding models behind a FastAPI REST API:
  POST /embed/semantic → BGE-M3 dense 1024d
  POST /embed/sparse   → BM25 sparse (indices + values)
  POST /embed/visual   → SigLIP text dense 1152d

Models MUST match those used during Qdrant indexing (qdrant_upsert.py):
  - BAAI/bge-m3               → keyframe-caption-dense (1024d)
  - Qdrant/bm25               → keyframe-object-sparse / keyframe-ocr-sparse
  - google/siglip-so400m-patch14-384 → keyframe-dense (1152d)

Target: Azure VM Standard_B4as_v2 (4 vCPU, 16GB RAM, NO GPU)
"""

from __future__ import annotations

import time
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIG — Must match qdrant_upsert.py exactly
# ══════════════════════════════════════════════════════════════════════════════
SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"  # 1152d — keyframe-dense
BGE_M3_MODEL_ID = "BAAI/bge-m3"                       # 1024d — keyframe-caption-dense
BM25_MODEL_ID   = "Qdrant/bm25"                       # sparse — keyframe-object/ocr-sparse

DEVICE = "cpu"  # NO GPU on target VM

# ── Global model refs ─────────────────────────────────────────────────────────
siglip_tokenizer = None
siglip_model = None
bge_m3_model = None
bm25_model = None


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN — Load all models at startup, free at shutdown
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global siglip_tokenizer, siglip_model, bge_m3_model, bm25_model

    logger.info("═" * 60)
    logger.info("  🚀 Embedding Service — Loading Models (CPU)")
    logger.info("═" * 60)

    # ── 1. BGE-M3 (Semantic Dense — 1024d) ────────────────────────────
    t0 = time.time()
    logger.info(f"[1/3] Loading BGE-M3: {BGE_M3_MODEL_ID}...")
    from sentence_transformers import SentenceTransformer
    bge_m3_model = SentenceTransformer(BGE_M3_MODEL_ID, device=DEVICE)
    logger.info(f"  ✅ BGE-M3 ready ({time.time()-t0:.1f}s)")

    # ── 2. BM25 (Sparse) ─────────────────────────────────────────────
    t1 = time.time()
    logger.info(f"[2/3] Loading BM25: {BM25_MODEL_ID}...")
    from fastembed import SparseTextEmbedding
    bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_ID)
    logger.info(f"  ✅ BM25 ready ({time.time()-t1:.1f}s)")

    # ── 3. SigLIP Text Encoder (Visual Dense — 1152d) ────────────────
    t2 = time.time()
    logger.info(f"[3/3] Loading SigLIP: {SIGLIP_MODEL_ID}...")
    from transformers import AutoTokenizer, AutoModel
    siglip_tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_ID)
    siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_ID).to(DEVICE).eval()
    logger.info(f"  ✅ SigLIP ready ({time.time()-t2:.1f}s)")

    total = time.time() - t0
    logger.info("═" * 60)
    logger.info(f"  ✅ All models loaded in {total:.1f}s — Server ready!")
    logger.info("═" * 60)

    yield  # ← Server runs here

    # ── Cleanup ───────────────────────────────────────────────────────
    logger.info("Shutting down — releasing models...")
    del bge_m3_model, bm25_model, siglip_model, siglip_tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Embedding-as-a-Service",
    description=(
        "Hosts 3 embedding models (BGE-M3, BM25, SigLIP) for the "
        "Multimodal Agentic RAG system. CPU-only, no GPU required."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192, description="Input text to embed")

class DenseResponse(BaseModel):
    embedding: List[float]
    model: str
    dim: int
    latency_ms: float

class SparseResponse(BaseModel):
    indices: List[int]
    values: List[float]
    model: str
    nnz: int  # number of non-zero entries
    latency_ms: float


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": DEVICE,
        "models": {
            "semantic": BGE_M3_MODEL_ID,
            "sparse": BM25_MODEL_ID,
            "visual": SIGLIP_MODEL_ID,
        },
    }


# ── 1. Semantic Dense (BGE-M3 → 1024d) ───────────────────────────────────────

@app.post("/embed/semantic", response_model=DenseResponse)
async def embed_semantic(request: EmbedRequest):
    """
    Encode text → BGE-M3 1024d dense vector.
    Matches 'keyframe-caption-dense' in Qdrant.
    """
    if bge_m3_model is None:
        raise HTTPException(503, "BGE-M3 model not loaded")

    t0 = time.time()
    vec = bge_m3_model.encode(request.text, normalize_embeddings=True)
    latency = (time.time() - t0) * 1000

    embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)

    return DenseResponse(
        embedding=embedding,
        model=BGE_M3_MODEL_ID,
        dim=len(embedding),
        latency_ms=round(latency, 2),
    )


# ── 2. Sparse (BM25) ─────────────────────────────────────────────────────────

@app.post("/embed/sparse", response_model=SparseResponse)
async def embed_sparse(request: EmbedRequest):
    """
    Encode text → BM25 sparse vector (indices + values).
    Matches 'keyframe-object-sparse' and 'keyframe-ocr-sparse' in Qdrant.
    """
    if bm25_model is None:
        raise HTTPException(503, "BM25 model not loaded")

    t0 = time.time()
    results = list(bm25_model.embed([request.text]))
    latency = (time.time() - t0) * 1000

    if not results or len(results[0].indices) == 0:
        return SparseResponse(
            indices=[], values=[], model=BM25_MODEL_ID,
            nnz=0, latency_ms=round(latency, 2),
        )

    s = results[0]
    return SparseResponse(
        indices=s.indices.tolist(),
        values=s.values.tolist(),
        model=BM25_MODEL_ID,
        nnz=len(s.indices),
        latency_ms=round(latency, 2),
    )


# ── 3. Visual Dense (SigLIP → 1152d) ─────────────────────────────────────────

@app.post("/embed/visual", response_model=DenseResponse)
async def embed_visual(request: EmbedRequest):
    """
    Encode text → SigLIP 1152d dense vector.
    Matches 'keyframe-dense' in Qdrant (cross-modal text→image).
    """
    if siglip_tokenizer is None or siglip_model is None:
        raise HTTPException(503, "SigLIP model not loaded")

    t0 = time.time()

    inputs = siglip_tokenizer(
        [request.text],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = siglip_model.get_text_features(**inputs)

    # Robust tensor extraction (same logic as qdrant_upsert.py)
    if isinstance(outputs, torch.Tensor):
        features = outputs
    elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        features = outputs.text_embeds
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        features = outputs.pooler_output
    else:
        features = outputs[0] if isinstance(outputs, tuple) else outputs
        if features.ndim == 3:
            features = features.mean(dim=1)

    vec = features[0].cpu().numpy().flatten().astype("float32")
    latency = (time.time() - t0) * 1000

    embedding = vec.tolist()

    return DenseResponse(
        embedding=embedding,
        model=SIGLIP_MODEL_ID,
        dim=len(embedding),
        latency_ms=round(latency, 2),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
