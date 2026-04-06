"""
embedding_service/server.py — Mock Embedding Service (Phase 1)
================================================================

Phase 1: Returns dummy embedding vectors for connectivity testing.
Phase 2: Will load real models (SigLIP, BGE-M3, BM25) on CPU.

Endpoints:
  POST /embed  — Accept text, return dummy vector + status
  GET  /health — Health check
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os

app = FastAPI(
    title="Embedding Service",
    description="Mock embedding service for Phase 1 connectivity testing",
    version="0.1.0-mock",
)

# ── Request/Response Models ───────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    text: str
    model: str = "mock"  # Phase 2: "siglip", "bge-m3", "bm25"

class EmbedResponse(BaseModel):
    embedding: List[float]
    model: str
    dim: int
    status: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "embedding_service",
        "mode": "mock",
        "gpu": False,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    """
    Phase 1 Mock: Returns a fixed dummy vector [0.1, 0.2, 0.3].
    Phase 2: Will route to real model based on request.model.
    """
    # Mock response — proof that the service is reachable
    dummy_vector = [0.1, 0.2, 0.3]

    return EmbedResponse(
        embedding=dummy_vector,
        model=request.model,
        dim=len(dummy_vector),
        status="CPU Worker OK",
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("EMBEDDING_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
