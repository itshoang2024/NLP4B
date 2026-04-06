"""
api_gateway/main.py — API Gateway (Phase 1)
==============================================

Lightweight FastAPI router that proxies requests to internal services.
All external clients connect through this gateway on port 8080.

Internal routing:
  POST /v1/embeddings  → embedding_service:8001/embed
  POST /v1/chat/completions → llm_service:8000/v1/chat/completions (Phase 2)
  GET  /health → self health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx
import os

app = FastAPI(
    title="API Gateway — Azure AI Provider",
    description="Unified entry point for all AI services",
    version="0.1.0",
)

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_URL = os.environ.get("EMBEDDING_SERVICE_URL", "http://embedding_service:8001")
LLM_URL = os.environ.get("LLM_SERVICE_URL", "http://llm_service:8000")
TIMEOUT = int(os.environ.get("GATEWAY_TIMEOUT", 5))


# ── Request/Response Models ───────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    input: str
    model: str = "mock"

class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int = 0

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    status: str
    source_service: str = "embedding_service"


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Gateway self-check + downstream service status."""
    services = {}

    # Check embedding service
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(f"{EMBEDDING_URL}/health")
            services["embedding_service"] = resp.json() if resp.status_code == 200 else "unhealthy"
    except Exception as e:
        services["embedding_service"] = f"unreachable: {str(e)}"

    # Check LLM service
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(f"{LLM_URL}/health")
            services["llm_service"] = resp.json() if resp.status_code == 200 else "unhealthy"
    except Exception:
        services["llm_service"] = "mock (sleep mode)"

    return {
        "status": "healthy",
        "service": "api_gateway",
        "downstream": services,
    }


# ── Embedding Proxy ──────────────────────────────────────────────────────────

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """
    OpenAI-compatible embedding endpoint.
    Proxies to internal embedding_service with timeout protection.
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{EMBEDDING_URL}/embed",
                json={"text": request.input, "model": request.model},
            )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Embedding service returned {resp.status_code}: {resp.text}",
            )

        data = resp.json()
        return EmbeddingResponse(
            data=[EmbeddingData(embedding=data["embedding"])],
            model=data.get("model", request.model),
            status=data.get("status", "ok"),
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Embedding service timeout ({TIMEOUT}s). Service may be overloaded.",
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot reach embedding_service. Is the container running?",
        )


# ── LLM Proxy (Phase 2) ─────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions():
    """Phase 2: Will proxy to vLLM service. Currently returns mock."""
    return {
        "status": "mock",
        "message": "LLM service is in sleep mode (Phase 1). No model loaded.",
        "hint": "Uncomment vLLM command in docker-compose.yml for Phase 2.",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("GATEWAY_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
