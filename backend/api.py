"""
api.py — FastAPI application entry point for LookUp.ai backend.
Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
from pathlib import Path

# ── Ensure backend/ is on sys.path for `from src.xxx` imports ─────────────────
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ── Fix Windows console encoding ─────────────────────────────────────────────
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Load .env before any config access ────────────────────────────────────────
from src.config import COLLECTION_NAME, DEVICE  # noqa: E402 — triggers dotenv load
from src.schemas import HealthResponse  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from src.routes.search_route import router as search_router  # noqa: E402


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LookUp.ai — Unified Retrieval API",
    description=(
        "Combines agentic (intent-aware multimodal) and heuristic "
        "(dense hybrid RRF) retrieval with cross-source reranking."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Quick health check. Does NOT load models — stays fast."""
    return HealthResponse(
        status="ok",
        collection=COLLECTION_NAME,
        device=DEVICE,
    )


app.include_router(search_router, tags=["Retrieval"])


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )