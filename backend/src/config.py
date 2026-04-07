"""
config.py — Unified environment configuration for the backend.

Loads .env from the backend root directory and exposes validated
configuration accessors used across all services.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# .env lives at backend/.env (same level as api.py)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH, override=False)


# ── Qdrant ────────────────────────────────────────────────────────────────────

def get_qdrant_url() -> str:
    url = os.getenv("QDRANT_URL", "").strip().rstrip(",")
    if not url:
        raise EnvironmentError("QDRANT_URL is not set in .env")
    return url


def get_qdrant_api_key() -> str:
    key = os.getenv("QDRANT_API_KEY", "").strip()
    if not key:
        raise EnvironmentError("QDRANT_API_KEY is not set in .env")
    return key


# ── Embedding API ─────────────────────────────────────────────────────────────

def get_embedding_api_url() -> str:
    url = os.getenv("EMBEDDING_API_BASE_URL", "").strip().rstrip("/")
    if not url:
        raise EnvironmentError("EMBEDDING_API_BASE_URL is not set in .env")
    return url


# ── Gemini / LLM ─────────────────────────────────────────────────────────────

def get_gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    key = key.strip()
    if not key:
        raise EnvironmentError(
            "Missing Gemini API key. Set GEMINI_API_KEY (preferred) or GOOGLE_API_KEY."
        )
    return key


# ── Constants ─────────────────────────────────────────────────────────────────

COLLECTION_NAME: str = "keyframes_v1"

SIGLIP_DIM: int = 1152
BGE_M3_DIM: int = 1024

DEFAULT_TOP_K: int = 10
DEVICE: str = "remote (Azure VM)"
