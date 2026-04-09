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


# ── API Keys & Base URLs ────────────────────────────────────────

def get_embedding_api_url() -> str:
    url = os.getenv("EMBEDDING_API_BASE_URL", "").strip().rstrip("/")
    if not url:
        raise EnvironmentError("EMBEDDING_API_BASE_URL is not set in .env")
    return url

def get_azure_blob_base_url() -> str:
    # Not raising error if empty to avoid breaking local dev
    return os.getenv("AZURE_BLOB_BASE_URL", "").strip().rstrip("/")


# ── Gemini / LLM ─────────────────────────────────────────────────────────────

def get_gemini_api_key() -> str:
    """Return Gemini API key — kept for backward compatibility."""
    key = os.getenv("GEMINI_API_KEY")
    key = key.strip()
    if not key:
        raise EnvironmentError(
            "Missing Gemini API key. Set GEMINI_API_KEY."
        )
    return key


# ── LLM Provider (abstraction layer) ─────────────────────────────────────────

def get_llm_backend() -> str:
    """Return the active LLM backend name (default: ``gemini``)."""
    return os.getenv("LLM_BACKEND", "gemini").strip().lower()


def get_llm_base_url() -> str:
    """Return the base URL for OpenAI-compatible LLM servers."""
    return os.getenv("LLM_BASE_URL", "http://localhost:8080").strip().rstrip("/")


def get_llm_api_key() -> str | None:
    """Return the LLM API key, or None if not set."""
    val = os.getenv("LLM_API_KEY", "").strip()
    return val or None


def get_llm_model_name() -> str | None:
    """Return the LLM model name override, or None for provider default."""
    val = os.getenv("LLM_MODEL_NAME", "").strip()
    return val or None


# ── Constants ─────────────────────────────────────────────────────────────────

COLLECTION_NAME: str = "keyframes_v1"

SIGLIP_DIM: int = 1152
BGE_M3_DIM: int = 1024

DEFAULT_TOP_K: int = 20
DEVICE: str = "remote"
