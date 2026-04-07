"""
config.py — Load environment variables for Qdrant connection.

The .env file lives at retrieval/.env (one level above this package).
We use python-dotenv to load it explicitly by path.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env is at retrieval/.env  →  two levels up from this file
# this file:         retrieval/heuristic_retrieval/config.py
# .env location:     retrieval/.env
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
else:
    raise FileNotFoundError(
        f".env not found at expected path: {_ENV_PATH}\n"
        "Please create retrieval/.env with QDRANT_URL and QDRANT_API_KEY."
    )


def get_qdrant_url() -> str:
    url = os.getenv("qdrant_url", "").strip().rstrip(",")  # strip trailing comma if any
    if not url:
        raise EnvironmentError("QDRANT_URL is not set in .env")
    return url


def get_qdrant_api_key() -> str:
    key = os.getenv("qdrant_api_key", "").strip()
    if not key:
        raise EnvironmentError("QDRANT_API_KEY is not set in .env")
    return key


def get_embedding_api_url() -> str:
    url = os.getenv("API_BASE_URL", "").strip().rstrip("/")
    if not url:
        raise EnvironmentError("API_BASE_URL is not set in .env")
    return url


# ── Constants ────────────────────────────────────────────────────────────────

COLLECTION_NAME: str = "keyframes_v1"

SIGLIP_DIM: int = 1152
BGE_M3_DIM: int  = 1024

DEFAULT_TOP_K: int        = 10
DEFAULT_PREFETCH: int     = 50

