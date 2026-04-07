"""
embedder.py — Remote embedding via Azure-deployed Embedding-as-a-Service.

Instead of loading models locally, all encoding is delegated to the
remote FastAPI server at API_BASE_URL (see retrieval/.env).

Endpoints used:
  POST /embed/visual   → SigLIP 1152d  (for 'keyframe-dense' in Qdrant)
  POST /embed/semantic → BGE-M3 1024d  (for 'keyframe-caption-dense' in Qdrant)

Contract reference: azure-ai-provider/API_CONTRACT.md
"""

import httpx
from config import get_embedding_api_url, SIGLIP_DIM, BGE_M3_DIM

# ── Used by api.py for the /health endpoint ───────────────────────────────────
DEVICE = "remote (Azure VM)"

# Connection timeout per request (seconds).
# Azure VM P95 latency: semantic ~1.2s, visual ~0.72s
_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)


def _get_base_url() -> str:
    """Return the embedding service base URL from .env, cached per call."""
    return get_embedding_api_url()


# ── Public encode functions ───────────────────────────────────────────────────

def encode_visual(query: str) -> list:
    """
    Encode a text query into a SigLIP dense vector (1152-d) via remote API.
    Maps to the 'keyframe-dense' vector field in Qdrant.

    Calls: POST {API_BASE_URL}/embed/visual
    Body:  {"text": query}
    Returns: List[float] of length 1152
    """
    url = f"{_get_base_url()}/embed/visual"

    try:
        response = httpx.post(url, json={"text": query}, timeout=_TIMEOUT)
        response.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot connect to embedding service at {_get_base_url()}. "
            "Check that the Azure VM is running and API_BASE_URL is correct in .env."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Embedding service timed out on /embed/visual (query: '{query[:50]}...')"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Embedding service /embed/visual returned {e.response.status_code}: "
            f"{e.response.text[:200]}"
        )

    data = response.json()
    vec: list = data["embedding"]

    if len(vec) != SIGLIP_DIM:
        raise ValueError(
            f"SigLIP remote returned dim={len(vec)}, expected {SIGLIP_DIM}. "
            "Check the Azure VM model version."
        )

    return vec


def encode_semantic(query: str) -> list:
    """
    Encode a text query into a BGE-M3 dense vector (1024-d) via remote API.
    Maps to the 'keyframe-caption-dense' vector field in Qdrant.

    Calls: POST {API_BASE_URL}/embed/semantic
    Body:  {"text": query}
    Returns: List[float] of length 1024
    """
    url = f"{_get_base_url()}/embed/semantic"

    try:
        response = httpx.post(url, json={"text": query}, timeout=_TIMEOUT)
        response.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot connect to embedding service at {_get_base_url()}. "
            "Check that the Azure VM is running and API_BASE_URL is correct in .env."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Embedding service timed out on /embed/semantic (query: '{query[:50]}...')"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Embedding service /embed/semantic returned {e.response.status_code}: "
            f"{e.response.text[:200]}"
        )

    data = response.json()
    vec: list = data["embedding"]

    if len(vec) != BGE_M3_DIM:
        raise ValueError(
            f"BGE-M3 remote returned dim={len(vec)}, expected {BGE_M3_DIM}. "
            "Check the Azure VM model version."
        )

    return vec
