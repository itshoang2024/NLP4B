"""
embedder.py — Lazy singleton embedding models: SigLIP (visual) + BGE-M3 (semantic).

Both models are loaded once on first call and reused across requests.
Runs on CUDA if available, falls back to CPU automatically.
"""

import torch
from typing import Optional, Any, Tuple

from config import (
    SIGLIP_MODEL_ID, BGE_M3_MODEL_ID,
    SIGLIP_DIM, BGE_M3_DIM,
)

# ── Device ────────────────────────────────────────────────────────────────────

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── Module-level singletons (None until first call) ───────────────────────────

_siglip_tokenizer: Optional[Any] = None
_siglip_model:     Optional[Any] = None
_bge_m3_model:     Optional[Any] = None


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_siglip() -> Tuple[Any, Any]:
    """Load SigLIP tokenizer + model onto DEVICE (singleton)."""
    global _siglip_tokenizer, _siglip_model

    if _siglip_tokenizer is None or _siglip_model is None:
        from transformers import AutoTokenizer, AutoModel

        print(f"[embedder] Loading SigLIP ({SIGLIP_MODEL_ID}) on {DEVICE}...")
        _siglip_tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_ID)
        _siglip_model = (
            AutoModel.from_pretrained(SIGLIP_MODEL_ID)
            .to(DEVICE)
            .eval()
        )
        print(f"[embedder] SigLIP ready on {DEVICE}.")

    return _siglip_tokenizer, _siglip_model


def _load_bge_m3() -> Any:
    """Load BGE-M3 SentenceTransformer onto DEVICE (singleton)."""
    global _bge_m3_model

    if _bge_m3_model is None:
        from sentence_transformers import SentenceTransformer

        print(f"[embedder] Loading BGE-M3 ({BGE_M3_MODEL_ID}) on {DEVICE}...")
        _bge_m3_model = SentenceTransformer(BGE_M3_MODEL_ID, device=DEVICE)
        print(f"[embedder] BGE-M3 ready on {DEVICE}.")

    return _bge_m3_model


# ── Public encode functions ───────────────────────────────────────────────────

def encode_visual(query: str) -> list:
    """
    Encode a text query into a SigLIP dense vector (1152-d).
    Used for the 'keyframe-dense' vector field in Qdrant.
    """
    tokenizer, model = _load_siglip()

    inputs = tokenizer(
        [query],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

    # Normalise various output shapes
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

    if len(vec) != SIGLIP_DIM:
        raise ValueError(f"SigLIP produced dim={len(vec)}, expected {SIGLIP_DIM}")

    return vec.tolist()


def encode_semantic(query: str) -> list:
    """
    Encode a text query into a BGE-M3 dense vector (1024-d).
    Used for the 'keyframe-caption-dense' vector field in Qdrant.
    """
    model = _load_bge_m3()

    vec = model.encode(query, normalize_embeddings=True, device=DEVICE)

    if hasattr(vec, "tolist"):
        vec = vec.tolist()

    if len(vec) != BGE_M3_DIM:
        raise ValueError(f"BGE-M3 produced dim={len(vec)}, expected {BGE_M3_DIM}")

    return vec
