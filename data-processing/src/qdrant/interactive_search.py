"""
interactive_search.py — Visual Search Tool for Qdrant (Colab)
==============================================================

Cross-modal text→image retrieval using the SAME SigLIP model
(google/siglip-so400m-patch14-384) used during keyframe embedding.

Usage in Colab:
  1. Set env vars: QDRANT_URL, QDRANT_API_KEY
  2. Run this cell
  3. Type a search query → see visual gallery of matching keyframes
"""

# ── 0. Install deps ──────────────────────────────────────────────────────────
import subprocess, sys

def _pip(*p):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *p])

try:
    from qdrant_client import QdrantClient
except ImportError:
    _pip("qdrant-client")
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    _pip("transformers", "torch", "torchvision")

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os
import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import NamedVector
from transformers import AutoTokenizer, AutoModel
from IPython.display import display, HTML, Image as IPImage, clear_output

# ── 2. Config & CLI Arguments ───────────────────────────────────────────────────
import argparse

parser = argparse.ArgumentParser(description="Qdrant Visual Search")
parser.add_argument("--top_k", type=int, default=20, help="Number of results to retrieve")
parser.add_argument("--threshold", type=float, default=0.15, help="Minimum score threshold")
# use parse_known_args to handle `%run` inside Jupyter/Colab without throwing errors on unknown flags
args, _ = parser.parse_known_args()

QDRANT_URL      = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY  = os.environ.get("QDRANT_API_KEY", "")
COLLECTION      = "keyframes_v1"
DENSE_NAME      = "keyframe-dense"

TOP_K            = args.top_k
SCORE_THRESHOLD  = args.threshold

SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 3. Load model (one-time) ─────────────────────────────────────────────────
print(f"🔄 Loading SigLIP model on [{DEVICE}] ...")
tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_ID)
model = AutoModel.from_pretrained(SIGLIP_MODEL_ID).to(DEVICE).eval()
print("✅ SigLIP loaded.")

# ── 4. Connect to Qdrant ─────────────────────────────────────────────────────
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
info = client.get_collection(COLLECTION)
print(f"✅ Connected to Qdrant | Collection '{COLLECTION}' | {info.points_count} points\n")


# ── 5. Text → 1152d vector ───────────────────────────────────────────────────
def encode_text(query: str) -> list[float]:
    """
    Encode text query using the SAME SigLIP model as embedding.py.
    Mirrors the image pipeline: get_image_features ↔ get_text_features
    No extra L2 normalization (matches embedding.py behavior).
    """
    inputs = tokenizer(
        [query],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

    # Extract the actual tensor (mirrors embedding.py logic for images)
    if isinstance(outputs, torch.Tensor):
        features = outputs
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        features = outputs.pooler_output
    elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        features = outputs.text_embeds
    elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        features = outputs.last_hidden_state.mean(dim=1)  # mean pool tokens
    else:
        # Tuple fallback
        features = outputs[0]
        if features.ndim == 3:
            features = features.mean(dim=1)

    embedding = features[0].cpu().numpy().flatten().astype("float32")

    # Sanity check: must be 1152d
    if len(embedding) != 1152:
        raise ValueError(f"Text embedding dim={len(embedding)}, expected 1152")

    return embedding.tolist()


# ── 6. Search and display ────────────────────────────────────────────────────
def search_and_display(query: str, top_k: int = TOP_K, threshold: float = SCORE_THRESHOLD):
    """Run Qdrant search and render a visual gallery."""

    print(f"\n🔍 Searching: \"{query}\"  (top_k={top_k}, threshold={threshold})")
    print("─" * 60)

    # Encode query
    query_vec = encode_text(query)

    # Search (qdrant-client >= 1.12 uses query_points instead of search)
    response = client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        using=DENSE_NAME,
        limit=top_k,
        with_payload=True,
        score_threshold=threshold,
    )
    hits = response.points

    if not hits:
        print("❌ No results found. Try lowering SCORE_THRESHOLD or a different query.")
        return

    print(f"📊 Found {len(hits)} results above threshold {threshold}\n")

    # Render each result
    for rank, hit in enumerate(hits, 1):
        vid   = hit.payload.get("video_id", "?")
        fidx  = hit.payload.get("frame_idx", "?")
        url   = hit.payload.get("azure_url", "")
        score = hit.score
        meta  = hit.payload.get("metadata", {})

        # ── Header ───────────────────────────────────────────────────
        score_bar = "█" * int(score * 40) + "░" * (40 - int(score * 40))
        display(HTML(f"""
        <div style="
            border: 1px solid #444;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            background: #1a1a2e;
            color: #eee;
            font-family: 'Segoe UI', monospace;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 18px; font-weight: bold; color: #00d4ff;">
                    #{rank}
                </span>
                <span style="font-size: 14px; color: #aaa;">
                    Score: <b style="color: #00ff88;">{score:.4f}</b>
                </span>
            </div>
            <div style="
                background: #333;
                border-radius: 4px;
                height: 8px;
                margin: 6px 0;
                overflow: hidden;
            ">
                <div style="
                    background: linear-gradient(90deg, #00ff88, #00d4ff);
                    height: 100%;
                    width: {score * 100:.1f}%;
                    border-radius: 4px;
                "></div>
            </div>
            <table style="width: 100%; color: #ccc; font-size: 13px;">
                <tr>
                    <td><b>Video:</b> {vid}</td>
                    <td><b>Frame:</b> {fidx}</td>
                </tr>
                {"<tr><td colspan='2'><b>Caption:</b> " + meta.get('caption', '') + "</td></tr>" if meta.get('caption') else ""}
                {"<tr><td colspan='2'><b>Tags:</b> " + ", ".join(meta.get('tags', [])) + "</td></tr>" if meta.get('tags') else ""}
                {"<tr><td colspan='2'><b>OCR:</b> " + meta.get('ocr', '') + "</td></tr>" if meta.get('ocr') else ""}
            </table>
            <div style="margin-top: 8px;">
                <a href="{url}" target="_blank" style="color: #00d4ff; font-size: 12px;">
                    🔗 {url}
                </a>
            </div>
        </div>
        """))

        # ── Image ────────────────────────────────────────────────────
        try:
            display(IPImage(url=url, width=480))
        except Exception:
            display(HTML(f"<p style='color:red;'>⚠️ Could not load image from: {url}</p>"))

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    vids_seen = list(dict.fromkeys(h.payload.get("video_id", "?") for h in hits))
    print(f"📹 Videos represented: {', '.join(vids_seen[:10])}")
    print(f"🏆 Best score: {hits[0].score:.4f} | Worst: {hits[-1].score:.4f}")
    print(f"{'─' * 60}\n")


# ── 7. Interactive loop ──────────────────────────────────────────────────────
def interactive_loop():
    """Interactive search loop. Type 'quit' to exit."""
    print("=" * 60)
    print("  🔎 QDRANT VISUAL SEARCH — Interactive Mode")
    print("=" * 60)
    print(f"  Collection:  {COLLECTION}")
    print(f"  Model:       {SIGLIP_MODEL_ID}")
    print(f"  Top-K:       {TOP_K}  (edit TOP_K variable above)")
    print(f"  Threshold:   {SCORE_THRESHOLD}  (edit SCORE_THRESHOLD variable above)")
    print(f"  Device:      {DEVICE}")
    print("=" * 60)
    print("  Type a search query, or 'quit' to exit.\n")

    while True:
        try:
            query = str(input("🔍 Query: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("👋 Exiting.")
            break

        # Allow inline parameter override: "pink outfit /k=10 /t=0.2"
        top_k = TOP_K
        threshold = SCORE_THRESHOLD
        parts = query.split()
        clean_parts = []
        for p in parts:
            if p.startswith("/k="):
                try:
                    top_k = int(p[3:])
                except ValueError:
                    pass
            elif p.startswith("/t="):
                try:
                    threshold = float(p[3:])
                except ValueError:
                    pass
            else:
                clean_parts.append(p)
        query = " ".join(clean_parts)

        if query:
            search_and_display(query, top_k, threshold)


# ── 8. Run ────────────────────────────────────────────────────────────────────
interactive_loop()
