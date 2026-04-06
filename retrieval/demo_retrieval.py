"""
demo_retrieval.py — Branch A: Dense-Only Hybrid Search (Colab T4)
==================================================================

Architecture:
  Qdrant Native Hybrid = Prefetch(SigLIP) + Prefetch(BGE-M3) → FusionQuery(RRF)
  NO sparse vectors (OCR/Object) — clean baseline without agent noise.

Usage (Colab):
  1. Set QDRANT_URL, QDRANT_API_KEY in the config block at the bottom
  2. Run all cells
"""

# ── 0. Install deps (Colab auto-install) ─────────────────────────────────────
import subprocess, sys

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

try: from qdrant_client import QdrantClient
except ImportError: _pip("qdrant-client"); from qdrant_client import QdrantClient

try: from rich.console import Console
except ImportError: _pip("rich"); from rich.console import Console

try: from sentence_transformers import SentenceTransformer
except ImportError: _pip("sentence-transformers"); from sentence_transformers import SentenceTransformer

try: from transformers import AutoTokenizer, AutoModel
except ImportError: _pip("transformers"); from transformers import AutoTokenizer, AutoModel


# ── 1. Imports ────────────────────────────────────────────────────────────────
import os
import time
import numpy as np
import torch
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Prefetch,
    FusionQuery,
    Fusion,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from IPython.display import display, HTML

console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# 2. EMBEDDING MODELS — Lazy CUDA Singletons
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_siglip_tok: Optional[Any] = None
_siglip_mdl: Optional[Any] = None
_bge_m3_mdl: Optional[Any] = None

SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
BGE_M3_MODEL_ID = "BAAI/bge-m3"
SIGLIP_DIM = 1152
BGE_M3_DIM = 1024


def _load_siglip():
    """Load SigLIP text encoder to CUDA (singleton)."""
    global _siglip_tok, _siglip_mdl
    if _siglip_tok is None:
        console.print(f"[bold cyan]🔄 Loading SigLIP on [{DEVICE}]...[/]")
        _siglip_tok = AutoTokenizer.from_pretrained(SIGLIP_MODEL_ID)
        _siglip_mdl = AutoModel.from_pretrained(SIGLIP_MODEL_ID).to(DEVICE).eval()
        console.print(f"[bold green]✅ SigLIP ready on {DEVICE}[/]")
    return _siglip_tok, _siglip_mdl


def _load_bge_m3():
    """Load BGE-M3 to CUDA (singleton)."""
    global _bge_m3_mdl
    if _bge_m3_mdl is None:
        console.print(f"[bold cyan]🔄 Loading BGE-M3 on [{DEVICE}]...[/]")
        _bge_m3_mdl = SentenceTransformer(BGE_M3_MODEL_ID, device=DEVICE)
        console.print(f"[bold green]✅ BGE-M3 ready on {DEVICE}[/]")
    return _bge_m3_mdl


# ══════════════════════════════════════════════════════════════════════════════
# 3. ENCODING FUNCTIONS — Pure, Decoupled
# ══════════════════════════════════════════════════════════════════════════════

def encode_visual(query: str) -> list:
    """Encode text → SigLIP 1152d dense vector (CUDA). For 'keyframe-dense'."""
    tokenizer, model = _load_siglip()
    inputs = tokenizer(
        [query], padding="max_length", truncation=True, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

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
    assert len(vec) == SIGLIP_DIM, f"SigLIP dim={len(vec)}, expected {SIGLIP_DIM}"
    return vec.tolist()


def encode_semantic(query: str) -> list:
    """Encode text → BGE-M3 1024d dense vector (CUDA). For 'keyframe-caption-dense'."""
    model = _load_bge_m3()
    vec = model.encode(query, normalize_embeddings=True, device=DEVICE)
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    assert len(vec) == BGE_M3_DIM, f"BGE-M3 dim={len(vec)}, expected {BGE_M3_DIM}"
    return vec


# ══════════════════════════════════════════════════════════════════════════════
# 4. QDRANT HYBRID SEARCH — Native Prefetch + RRF Fusion
# ══════════════════════════════════════════════════════════════════════════════

def execute_qdrant_rrf(
    client: QdrantClient,
    collection: str,
    bge_vector: list,
    siglip_vector: list,
    top_k: int = 10,
    prefetch_limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Qdrant-native hybrid search:
      Prefetch(BGE-M3 on keyframe-caption-dense)
      Prefetch(SigLIP on keyframe-dense)
      → FusionQuery(Fusion.RRF) → Top K
    """
    results = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(
                query=bge_vector,
                using="keyframe-caption-dense",
                limit=prefetch_limit,
            ),
            Prefetch(
                query=siglip_vector,
                using="keyframe-dense",
                limit=prefetch_limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "point_id": str(hit.id),
            "rrf_score": hit.score,
            "payload": hit.payload or {},
        }
        for hit in results
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 5. DISPLAY — Rich Table + Colab Visual Cards
# ══════════════════════════════════════════════════════════════════════════════

def print_rich_table(results: List[Dict], query: str) -> None:
    """Print a Rich table with RRF scores."""
    table = Table(
        title=f"🔍 Query: \"{query}\"",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold magenta",
        header_style="bold cyan",
        border_style="bright_blue",
    )
    table.add_column("#", style="bold white", width=3, justify="center")
    table.add_column("Video", style="green", width=14)
    table.add_column("Frame", style="yellow", width=8, justify="center")
    table.add_column("Time", style="yellow", width=6, justify="center")
    table.add_column("RRF Score", style="bold red", width=12, justify="center")
    table.add_column("Caption", style="white", max_width=55)

    for i, r in enumerate(results, 1):
        p = r["payload"]
        table.add_row(
            str(i),
            p.get("video_id", "?"),
            str(p.get("frame_idx", "?")),
            f"{p.get('timestamp_sec', 0)}s",
            f"{r['rrf_score']:.6f}",
            (p.get("caption", "") or p.get("detailed_caption", ""))[:55],
        )

    console.print()
    console.print(table)
    console.print()


def render_results_in_colab(results: List[Dict], max_display: int = 5) -> None:
    """Render visual cards in Colab with images + clickable YouTube links."""
    html_parts = ["""
    <style>
        .rc{border:1px solid #444;border-radius:12px;padding:16px;margin:12px 0;
            background:linear-gradient(135deg,#1a1a2e,#16213e);color:#eee;
            font-family:'Segoe UI',sans-serif;display:flex;gap:16px;align-items:flex-start}
        .rc img{border-radius:8px;max-width:320px;max-height:200px;object-fit:cover;border:2px solid #0f3460}
        .ri{flex:1;min-width:0}
        .rr{font-size:22px;font-weight:bold;color:#00d4ff;margin-bottom:8px}
        .sb{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:13px}
        .sl{width:55px;text-align:right;color:#aaa}
        .sbg{flex:1;height:12px;background:#333;border-radius:6px;overflow:hidden;max-width:220px}
        .sf{height:100%;border-radius:6px}
        .sv{width:70px;font-family:monospace;color:#ccc}
        .mt{margin-top:10px;font-size:13px;color:#bbb}
        .mt td{padding:2px 8px}
        .mk{color:#888;font-weight:bold}
        .yt{display:inline-block;margin-top:10px;padding:8px 16px;background:#ff0000;
            color:white;border-radius:6px;text-decoration:none;font-size:14px;font-weight:bold}
        .yt:hover{background:#cc0000}
    </style>
    """]

    for i, r in enumerate(results[:max_display], 1):
        p = r["payload"]
        img_url = p.get("azure_url", "")
        yt_link = p.get("youtube_link", "")
        caption = p.get("caption", "") or p.get("detailed_caption", "")
        ocr = p.get("ocr_text", "")
        rrf = r["rrf_score"]
        ts = p.get("timestamp_sec", 0)

        img_html = (
            f'<img src="{img_url}" onerror="this.style.display=\'none\'">'
            if img_url else
            '<div style="width:320px;height:180px;background:#333;border-radius:8px;'
            'display:flex;align-items:center;justify-content:center;color:#666">No Image</div>'
        )
        yt_html = (
            f'<a href="{yt_link}" target="_blank" class="yt">▶ YouTube (t={ts}s)</a>'
            if yt_link else ""
        )

        html_parts.append(f"""
        <div class="rc">
            <div>{img_html}</div>
            <div class="ri">
                <div class="rr">#{i}</div>
                <div class="sb">
                    <span class="sl">RRF</span>
                    <div class="sbg">
                        <div class="sf" style="width:{min(rrf*1500,100):.0f}%;
                             background:linear-gradient(90deg,#ff6b6b,#feca57)"></div>
                    </div>
                    <span class="sv">{rrf:.6f}</span>
                </div>
                <table class="mt">
                    <tr><td class="mk">Video:</td><td>{p.get('video_id','?')}</td></tr>
                    <tr><td class="mk">Frame:</td><td>{p.get('frame_idx','?')} (t={ts}s)</td></tr>
                    {"<tr><td class='mk'>Caption:</td><td>"+caption[:120]+"</td></tr>" if caption else ""}
                    {"<tr><td class='mk'>OCR:</td><td>"+ocr[:100]+"</td></tr>" if ocr else ""}
                </table>
                {yt_html}
            </div>
        </div>
        """)

    display(HTML("".join(html_parts)))


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_search(
    query: str,
    top_k: int,
    qdrant_url: str,
    qdrant_api_key: str,
    collection: str,
    prefetch_limit: int = 50,
) -> None:
    """
    End-to-end:
      1. Encode query → 2 vectors (CUDA)
      2. Execute Qdrant native RRF
      3. Display rich table + visual cards
    """
    console.print(Panel.fit(
        f"[bold white]🔍 Query:[/] [italic]{query}[/]\n"
        f"[bold white]📊 Top-K:[/] {top_k}  |  "
        f"[bold white]🗄️  Collection:[/] {collection}  |  "
        f"[bold white]🖥️  Device:[/] {DEVICE}",
        title="[bold cyan]Branch A — Dense-Only Hybrid Search[/]",
        border_style="bright_blue",
    ))

    # ── Step 1: Encode ────────────────────────────────────────────────
    console.print("\n[bold]Step 1/3:[/] Encoding query vectors...")
    t0 = time.time()

    with console.status("[cyan]Encoding SigLIP (visual)...[/]"):
        siglip_vec = encode_visual(query)
    t_siglip = time.time() - t0

    t1 = time.time()
    with console.status("[cyan]Encoding BGE-M3 (semantic)...[/]"):
        bge_vec = encode_semantic(query)
    t_bge = time.time() - t1

    console.print(
        f"  ├─ SigLIP: {SIGLIP_DIM}d in [green]{t_siglip:.2f}s[/]\n"
        f"  └─ BGE-M3: {BGE_M3_DIM}d in [green]{t_bge:.2f}s[/]"
    )

    # ── Step 2: Qdrant RRF ────────────────────────────────────────────
    console.print("\n[bold]Step 2/3:[/] Executing Qdrant native RRF...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    t2 = time.time()
    with console.status("[cyan]Prefetch + Fusion.RRF...[/]"):
        results = execute_qdrant_rrf(
            client, collection, bge_vec, siglip_vec,
            top_k=top_k, prefetch_limit=prefetch_limit,
        )
    t_search = time.time() - t2
    console.print(f"  └─ {len(results)} results in [green]{t_search:.3f}s[/]")

    if not results:
        console.print("[bold red]❌ No results found.[/]")
        return

    # ── Step 3: Display ───────────────────────────────────────────────
    console.print("\n[bold]Step 3/3:[/] Rendering results...\n")
    print_rich_table(results, query)
    render_results_in_colab(results, max_display=top_k)

    # ── Summary ───────────────────────────────────────────────────────
    total = time.time() - t0
    console.print(Panel.fit(
        f"[green]✅ Done in {total:.2f}s[/]  |  "
        f"Encode: {t_siglip + t_bge:.2f}s  |  "
        f"Search: {t_search:.3f}s",
        border_style="green",
    ))


# ══════════════════════════════════════════════════════════════════════════════
# 7. CONFIGURATION BLOCK — Edit these values and run!
# ══════════════════════════════════════════════════════════════════════════════

QDRANT_URL        = os.environ.get("QDRANT_URL", "")      # 👈 Paste your Qdrant Cloud URL
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY", "")   # 👈 Paste your Qdrant API Key
COLLECTION_NAME   = "keyframes_v1"                         # 👈 Your collection name

USER_QUERY        = "người mặc áo đỏ đang nấu ăn"        # 👈 Your search query
TOP_K             = 5                                      # 👈 Number of results
PREFETCH_LIMIT    = 50                                     # 👈 Candidates per vector before RRF


# ══════════════════════════════════════════════════════════════════════════════
# 8. EXECUTE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not QDRANT_URL or not QDRANT_API_KEY:
        console.print("[bold red]❌ Missing QDRANT_URL or QDRANT_API_KEY![/]")
        console.print("[yellow]Set them in the CONFIGURATION block above.[/]")
    else:
        run_search(
            query=USER_QUERY,
            top_k=TOP_K,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            collection=COLLECTION_NAME,
            prefetch_limit=PREFETCH_LIMIT,
        )
