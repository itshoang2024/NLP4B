"""
run_agentic_demo.py — End-to-end demo using real services.

Migrated from: retrieval/agentic_retrieval/run_agentic_demo.py
Updated to use the new backend package structure.

Usage:
    cd backend
    python test/run_agentic_demo.py
    python test/run_agentic_demo.py --query "a red car driving on a highway"
    python test/run_agentic_demo.py --query "Khung hình có chữ 'Quân A.P'" --top_k 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ── OpenMP Workaround ────────────────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── path setup (ensure backend/ is on sys.path) ─────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ── load .env ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(_BACKEND_DIR / ".env")

# ── fix Windows console encoding ─────────────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── project imports ──────────────────────────────────────────────────────────
from src.services.agentic_retrieve.service import AgenticRetrieveService
from src.services.translator import detect_language, translate_to_english


SEPARATOR = "═" * 90
THIN_SEP  = "─" * 90

SAMPLE_QUERIES: list[str] = [
    "Tìm video có một diễn giả mặc áo đỏ phát biểu ngoài trời",
    "a person standing in front of a whiteboard explaining a diagram",
    "cảnh đường phố ban đêm có đèn neon",
    "someone holding a microphone on stage",
    "Khung hình có chữ 'Quân A.P'",
    "Hai người đàn ông ngoài trời trong chương trình thực tế, một người hóa trang chú hề mặc đồ đỏ trắng, mặt trang điểm trắng, người còn lại cầm micro có bông lọc gió đang nói; bối cảnh đường phố với biển báo giao thông. Chữ trên ảnh: 'THỬ THÁCH NHẬP CUỘC', logo 'FOREST STUDIO'."
]


def _fmt_score(value: float) -> str:
    return f"{value:.4f}"


def _print_header(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def _print_section(title: str) -> None:
    print(f"\n{THIN_SEP}")
    print(f"  {title}")
    print(THIN_SEP)


def _print_candidate(idx: int, item: Dict[str, Any], verbose: bool = False) -> None:
    evidence = item.get("evidence", [])
    source_scores = item.get("source_scores", {})

    print(
        f"  {idx:02d}. video={item['video_id']:<14s} | "
        f"frame={item['frame_id']:<6d} | "
        f"agent_score={_fmt_score(item.get('agent_score', 0.0))} | "
        f"evidence={evidence}"
    )

    if verbose and source_scores:
        parts = [f"{k}={v:.4f}" for k, v in source_scores.items()]
        print(f"      └── source_scores: {', '.join(parts)}")


def _build_query_bundle(raw_query: str) -> Dict[str, Any]:
    """Replicate the middleware logic for standalone demo usage."""
    cleaned = " ".join(raw_query.strip().split())
    lang = detect_language(cleaned)
    translated_en = translate_to_english(cleaned, lang)

    rewrites = []
    if translated_en and translated_en != cleaned:
        rewrites.append(translated_en)
    rewrites.append(cleaned)
    seen = set()
    final = []
    for x in rewrites:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            final.append(x)

    return {
        "raw": raw_query,
        "cleaned": cleaned,
        "lang": lang,
        "translated_en": translated_en,
        "rewrites": final[:3],
    }


def run_demo(
    query: str,
    top_k: int = 20,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run a single query through the agentic retrieval pipeline."""

    # ---- build query bundle (replicating middleware) ----
    _print_header(f"QUERY: {query}")
    query_bundle = _build_query_bundle(query)

    _print_section("QUERY BUNDLE")
    for k, v in query_bundle.items():
        print(f"  {k:<16s}: {v}")

    # ---- init & run ----
    _print_header("INITIALIZING AGENTIC SERVICE")
    t0 = time.perf_counter()
    service = AgenticRetrieveService()
    t_init = time.perf_counter() - t0
    print(f"  ⏱ Init took {t_init:.2f}s")

    t1 = time.perf_counter()
    # retrieve from graph
    initial_state: Dict[str, Any] = {"query_bundle": query_bundle}
    try:
        final_state: Dict[str, Any] = service.graph.invoke(initial_state)
    except Exception as exc:
        print(f"Agentic pipeline failed: {exc}")
        final_state = {}
        
    candidates = final_state.get("agent_topk", [])[:top_k]
    t_pipeline = time.perf_counter() - t1

    # ---- display ----
    _print_header(f"AGENT TOP-{top_k}  ({len(candidates)} results)")
    if not candidates:
        print("  (no results)")
    for idx, item in enumerate(candidates, start=1):
        _print_candidate(idx, item, verbose=verbose)

    if verbose:
        _print_section("TRACE LOGS")
        trace_logs = final_state.get("trace_logs", [])
        if trace_logs:
            print(json.dumps(trace_logs, indent=2, ensure_ascii=False))
        else:
            print("  (no trace logs)")

    _print_section("TIMING SUMMARY")
    print(f"  Service init : {t_init:.2f}s")
    print(f"  Pipeline     : {t_pipeline:.2f}s")
    print(f"  Total        : {t_init + t_pipeline:.2f}s")
    print()

    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Retrieval — standalone demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sample queries:\n" + "\n".join(f"  • {q}" for q in SAMPLE_QUERIES),
    )
    parser.add_argument("--query", "-q", type=str, default=SAMPLE_QUERIES[5])
    parser.add_argument("--top_k", "-k", type=int, default=20)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--all-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all_samples:
        for i, q in enumerate(SAMPLE_QUERIES, start=1):
            print(f"\n{'#' * 90}")
            print(f"  SAMPLE {i}/{len(SAMPLE_QUERIES)}")
            print(f"{'#' * 90}")
            run_demo(query=q, top_k=args.top_k, verbose=args.verbose)
    else:
        run_demo(query=args.query, top_k=args.top_k, verbose=args.verbose)


if __name__ == "__main__":
    main()
