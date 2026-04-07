"""
run_agentic_demo.py — End-to-end demo using **real** services.

Prerequisites
─────────────
1. pip install -r requirements.txt
2. Create (or update) a .env file in this directory with:
     QDRANT_URL=<your-qdrant-cloud-url>
     QDRANT_API_KEY=<your-qdrant-api-key>
     GEMINI_API_KEY=<your-gemini-api-key>
     EMBEDDING_API_BASE_URL=http://<azure-vm-ip>:8000

Usage
─────
    cd retrieval/agentic_retrieval
    python run_agentic_demo.py
    python run_agentic_demo.py --query "a red car driving on a highway"
    python run_agentic_demo.py --query "Tìm video có chữ 'SALE' trên biển quảng cáo" --top_k 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ── load .env ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# ── fix Windows console encoding ─────────────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── project imports ──────────────────────────────────────────────────────────
from graph import build_agentic_retrieval_graph
from services.llm_service import LLMService
from services.qdrant_search import QdrantSearchService

# ── pretty helpers ───────────────────────────────────────────────────────────
SEPARATOR = "═" * 90
THIN_SEP  = "─" * 90

SAMPLE_QUERIES: list[str] = [
    "Tìm video có một diễn giả mặc áo đỏ phát biểu ngoài trời",
    "a person standing in front of a whiteboard explaining a diagram",
    "cảnh đường phố ban đêm có đèn neon",
    "someone holding a microphone on stage",
    "Có chữ 'DANGER' trên biển báo",
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


# ── main ─────────────────────────────────────────────────────────────────────
def run_demo(
    query: str,
    top_k: int = 20,
    top_k_per_source: int = 20,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single query through the full agentic retrieval pipeline."""

    # ---- validate env ----
    missing = []
    for var in ("QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY"):
        if not os.getenv(var):
            missing.append(var)
    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        print("        Please set them in .env or export them before running.")
        sys.exit(1)

    # ---- init services ----
    _print_header("INITIALIZING SERVICES")
    t0 = time.perf_counter()

    llm = LLMService()
    print(f"  ✓ LLMService ready  (model={llm.model_name})")

    search_service = QdrantSearchService()
    print(f"  ✓ QdrantSearchService ready  (collection={search_service.collection_name})")

    t_init = time.perf_counter() - t0
    print(f"  ⏱ Service init took {t_init:.2f}s")

    # ---- build graph ----
    graph = build_agentic_retrieval_graph(llm, search_service)

    # ---- run pipeline ----
    _print_header(f"QUERY: {query}")
    t1 = time.perf_counter()

    initial_state: Dict[str, Any] = {"raw_query": query}
    final_state: Dict[str, Any] = graph.invoke(initial_state)

    t_pipeline = time.perf_counter() - t1
    print(f"\n  ⏱ Pipeline completed in {t_pipeline:.2f}s")

    # ---- check for pipeline errors ----
    if final_state.get("error"):
        print(f"\n  [PIPELINE ERROR] {final_state['error']}")

    # ---- display query bundle ----
    qb = final_state.get("query_bundle", {})
    if qb:
        _print_section("QUERY BUNDLE")
        print(f"  raw          : {qb.get('raw', '')}")
        print(f"  cleaned      : {qb.get('cleaned', '')}")
        print(f"  lang         : {qb.get('lang', '?')}")
        print(f"  translated_en: {qb.get('translated_en', '')}")
        print(f"  rewrites     : {qb.get('rewrites', [])}")

    # ---- display intent ----
    qi = final_state.get("query_intent", {})
    if qi:
        _print_section("QUERY INTENT (LLM extracted)")
        for key in ("objects", "attributes", "actions", "scene", "text_cues", "metadata_cues", "query_type"):
            val = qi.get(key, "—")
            if isinstance(val, list):
                val = ", ".join(val) if val else "[]"
            print(f"  {key:<16s}: {val}")

    # ---- display routing weights ----
    rw = final_state.get("routing_weights", {})
    if rw:
        _print_section("ROUTING WEIGHTS")
        for modality, w in sorted(rw.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 40)
            print(f"  {modality:<12s}: {w:.4f}  {bar}")

    # ---- display retrieval counts ----
    rr = final_state.get("retrieval_results", {})
    if rr:
        _print_section("RETRIEVAL RESULTS (per source)")
        for source, items in rr.items():
            print(f"  {source:<12s}: {len(items):>3d} candidates")

    # ---- display top-k ----
    agent_topk: List[Dict[str, Any]] = final_state.get("agent_topk", [])
    truncated = agent_topk[:top_k]

    _print_header(f"AGENT TOP-{top_k}  ({len(truncated)} results)")
    if not truncated:
        print("  (no results)")
    for idx, item in enumerate(truncated, start=1):
        _print_candidate(idx, item, verbose=verbose)

    # ---- display trace logs (verbose) ----
    if verbose:
        trace_logs = final_state.get("trace_logs", [])
        if trace_logs:
            _print_section("TRACE LOGS")
            for log in trace_logs:
                print(f"\n  [{log['node']}]")
                print(json.dumps(log["payload"], ensure_ascii=False, indent=4))

    # ---- timing summary ----
    _print_section("TIMING SUMMARY")
    print(f"  Service init : {t_init:.2f}s")
    print(f"  Pipeline     : {t_pipeline:.2f}s")
    print(f"  Total        : {t_init + t_pipeline:.2f}s")
    print()

    return final_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Retrieval — real-service demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sample queries:\n" + "\n".join(f"  • {q}" for q in SAMPLE_QUERIES),
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=SAMPLE_QUERIES[0],
        help="User query to search for (default: first sample query)",
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=20,
        help="Number of final results to display (default: 20)",
    )
    parser.add_argument(
        "--top_k_per_source",
        type=int,
        default=20,
        help="Number of candidates per retrieval source (default: 20)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-source scores and full trace logs",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Run all built-in sample queries sequentially",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all_samples:
        for i, q in enumerate(SAMPLE_QUERIES, start=1):
            print(f"\n{'#' * 90}")
            print(f"  SAMPLE {i}/{len(SAMPLE_QUERIES)}")
            print(f"{'#' * 90}")
            run_demo(
                query=q,
                top_k=args.top_k,
                top_k_per_source=args.top_k_per_source,
                verbose=args.verbose,
            )
    else:
        run_demo(
            query=args.query,
            top_k=args.top_k,
            top_k_per_source=args.top_k_per_source,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()