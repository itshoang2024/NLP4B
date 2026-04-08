"""
bench_latency.py — Retrieval latency benchmark.

Instruments every stage of BOTH retrieval pipelines:

  AGENTIC pipeline:
    1. Translation          (LLM call)
    2. Intent Extraction     (LLM call)
    3. Modality Routing      (CPU)
    4. Parallel Retrieval    (Qdrant + Embedding API network)
    5. Candidate Fusion      (CPU)
    6. Frame Reranking       (CPU)

  HEURISTIC pipeline:
    1. Embedding             (/embed/query — 1 HTTP call)
    2. Fallback Search       (Qdrant 2-tier, 4 streams)
    3. RRF Fusion            (CPU)
    4. Count Bonus           (CPU)

Reports per-node latency breakdown and highlights the bottleneck.
Also logs which LLM backend (gemini / llama_cpp) was used for agentic.

Usage:
    cd backend
    python test/bench_latency.py                                # agentic only (default)
    python test/bench_latency.py --strategy both --runs 3
    python test/bench_latency.py --strategy heuristic --runs 2
    python test/bench_latency.py --strategy agentic --backend llama_cpp --runs 3
    python test/bench_latency.py --all-samples --strategy both
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ── OpenMP workaround ────────────────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Path setup ───────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from dotenv import load_dotenv

# ── Fix Windows console encoding ─────────────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_QUERIES: list[str] = [
    "Khung hình có chữ 'Quân A.P'",
    "Tìm video có một diễn giả mặc áo đỏ phát biểu ngoài trời",
    "a person standing in front of a whiteboard explaining a diagram",
    "cảnh đường phố ban đêm có đèn neon",
    "someone holding a microphone on stage",
]

AGENTIC_NODES = [
    "translation",
    "query_intent_extraction",
    "modality_routing",
    "parallel_retrieval",
    "candidate_fusion",
    "frame_reranking",
]

HEURISTIC_NODES = [
    "embedding_query",
    "fallback_search",
    "rrf_fusion",
    "count_bonus",
]

SEPARATOR = "═" * 94
THIN_SEP = "─" * 94


# ═══════════════════════════════════════════════════════════════════════════════
# Timing infrastructure
# ═══════════════════════════════════════════════════════════════════════════════


def _wrap_node(fn: Callable, node_name: str, timings: Dict[str, float]) -> Callable:
    """Wrap a LangGraph node function to record its execution time."""
    def timed_node(state):
        t0 = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - t0
        timings[node_name] = elapsed
        return result
    return timed_node


def _time_call(fn: Callable, timings: Dict[str, float], name: str, *args, **kwargs):
    """Time a function call and store the result."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    timings[name] = time.perf_counter() - t0
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Build query bundle (shared middleware logic)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_query_bundle(query: str, timings: Dict[str, float]) -> Dict[str, Any]:
    """Build query_bundle and time the translation step."""
    from src.services.translator import detect_language, translate_to_english

    cleaned = " ".join(query.strip().split())
    lang = detect_language(cleaned)

    t0 = time.perf_counter()
    translated_en = translate_to_english(cleaned, lang)
    timings["translation"] = time.perf_counter() - t0

    rewrites: list[str] = []
    if translated_en and translated_en != cleaned:
        rewrites.append(translated_en)
    rewrites.append(cleaned)
    seen: set[str] = set()
    final: list[str] = []
    for x in rewrites:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            final.append(x)

    return {
        "raw": query,
        "cleaned": cleaned,
        "lang": lang,
        "translated_en": translated_en,
        "rewrites": final[:3],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Agentic benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def run_agentic_benchmark(query: str, top_k: int = 20) -> Dict[str, Any]:
    """Run a single query through the agentic pipeline with per-node timing."""
    from src.services.agentic_retrieve.llm_service import LLMService
    from src.services.agentic_retrieve.qdrant_search import QdrantSearchService
    from src.services.agentic_retrieve.state import AgentState
    from src.services.agentic_retrieve.nodes.intent_extraction import query_intent_extraction_node_factory
    from src.services.agentic_retrieve.nodes.routing import modality_routing_node
    from src.services.agentic_retrieve.nodes.retrieval import parallel_retrieval_node_factory
    from src.services.agentic_retrieve.nodes.fusion import candidate_fusion_node
    from src.services.agentic_retrieve.nodes.rerank import frame_reranking_node
    from src.services.llm import get_llm_provider
    from langgraph.graph import END, StateGraph

    node_timings: Dict[str, float] = {}

    # Step 1: Translation
    query_bundle = _build_query_bundle(query, node_timings)

    # Steps 2-6: Instrumented graph
    llm = LLMService()
    search_service = QdrantSearchService()

    intent_fn = query_intent_extraction_node_factory(llm)
    retrieval_fn = parallel_retrieval_node_factory(search_service, top_k_per_source=top_k)

    builder = StateGraph(AgentState)
    builder.add_node("query_intent_extraction", _wrap_node(intent_fn, "query_intent_extraction", node_timings))
    builder.add_node("modality_routing", _wrap_node(modality_routing_node, "modality_routing", node_timings))
    builder.add_node("parallel_retrieval", _wrap_node(retrieval_fn, "parallel_retrieval", node_timings))
    builder.add_node("candidate_fusion", _wrap_node(candidate_fusion_node, "candidate_fusion", node_timings))
    builder.add_node("frame_reranking", _wrap_node(frame_reranking_node, "frame_reranking", node_timings))

    builder.set_entry_point("query_intent_extraction")
    builder.add_edge("query_intent_extraction", "modality_routing")
    builder.add_edge("modality_routing", "parallel_retrieval")
    builder.add_edge("parallel_retrieval", "candidate_fusion")
    builder.add_edge("candidate_fusion", "frame_reranking")
    builder.add_edge("frame_reranking", END)
    graph = builder.compile()

    initial_state: Dict[str, Any] = {"query_bundle": query_bundle}
    t_total_start = time.perf_counter()
    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        print(f"  ⚠ Agentic pipeline error: {exc}")
        final_state = {}
    t_total = time.perf_counter() - t_total_start

    n_results = len(final_state.get("agent_topk", []))
    provider = get_llm_provider()

    return {
        "strategy": "agentic",
        "query": query,
        "lang": query_bundle["lang"],
        "llm_backend": os.getenv("LLM_BACKEND", "gemini"),
        "llm_model": provider.model_name,
        "node_names": AGENTIC_NODES,
        "node_timings": node_timings,
        "total_pipeline_s": t_total,
        "total_e2e_s": t_total + node_timings.get("translation", 0.0),
        "n_results": n_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Heuristic benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def run_heuristic_benchmark(query: str, top_k: int = 20) -> Dict[str, Any]:
    """Run a single query through the heuristic pipeline with per-step timing."""
    from src.services.heuristic_retrieve.service import (
        HeuristicRetrieveService,
        EmbedQueryClient,
        execute_fallback_search,
        compute_rrf,
        apply_count_bonus,
    )
    from src.config import get_qdrant_url, get_qdrant_api_key, get_embedding_api_url
    from qdrant_client import QdrantClient

    node_timings: Dict[str, float] = {}

    # We need the query text the same way the service builds it
    # (translation happens at middleware level, before reaching either service)
    query_bundle = _build_query_bundle(query, node_timings)
    query_text = (
        query_bundle.get("translated_en")
        or query_bundle.get("cleaned")
        or query_bundle.get("raw", "")
    ).strip()

    # Init clients (not timed — same as agentic)
    qdrant = QdrantClient(url=get_qdrant_url(), api_key=get_qdrant_api_key(), timeout=60)
    embed_client = EmbedQueryClient(base_url=get_embedding_api_url(), timeout=90)

    t_total_start = time.perf_counter()
    n_results = 0

    try:
        # Step 1: Embedding (/embed/query)
        embed_resp = _time_call(embed_client.query, node_timings, "embedding_query", query_text)
        if embed_resp is None:
            print("  ⚠ Embedding API returned None")
            t_total = time.perf_counter() - t_total_start
            return _heuristic_result(query, query_bundle, node_timings, t_total, 0)

        nlp_analysis = embed_resp.get("nlp_analysis", {})

        # Step 2: Fallback search (Qdrant 2-tier)
        streams = _time_call(execute_fallback_search, node_timings, "fallback_search", qdrant, embed_resp, top_k)

        # Step 3: RRF fusion
        rrf_pool = _time_call(compute_rrf, node_timings, "rrf_fusion", streams)

        # Step 4: Count bonus
        ranked = _time_call(apply_count_bonus, node_timings, "count_bonus", rrf_pool, nlp_analysis, top_k)

        n_results = len(ranked)

    except Exception as exc:
        print(f"  ⚠ Heuristic pipeline error: {exc}")

    t_total = time.perf_counter() - t_total_start
    return _heuristic_result(query, query_bundle, node_timings, t_total, n_results)


def _heuristic_result(
    query: str,
    query_bundle: Dict[str, Any],
    node_timings: Dict[str, float],
    t_total: float,
    n_results: int,
) -> Dict[str, Any]:
    return {
        "strategy": "heuristic",
        "query": query,
        "lang": query_bundle["lang"],
        "llm_backend": os.getenv("LLM_BACKEND", "gemini"),
        "llm_model": "n/a (heuristic)",
        "node_names": ["translation"] + HEURISTIC_NODES,
        "node_timings": node_timings,
        "total_pipeline_s": t_total,
        "total_e2e_s": t_total + node_timings.get("translation", 0.0),
        "n_results": n_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════════

_BAR_CHAR = "█"
_BAR_MAX_WIDTH = 30


def _bar(value: float, max_value: float) -> str:
    if max_value <= 0:
        return ""
    width = int((value / max_value) * _BAR_MAX_WIDTH)
    return _BAR_CHAR * max(width, 1)


def _fmt(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}µs"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.3f}s"


def _pct(part: float, total: float) -> str:
    if total <= 0:
        return "  -  "
    return f"{part / total * 100:5.1f}%"


def print_single_run(result: Dict[str, Any]) -> None:
    """Print a detailed breakdown for a single run."""
    nt = result["node_timings"]
    nodes = result["node_names"]
    total_e2e = result["total_e2e_s"]
    max_t = max(nt.values()) if nt else 0.001

    print(f"\n  {'Node':<28s} {'Time':>10s} {'%':>7s}  Bar")
    print(f"  {'─' * 28} {'─' * 10} {'─' * 7}  {'─' * _BAR_MAX_WIDTH}")

    for node in nodes:
        t = nt.get(node, 0.0)
        print(f"  {node:<28s} {_fmt(t):>10s} {_pct(t, total_e2e):>7s}  {_bar(t, max_t)}")

    print(f"  {'─' * 28} {'─' * 10} {'─' * 7}")
    print(f"  {'TOTAL (end-to-end)':<28s} {_fmt(total_e2e):>10s} {'100.0%':>7s}")
    print(f"  {'Results':<28s} {result['n_results']:>10d}")


def print_aggregate_summary(
    all_results: List[Dict[str, Any]],
    strategy: str,
) -> None:
    """Print aggregate statistics for a given strategy."""
    if not all_results:
        return

    n_runs = len(all_results)
    nodes = all_results[0]["node_names"]
    backend = all_results[0].get("llm_backend", "")
    model = all_results[0].get("llm_model", "")

    print(f"\n{SEPARATOR}")
    print(f"  AGGREGATE: {strategy.upper()}  —  {n_runs} run(s)  |  backend={backend}  |  model={model}")
    print(SEPARATOR)

    node_data: Dict[str, List[float]] = {n: [] for n in nodes}
    e2e_times: List[float] = []

    for r in all_results:
        nt = r["node_timings"]
        for n in nodes:
            node_data[n].append(nt.get(n, 0.0))
        e2e_times.append(r["total_e2e_s"])

    avg_e2e = statistics.mean(e2e_times) if e2e_times else 0.001

    print(
        f"\n  {'Node':<28s} {'Min':>9s} {'Avg':>9s} {'Median':>9s} "
        f"{'Max':>9s} {'Avg %':>7s}  {'Avg Bar'}"
    )
    print(
        f"  {'─' * 28} {'─' * 9} {'─' * 9} {'─' * 9} "
        f"{'─' * 9} {'─' * 7}  {'─' * _BAR_MAX_WIDTH}"
    )

    max_avg = max((statistics.mean(v) for v in node_data.values() if v), default=0.001)

    bottleneck_node = ""
    bottleneck_avg = 0.0

    for node in nodes:
        vals = node_data[node]
        if not vals:
            continue
        mn, avg, med, mx = min(vals), statistics.mean(vals), statistics.median(vals), max(vals)
        if avg > bottleneck_avg:
            bottleneck_avg = avg
            bottleneck_node = node
        print(
            f"  {node:<28s} {_fmt(mn):>9s} {_fmt(avg):>9s} {_fmt(med):>9s} "
            f"{_fmt(mx):>9s} {_pct(avg, avg_e2e):>7s}  {_bar(avg, max_avg)}"
        )

    print(f"  {'─' * 28} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 7}")

    e2e_avg = statistics.mean(e2e_times)
    e2e_med = statistics.median(e2e_times)
    print(
        f"  {'TOTAL (end-to-end)':<28s} {_fmt(min(e2e_times)):>9s} {_fmt(e2e_avg):>9s} "
        f"{_fmt(e2e_med):>9s} {_fmt(max(e2e_times)):>9s} {'100.0%':>7s}"
    )

    # Bottleneck
    print(f"\n  🔥 Bottleneck: {bottleneck_node}  (avg {_fmt(bottleneck_avg)}, "
          f"{_pct(bottleneck_avg, avg_e2e).strip()} of total)")

    # LLM / network breakdown (agentic specific)
    if strategy == "agentic":
        llm_nodes = {"translation", "query_intent_extraction"}
        llm_avg = sum(statistics.mean(node_data[n]) for n in llm_nodes if node_data.get(n))
        non_llm_avg = avg_e2e - llm_avg
        print(f"\n  ⏱  LLM-dependent time (avg): {_fmt(llm_avg)} ({_pct(llm_avg, avg_e2e).strip()})")
        print(f"  ⏱  Non-LLM time (avg):       {_fmt(non_llm_avg)} ({_pct(non_llm_avg, avg_e2e).strip()})")

    if strategy == "heuristic":
        net_nodes = {"translation", "embedding_query", "fallback_search"}
        net_avg = sum(statistics.mean(node_data[n]) for n in net_nodes if node_data.get(n))
        cpu_avg = avg_e2e - net_avg
        print(f"\n  ⏱  Network time (avg):  {_fmt(net_avg)} ({_pct(net_avg, avg_e2e).strip()})")
        print(f"  ⏱  CPU-only time (avg): {_fmt(cpu_avg)} ({_pct(cpu_avg, avg_e2e).strip()})")

    print()


def print_comparison(
    agentic_results: List[Dict[str, Any]],
    heuristic_results: List[Dict[str, Any]],
) -> None:
    """Print a side-by-side comparison of the two strategies."""
    if not agentic_results or not heuristic_results:
        return

    a_avg = statistics.mean(r["total_e2e_s"] for r in agentic_results)
    h_avg = statistics.mean(r["total_e2e_s"] for r in heuristic_results)
    a_res = statistics.mean(r["n_results"] for r in agentic_results)
    h_res = statistics.mean(r["n_results"] for r in heuristic_results)

    faster = "HEURISTIC" if h_avg < a_avg else "AGENTIC"
    speedup = max(a_avg, h_avg) / max(min(a_avg, h_avg), 0.001)

    print(f"\n{SEPARATOR}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(SEPARATOR)
    print(f"\n  {'Metric':<28s} {'Agentic':>12s} {'Heuristic':>12s}")
    print(f"  {'─' * 28} {'─' * 12} {'─' * 12}")
    print(f"  {'Avg E2E latency':<28s} {_fmt(a_avg):>12s} {_fmt(h_avg):>12s}")
    print(f"  {'Avg results':<28s} {a_res:>12.0f} {h_res:>12.0f}")
    print(f"  {'LLM backend':<28s} {agentic_results[0]['llm_backend']:>12s} {'n/a':>12s}")
    print(f"\n  🏆 {faster} is faster by {speedup:.1f}x")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CSV output
# ═══════════════════════════════════════════════════════════════════════════════


ALL_NODE_NAMES = (
    ["translation"]
    + [f"ag_{n}" for n in AGENTIC_NODES if n != "translation"]
    + [f"hr_{n}" for n in HEURISTIC_NODES]
)


def save_results_csv(all_results: List[Dict[str, Any]], output_path: Path) -> None:
    """Write raw run data to CSV for further analysis."""
    fieldnames = [
        "timestamp", "strategy", "query", "lang", "llm_backend", "llm_model",
        "n_results", "total_e2e_s",
    ] + [f"node_{n}_s" for n in ALL_NODE_NAMES]

    rows = []
    for r in all_results:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "strategy": r["strategy"],
            "query": r["query"],
            "lang": r["lang"],
            "llm_backend": r["llm_backend"],
            "llm_model": r["llm_model"],
            "n_results": r["n_results"],
            "total_e2e_s": round(r["total_e2e_s"], 6),
        }
        prefix = "ag_" if r["strategy"] == "agentic" else "hr_"
        for n in r["node_names"]:
            col = n if n == "translation" else f"{prefix}{n}"
            row[f"node_{col}_s"] = round(r["node_timings"].get(n, 0.0), 6)
        rows.append(row)

    file_exists = output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"  📄 Results appended to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval latency benchmark — agentic & heuristic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test/bench_latency.py --strategy both --runs 3\n"
            "  python test/bench_latency.py --strategy heuristic --all-samples\n"
            "  python test/bench_latency.py --strategy agentic --backend gemini --runs 5\n"
        ),
    )
    parser.add_argument(
        "--strategy", "-s", type=str, default="both",
        choices=["agentic", "heuristic", "both"],
        help="Which retrieval strategy to benchmark (default: both).",
    )
    parser.add_argument(
        "--backend", "-b", type=str, default=None,
        choices=["gemini", "llama_cpp", "openai_compat"],
        help="LLM backend for agentic (overrides .env).",
    )
    parser.add_argument("--query", "-q", type=str, default=None)
    parser.add_argument("--all-samples", action="store_true", help="Run all sample queries.")
    parser.add_argument("--runs", "-n", type=int, default=1, help="Repeat each query N times.")
    parser.add_argument("--top_k", "-k", type=int, default=20)
    parser.add_argument("--csv", type=str, default=None, help="CSV output path.")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv(_BACKEND_DIR / ".env", override=False)
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend

    backend = os.getenv("LLM_BACKEND", "gemini")

    if args.all_samples:
        queries = SAMPLE_QUERIES
    elif args.query:
        queries = [args.query]
    else:
        queries = [SAMPLE_QUERIES[0]]

    run_agentic = args.strategy in ("agentic", "both")
    run_heuristic = args.strategy in ("heuristic", "both")

    strategies = []
    if run_agentic:
        strategies.append("agentic")
    if run_heuristic:
        strategies.append("heuristic")

    total_runs = len(queries) * args.runs * len(strategies)

    # Banner
    print(f"\n{SEPARATOR}")
    print(f"  LATENCY BENCHMARK")
    print(f"  Strategies : {', '.join(strategies)}")
    print(f"  LLM Backend: {backend}" if run_agentic else f"  LLM Backend: n/a")
    print(f"  Queries    : {len(queries)}  ×  {args.runs} run(s)  ×  {len(strategies)} strategy  =  {total_runs} total")
    print(f"  Top-K      : {args.top_k}")
    print(SEPARATOR)

    # Suppress noisy loggers
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for noisy in (
        "httpx", "openai", "httpcore", "qdrant_client",
        "QdrantSearch", "src", "src.services", "langgraph",
        "src.services.llm", "src.services.translator",
        "src.services.agentic_retrieve", "src.services.heuristic_retrieve",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    agentic_results: List[Dict[str, Any]] = []
    heuristic_results: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    for qi, query in enumerate(queries, 1):
        for run in range(1, args.runs + 1):
            # ── Agentic ──────────────────────────────────────────────────
            if run_agentic:
                label = f"Q{qi} R{run}/{args.runs} AGENTIC"
                print(f"\n{THIN_SEP}")
                print(f"  [{label}] {query[:65]}")
                print(THIN_SEP)

                result = run_agentic_benchmark(query, top_k=args.top_k)
                agentic_results.append(result)
                all_results.append(result)
                print_single_run(result)

            # ── Heuristic ────────────────────────────────────────────────
            if run_heuristic:
                label = f"Q{qi} R{run}/{args.runs} HEURISTIC"
                print(f"\n{THIN_SEP}")
                print(f"  [{label}] {query[:65]}")
                print(THIN_SEP)

                result = run_heuristic_benchmark(query, top_k=args.top_k)
                heuristic_results.append(result)
                all_results.append(result)
                print_single_run(result)

    # ── Aggregate summaries ───────────────────────────────────────────────
    if agentic_results:
        print_aggregate_summary(agentic_results, strategy="agentic")
    if heuristic_results:
        print_aggregate_summary(heuristic_results, strategy="heuristic")
    if agentic_results and heuristic_results:
        print_comparison(agentic_results, heuristic_results)

    # ── CSV ────────────────────────────────────────────────────────────────
    if not args.no_csv and all_results:
        csv_path = Path(args.csv) if args.csv else (_BACKEND_DIR / "test" / "bench_results.csv")
        save_results_csv(all_results, csv_path)


if __name__ == "__main__":
    main()
