"""
inference.py — LongVALE Benchmark Inference Script
====================================================

Sends LongVALE test queries to the deployed API and records per-query
results (video_id, frame_id, score, timestamps, latency) to CSV.

Metrics calculation (MRR, HitRatio@5, Temporal IoU) is handled by a
separate eval.py script that reads the CSV output from this script.

Usage:
    cd data-processing

    # All 3 strategies, full dataset
    python -m src.evaluation.inference --strategy all

    # Batch run for parallel team execution (4 members)
    python -m src.evaluation.inference --strategy all --batch 1 --total_batches 4
    python -m src.evaluation.inference --strategy all --batch 2 --total_batches 4
    python -m src.evaluation.inference --strategy all --batch 3 --total_batches 4
    python -m src.evaluation.inference --strategy all --batch 4 --total_batches 4

    # Single strategy with custom settings
    python -m src.evaluation.inference --strategy agentic --top_k 5 --delay 0.3

    # Resume after interruption
    python -m src.evaluation.inference --strategy agentic --batch 1 --total_batches 4 --resume
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:
    print("httpx is required. Install with: pip install httpx")
    sys.exit(1)


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FMT = "[%(asctime)s] %(levelname)-8s %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt=DATE_FMT)
logger = logging.getLogger("inference")


# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_API_URL = "https://nlp4b.vercel.app"
# DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_CSV_PATH = "data/longvale_test.csv"
DEFAULT_OUTPUT_DIR = "output/evaluation"
DEFAULT_TOP_K = 5
DEFAULT_DELAY = 0.5
DEFAULT_TIMEOUT = 30
DEFAULT_TOTAL_BATCHES = 4

# Strategy → API endpoint mapping
STRATEGY_ENDPOINTS = {
    "agentic": "/search/agentic",
    "heuristic": "/search/heuristic",
    "both": "/search",
}


# ── CSV Loading ───────────────────────────────────────────────────────────────
def load_longvale_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load LongVALE test CSV and parse timestamp fields.

    Returns list of dicts with keys:
        query_idx, video_id, duration, timestamp_start, timestamp_end, sentences
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            ts = ast.literal_eval(row["time_stamp"])
            rows.append({
                "query_idx": idx,
                "video_id": row["video_id"],
                "duration": float(row["duration"]),
                "timestamp_start": float(ts[0]),
                "timestamp_end": float(ts[1]),
                "sentences": row["sentences"],
            })
    logger.info("Loaded %d queries from %s", len(rows), csv_path)
    return rows


def split_batch(rows: List[Dict], batch: int, total_batches: int) -> List[Dict]:
    """Split rows into equal-sized batches. batch is 1-indexed."""
    n = len(rows)
    chunk_size = math.ceil(n / total_batches)
    start = (batch - 1) * chunk_size
    end = min(start + chunk_size, n)
    subset = rows[start:end]
    logger.info(
        "Batch %d/%d: rows %d–%d (%d queries)",
        batch, total_batches, start, end - 1, len(subset),
    )
    return subset


# ── API Client ────────────────────────────────────────────────────────────────
def call_api(
    client: httpx.Client,
    api_url: str,
    query_text: str,
    strategy: str,
    top_k: int,
    timeout: float,
) -> Tuple[Optional[Dict[str, Any]], float, str]:
    """Call the search API and return (response_dict, latency_ms, error_msg).

    latency_ms is measured client-side (end-to-end including network).
    error_msg is empty string on success.
    """
    endpoint = STRATEGY_ENDPOINTS[strategy]
    url = f"{api_url}{endpoint}"
    payload = {"raw_query": query_text, "top_k": top_k}

    t_start = time.perf_counter()
    try:
        resp = client.post(url, json=payload, timeout=timeout)
        latency_ms = (time.perf_counter() - t_start) * 1000

        if resp.status_code != 200:
            return None, latency_ms, f"HTTP {resp.status_code}: {resp.text[:200]}"

        data = resp.json()
        return data, latency_ms, ""

    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - t_start) * 1000
        return None, latency_ms, "TIMEOUT"

    except Exception as exc:
        latency_ms = (time.perf_counter() - t_start) * 1000
        return None, latency_ms, f"ERROR: {exc}"


# ── Result Row Builder ────────────────────────────────────────────────────────
def _csv_header(top_k: int) -> List[str]:
    """Build CSV header row."""
    base = [
        "query_idx",
        "video_id_gt",
        "timestamp_start_gt",
        "timestamp_end_gt",
        "query_text",
        "strategy",
        "latency_server_total_ms",
    ]
    for i in range(1, top_k + 1):
        base.append(f"keyframe_{i}")
    return base


def build_result_row(
    query_row: Dict[str, Any],
    strategy: str,
    top_k: int,
    response: Optional[Dict[str, Any]],
    latency_ms: float,
    error: str,
) -> Dict[str, Any]:
    """Build a flat dict representing one CSV row."""
    row: Dict[str, Any] = {
        "query_idx": query_row["query_idx"],
        "video_id_gt": query_row["video_id"],
        "timestamp_start_gt": query_row["timestamp_start"],
        "timestamp_end_gt": query_row["timestamp_end"],
        "query_text": query_row["sentences"],
        "strategy": strategy,
        "latency_server_total_ms": 0.0,
    }

    # Fill result columns with empty defaults
    for i in range(1, top_k + 1):
        row[f"keyframe_{i}"] = ""

    if response is None:
        return row

    # Server-reported latency
    latency_data = response.get("latency_ms", {})
    row["latency_server_total_ms"] = latency_data.get("total_ms", 0.0)

    results = response.get("results", [])

    for i, item in enumerate(results[:top_k], start=1):
        vid = item.get("video_id", "")
        fid = item.get("frame_id", "")
        if vid and str(fid):
            row[f"keyframe_{i}"] = f"{vid}_{fid}"

    # For business metrics array we inject back error and num_results + total latency silently into row
    row["_error"] = error
    row["_num_results"] = len(results)
    row["_latency_total_ms"] = round(latency_ms, 2)

    return row


# ── CSV I/O ───────────────────────────────────────────────────────────────────
def output_filename(strategy: str, batch: int) -> str:
    """Generate output filename."""
    if batch > 0:
        return f"inference_results_{strategy}_batch{batch}.csv"
    return f"inference_results_{strategy}.csv"


def save_csv(
    rows: List[Dict[str, Any]],
    filepath: str,
    header: List[str],
) -> None:
    """Write result rows to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d rows to %s", len(rows), filepath)


def load_resume_state(filepath: str) -> set:
    """Load already-completed query_idx values from existing CSV."""
    done = set()
    if not os.path.isfile(filepath):
        return done
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = row.get("query_idx", "")
                if idx:
                    done.add(int(idx))
        logger.info("Resume: found %d already-completed queries in %s", len(done), filepath)
    except Exception as exc:
        logger.warning("Could not load resume state from %s: %s", filepath, exc)
    return done


def load_existing_rows(filepath: str, header: List[str]) -> List[Dict[str, Any]]:
    """Load existing rows from CSV for resume mode."""
    rows = []
    if not os.path.isfile(filepath):
        return rows
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        pass
    return rows


# ── Business Metrics & Summary ────────────────────────────────────────────────
def compute_business_metrics(
    results: List[Dict[str, Any]],
    total_runtime_sec: float,
) -> Dict[str, Any]:
    """Compute business metrics from result rows."""
    total = len(results)
    if total == 0:
        return {}

    # Latency distribution (client-side)
    latencies = [
        float(r["_latency_total_ms"])
        for r in results
        if r.get("_error") == ""
    ]

    # Error analysis
    errors = [r for r in results if r.get("_error") != ""]
    timeouts = [r for r in errors if "TIMEOUT" in str(r.get("_error", ""))]
    empty_results = [r for r in results if r.get("_error") == "" and int(r.get("_num_results", 0)) == 0]

    # Latency gap (client vs server)
    latency_gaps = []
    for r in results:
        if r.get("_error") == "" and float(r.get("latency_server_total_ms", 0)) > 0:
            gap = float(r["_latency_total_ms"]) - float(r["latency_server_total_ms"])
            latency_gaps.append(gap)

    cold_starts = [g for g in latency_gaps if g > float(statistics.mean(latency_gaps)) * 2] if latency_gaps else []

    metrics: Dict[str, Any] = {
        "error_rate_pct": round(len(errors) / total * 100, 2),
        "timeout_rate_pct": round(len(timeouts) / total * 100, 2),
        "empty_result_rate_pct": round(len(empty_results) / total * 100, 2),
    }

    if latencies:
        sorted_lat = sorted(latencies)
        metrics.update({
            "latency_mean_ms": round(statistics.mean(sorted_lat), 2),
            "latency_p50_ms": round(sorted_lat[len(sorted_lat) // 2], 2),
            "latency_p95_ms": round(sorted_lat[int(len(sorted_lat) * 0.95)], 2),
            "latency_p99_ms": round(sorted_lat[int(len(sorted_lat) * 0.99)], 2),
            "latency_min_ms": round(sorted_lat[0], 2),
            "latency_max_ms": round(sorted_lat[-1], 2),
        })

    if latency_gaps:
        metrics["network_overhead_mean_ms"] = round(statistics.mean(latency_gaps), 2)
        metrics["cold_start_rate_pct"] = round(len(cold_starts) / len(latency_gaps) * 100, 2)

    metrics["throughput_qps"] = round(total / total_runtime_sec, 4) if total_runtime_sec > 0 else 0
    metrics["total_runtime_sec"] = round(total_runtime_sec, 2)

    return metrics


def save_summary_json(
    strategy: str,
    batch: int,
    total_batches: int,
    api_url: str,
    top_k: int,
    results: List[Dict[str, Any]],
    total_runtime_sec: float,
    output_dir: str,
) -> None:
    """Save summary JSON with business metrics."""
    total = len(results)
    successful = sum(1 for r in results if r.get("_error") == "")
    failed = total - successful
    empty = sum(1 for r in results if r.get("_error") == "" and int(r.get("_num_results", 0)) == 0)

    summary = {
        "strategy": strategy,
        "batch": batch,
        "total_batches": total_batches,
        "api_url": api_url,
        "top_k": top_k,
        "total_queries": total,
        "successful_queries": successful,
        "failed_queries": failed,
        "empty_result_queries": empty,
        "business_metrics": compute_business_metrics(results, total_runtime_sec),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    suffix = f"_batch{batch}" if batch > 0 else ""
    filepath = os.path.join(output_dir, f"inference_summary_{strategy}{suffix}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary saved to %s", filepath)


# ── Progress Display ──────────────────────────────────────────────────────────
def _progress_bar(current: int, total: int, width: int = 40) -> str:
    """Simple text progress bar."""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({pct:.1%})"


# ── Main Inference Loop ──────────────────────────────────────────────────────
def run_inference(
    queries: List[Dict[str, Any]],
    strategy: str,
    api_url: str,
    top_k: int,
    delay: float,
    timeout: float,
    output_dir: str,
    batch: int,
    total_batches: int,
    resume: bool,
) -> None:
    """Run inference for a single strategy."""
    header = _csv_header(top_k)
    fname = output_filename(strategy, batch)
    filepath = os.path.join(output_dir, fname)

    # Resume support
    existing_rows: List[Dict[str, Any]] = []
    done_indices: set = set()
    if resume:
        done_indices = load_resume_state(filepath)
        existing_rows = load_existing_rows(filepath, header)
        logger.info("Resuming: %d queries already done, %d remaining",
                     len(done_indices), len(queries) - len(done_indices))

    results = list(existing_rows)
    total = len(queries)
    pending = [q for q in queries if q["query_idx"] not in done_indices]

    logger.info(
        "═" * 70 + "\n"
        "  STRATEGY: %s | API: %s | top_k: %d\n"
        "  Total: %d | Pending: %d | Delay: %.1fs\n" +
        "═" * 70,
        strategy, api_url, top_k, total, len(pending), delay,
    )

    t_start_total = time.perf_counter()

    with httpx.Client() as client:
        for i, query_row in enumerate(pending):
            idx = query_row["query_idx"]

            # Call API
            response, latency_ms, error = call_api(
                client=client,
                api_url=api_url,
                query_text=query_row["sentences"],
                strategy=strategy,
                top_k=top_k,
                timeout=timeout,
            )

            # Build result row
            row = build_result_row(query_row, strategy, top_k, response, latency_ms, error)
            
            # Status indicator
            status = "✓" if error == "" else f"✗ {error[:40]}"
            num_res = row["_num_results"]
            progress = _progress_bar(i + 1, len(pending))
            print(
                f"\r  {progress}  idx={idx}  lat={latency_ms:.0f}ms  "
                f"n={num_res}  {status}          ",
                end="", flush=True,
            )

            # Keep only the CSV requested headers by stripping "_" prefixed private fields when adding to results
            # We must keep them for JSON summary, so we save a copy of the dict with them
            results.append(row)
            
            # Create a clean version just for CSV saving
            clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
            
            # Periodic save (every 50 queries)
            if (i + 1) % 50 == 0:
                clean_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
                save_csv(clean_results, filepath, header)

            # Delay between requests
            if i < len(pending) - 1:
                time.sleep(delay)

    print()  # newline after progress bar
    total_runtime = time.perf_counter() - t_start_total

    # Final save
    clean_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    save_csv(clean_results, filepath, header)
    save_summary_json(
        strategy=strategy,
        batch=batch,
        total_batches=total_batches,
        api_url=api_url,
        top_k=top_k,
        results=results,
        total_runtime_sec=total_runtime,
        output_dir=output_dir,
    )

    # Print quick summary
    successful = sum(1 for r in results if r.get("_error") == "")
    failed = len(results) - successful
    metrics = compute_business_metrics(results, total_runtime)

    logger.info(
        "─" * 70 + "\n"
        "  DONE [%s] — %d ok / %d fail / %.1fs total\n"
        "  Latency: mean=%.0fms  p50=%.0fms  p95=%.0fms  p99=%.0fms\n"
        "  Error rate: %.1f%%  |  Empty result rate: %.1f%%  |  QPS: %.3f\n" +
        "─" * 70,
        strategy, successful, failed, total_runtime,
        metrics.get("latency_mean_ms", 0),
        metrics.get("latency_p50_ms", 0),
        metrics.get("latency_p95_ms", 0),
        metrics.get("latency_p99_ms", 0),
        metrics.get("error_rate_pct", 0),
        metrics.get("empty_result_rate_pct", 0),
        metrics.get("throughput_qps", 0),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LongVALE Benchmark Inference — run queries against deployed API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.evaluation.inference --strategy all\n"
            "  python -m src.evaluation.inference --strategy agentic --batch 1 --total_batches 4\n"
            "  python -m src.evaluation.inference --strategy both --top_k 10 --delay 0.3\n"
            "  python -m src.evaluation.inference --strategy heuristic --resume\n"
        ),
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV_PATH,
        help=f"Path to LongVALE test CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--api_url", default=DEFAULT_API_URL,
        help=f"API base URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--strategy", default="all",
        choices=["agentic", "heuristic", "both", "all"],
        help="Search strategy: agentic, heuristic, both, or all (default: all)",
    )
    parser.add_argument(
        "--top_k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of results to request (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--batch", type=int, default=0,
        help="Batch index, 1-based. 0 = run all rows (default: 0)",
    )
    parser.add_argument(
        "--total_batches", type=int, default=DEFAULT_TOTAL_BATCHES,
        help=f"Total number of batches for splitting (default: {DEFAULT_TOTAL_BATCHES})",
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Seconds between API calls (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout per request in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max number of queries to run. 0 = no limit (default: 0)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last completed query (skip already-done rows)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate
    if args.batch < 0 or args.batch > args.total_batches:
        logger.error("--batch must be 0 (all) or between 1 and --total_batches (%d)", args.total_batches)
        sys.exit(1)

    # Load data
    all_queries = load_longvale_csv(args.csv)

    # Batch slicing
    if args.batch > 0:
        queries = split_batch(all_queries, args.batch, args.total_batches)
    else:
        queries = all_queries

    # Limit
    if args.limit > 0:
        queries = queries[:args.limit]
        logger.info("Limited to %d queries (--limit %d)", len(queries), args.limit)

    # Determine strategies
    strategies = list(STRATEGY_ENDPOINTS.keys()) if args.strategy == "all" else [args.strategy]

    logger.info(
        "╔" + "═" * 68 + "╗\n"
        "║  LongVALE Inference Benchmark                                      ║\n"
        "║  API: %-60s ║\n"
        "║  Strategies: %-53s ║\n"
        "║  Queries: %-57s ║\n"
        "║  Batch: %-59s ║\n"
        "╚" + "═" * 68 + "╝",
        args.api_url,
        ", ".join(strategies),
        f"{len(queries)} (of {len(all_queries)} total)",
        f"{args.batch}/{args.total_batches}" if args.batch > 0 else "all",
    )

    # Health check
    try:
        resp = httpx.get(f"{args.api_url}/health", timeout=10)
        health = resp.json()
        logger.info("API health: %s", health)
    except Exception as exc:
        logger.error("API health check failed: %s — is the API running?", exc)
        sys.exit(1)

    # Run strategies
    for strat in strategies:
        run_inference(
            queries=queries,
            strategy=strat,
            api_url=args.api_url,
            top_k=args.top_k,
            delay=args.delay,
            timeout=args.timeout,
            output_dir=args.output_dir,
            batch=args.batch,
            total_batches=args.total_batches,
            resume=args.resume,
        )

    logger.info("All strategies completed.")


if __name__ == "__main__":
    main()
