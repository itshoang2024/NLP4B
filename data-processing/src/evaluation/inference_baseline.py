"""
inference_baseline.py — LongVALE Baseline Inference Script
===========================================================

Baseline retrieval: embed query text via SigLIP (/embed/visual) then
perform a dense-only nearest-neighbour search on the Qdrant `keyframe-dense`
vector.  Output CSV schema is identical to inference.py so the same eval.py
can be used without modification.

Usage:
    cd data-processing

    # Full dataset, single machine
    python -m src.evaluation.inference_baseline

    # Quick smoke-test (first 10 queries)
    python -m src.evaluation.inference_baseline --limit 10

    # Parallel team execution (4 members)
    python -m src.evaluation.inference_baseline --batch 1 --total_batches 4
    python -m src.evaluation.inference_baseline --batch 2 --total_batches 4
    python -m src.evaluation.inference_baseline --batch 3 --total_batches 4
    python -m src.evaluation.inference_baseline --batch 4 --total_batches 4

    # Resume after interruption
    python -m src.evaluation.inference_baseline --batch 1 --total_batches 4 --resume

Environment (read from backend/.env or data-processing/.env):
    QDRANT_URL        — Qdrant cloud / local URL
    QDRANT_API_KEY    — Qdrant API key
    EMBEDDING_API_BASE_URL — Base URL for the embedding service
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

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

try:
    from qdrant_client import QdrantClient
    from qdrant_client import models as qdrant_models
except ImportError:
    print("qdrant-client is required. Install with: pip install qdrant-client")
    sys.exit(1)


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FMT = "[%(asctime)s] %(levelname)-8s %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt=DATE_FMT)
logger = logging.getLogger("inference_baseline")


# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_CSV_PATH = "data/longvale_test.csv"
DEFAULT_OUTPUT_DIR = "output/evaluation"
DEFAULT_TOP_K = 5
DEFAULT_DELAY = 0.1          # much faster than the agentic pipeline
DEFAULT_TIMEOUT = 30
DEFAULT_TOTAL_BATCHES = 4
DEFAULT_COLLECTION = "keyframes_v1"
VEC_DENSE = "keyframe-dense"  # SigLIP 1152d vector name in Qdrant

EMBED_VISUAL_PATH = "/embed/visual"

STRATEGY_NAME = "baseline_siglip"


# ── .env loader ───────────────────────────────────────────────────────────────
def _load_env() -> None:
    if load_dotenv is None:
        return
    for candidate in [
        Path(__file__).resolve().parent.parent.parent.parent / "backend" / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            logger.info("Loaded .env from %s", candidate)
            return


# ── CSV Loading ───────────────────────────────────────────────────────────────
def load_longvale_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load LongVALE test CSV and parse timestamp fields."""
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
    """Split rows into equal-sized batches (1-indexed)."""
    n = len(rows)
    chunk_size = math.ceil(n / total_batches)
    start = (batch - 1) * chunk_size
    end = min(start + chunk_size, n)
    subset = rows[start:end]
    logger.info(
        "Batch %d/%d: rows %d-%d (%d queries)",
        batch, total_batches, start, end - 1, len(subset),
    )
    return subset


# ── Embedding: SigLIP via HTTP ────────────────────────────────────────────────
def embed_visual(text: str, embed_api_url: str, timeout: float) -> Optional[List[float]]:
    """Call /embed/visual → returns SigLIP 1152d embedding, or None on failure."""
    url = f"{embed_api_url.rstrip('/')}{EMBED_VISUAL_PATH}"
    try:
        resp = httpx.post(url, json={"text": text.strip()}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            logger.warning("Unexpected visual embedding response shape: %s", type(embedding))
            return None
        return [float(x) for x in embedding]
    except Exception as exc:
        logger.warning("embed_visual failed: %s", exc)
        return None


# ── Qdrant Dense Search ───────────────────────────────────────────────────────
def search_dense(
    client: QdrantClient,
    collection: str,
    query_vector: List[float],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Run dense ANN search on keyframe-dense vector."""
    try:
        if hasattr(client, "query_points"):
            results = client.query_points(
                collection_name=collection,
                query=query_vector,
                using=VEC_DENSE,
                limit=top_k,
                with_payload=["video_id", "frame_idx"],
                with_vectors=False,
            )
            points = results.points if hasattr(results, "points") else results[0] if isinstance(results, tuple) else results
        else:
            raise AttributeError("query_points not found")
    except (AttributeError, TypeError):
        results = client.search(
            collection_name=collection,
            query_vector=(VEC_DENSE, query_vector),
            limit=top_k,
            with_payload=["video_id", "frame_idx"],
            with_vectors=False,
        )
        points = results

    hits = []
    for point in points:
        payload = getattr(point, "payload", {}) or {}
        hits.append({
            "video_id": payload.get("video_id", ""),
            "frame_idx": payload.get("frame_idx", ""),
            "score": point.score,
        })
    return hits


# ── CSV Schema (identical to inference.py, strategy=baseline_siglip) ──────────
def _csv_header(top_k: int) -> List[str]:
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
    top_k: int,
    hits: List[Dict[str, Any]],
    embed_latency_ms: float,
    search_latency_ms: float,
    error: str,
) -> Dict[str, Any]:
    """Build flat CSV row, with private _* fields for business metrics."""
    row: Dict[str, Any] = {
        "query_idx": query_row["query_idx"],
        "video_id_gt": query_row["video_id"],
        "timestamp_start_gt": query_row["timestamp_start"],
        "timestamp_end_gt": query_row["timestamp_end"],
        "query_text": query_row["sentences"],
        "strategy": STRATEGY_NAME,
        "latency_server_total_ms": round(embed_latency_ms + search_latency_ms, 2),
    }
    for i in range(1, top_k + 1):
        row[f"keyframe_{i}"] = ""

    for i, hit in enumerate(hits[:top_k], start=1):
        vid = hit.get("video_id", "")
        fidx = hit.get("frame_idx", "")
        if vid and str(fidx):
            row[f"keyframe_{i}"] = f"{vid}_{fidx}"

    # Private fields for business metric computation (stripped before CSV write)
    row["_error"] = error
    row["_num_results"] = len(hits)
    row["_latency_total_ms"] = round(embed_latency_ms + search_latency_ms, 2)
    row["_embed_ms"] = round(embed_latency_ms, 2)
    row["_search_ms"] = round(search_latency_ms, 2)

    return row


# ── Resume Support ────────────────────────────────────────────────────────────
def load_resume_state(filepath: str) -> set:
    done: set = set()
    if not os.path.isfile(filepath):
        return done
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = row.get("query_idx", "")
                if idx:
                    done.add(int(idx))
        logger.info("Resume: %d already-done queries in %s", len(done), filepath)
    except Exception as exc:
        logger.warning("Could not load resume state: %s", exc)
    return done



def _restore_private_fields(row: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    """Restore private runtime fields for rows reloaded from CSV in resume mode."""
    restored = dict(row)

    if "_error" not in restored:
        restored["_error"] = ""

    if "_num_results" not in restored:
        num_results = 0
        for i in range(1, top_k + 1):
            if str(restored.get(f"keyframe_{i}", "")).strip():
                num_results += 1
        restored["_num_results"] = num_results

    if "_latency_total_ms" not in restored:
        try:
            restored["_latency_total_ms"] = round(float(restored.get("latency_server_total_ms", 0) or 0), 2)
        except Exception:
            restored["_latency_total_ms"] = 0.0

    if "_embed_ms" not in restored:
        restored["_embed_ms"] = 0.0

    if "_search_ms" not in restored:
        restored["_search_ms"] = 0.0

    return restored


def load_existing_rows(filepath: str, top_k: int) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.isfile(filepath):
        return rows
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(_restore_private_fields(row, top_k))
    except Exception:
        pass
    return rows


# ── CSV Save ──────────────────────────────────────────────────────────────────
def save_csv(rows: List[Dict[str, Any]], filepath: str, header: List[str]) -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(clean)
    logger.info("Saved %d rows to %s", len(clean), filepath)


# ── Business Metrics ──────────────────────────────────────────────────────────
def compute_business_metrics(
    results: List[Dict[str, Any]],
    total_runtime_sec: float,
) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    latencies = [float(r["_latency_total_ms"]) for r in results if r.get("_error") == ""]
    errors = [r for r in results if r.get("_error") != ""]
    empty = [r for r in results if r.get("_error") == "" and int(r.get("_num_results", 0)) == 0]

    metrics: Dict[str, Any] = {
        "error_rate_pct": round(len(errors) / total * 100, 2),
        "empty_result_rate_pct": round(len(empty) / total * 100, 2),
    }
    if latencies:
        sl = sorted(latencies)
        metrics.update({
            "latency_mean_ms": round(statistics.mean(sl), 2),
            "latency_p50_ms": round(sl[len(sl) // 2], 2),
            "latency_p95_ms": round(sl[int(len(sl) * 0.95)], 2),
            "latency_p99_ms": round(sl[int(len(sl) * 0.99)], 2),
            "latency_min_ms": round(sl[0], 2),
            "latency_max_ms": round(sl[-1], 2),
        })
    metrics["throughput_qps"] = round(total / total_runtime_sec, 4) if total_runtime_sec > 0 else 0
    metrics["total_runtime_sec"] = round(total_runtime_sec, 2)
    return metrics


def save_summary_json(
    batch: int,
    total_batches: int,
    results: List[Dict[str, Any]],
    total_runtime_sec: float,
    output_dir: str,
) -> None:
    total = len(results)
    successful = sum(1 for r in results if r.get("_error") == "")
    empty = sum(1 for r in results if r.get("_error") == "" and int(r.get("_num_results", 0)) == 0)
    suffix = f"_batch{batch}" if batch > 0 else ""

    summary = {
        "strategy": STRATEGY_NAME,
        "batch": batch,
        "total_batches": total_batches,
        "total_queries": total,
        "successful_queries": successful,
        "failed_queries": total - successful,
        "empty_result_queries": empty,
        "business_metrics": compute_business_metrics(results, total_runtime_sec),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    filepath = os.path.join(output_dir, f"inference_summary_{STRATEGY_NAME}{suffix}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary saved to %s", filepath)


# ── Progress Bar ──────────────────────────────────────────────────────────────
def _progress_bar(current: int, total: int, width: int = 40) -> str:
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({pct:.1%})"


# ── Main Inference Loop ───────────────────────────────────────────────────────
def run_baseline(
    queries: List[Dict[str, Any]],
    qdrant_client: QdrantClient,
    embed_api_url: str,
    top_k: int,
    delay: float,
    embed_timeout: float,
    qdrant_timeout: float,
    collection: str,
    output_dir: str,
    batch: int,
    total_batches: int,
    resume: bool,
) -> None:
    header = _csv_header(top_k)
    suffix = f"_batch{batch}" if batch > 0 else ""
    filepath = os.path.join(output_dir, f"inference_results_{STRATEGY_NAME}{suffix}.csv")

    existing_rows: List[Dict[str, Any]] = []
    done_indices: set = set()
    if resume:
        done_indices = load_resume_state(filepath)
        existing_rows = load_existing_rows(filepath, top_k)

    results = list(existing_rows)
    pending = [q for q in queries if q["query_idx"] not in done_indices]

    logger.info(
        "Baseline SigLIP | top_k=%d | Collection=%s\n"
        "  Total: %d | Pending: %d | Delay: %.2fs",
        top_k, collection, len(queries), len(pending), delay,
    )

    t_start_total = time.perf_counter()

    for i, query_row in enumerate(pending):
        idx = query_row["query_idx"]
        query_text = query_row["sentences"]
        error = ""
        hits: List[Dict[str, Any]] = []
        embed_ms = 0.0
        search_ms = 0.0

        # Step 1: Embed via SigLIP
        t0 = time.perf_counter()
        vector = embed_visual(query_text, embed_api_url, embed_timeout)
        embed_ms = (time.perf_counter() - t0) * 1000

        if vector is None:
            error = "EMBED_FAIL"
        else:
            # Step 2: Qdrant dense search
            t1 = time.perf_counter()
            try:
                hits = search_dense(qdrant_client, collection, vector, top_k)
                search_ms = (time.perf_counter() - t1) * 1000
            except Exception as exc:
                search_ms = (time.perf_counter() - t1) * 1000
                error = f"QDRANT_ERR: {str(exc)[:60]}"

        row = build_result_row(query_row, top_k, hits, embed_ms, search_ms, error)
        results.append(row)

        status = "ok" if error == "" else error[:30]
        print(
            f"\r  {_progress_bar(i + 1, len(pending))}  "
            f"idx={idx}  emb={embed_ms:.0f}ms  srch={search_ms:.0f}ms  "
            f"n={len(hits)}  {status}          ",
            end="", flush=True,
        )

        if (i + 1) % 50 == 0:
            save_csv(results, filepath, header)

        if i < len(pending) - 1:
            time.sleep(delay)

    print()
    total_runtime = time.perf_counter() - t_start_total

    save_csv(results, filepath, header)
    save_summary_json(batch, total_batches, results, total_runtime, output_dir)

    successful = sum(1 for r in results if r.get("_error") == "")
    failed = len(results) - successful
    metrics = compute_business_metrics(results, total_runtime)
    logger.info(
        "DONE [%s] — %d ok / %d fail / %.1fs total\n"
        "  Latency: mean=%.0fms  p50=%.0fms  p95=%.0fms\n"
        "  Error rate: %.1f%%  |  Empty: %.1f%%  |  QPS: %.3f",
        STRATEGY_NAME, successful, failed, total_runtime,
        metrics.get("latency_mean_ms", 0),
        metrics.get("latency_p50_ms", 0),
        metrics.get("latency_p95_ms", 0),
        metrics.get("error_rate_pct", 0),
        metrics.get("empty_result_rate_pct", 0),
        metrics.get("throughput_qps", 0),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LongVALE Baseline Inference — SigLIP embed + Qdrant dense search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH,
                        help=f"LongVALE test CSV (default: {DEFAULT_CSV_PATH})")
    parser.add_argument("--embed_api_url", default=None,
                        help="Embedding service base URL (fallback: EMBEDDING_API_BASE_URL env)")
    parser.add_argument("--qdrant_url", default=None,
                        help="Qdrant URL (fallback: QDRANT_URL env)")
    parser.add_argument("--qdrant_key", default=None,
                        help="Qdrant API key (fallback: QDRANT_API_KEY env)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help=f"Results per query (default: {DEFAULT_TOP_K})")
    parser.add_argument("--batch", type=int, default=0,
                        help="Batch index 1-based; 0 = all rows (default: 0)")
    parser.add_argument("--total_batches", type=int, default=DEFAULT_TOTAL_BATCHES,
                        help=f"Total batches (default: {DEFAULT_TOTAL_BATCHES})")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max queries to run; 0 = no limit (default: 0)")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"Seconds between requests (default: {DEFAULT_DELAY})")
    parser.add_argument("--embed_timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"HTTP timeout for embed API (default: {DEFAULT_TIMEOUT}s)")
    parser.add_argument("--qdrant_timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"Qdrant client timeout (default: {DEFAULT_TIMEOUT}s)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed queries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_env()

    # Resolve credentials (CLI args override env)
    embed_api_url = (
        args.embed_api_url
        or os.getenv("EMBEDDING_API_BASE_URL", "").strip().rstrip("/")
    )
    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL", "").strip().rstrip(",")
    qdrant_key = args.qdrant_key or os.getenv("QDRANT_API_KEY", "").strip()

    if not embed_api_url:
        logger.error(
            "Embedding API URL not set. "
            "Use --embed_api_url or set EMBEDDING_API_BASE_URL in .env"
        )
        sys.exit(1)
    if not qdrant_url or not qdrant_key:
        logger.error("QDRANT_URL and QDRANT_API_KEY must be set (env or CLI flags)")
        sys.exit(1)

    # Validate
    if args.batch < 0 or (args.batch > args.total_batches):
        logger.error("--batch must be 0 (all) or between 1 and --total_batches (%d)", args.total_batches)
        sys.exit(1)

    # Connect to Qdrant
    logger.info("Connecting to Qdrant at %s ...", qdrant_url)
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_key,
        timeout=args.qdrant_timeout,
    )
    try:
        info = qdrant_client.get_collection(args.collection)
        logger.info("Collection '%s' OK — %d points", args.collection, info.points_count)
    except Exception as exc:
        logger.error("Cannot connect to Qdrant collection '%s': %s", args.collection, exc)
        sys.exit(1)

    # Load dataset
    all_queries = load_longvale_csv(args.csv)

    # Batch slicing
    queries = split_batch(all_queries, args.batch, args.total_batches) if args.batch > 0 else all_queries

    # Limit
    if args.limit > 0:
        queries = queries[:args.limit]
        logger.info("Limited to %d queries (--limit %d)", len(queries), args.limit)

    logger.info(
        "Embedding API: %s | Collection: %s\n"
        "  Queries: %d (of %d total) | Batch: %s | top_k: %d",
        embed_api_url, args.collection,
        len(queries), len(all_queries),
        f"{args.batch}/{args.total_batches}" if args.batch > 0 else "all",
        args.top_k,
    )

    run_baseline(
        queries=queries,
        qdrant_client=qdrant_client,
        embed_api_url=embed_api_url,
        top_k=args.top_k,
        delay=args.delay,
        embed_timeout=args.embed_timeout,
        qdrant_timeout=args.qdrant_timeout,
        collection=args.collection,
        output_dir=args.output_dir,
        batch=args.batch,
        total_batches=args.total_batches,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
