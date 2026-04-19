"""
eval.py — LongVALE Benchmark Evaluation Script
================================================

Reads inference result CSVs produced by inference.py, fetches predicted
timestamps from Qdrant (using deterministic UUID5 point IDs), then computes:
  - MRR@5             (Strict Hit: same video_id AND timestamp overlap with GT)
  - HitRatio@5        (Strict Hit: same video_id AND timestamp overlap with GT)
  - Temporal IoU R@1  (IoU >= θ for θ ∈ {0.3, 0.5, 0.7}, Top-1 only)

Outputs per-query detail CSV and a summary JSON/console table.

Usage:
    cd data-processing

    # Evaluate a single strategy file
    python -m src.evaluation.eval --input output/evaluation/inference_results_agentic.csv

    # Evaluate all CSVs in a folder (auto-merges batches)
    python -m src.evaluation.eval --input_dir output/evaluation

    # Evaluate and compare multiple strategies side-by-side
    python -m src.evaluation.eval --input_dir output/evaluation --compare

    # Dry-run without Qdrant (skips IoU, treats all keyframe hits as video-only)
    python -m src.evaluation.eval --input output/evaluation/inference_results_agentic.csv --no_qdrant

Notes for eval.py developer context:
  - "Hit" definition: STRICT — same video_id AND predicted timestamp overlaps GT interval.
  - Temporal IoU only computed for keyframe_1 (Top-1).
  - Predicted timestamps fetched from Qdrant via client.retrieve() using
    deterministic UUID5 IDs: uuid5(NAMESPACE_URL, f"{video_id}_{frame_id}").
  - GT timestamps are in columns: timestamp_start_gt, timestamp_end_gt.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

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
logger = logging.getLogger("eval")

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_COLLECTION = "keyframes_v1"
DEFAULT_TOP_K = 5
IOU_THRESHOLDS = [0.3, 0.5, 0.7]
QDRANT_BATCH_SIZE = 500
EPS = 1e-9  # Avoid zero-division on degenerate GT intervals


# ── Load .env ─────────────────────────────────────────────────────────────────
def _load_env() -> None:
    """Try to load .env from backend/ or data-processing/ root."""
    for candidate in [
        Path(__file__).resolve().parent.parent.parent.parent / "backend" / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]:
        if candidate.exists():
            if load_dotenv:
                load_dotenv(dotenv_path=candidate, override=False)
                logger.info("Loaded .env from %s", candidate)
            return


# ── Deterministic ID (must match qdrant_upsert.py) ───────────────────────────
def deterministic_id(video_id: str, frame_idx: int) -> str:
    """UUID5 point ID — mirrors qdrant_upsert.deterministic_id()."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_{frame_idx}"))


# ── Parse keyframe_i column ("video_id_frame_idx") ───────────────────────────
def parse_keyframe(value: str) -> Optional[Tuple[str, int]]:
    """Split 'video_id_frame_idx' → (video_id, frame_idx).

    Frame idx is guaranteed to be the last underscore-delimited token.
    Returns None if value is empty or malformed.
    """
    if not value:
        return None
    parts = value.rsplit("_", 1)
    if len(parts) != 2:
        return None
    try:
        return parts[0], int(parts[1])
    except ValueError:
        return None


# ── CSV Loading ───────────────────────────────────────────────────────────────
def load_inference_csv(path: str) -> List[Dict[str, Any]]:
    """Load a single inference result CSV."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    logger.info("Loaded %d rows from %s", len(rows), path)
    return rows


def load_inference_dir(directory: str) -> List[Dict[str, Any]]:
    """Auto-discover and merge all inference_results_*.csv in a directory."""
    all_rows: List[Dict[str, Any]] = []
    csv_files = sorted(Path(directory).glob("inference_results_*.csv"))
    if not csv_files:
        logger.error("No inference_results_*.csv found in %s", directory)
        sys.exit(1)
    for p in csv_files:
        all_rows.extend(load_inference_csv(str(p)))
    logger.info("Total rows loaded: %d (from %d files)", len(all_rows), len(csv_files))
    return all_rows


# ── Qdrant Timestamp Fetcher ──────────────────────────────────────────────────
def build_qdrant_client() -> QdrantClient:
    """Build Qdrant client from env vars."""
    url = os.getenv("QDRANT_URL", "").strip().rstrip(",")
    key = os.getenv("QDRANT_API_KEY", "").strip()
    if not url or not key:
        raise EnvironmentError(
            "QDRANT_URL and QDRANT_API_KEY must be set (check your .env)"
        )
    return QdrantClient(url=url, api_key=key)


def fetch_timestamps_from_qdrant(
    client: QdrantClient,
    keyframe_pairs: List[Tuple[str, int]],  # [(video_id, frame_idx), ...]
    collection: str,
    batch_size: int = QDRANT_BATCH_SIZE,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Batch-retrieve timestamp_start/end for a list of (video_id, frame_idx).

    Returns dict keyed by "video_id_frame_idx" → (start, end).
    Both None if the point has no timestamp fields in its payload.
    """
    # Deduplicate
    unique_pairs = list(set(keyframe_pairs))
    logger.info("Fetching timestamps for %d unique keyframes from Qdrant...", len(unique_pairs))

    id_to_key: Dict[str, str] = {}
    for vid, fidx in unique_pairs:
        pid = deterministic_id(vid, fidx)
        id_to_key[pid] = f"{vid}_{fidx}"

    point_ids = list(id_to_key.keys())
    ts_cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    # Batch retrieve
    n_found = n_missing = 0
    for i in range(0, len(point_ids), batch_size):
        batch_ids = point_ids[i : i + batch_size]
        try:
            points = client.retrieve(
                collection_name=collection,
                ids=batch_ids,
                with_payload=["timestamp_start", "timestamp_end"],
                with_vectors=False,
            )
            for point in points:
                key = id_to_key[str(point.id)]
                payload = point.payload or {}
                ts_start = payload.get("timestamp_start")
                ts_end = payload.get("timestamp_end")
                ts_cache[key] = (
                    float(ts_start) if ts_start is not None else None,
                    float(ts_end) if ts_end is not None else None,
                )
                n_found += 1
        except Exception as exc:
            logger.warning("Qdrant batch retrieve failed (batch %d): %s", i // batch_size, exc)

    # Mark missing pairs
    for vid, fidx in unique_pairs:
        key = f"{vid}_{fidx}"
        if key not in ts_cache:
            ts_cache[key] = (None, None)
            n_missing += 1

    logger.info(
        "Timestamp fetch done: %d found, %d missing payload",
        n_found, n_missing,
    )
    return ts_cache


# ── IoU Calculation ───────────────────────────────────────────────────────────
def temporal_iou(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
) -> float:
    """Compute Temporal IoU between two 1D intervals."""
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)
    if intersection == 0.0:
        return 0.0
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    return intersection / max(union, EPS)


# ── Strict Hit Check ──────────────────────────────────────────────────────────
def is_strict_hit(
    pred_video_id: str,
    pred_start: Optional[float],
    pred_end: Optional[float],
    gt_video_id: str,
    gt_start: float,
    gt_end: float,
) -> bool:
    """Strict Hit: same video AND predicted period overlaps GT interval.

    Falls back to video-only match if timestamp unavailable.
    """
    if pred_video_id != gt_video_id:
        return False
    # If no timestamp available, treat as loose (video-only) hit
    if pred_start is None or pred_end is None:
        return True
    # Overlap check: not (pred_end <= gt_start or pred_start >= gt_end)
    return not (pred_end <= gt_start or pred_start >= gt_end)


# ── Per-Query Evaluation ──────────────────────────────────────────────────────
def evaluate_query(
    row: Dict[str, Any],
    ts_cache: Dict[str, Tuple[Optional[float], Optional[float]]],
    top_k: int,
) -> Dict[str, Any]:
    """Compute per-query RR, hit, IoU metrics."""
    gt_video_id = row["video_id_gt"]
    gt_start = float(row["timestamp_start_gt"])
    gt_end = float(row["timestamp_end_gt"])
    strategy = row.get("strategy", "unknown")

    # ── Reciprocal Rank and HitRatio@K ─────────────────────────────────
    rr = 0.0
    hit_at_k = 0
    first_hit_rank = -1

    for rank in range(1, top_k + 1):
        kf_val = row.get(f"keyframe_{rank}", "").strip()
        parsed = parse_keyframe(kf_val)
        if parsed is None:
            continue
        vid, fidx = parsed
        pred_start, pred_end = ts_cache.get(f"{vid}_{fidx}", (None, None))

        if is_strict_hit(vid, pred_start, pred_end, gt_video_id, gt_start, gt_end):
            if first_hit_rank == -1:
                first_hit_rank = rank
                rr = 1.0 / rank
            hit_at_k = 1

    # ── Temporal IoU (Top-1 only) ───────────────────────────────────────
    iou_score = 0.0
    recall_at_1: Dict[str, int] = {}

    kf1_val = row.get("keyframe_1", "").strip()
    parsed_top1 = parse_keyframe(kf1_val)

    if parsed_top1 is not None:
        vid1, fidx1 = parsed_top1
        pred_start1, pred_end1 = ts_cache.get(f"{vid1}_{fidx1}", (None, None))

        if pred_start1 is not None and pred_end1 is not None:
            iou_score = temporal_iou(pred_start1, pred_end1, gt_start, gt_end)

    for theta in IOU_THRESHOLDS:
        key = f"recall_at_1_iou{int(theta * 10)}"
        recall_at_1[key] = 1 if iou_score >= theta else 0

    return {
        "query_idx": row.get("query_idx", ""),
        "video_id_gt": gt_video_id,
        "timestamp_start_gt": gt_start,
        "timestamp_end_gt": gt_end,
        "query_text": row.get("query_text", ""),
        "strategy": strategy,
        "latency_server_total_ms": row.get("latency_server_total_ms", ""),
        "keyframe_1": kf1_val,
        "first_hit_rank": first_hit_rank,
        "rr": round(rr, 6),
        "hit_at_k": hit_at_k,
        "iou_top1": round(iou_score, 6),
        **recall_at_1,
    }


# ── Aggregate Metrics ─────────────────────────────────────────────────────────
def aggregate_metrics(
    per_query: List[Dict[str, Any]],
    strategy: str,
    top_k: int,
) -> Dict[str, Any]:
    """Compute aggregate metrics from per-query results."""
    n = len(per_query)
    if n == 0:
        return {}

    mrr = sum(r["rr"] for r in per_query) / n
    hit_ratio = sum(r["hit_at_k"] for r in per_query) / n
    mean_iou = sum(r["iou_top1"] for r in per_query) / n

    recall_by_theta: Dict[str, float] = {}
    for theta in IOU_THRESHOLDS:
        key = f"recall_at_1_iou{int(theta * 10)}"
        recall_by_theta[f"R@1_IoU{theta}"] = round(
            sum(r.get(key, 0) for r in per_query) / n, 4
        )

    latencies = [
        float(r["latency_server_total_ms"])
        for r in per_query
        if r.get("latency_server_total_ms") not in ("", None)
    ]
    lat_mean = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "strategy": strategy,
        "top_k": top_k,
        "n_queries": n,
        "MRR": round(mrr, 4),
        f"HitRatio@{top_k}": round(hit_ratio, 4),
        "mean_IoU_Top1": round(mean_iou, 4),
        **recall_by_theta,
        "latency_mean_ms": round(lat_mean, 2),
    }


# ── Console Report ────────────────────────────────────────────────────────────
def print_report(all_metrics: List[Dict[str, Any]]) -> None:
    """Print a clean comparison table to console."""
    if not all_metrics:
        return

    keys = [k for k in all_metrics[0].keys() if k not in ("strategy", "top_k", "n_queries")]
    col_w = 20
    header_strat = f"{'Strategy':<{col_w}}"
    header_n = f"{'N Queries':<{col_w}}"
    header_keys = "".join(f"{k:<{col_w}}" for k in keys)

    sep = "-" * (col_w * (2 + len(keys)))
    print("\n" + sep)
    print(f"  LongVALE Benchmark Evaluation Results")
    print(sep)
    print(header_strat + header_n + header_keys)
    print(sep)
    for m in all_metrics:
        row = f"{m['strategy']:<{col_w}}{m['n_queries']:<{col_w}}"
        for k in keys:
            val = m.get(k, "")
            row += f"{str(val):<{col_w}}"
        print(row)
    print(sep + "\n")


# ── CSV / JSON Output ─────────────────────────────────────────────────────────
def save_per_query_csv(rows: List[Dict[str, Any]], filepath: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Per-query detail saved to %s", filepath)


def save_summary_json(metrics_list: List[Dict[str, Any]], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": metrics_list,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary saved to %s", filepath)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LongVALE Evaluation — compute MRR, HitRatio@K, Temporal IoU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.evaluation.eval --input output/evaluation/inference_results_agentic.csv\n"
            "  python -m src.evaluation.eval --input_dir output/evaluation\n"
            "  python -m src.evaluation.eval --input_dir output/evaluation --no_qdrant\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to a single inference results CSV")
    group.add_argument("--input_dir", help="Directory containing inference_results_*.csv files")

    parser.add_argument(
        "--top_k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of ranked results to evaluate (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--output_dir", default="output/evaluation",
        help="Output directory for per-query CSV and summary JSON (default: output/evaluation)",
    )
    parser.add_argument(
        "--no_qdrant", action="store_true",
        help="Skip Qdrant timestamp fetching (video-only Hit, no Temporal IoU)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="When using --input_dir, compare all strategies side-by-side",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    _load_env()

    # ── Load inference rows ───────────────────────────────────────────────────
    if args.input:
        all_rows = load_inference_csv(args.input)
    else:
        all_rows = load_inference_dir(args.input_dir)

    if not all_rows:
        logger.error("No rows loaded. Exiting.")
        sys.exit(1)

    # ── Deduplicate (e.g. batch files may overlap if batches were merged manually)
    seen = set()
    deduped = []
    for r in all_rows:
        key = (r.get("query_idx"), r.get("strategy"))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    if len(deduped) < len(all_rows):
        logger.warning("Removed %d duplicate rows", len(all_rows) - len(deduped))
    all_rows = deduped

    # ── Group by strategy ─────────────────────────────────────────────────────
    by_strategy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_strategy[row.get("strategy", "unknown")].append(row)

    logger.info("Strategies found: %s", list(by_strategy.keys()))

    # ── Collect all keyframe pairs for Qdrant batch fetch ─────────────────────
    ts_cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    if not args.no_qdrant:
        if not QDRANT_AVAILABLE:
            logger.error("qdrant-client not installed. Run: pip install qdrant-client")
            sys.exit(1)

        all_pairs: List[Tuple[str, int]] = []
        for row in all_rows:
            for rank in range(1, args.top_k + 1):
                kf = row.get(f"keyframe_{rank}", "").strip()
                parsed = parse_keyframe(kf)
                if parsed:
                    all_pairs.append(parsed)

        try:
            client = build_qdrant_client()
            ts_cache = fetch_timestamps_from_qdrant(
                client, all_pairs, args.collection
            )
        except Exception as exc:
            logger.error("Cannot connect to Qdrant: %s", exc)
            logger.warning("Falling back to video-only Hit (no Temporal IoU).")

    # ── Evaluate per strategy ─────────────────────────────────────────────────
    all_summary: List[Dict[str, Any]] = []
    all_per_query: List[Dict[str, Any]] = []

    for strategy, rows in sorted(by_strategy.items()):
        logger.info("Evaluating strategy: %s (%d queries)", strategy, len(rows))
        per_query = [evaluate_query(r, ts_cache, args.top_k) for r in rows]
        all_per_query.extend(per_query)
        metrics = aggregate_metrics(per_query, strategy, args.top_k)
        all_summary.append(metrics)

    # ── Print report ──────────────────────────────────────────────────────────
    print_report(all_summary)

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    per_query_path = os.path.join(args.output_dir, "eval_per_query.csv")
    save_per_query_csv(all_per_query, per_query_path)

    summary_path = os.path.join(args.output_dir, "metrics_report.json")
    save_summary_json(all_summary, summary_path)

    # Also save summary as CSV for easy Excel import
    summary_csv_path = os.path.join(args.output_dir, "metrics_report.csv")
    if all_summary:
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summary[0].keys()))
            writer.writeheader()
            writer.writerows(all_summary)
        logger.info("Summary CSV saved to %s", summary_csv_path)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
