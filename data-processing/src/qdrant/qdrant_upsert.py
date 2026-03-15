"""
qdrant_upsert.py — Upsert SigLIP dense + optional YOLO/Florence sparse vectors into Qdrant Cloud
==================================================================================================

Designed for Google Colab:
  1. !git clone <repo> && cd <repo>
  2. Set env vars:  QDRANT_URL, QDRANT_API_KEY, AZURE_BLOB_BASE_URL
  3. !python qdrant_upsert.py --embeddings_dir /content/embeddings --detections_dir /content/detections

Features:
  - Deterministic point IDs via uuid5  →  safe re-runs / idempotent upserts
  - Graceful degradation              →  missing detection JSONs  =  dense-only upload
  - Batched upserts                   →  memory-efficient at scale
  - Structured logging                →  console INFO + file ERROR
"""

from __future__ import annotations

# ── 0. Auto-install (Colab-friendly) ──────────────────────────────────────────
import subprocess
import sys


def _pip(*pkgs: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


try:
    from qdrant_client import QdrantClient  # noqa: F401
except ImportError:
    _pip("qdrant-client[fastembed]")

try:
    from fastembed import SparseTextEmbedding  # noqa: F401
except ImportError:
    _pip("fastembed")

# ── 1. Imports ────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseVector,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)

# ── 2. Logging ────────────────────────────────────────────────────────────────
LOG_FMT = "[%(asctime)s] %(levelname)-8s %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

console_h = logging.StreamHandler(sys.stdout)
console_h.setLevel(logging.INFO)
console_h.setFormatter(logging.Formatter(LOG_FMT, datefmt=DATE_FMT))

file_h = logging.FileHandler("qdrant_upsert_errors.log", encoding="utf-8")
file_h.setLevel(logging.ERROR)
file_h.setFormatter(logging.Formatter(LOG_FMT, datefmt=DATE_FMT))

logger = logging.getLogger("qdrant_upsert")
logger.setLevel(logging.DEBUG)
logger.addHandler(console_h)
logger.addHandler(file_h)

# ── 3. Constants ──────────────────────────────────────────────────────────────
DENSE_DIM = 1152          # SigLIP google/siglip-so400m-patch14-384
DENSE_NAME = "keyframe-dense"
SPARSE_NAME = "keyframe-sparse"
SPARSE_MODEL = "Qdrant/bm25"


# ── 4. CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upsert SigLIP + optional YOLO/Florence vectors into Qdrant Cloud."
    )
    p.add_argument(
        "--embeddings_dir",
        type=Path,
        required=True,
        help="Dir with SigLIP outputs: <video_id>.npy and <video_id>_frames.json",
    )
    p.add_argument(
        "--detections_dir",
        type=Path,
        default=None,
        help="(Optional) Dir with YOLO/Florence detection JSONs: <video_id>.json",
    )
    p.add_argument(
        "--collection_name",
        type=str,
        default="keyframes_v1",
        help="Qdrant collection name (default: keyframes_v1)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Points per upsert batch (default: 100)",
    )
    return p.parse_args()


# ── 5. Helpers ────────────────────────────────────────────────────────────────
def deterministic_id(video_id: str, frame_idx: int) -> str:
    """Generate a stable UUID-5 string from video_id + frame_idx."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_{frame_idx}"))


def load_frames_json(embeddings_dir: Path, video_id: str) -> list[str] | None:
    """
    Try to load <video_id>_frames.json which maps index → frame filename.
    Returns a list of frame filename strings, or None if not found.
    """
    frames_path = embeddings_dir / f"{video_id}_frames.json"
    if not frames_path.is_file():
        return None
    try:
        with open(frames_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Could not parse {frames_path}: {exc}")
        return None


def load_detection(detections_dir: Path | None, video_id: str) -> dict | None:
    """
    Load <detections_dir>/<video_id>.json.
    Expected schema (per frame):
      {
        "frames": {
           "0": {"tags": ["car", "person"], "caption": "A red car ..."},
           "1": { ... }
        }
      }
    Returns parsed dict, or None if unavailable.
    """
    if detections_dir is None:
        return None
    det_path = detections_dir / f"{video_id}.json"
    if not det_path.is_file():
        return None
    try:
        with open(det_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Could not parse detection file {det_path}: {exc}")
        return None


# ── 6. Sparse embedding helper ───────────────────────────────────────────────
_sparse_model: SparseTextEmbedding | None = None


def get_sparse_model() -> SparseTextEmbedding:
    """Lazy singleton — only loaded if at least one detection JSON exists."""
    global _sparse_model
    if _sparse_model is None:
        logger.info(f"Loading sparse embedding model: {SPARSE_MODEL} ...")
        from fastembed import SparseTextEmbedding
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
        logger.info("Sparse model loaded successfully.")
    return _sparse_model


def encode_sparse(text: str) -> SparseVector | None:
    """Encode a single text string to a Qdrant SparseVector using BM25."""
    if not text or not text.strip():
        return None
    try:
        model = get_sparse_model()
        results = list(model.embed([text]))
        if not results:
            return None
        sparse = results[0]
        return SparseVector(
            indices=sparse.indices.tolist(),
            values=sparse.values.tolist(),
        )
    except Exception as exc:
        logger.warning(f"Sparse encoding failed for text '{text[:80]}...': {exc}")
        return None


# ── 7. Collection setup ──────────────────────────────────────────────────────
def ensure_collection(client: QdrantClient, name: str) -> None:
    """Create collection if it doesn't exist with correct vector config."""
    collections = [c.name for c in client.get_collections().collections]

    if name in collections:
        logger.info(f"Collection '{name}' already exists.")
        return

    logger.info(f"Creating collection '{name}' ...")
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_NAME: VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    logger.info(f"Collection '{name}' created with dense ({DENSE_DIM}d) + sparse vectors.")


# ── 8. Build points for a single video ───────────────────────────────────────
def build_points_for_video(
    video_id: str,
    embeddings: np.ndarray,
    frame_names: list[str] | None,
    detection_data: dict | None,
    azure_base_url: str,
) -> list[PointStruct]:
    """
    Build PointStruct list for one video.
    embeddings shape: (N, 1152)
    """
    points: list[PointStruct] = []
    n_frames = embeddings.shape[0]

    # Pre-extract the per-frame detection lookup if available
    det_frames: dict | None = None
    if detection_data is not None and "frames" in detection_data:
        det_frames = detection_data["frames"]

    for idx in range(n_frames):
        # ── Frame identifier ─────────────────────────────────────────────
        # _frames.json may contain:
        #   A) raw ints like [872, 1230, ...]        → original video frame index
        #   B) strings  like ["video_00872.jpg", ...] → full filename
        #   C) not exist at all                       → fallback to sequential idx
        raw_entry = frame_names[idx] if (frame_names and idx < len(frame_names)) else None

        if raw_entry is not None:
            if isinstance(raw_entry, (int, float)):
                # Case A: raw frame index number → build filename
                frame_idx = int(raw_entry)
                frame_filename = f"{video_id}_{frame_idx:05d}.jpg"
            elif isinstance(raw_entry, str) and raw_entry.replace(".", "").replace("_", "").isalnum():
                # Case B: full filename string
                frame_filename = raw_entry
                # Try to extract frame index from filename like "video_00872.jpg"
                try:
                    stem = Path(raw_entry).stem  # "video_00872"
                    frame_idx = int(stem.rsplit("_", 1)[-1])
                except (ValueError, IndexError):
                    frame_idx = idx
            else:
                frame_filename = str(raw_entry)
                frame_idx = idx
        else:
            # Case C: no _frames.json → sequential index (likely wrong for Azure)
            frame_idx = idx
            frame_filename = f"{video_id}_{frame_idx:05d}.jpg"

        # ── Deterministic ID ─────────────────────────────────────────────
        point_id = deterministic_id(video_id, frame_idx)

        # ── Dense vector ─────────────────────────────────────────────────
        dense_vec = embeddings[idx].tolist()

        # ── Base payload (always present) ────────────────────────────────
        azure_url = f"{azure_base_url}/{video_id}/{frame_filename}"
        payload: dict = {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "azure_url": azure_url,
        }

        # ── Optional: detection metadata + sparse vector ─────────────────
        vectors: dict = {DENSE_NAME: dense_vec}

        if det_frames is not None:
            frame_det = det_frames.get(str(frame_idx))
            if frame_det:
                tags = frame_det.get("tags", [])
                caption = frame_det.get("caption", "")
                ocr_text = frame_det.get("ocr", "")

                # Build metadata sub-payload
                payload["metadata"] = {
                    "tags": tags,
                    "caption": caption,
                }
                if ocr_text:
                    payload["metadata"]["ocr"] = ocr_text

                # Build sparse text from caption + OCR + tags
                sparse_text = " ".join(
                    filter(None, [caption, ocr_text, " ".join(tags)])
                )
                sparse_vec = encode_sparse(sparse_text)
                if sparse_vec is not None:
                    vectors[SPARSE_NAME] = sparse_vec

        # ── Assemble point ───────────────────────────────────────────────
        points.append(
            PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload,
            )
        )

    return points


# ── 9. Batch upsert ──────────────────────────────────────────────────────────
def batch_upsert(
    client: QdrantClient,
    collection_name: str,
    points: list[PointStruct],
    batch_size: int,
) -> tuple[int, int]:
    """
    Upsert points in batches.
    Returns (success_count, fail_count).
    """
    total = len(points)
    success = 0
    fail = 0

    for start in range(0, total, batch_size):
        batch = points[start : start + batch_size]
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch,
            )
            success += len(batch)
            logger.info(
                f"  Upserted batch [{start + 1}–{start + len(batch)}] / {total}"
            )
        except Exception as exc:
            fail += len(batch)
            logger.error(
                f"  Failed batch [{start + 1}–{start + len(batch)}] / {total}: {exc}"
            )

    return success, fail


# ── 10. Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Validate environment ─────────────────────────────────────────────
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    azure_base = os.environ.get("AZURE_BLOB_BASE_URL", "")

    if not qdrant_url or not qdrant_key:
        logger.error(
            "Missing env var(s): QDRANT_URL and/or QDRANT_API_KEY. "
            "Set them before running this script."
        )
        sys.exit(1)

    if not azure_base:
        logger.warning(
            "AZURE_BLOB_BASE_URL is not set. "
            "azure_url fields in payloads will use relative paths."
        )

    # ── Connect to Qdrant ────────────────────────────────────────────────
    logger.info("Connecting to Qdrant Cloud ...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    ensure_collection(client, args.collection_name)

    # ── Discover videos from embeddings dir ──────────────────────────────
    if not args.embeddings_dir.is_dir():
        logger.error(f"Embeddings directory not found: {args.embeddings_dir}")
        sys.exit(1)

    npy_files = sorted(args.embeddings_dir.rglob("*.npy"))
    if not npy_files:
        logger.error(f"No .npy files found in {args.embeddings_dir}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Qdrant Vector Upsert Pipeline")
    logger.info("=" * 60)
    logger.info(f"Embeddings dir:   {args.embeddings_dir}")
    logger.info(f"Detections dir:   {args.detections_dir or '(not provided)'}")
    logger.info(f"Collection:       {args.collection_name}")
    logger.info(f"Batch size:       {args.batch_size}")
    logger.info(f"Videos found:     {len(npy_files)}")
    logger.info("")

    start_time = time.time()
    total_success = 0
    total_fail = 0
    total_points = 0
    videos_with_detections = 0
    videos_dense_only = 0

    for npy_path in npy_files:
        video_id = npy_path.stem  # e.g. "abc123"

        logger.info(f"Processing video: {video_id}")

        # ── Load dense embeddings ────────────────────────────────────────
        try:
            embeddings = np.load(npy_path)
        except Exception as exc:
            logger.error(f"  Failed to load {npy_path}: {exc}")
            continue

        if embeddings.ndim != 2 or embeddings.shape[1] != DENSE_DIM:
            logger.error(
                f"  Unexpected shape {embeddings.shape} for {video_id} "
                f"(expected (N, {DENSE_DIM})). Skipping."
            )
            continue

        n_frames = embeddings.shape[0]
        logger.info(f"  Loaded {n_frames} embeddings ({embeddings.shape})")

        # ── Load frame names (optional) ──────────────────────────────────
        frame_names = load_frames_json(args.embeddings_dir, video_id)

        # ── Load detection data (optional, graceful) ─────────────────────
        detection_data = load_detection(args.detections_dir, video_id)
        if detection_data is not None:
            videos_with_detections += 1
            logger.info(f"  Detection JSON found → including sparse vectors + metadata")
        else:
            videos_dense_only += 1
            logger.warning(
                f"  No detection JSON for '{video_id}' → dense-only upload"
            )

        # ── Build points ─────────────────────────────────────────────────
        try:
            points = build_points_for_video(
                video_id=video_id,
                embeddings=embeddings,
                frame_names=frame_names,
                detection_data=detection_data,
                azure_base_url=azure_base,
            )
        except Exception as exc:
            logger.error(f"  Failed to build points for {video_id}: {exc}")
            continue

        total_points += len(points)

        # ── Upsert ───────────────────────────────────────────────────────
        ok, fail = batch_upsert(client, args.collection_name, points, args.batch_size)
        total_success += ok
        total_fail += fail

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    logger.info("")
    logger.info("=" * 60)
    logger.info("Upsert Complete!")
    logger.info("=" * 60)
    logger.info(f"Videos processed:      {len(npy_files)}")
    logger.info(f"  with detections:     {videos_with_detections}")
    logger.info(f"  dense-only:          {videos_dense_only}")
    logger.info(f"Total points built:    {total_points}")
    logger.info(f"Successfully upserted: {total_success}")
    logger.info(f"Failed:                {total_fail}")
    logger.info(f"Elapsed time:          {elapsed:.1f}s")

    if total_fail > 0:
        logger.warning(
            f"{total_fail} point(s) failed. Check qdrant_upsert_errors.log for details."
        )


if __name__ == "__main__":
    main()
