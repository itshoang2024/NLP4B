"""
qdrant_upsert.py V2 — 4-Vector Azure-Streaming Upsert Pipeline (RAM-optimized)
================================================================================

Streams .npy embeddings and detection JSONs directly from Azure Blob Storage
into Qdrant Cloud. Designed for Colab T4 (15 GB VRAM / 12 GB RAM).

RAM Strategy:
  - Generator-based (yield): never accumulates full point lists in memory
  - Per-video processing: load → build → upsert → free, one video at a time
  - Lazy model loading: BM25/BGE-M3 only loaded when first needed
  - Aggressive gc.collect() after each video to reclaim memory

4-Vector Architecture:
  1. keyframe-dense        (SigLIP 1152d)   → visual "vibe", abstract context
  2. keyframe-object-sparse (BM25)          → object presence (unique tags)
  3. keyframe-caption-dense (BGE-M3 1024d)  → multilingual logical understanding
  4. keyframe-ocr-sparse   (BM25)           → STUB (future OCR pipeline)

Env vars:
  AZURE_STORAGE_CONNECTION_STRING
  QDRANT_URL, QDRANT_API_KEY, AZURE_BLOB_BASE_URL
"""

from __future__ import annotations

# ── 0. Auto-install ───────────────────────────────────────────────────────────
import subprocess
import sys


def _pip(*pkgs: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


try:
    from qdrant_client import QdrantClient
except ImportError:
    _pip("qdrant-client[fastembed]")

try:
    from azure.storage.blob import ContainerClient
except ImportError:
    _pip("azure-storage-blob")

try:
    from fastembed import SparseTextEmbedding
except ImportError:
    _pip("fastembed")

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    _pip("FlagEmbedding")

# ── 1. Imports ────────────────────────────────────────────────────────────────
import argparse
import gc
import io
import json
import logging
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Generator

import numpy as np
from azure.storage.blob import ContainerClient
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
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
SIGLIP_DIM = 1152
BGE_M3_DIM = 1024

VEC_DENSE = "keyframe-dense"
VEC_CAPTION_DENSE = "keyframe-caption-dense"
VEC_OBJECT_SPARSE = "keyframe-object-sparse"
VEC_OCR_SPARSE = "keyframe-ocr-sparse"

BM25_MODEL = "Qdrant/bm25"
BGE_M3_MODEL = "BAAI/bge-m3"


# ── 4. CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="4-Vector Qdrant Upsert — Azure Streaming (RAM-optimized)"
    )
    p.add_argument("--collection_name", default="keyframes_v1")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Points per upsert batch (lower = less RAM, default: 64)")
    p.add_argument("--embeddings_container", default="embeddings")
    p.add_argument("--detections_container", default="object-detection")
    return p.parse_args()


# ── 5. Azure Blob streaming ─────────────────────────────────────────────────
def get_container_client(conn_str: str, container: str) -> ContainerClient:
    return ContainerClient.from_connection_string(conn_str, container)


def stream_blob_bytes(container: ContainerClient, blob_name: str) -> bytes | None:
    """Download blob into memory. Returns None if not found."""
    try:
        return container.get_blob_client(blob_name).download_blob().readall()
    except Exception as exc:
        if "BlobNotFound" in str(exc) or "404" in str(exc):
            return None
        logger.warning(f"Blob stream failed '{blob_name}': {exc}")
        return None


def stream_npy(container: ContainerClient, blob_name: str) -> np.ndarray | None:
    data = stream_blob_bytes(container, blob_name)
    if data is None:
        return None
    arr = np.load(io.BytesIO(data))
    del data  # free raw bytes immediately
    return arr


def stream_json(container: ContainerClient, blob_name: str) -> dict | list | None:
    data = stream_blob_bytes(container, blob_name)
    if data is None:
        return None
    parsed = json.loads(data.decode("utf-8"))
    del data
    return parsed


def discover_video_ids(container: ContainerClient) -> list[str]:
    """List unique video_ids by scanning for *.npy blobs."""
    return sorted({
        Path(b.name).stem for b in container.list_blobs() if b.name.endswith(".npy")
    })


# ── 6. Lazy model singletons ─────────────────────────────────────────────────
_bm25: SparseTextEmbedding | None = None
_bge: object | None = None  # BGEM3FlagModel


def get_bm25() -> SparseTextEmbedding:
    global _bm25
    if _bm25 is None:
        logger.info(f"Loading BM25: {BM25_MODEL} ...")
        _bm25 = SparseTextEmbedding(model_name=BM25_MODEL)
    return _bm25


def get_bge_m3():
    global _bge
    if _bge is None:
        logger.info(f"Loading BGE-M3: {BGE_M3_MODEL} (fp16) ...")
        from FlagEmbedding import BGEM3FlagModel
        _bge = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=True)
        logger.info("BGE-M3 loaded.")
    return _bge


# ── 7. Encoding (single-item, no batching to save RAM) ───────────────────────
def encode_bm25(text: str) -> SparseVector | None:
    if not text or not text.strip():
        return None
    try:
        results = list(get_bm25().embed([text]))
        if not results:
            return None
        s = results[0]
        return SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
    except Exception as exc:
        logger.warning(f"BM25 failed: {exc}")
        return None


def encode_bge_m3(text: str) -> list[float] | None:
    if not text or not text.strip():
        return None
    try:
        result = get_bge_m3().encode([text])
        vec = result["dense_vecs"][0]
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        if len(vec) != BGE_M3_DIM:
            logger.warning(f"BGE-M3 dim={len(vec)}, expected {BGE_M3_DIM}")
            return None
        return vec
    except Exception as exc:
        logger.warning(f"BGE-M3 failed: {exc}")
        return None


# ── 8. Detection JSON parsing ────────────────────────────────────────────────
def build_det_lookup(detection_data: dict | None) -> dict[str, dict]:
    """Build image_id → frame_result lookup from detection JSON."""
    if not detection_data:
        return {}
    return {r.get("image_id", ""): r for r in detection_data.get("results", [])}


def extract_frame_metadata(frame_result: dict) -> dict:
    """Extract tags, counts, captions from a single frame's detection result."""
    gd = frame_result.get("global_descriptions", {})
    objects = frame_result.get("objects", [])
    all_labels = [o.get("label", "") for o in objects if o.get("label")]

    return {
        "tags": gd.get("tags", []),
        "caption": gd.get("caption", ""),
        "detailed_caption": gd.get("detailed_caption", ""),
        "object_counts": dict(Counter(all_labels)),
        "unique_tags": sorted(set(all_labels)),
    }


# ── 9. OCR stub ──────────────────────────────────────────────────────────────
def load_ocr_data(video_id: str, frame_idx: int) -> str | None:
    """STUB: returns None until OCR pipeline is ready."""
    return None


# ── 10. Collection setup ─────────────────────────────────────────────────────
def ensure_collection(client: QdrantClient, name: str) -> None:
    if name in [c.name for c in client.get_collections().collections]:
        logger.info(f"Collection '{name}' exists.")
        return
    logger.info(f"Creating collection '{name}' ...")
    client.create_collection(
        collection_name=name,
        vectors_config={
            VEC_DENSE: VectorParams(size=SIGLIP_DIM, distance=Distance.COSINE),
            VEC_CAPTION_DENSE: VectorParams(size=BGE_M3_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            VEC_OBJECT_SPARSE: SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            VEC_OCR_SPARSE: SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )
    logger.info(f"Collection created: {SIGLIP_DIM}d + {BGE_M3_DIM}d + 2 sparse.")


# ── 11. ID generator ─────────────────────────────────────────────────────────
def deterministic_id(video_id: str, frame_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_{frame_idx}"))


# ── 12. GENERATOR: yield one PointStruct at a time ──────────────────────────
def generate_points(
    video_id: str,
    embeddings: np.ndarray,
    frame_indices: list[int] | None,
    det_lookup: dict[str, dict],
    azure_base_url: str,
) -> Generator[PointStruct, None, None]:
    """
    Yield PointStruct one-by-one. Never holds the full list in memory.
    After yielding, the caller can batch and upsert in fixed-size chunks.
    """
    n_frames = embeddings.shape[0]

    for idx in range(n_frames):
        # ── Frame index ──────────────────────────────────────────────
        frame_idx = int(frame_indices[idx]) if (frame_indices and idx < len(frame_indices)) else idx
        frame_filename = f"{video_id}_{frame_idx:05d}.jpg"
        image_id = f"{video_id}_{frame_idx:05d}"

        # ── Dense vector (convert row then free ref) ─────────────────
        dense_vec = embeddings[idx].tolist()

        # ── Base payload ─────────────────────────────────────────────
        vectors: dict = {VEC_DENSE: dense_vec}
        payload: dict = {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "azure_url": f"{azure_base_url}/{video_id}/{frame_filename}",
        }

        # ── Detection metadata (optional) ────────────────────────────
        frame_result = det_lookup.get(image_id) or det_lookup.get(f"{video_id}_{frame_idx}")
        if frame_result:
            meta = extract_frame_metadata(frame_result)

            payload["tags"] = meta["tags"]
            payload["caption"] = meta["caption"]
            payload["detailed_caption"] = meta["detailed_caption"]
            payload["object_counts"] = meta["object_counts"]

            # Object sparse (BM25 on unique tags)
            if meta["unique_tags"]:
                obj_sparse = encode_bm25(" ".join(meta["unique_tags"]))
                if obj_sparse:
                    vectors[VEC_OBJECT_SPARSE] = obj_sparse

            # Caption dense (BGE-M3 on detailed_caption, lazy-loaded)
            if meta["detailed_caption"]:
                cap_vec = encode_bge_m3(meta["detailed_caption"])
                if cap_vec:
                    vectors[VEC_CAPTION_DENSE] = cap_vec

        # ── OCR sparse (stub) ────────────────────────────────────────
        ocr_text = load_ocr_data(video_id, frame_idx)
        if ocr_text:
            ocr_sparse = encode_bm25(ocr_text)
            if ocr_sparse:
                vectors[VEC_OCR_SPARSE] = ocr_sparse

        yield PointStruct(
            id=deterministic_id(video_id, frame_idx),
            vector=vectors,
            payload=payload,
        )


# ── 13. Streaming batch upsert (consumes generator in chunks) ────────────────
def stream_upsert(
    client: QdrantClient,
    collection: str,
    point_gen: Generator[PointStruct, None, None],
    batch_size: int,
) -> tuple[int, int]:
    """
    Consume a point generator in fixed-size batches.
    Each batch is upserted then discarded → constant memory usage.
    """
    ok = fail = batch_num = 0
    batch: list[PointStruct] = []

    for point in point_gen:
        batch.append(point)

        if len(batch) >= batch_size:
            batch_num += 1
            try:
                client.upsert(collection_name=collection, points=batch)
                ok += len(batch)
                logger.info(f"  Batch {batch_num}: upserted {len(batch)} pts (total OK: {ok})")
            except Exception as exc:
                fail += len(batch)
                logger.error(f"  Batch {batch_num} failed: {exc}")
            batch.clear()  # free memory immediately

    # Flush remaining
    if batch:
        batch_num += 1
        try:
            client.upsert(collection_name=collection, points=batch)
            ok += len(batch)
            logger.info(f"  Batch {batch_num} (final): upserted {len(batch)} pts (total OK: {ok})")
        except Exception as exc:
            fail += len(batch)
            logger.error(f"  Batch {batch_num} (final) failed: {exc}")
        batch.clear()

    return ok, fail


# ── 14. Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Env vars ─────────────────────────────────────────────────────
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    qdrant_url = os.environ.get("QDRANT_URL", "")
    qdrant_key = os.environ.get("QDRANT_API_KEY", "")
    azure_base = os.environ.get("AZURE_BLOB_BASE_URL", "")

    if not conn_str:
        logger.error("Missing AZURE_STORAGE_CONNECTION_STRING"); sys.exit(1)
    if not qdrant_url or not qdrant_key:
        logger.error("Missing QDRANT_URL / QDRANT_API_KEY"); sys.exit(1)

    # ── Connect ──────────────────────────────────────────────────────
    emb_container = get_container_client(conn_str, args.embeddings_container)
    det_container = get_container_client(conn_str, args.detections_container)
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    ensure_collection(qdrant, args.collection_name)

    # ── Discover videos ──────────────────────────────────────────────
    video_ids = discover_video_ids(emb_container)
    if not video_ids:
        logger.error(f"No .npy blobs in container '{args.embeddings_container}'")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Qdrant V2: 4-Vector Azure-Streaming Upsert (RAM-Optimized)")
    logger.info("=" * 60)
    logger.info(f"Embeddings:  {args.embeddings_container}")
    logger.info(f"Detections:  {args.detections_container}")
    logger.info(f"Collection:  {args.collection_name}")
    logger.info(f"Batch size:  {args.batch_size}")
    logger.info(f"Videos:      {len(video_ids)}")
    logger.info("")

    t0 = time.time()
    total_ok = total_fail = 0
    n_det = n_dense = 0

    for i, vid in enumerate(video_ids, 1):
        logger.info(f"[{i}/{len(video_ids)}] Processing: {vid}")

        # ── Stream embeddings (largest object per video) ─────────────
        embeddings = stream_npy(emb_container, f"{vid}.npy")
        if embeddings is None:
            logger.error(f"  Could not load {vid}.npy"); continue
        if embeddings.ndim != 2 or embeddings.shape[1] != SIGLIP_DIM:
            logger.error(f"  Bad shape {embeddings.shape}"); del embeddings; continue

        n = embeddings.shape[0]
        logger.info(f"  {n} frames streamed ({embeddings.nbytes / 1e6:.1f} MB)")

        # ── Stream frame indices ─────────────────────────────────────
        frame_indices = stream_json(emb_container, f"{vid}_frames.json")

        # ── Stream detection JSON ────────────────────────────────────
        det_data = stream_json(det_container, f"{vid}.json")
        det_lookup = build_det_lookup(det_data)
        del det_data  # free raw JSON, keep only lookup

        if det_lookup:
            n_det += 1
            logger.info(f"  Detection: {len(det_lookup)} frames → objects + caption")
        else:
            n_dense += 1
            logger.warning(f"  No detection JSON → dense-only")

        # ── Stream-upsert via generator ──────────────────────────────
        try:
            point_gen = generate_points(
                vid, embeddings, frame_indices, det_lookup, azure_base
            )
            ok, fail = stream_upsert(qdrant, args.collection_name, point_gen, args.batch_size)
            total_ok += ok
            total_fail += fail
        except Exception as exc:
            logger.error(f"  Pipeline failed for {vid}: {exc}")

        # ── Aggressive memory cleanup after each video ───────────────
        del embeddings, frame_indices, det_lookup
        gc.collect()

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("Upsert V2 Complete!")
    logger.info("=" * 60)
    logger.info(f"Videos:     {len(video_ids)} ({n_det} w/ detections, {n_dense} dense-only)")
    logger.info(f"Upserted:   {total_ok}")
    logger.info(f"Failed:     {total_fail}")
    logger.info(f"Time:       {elapsed:.1f}s")
    if total_fail:
        logger.warning(f"{total_fail} failures — see qdrant_upsert_errors.log")


if __name__ == "__main__":
    main()
