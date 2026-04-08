"""
qdrant_upsert.py V2.1 — 4-Vector Azure-Streaming Upsert Pipeline (RAM-optimized)
================================================================================

Streams .npy embeddings, detection JSONs, and OCR JSONs directly from Azure Blob 
Storage into Qdrant Cloud. Incorporates metadata processing for YouTube links.
Designed for Colab T4 (15 GB VRAM / 12 GB RAM).

RAM Strategy:
  - Generator-based (yield): never accumulates full point lists in memory
  - Per-video processing: load → build → upsert → free, one video at a time
  - Lazy model loading: BM25/BGE-M3 only loaded when first needed
  - O(1) OCR parsing using standard json.loads (since chunk sizes are small enough)
  - Aggressive gc.collect() after each video to reclaim memory

4-Vector Architecture:
  1. keyframe-dense        (SigLIP 1152d)   → visual "vibe", abstract context
  2. keyframe-object-sparse (BM25)          → object presence (unique tags)
  3. keyframe-caption-dense (BGE-M3 1024d)  → multilingual logical understanding
  4. keyframe-ocr-sparse   (BM25)           → OCR parsed text from Azure

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
    from sentence_transformers import SentenceTransformer
except ImportError:
    _pip("sentence-transformers")

try:
    import ijson
except ImportError:
    _pip("ijson")
    import ijson


# ── 1. Imports ────────────────────────────────────────────────────────────────
import argparse
import csv
import gc
import io
import json
import logging
import os
import re
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
    PointVectors,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.http.models import SetPayloadOperation, SetPayload

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
        description="4-Vector Qdrant Upsert — Azure Streaming (RAM-optimized) with OCR & Metadata"
    )
    p.add_argument("--mode", choices=["upsert", "update"], default="upsert",
                   help="'upsert': Full point creation. 'update': Update specific payloads/vectors on existing points.")
    p.add_argument("--update_payloads", default="ocr_text,youtube_link,timestamp_sec",
                   help="Comma-separated payload fields to set (mode='update').")
    p.add_argument("--update_vectors", default="keyframe-ocr-sparse",
                   help="Comma-separated vector names to update (mode='update').")
    p.add_argument("--collection_name", default="keyframes_v1")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Points per upsert batch (lower = less RAM, default: 64)")
    p.add_argument("--embeddings_container", default="embeddings")
    p.add_argument("--detections_container", default="object-detection")
    p.add_argument("--ocr_container", default="ocr", help="Azure container housing OCR results")
    p.add_argument("--video_metadata_link", default="", required=False, help="Path to local metadata CSV (optional)")
    return p.parse_args()


# ── 5. Azure Blob & Metadata streaming ────────────────────────────────────────
def load_metadata_csv(csv_path: str) -> dict[str, dict]:
    """Parse Video Metadata into lookup dict."""
    if not csv_path or not os.path.isfile(csv_path):
        logger.warning(f"Metadata CSV '{csv_path}' not found or not provided. Generating empty metadata.")
        return {}
        
    meta_dict = {}
    logger.info(f"Loading metadata from {csv_path}...")
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "")
            if vid:
                meta_dict[vid] = {
                    "source_url": row.get("source_url", ""),
                    "fps": float(row.get("fps", 1.0) or 1.0)
                }
    logger.info(f"Loaded metadata for {len(meta_dict)} videos.")
    return meta_dict

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

def discover_video_ids(container: ContainerClient) -> dict[str, str]:
    """Find unique video_ids and their blob prefixes by scanning for *.npy blobs."""
    video_map = {}
    for blob in container.list_blobs():
        if blob.name.endswith(".npy"):
            vid = Path(blob.name).stem
            video_map[vid] = blob.name[:-4]  # exact path without .npy
    return video_map

def discover_detection_jsons(container: ContainerClient) -> dict[str, str]:
    """Find detection JSON blob paths by scanning detections container."""
    det_map = {}
    for blob in container.list_blobs():
        if blob.name.endswith(".json") or blob.name.endswith(".js"):
            stem = Path(blob.name).stem
            if stem.endswith("_object_detection"):
                vid = stem.replace("_object_detection", "")
            else:
                vid = stem
            det_map[vid] = blob.name
    return det_map


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
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading BGE-M3: {BGE_M3_MODEL} on [{device}] ...")
        from sentence_transformers import SentenceTransformer
        _bge = SentenceTransformer(BGE_M3_MODEL, device=device)
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
        if len(s.indices) == 0:
            return None
        return SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
    except Exception as exc:
        logger.warning(f"BM25 failed: {exc}")
        return None

def encode_bge_m3(text: str) -> list[float] | None:
    if not text or not text.strip():
        return None
    try:
        vec = get_bge_m3().encode(text, normalize_embeddings=True)
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        if len(vec) != BGE_M3_DIM:
            logger.warning(f"BGE-M3 dim={len(vec)}, expected {BGE_M3_DIM}")
            return None
        return vec
    except Exception as exc:
        logger.warning(f"BGE-M3 failed: {exc}")
        return None


# ── 8. Detection and OCR JSON parsing ────────────────────────────────────────
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

def clean_ocr_text(text: str) -> str:
    """Limits repetitive chars, drops special chars."""
    if not text:
        return ""
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'(.)\1{10,}', r'\1', text)
    text = " ".join(text.split())
    return text

def build_ocr_lookup(container: ContainerClient, video_id: str) -> dict[int, str]:
    """Loads OCR JSON array completely using stream_json to build an exact lookup."""
    blob_name = f"{video_id}/{video_id}_ocr.json"  # Azure blob name inside the ocr container
    lookup = {}
    try:
        data = stream_json(container, blob_name)
        if data:
            for item in data:
                img_name = item.get("image", "")
                raw_text = item.get("ocr_text", "")
                cleaned = clean_ocr_text(raw_text)
                
                if cleaned:    
                    stem = Path(img_name).stem
                    parts = stem.split("_")
                    if len(parts) >= 2:
                        try:
                            f_idx = int(parts[-1])
                            lookup[f_idx] = cleaned
                        except ValueError:
                            pass
    except Exception as exc:
        if "BlobNotFound" not in str(exc) and "404" not in str(exc):
            logger.warning(f"Failed OCR lookup {blob_name}: {exc}")
    return lookup


# ── 10. Collection setup ─────────────────────────────────────────────────────
REQUIRED_DENSE = {VEC_DENSE, VEC_CAPTION_DENSE}
REQUIRED_SPARSE = {VEC_OBJECT_SPARSE, VEC_OCR_SPARSE}

def ensure_collection(client: QdrantClient, name: str) -> None:
    existing = [c.name for c in client.get_collections().collections]

    if name in existing:
        # Ensure payload index on 'tags' exists for heuristic filtering
        try:
            client.create_payload_index(
                collection_name=name,
                field_name="tags",
                field_schema="keyword",
            )
            logger.info(f"Checked/Created payload index on 'tags' for existing collection '{name}'.")
        except Exception as e:
            pass # Index likely already exists

        info = client.get_collection(name)
        existing_vecs = set(info.config.params.vectors.keys()) if info.config.params.vectors else set()
        existing_sparse = set(info.config.params.sparse_vectors.keys()) if info.config.params.sparse_vectors else set()

        missing_dense = REQUIRED_DENSE - existing_vecs
        missing_sparse = REQUIRED_SPARSE - existing_sparse

        if not missing_dense and not missing_sparse:
            logger.info(f"Collection '{name}' exists with correct 4-vector schema.")
            return

        # Add missing vectors using update_collection to safely patch the schema
        new_dense = {}
        for vec, size in [(VEC_DENSE, SIGLIP_DIM), (VEC_CAPTION_DENSE, BGE_M3_DIM)]:
            if vec not in existing_vecs: 
                new_dense[vec] = VectorParams(size=size, distance=Distance.COSINE)
        
        new_sparse = {}
        for vec in [VEC_OBJECT_SPARSE, VEC_OCR_SPARSE]:
            if vec not in existing_sparse: 
                # Note: Sparse vectors don't have dimensions, they just store index/value pairs.
                new_sparse[vec] = SparseVectorParams(index=SparseIndexParams(on_disk=False))
            
        if new_dense or new_sparse:
            try:
                client.update_collection(
                    collection_name=name, 
                    vectors_config=new_dense or None, 
                    sparse_vectors_config=new_sparse or None
                )
                logger.info(f"Safely updated collection schema with missing vectors: {list(new_dense.keys()) + list(new_sparse.keys())}.")
            except Exception as e:
                logger.error(f"Failed to update collection '{name}' securely: {e}")
        return

    logger.info(f"Creating collection '{name}' with 4-vector schema ...")
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
    # create payload index
    try:
        client.create_payload_index(
            collection_name=name,
            field_name="tags",
            field_schema="keyword",
        )
        logger.info("Payload index created on 'tags'.")
    except Exception as e:
        logger.error(f"Failed to create payload index on 'tags': {e}")
        
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
    ocr_lookup: dict[int, str],
    meta: dict,
    azure_base_url: str,
) -> Generator[PointStruct, None, None]:
    """
    Yield PointStruct one-by-one. Never holds the full list in memory.
    Yields 4 vectors immediately at upsert time and bundles OCR + YouTube link payloads.
    """
    n_frames = embeddings.shape[0]
    
    fps = meta.get("fps", 1.0)
    source_url = meta.get("source_url", "")

    for idx in range(n_frames):
        # ── Frame index ──────────────────────────────────────────────
        frame_idx = int(frame_indices[idx]) if (frame_indices and idx < len(frame_indices)) else idx
        frame_filename = f"{video_id}_{frame_idx:05d}.jpg"
        image_id = f"{video_id}_{frame_idx:05d}"
        
        # Calculate Timestamp & YouTube Link
        timestamp_sec = int(frame_idx / float(fps)) if fps > 0 else 0
        youtube_link = f"{source_url}&t={timestamp_sec}s" if source_url else ""

        # ── Dense vector (convert row then free ref) ─────────────────
        dense_vec = embeddings[idx].tolist()

        # ── Base payload ─────────────────────────────────────────────
        vectors: dict = {VEC_DENSE: dense_vec}
        payload: dict = {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "youtube_link": youtube_link,
            "azure_url": f"{azure_base_url}/{video_id}/{frame_filename}",
        }

        # ── Detection metadata (optional) ────────────────────────────
        frame_result = det_lookup.get(image_id) or det_lookup.get(f"{video_id}_{frame_idx}")
        if frame_result:
            m = extract_frame_metadata(frame_result)

            payload["tags"] = m["tags"]
            payload["caption"] = m["caption"]
            payload["detailed_caption"] = m["detailed_caption"]
            payload["object_counts"] = m["object_counts"]

            if m["unique_tags"]:
                obj_sparse = encode_bm25(" ".join(m["unique_tags"]))
                if obj_sparse:
                    vectors[VEC_OBJECT_SPARSE] = obj_sparse

            if m["detailed_caption"]:
                cap_vec = encode_bge_m3(m["detailed_caption"])
                if cap_vec:
                    vectors[VEC_CAPTION_DENSE] = cap_vec

        # ── OCR processing ───────────────────────────────────────────
        ocr_text = ocr_lookup.get(frame_idx, "")
        if ocr_text:
            payload["ocr_text"] = ocr_text
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
            batch.clear()

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


# ── 13b. Update Existing (Fast Mode) generators ──────────────────────────────
def generate_updates(
    video_id: str,
    frame_indices: list[int] | None,
    det_lookup: dict[str, dict],
    ocr_lookup: dict[int, str],
    meta: dict,
    update_payloads: list[str],
    update_vectors: list[str]
) -> Generator[tuple[str, dict, dict], None, None]:
    """Yields (point_id, payload_dict, vectors_dict) for existing points."""
    fps = meta.get("fps", 1.0)
    source_url = meta.get("source_url", "")

    all_frames = set(frame_indices) if frame_indices else set()
    if not all_frames:
        for image_id in det_lookup.keys():
            parts = image_id.split("_")
            if parts:
                try: all_frames.add(int(parts[-1]))
                except ValueError: pass
        all_frames.update(ocr_lookup.keys())

    for frame_idx in sorted(all_frames):
        payload = {}
        vectors = {}
        
        # 1. Payloads
        timestamp_sec = int(frame_idx / float(fps)) if fps > 0 else 0
        youtube_link = f"{source_url}&t={timestamp_sec}s" if source_url else ""
        
        if "youtube_link" in update_payloads:
            payload["youtube_link"] = youtube_link
        if "timestamp_sec" in update_payloads:
            payload["timestamp_sec"] = timestamp_sec
        if "video_id" in update_payloads:
            payload["video_id"] = video_id
        if "frame_idx" in update_payloads:
            payload["frame_idx"] = frame_idx
            
        image_id = f"{video_id}_{frame_idx:05d}"
        frame_result = det_lookup.get(image_id) or det_lookup.get(f"{video_id}_{frame_idx}")
        if frame_result:
            m = extract_frame_metadata(frame_result)
            for k in ["tags", "caption", "detailed_caption", "object_counts"]:
                if k in update_payloads and k in m:
                    payload[k] = m[k]
                    
            if VEC_OBJECT_SPARSE in update_vectors and m["unique_tags"]:
                obj_sparse = encode_bm25(" ".join(m["unique_tags"]))
                if obj_sparse: vectors[VEC_OBJECT_SPARSE] = obj_sparse
            if VEC_CAPTION_DENSE in update_vectors and m["detailed_caption"]:
                cap_vec = encode_bge_m3(m["detailed_caption"])
                if cap_vec: vectors[VEC_CAPTION_DENSE] = cap_vec
                
        ocr_text = ocr_lookup.get(frame_idx, "")
        if "ocr_text" in update_payloads and ocr_text:
            payload["ocr_text"] = ocr_text
            
        if VEC_OCR_SPARSE in update_vectors and ocr_text:
            ocr_sparse = encode_bm25(ocr_text)
            if ocr_sparse: vectors[VEC_OCR_SPARSE] = ocr_sparse
                
        if payload or vectors:
            point_id = deterministic_id(video_id, frame_idx)
            yield (point_id, payload, vectors)

def stream_updates(
    client: QdrantClient, col: str, op_gen: Generator, batch_size: int
) -> tuple[int, int]:
    ok = fail = b_num = 0
    payload_batch = []
    vector_batch = []
    
    def _flush():
        nonlocal ok, fail, b_num
        b_num += 1
        try:
            if payload_batch:
                client.batch_update_points(collection_name=col, update_operations=payload_batch)
            if vector_batch:
                client.update_vectors(collection_name=col, points=vector_batch)
            ok += max(len(payload_batch), len(vector_batch))
            logger.info(f"  Batch {b_num}: updated payload/vectors for pts (total OK: {ok})")
        except Exception as e:
            fail += max(len(payload_batch), len(vector_batch))
            logger.error(f"  Batch {b_num} failed: {e}")
        payload_batch.clear()
        vector_batch.clear()

    for pid, payload, vecs in op_gen:
        if payload: payload_batch.append(SetPayloadOperation(set_payload=SetPayload(payload=payload, points=[pid])))
        if vecs: vector_batch.append(PointVectors(id=pid, vector=vecs))
        
        if len(payload_batch) >= batch_size or len(vector_batch) >= batch_size:
            _flush()

    if payload_batch or vector_batch:
        _flush()
            
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
    ocr_container = get_container_client(conn_str, args.ocr_container)
    
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    ensure_collection(qdrant, args.collection_name)
    
    meta_dict = load_metadata_csv(args.video_metadata_link)

    # ── Discover videos ──────────────────────────────────────────────
    video_map = discover_video_ids(emb_container)
    if not video_map:
        logger.error(f"No .npy blobs in container '{args.embeddings_container}'")
        sys.exit(1)
        
    det_map = discover_detection_jsons(det_container)

    logger.info("=" * 60)
    logger.info("Qdrant V2.1: 4-Vector Azure-Streaming Upsert (RAM-Optimized)")
    logger.info("=" * 60)
    logger.info(f"Embeddings:  {args.embeddings_container}")
    logger.info(f"Detections:  {args.detections_container}")
    logger.info(f"OCR Stream:  {args.ocr_container}")
    logger.info(f"Collection:  {args.collection_name}")
    logger.info(f"Batch size:  {args.batch_size}")
    logger.info(f"Videos:      {len(video_map)}")
    logger.info("")

    t0 = time.time()
    total_ok = total_fail = 0
    n_det = n_dense = 0

    video_ids = sorted(video_map.keys())
    
    update_payloads = [f.strip() for f in args.update_payloads.split(",") if f.strip()]
    update_vectors = [f.strip() for f in args.update_vectors.split(",") if f.strip()]
    
    for i, vid in enumerate(video_ids, 1):
        logger.info(f"[{i}/{len(video_ids)}] Processing: {vid}")

        emb_prefix = video_map[vid]
        det_blob = det_map.get(vid)

        # ── Stream frame indices ─────────────────────────────────────
        frame_indices = stream_json(emb_container, f"{emb_prefix}_frames.json")

        # ── Stream detection JSON ────────────────────────────────────
        det_data = stream_json(det_container, det_blob) if det_blob else None
        det_lookup = build_det_lookup(det_data)
        
        if det_data is not None:
             del det_data

        if det_lookup:
            n_det += 1
            logger.info(f"  Detection: {len(det_lookup)} frames loaded.")
        else:
            n_dense += 1
            logger.warning(f"  No detection JSON → dense/OCR only")

        # ── Setup OCR lookup & Meta ──────────────────────────────────
        ocr_lookup = build_ocr_lookup(ocr_container, vid)
        if ocr_lookup:
            logger.info(f"  OCR Text: {len(ocr_lookup)} mapped frames parsed via JSON.")
            
        vid_meta = meta_dict.get(vid, {})

        if args.mode == "update":
            # ── Update Payload/Vector Mode ─────────────────────────────
            try:
                op_gen = generate_updates(
                    vid, frame_indices, det_lookup, ocr_lookup, vid_meta, update_payloads, update_vectors
                )
                ok, fail = stream_updates(qdrant, args.collection_name, op_gen, args.batch_size)
                total_ok += ok
                total_fail += fail
            except Exception as exc:
                logger.error(f"  Update failed for {vid}: {exc}")
            
            del frame_indices, det_lookup, ocr_lookup
            gc.collect()
            continue

        # ── Full Upsert Mode (requires embeddings.npy) ───────────────
        embeddings = stream_npy(emb_container, f"{emb_prefix}.npy")
        if embeddings is None:
            logger.error(f"  Could not load {emb_prefix}.npy (required for full upsert)"); continue
        if embeddings.ndim != 2 or embeddings.shape[1] != SIGLIP_DIM:
            logger.error(f"  Bad shape {embeddings.shape}"); del embeddings; continue

        n = embeddings.shape[0]
        logger.info(f"  {n} frames streamed ({embeddings.nbytes / 1e6:.1f} MB)")

        # ── Stream-upsert via generator ──────────────────────────────
        try:
            point_gen = generate_points(
                vid, embeddings, frame_indices, det_lookup, ocr_lookup, vid_meta, azure_base
            )
            ok, fail = stream_upsert(qdrant, args.collection_name, point_gen, args.batch_size)
            total_ok += ok
            total_fail += fail
        except Exception as exc:
            logger.error(f"  Pipeline failed for {vid}: {exc}")

        # ── Aggressive memory cleanup after each video ───────────────
        del embeddings, frame_indices, det_lookup, ocr_lookup
        gc.collect()

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("Upsert V2.1 Complete!")
    logger.info("=" * 60)
    logger.info(f"Videos:     {len(video_ids)} ({n_det} w/ detections, {n_dense} dense-only)")
    logger.info(f"Upserted:   {total_ok}")
    logger.info(f"Failed:     {total_fail}")
    logger.info(f"Time:       {elapsed:.1f}s")
    if total_fail:
        logger.warning(f"{total_fail} failures — see qdrant_upsert_errors.log")


if __name__ == "__main__":
    main()