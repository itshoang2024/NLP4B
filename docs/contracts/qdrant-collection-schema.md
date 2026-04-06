# Qdrant Collection Schema — `keyframes_v1`

> **This is a schema contract.** Changes to this schema must be coordinated between the producer (`qdrant_upsert.py`) and all consumers (`qdrant_search.py`, `run_agentic_demo.py`, potentially Streamlit UI).

## Purpose

The `keyframes_v1` Qdrant collection is the central shared data structure between the offline data processing pipeline and the online agentic retrieval system. Each point represents a single video keyframe.

## Producer

- **Script:** [`data-processing/src/qdrant/qdrant_upsert.py`](../../data-processing/src/qdrant/qdrant_upsert.py)
- **Data sources:** Azure Blob Storage containers (`embeddings`, `object-detection`, `ocr`)
- **Point ID:** Deterministic UUID5 from `uuid.uuid5(NAMESPACE_URL, "{video_id}_{frame_idx}")`

## Consumers

- [`retrieval/agentic_retrieval/services/qdrant_search.py`](../../retrieval/agentic_retrieval/services/qdrant_search.py) — query-time vector search
- [`retrieval/agentic_retrieval/run_agentic_demo.py`](../../retrieval/agentic_retrieval/run_agentic_demo.py) — end-to-end demo
- [`azure-ai-provider/embedding_service/app.py`](../../azure-ai-provider/embedding_service/app.py) — must produce compatible embeddings

## Named Vectors

| Vector Name | Type | Dimensions | Distance | Model | Description |
|---|---|---|---|---|---|
| `keyframe-dense` | Dense | 1152 | Cosine | `google/siglip-so400m-patch14-384` | SigLIP image embedding — visual similarity search |
| `keyframe-caption-dense` | Dense | 1024 | Cosine | `BAAI/bge-m3` | BGE-M3 embedding of `detailed_caption` — semantic text search |
| `keyframe-object-sparse` | Sparse | — | — | `Qdrant/bm25` | BM25 encoding of unique object tags — lexical object matching |
| `keyframe-ocr-sparse` | Sparse | — | — | `Qdrant/bm25` | BM25 encoding of OCR text — text-in-image search |

### Vector population rules

- `keyframe-dense` is **always present** (required — comes from SigLIP `.npy` embeddings)
- `keyframe-caption-dense` is present **only if** the object detection JSON contains a non-empty `detailed_caption` for this frame
- `keyframe-object-sparse` is present **only if** the detection JSON contains at least one object label for this frame
- `keyframe-ocr-sparse` is present **only if** the OCR JSON contains non-empty text for this frame

## Payload Fields

| Field | Type | Required | Source | Description |
|---|---|---|---|---|
| `video_id` | `str` | ✅ | Embedding blob path | YouTube video ID or video folder name |
| `frame_idx` | `int` | ✅ | `_frames.json` | Original frame index in the source video |
| `azure_url` | `str` | ✅ | Constructed | Full URL to keyframe image in Azure Blob (`{base_url}/{video_id}/{filename}`) |
| `timestamp_sec` | `int` | ✅ | Computed | Estimated timestamp: `frame_idx / fps` |
| `youtube_link` | `str` | ✅ | video_metadata.csv | YouTube URL with `&t={timestamp_sec}s` parameter |
| `tags` | `list[str]` | ❌ | Object detection JSON | Sorted union of YOLO + Florence-2 labels |
| `caption` | `str` | ❌ | Object detection JSON | Florence-2 `<CAPTION>` output |
| `detailed_caption` | `str` | ❌ | Object detection JSON | Florence-2 `<MORE_DETAILED_CAPTION>` output |
| `object_counts` | `dict[str, int]` | ❌ | Object detection JSON | Label frequency map (e.g., `{"person": 3, "car": 1}`) |
| `ocr_text` | `str` | ❌ | OCR JSON | Cleaned OCR text extracted from the keyframe |

### Fields NOT currently present but referenced in search code

| Field | Status | Notes |
|---|---|---|
| `title` / `video_title` / `youtube_title` | **Not indexed** | `qdrant_search.py` probes for these fields. Currently returns `[]` for metadata searches. Adding a title field requires a schema update + re-upsert. |

## Naming Conventions

- **Frame filename:** `{video_id}_{frame_idx:05d}.jpg` (e.g., `L01_V001_00142.jpg`)
- **Point ID:** `uuid5(NAMESPACE_URL, "{video_id}_{frame_idx}")` — deterministic, so re-upserts are idempotent
- **Azure blob path:** `keyframes/{video_id}/{filename}` and `embeddings/{video_id}/{filename}`

## Collection Configuration

```python
# Created by ensure_collection() in qdrant_upsert.py
vectors_config = {
    "keyframe-dense":          VectorParams(size=1152, distance=Distance.COSINE),
    "keyframe-caption-dense":  VectorParams(size=1024, distance=Distance.COSINE),
}
sparse_vectors_config = {
    "keyframe-object-sparse":  SparseVectorParams(index=SparseIndexParams(on_disk=False)),
    "keyframe-ocr-sparse":     SparseVectorParams(index=SparseIndexParams(on_disk=False)),
}
```

## Failure Modes

| Failure | Cause | Symptom |
|---|---|---|
| Missing `keyframe-dense` vector | Embedding `.npy` not found or wrong shape | Point not created (upsert skipped) |
| Mismatched vector dimensions | Model version drift | Qdrant rejects upsert with dimension mismatch |
| Missing payload fields | No detection/OCR JSON for video | Downstream search returns results with empty `tags`, `caption`, `ocr_text` |
| Duplicate points | Re-running upsert | Overwritten safely (deterministic UUID5 IDs) |
| AV1 codec video | YouTube VP9/AV1 default | Embedding step may fail; `LMSKE.py` auto-transcodes to H.264 |

## Compatibility Risks

1. **Changing model IDs** in `qdrant_upsert.py` without re-indexing all points will put new points in a different vector space than old ones
2. **Changing the hash vocabulary size** (`_BM25_VOCAB_SIZE = 2**16`) in `qdrant_search.py` will make sparse query vectors incompatible with indexed ones (upsert uses `fastembed` BM25, search uses a hash-based approximation)
3. **Adding new payload fields** is safe — Qdrant is schema-flexible for payloads
4. **Adding new named vectors** requires calling `update_collection()` — existing points won't have the new vector until re-upserted

## Validation Checklist

Before changing the collection schema:

- [ ] Confirm all vector dimensions match between producer and consumer model versions
- [ ] Verify `COLLECTION_NAME` constant is identical in `qdrant_upsert.py` and `qdrant_search.py`
- [ ] Verify `VEC_*` vector name constants are identical across both files
- [ ] Test that `search_keyframe()`, `search_caption()`, `search_object()`, `search_ocr()` return results
- [ ] Check `DEFAULT_PAYLOAD_FIELDS` in search service includes all fields you need
- [ ] Run `retrieval/agentic_retrieval/test/test_qdrant_search.py` after changes
