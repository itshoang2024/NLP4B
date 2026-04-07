# 📡 API Contract — Embedding-as-a-Service

> **Base URL:** `http://<VM_IP>:8000`
> **Version:** 1.0.0
> **Auth:** None (internal network only)

---

## Endpoints Overview

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| `GET` | `/health` | Service health check | JSON |
| `POST` | `/embed/semantic` | BGE-M3 dense vector (1024d) | `DenseResponse` |
| `POST` | `/embed/sparse` | BM25 sparse vector | `SparseResponse` |
| `POST` | `/embed/visual` | SigLIP dense vector (1152d) | `DenseResponse` |
| `POST` | **`/embed/query`** | **⭐ Unified — all 4 vectors + NLP** | **`QueryResponse`** |
| `GET` | `/docs` | Interactive Swagger UI | HTML |

---

## Models & Qdrant Mapping

| Endpoint | Model | Vector Name in Qdrant | Dimension |
|----------|-------|-----------------------|-----------|
| `/embed/semantic` | `BAAI/bge-m3` | `keyframe-caption-dense` | 1024 (float32) |
| `/embed/sparse` | `Qdrant/bm25` | `keyframe-ocr-sparse` / `keyframe-object-sparse` | dynamic |
| `/embed/visual` | `google/siglip-so400m-patch14-384` | `keyframe-dense` | 1152 (float32) |

---

## `GET /health`

Health check — no auth required.

**Response `200 OK`:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "models": {
    "semantic": "BAAI/bge-m3",
    "sparse": "Qdrant/bm25",
    "visual": "google/siglip-so400m-patch14-384"
  }
}
```

---

## `POST /embed/semantic`

Encode text → **BGE-M3 1024d** dense vector (L2-normalized).

### Request

```json
{
  "text": "người mặc áo đỏ đang nấu ăn"
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `text` | `string` | ✅ | min 1 char, max 8192 chars |

### Response `200 OK` — `DenseResponse`

```json
{
  "embedding": [0.0123, -0.0456, ...],
  "model": "BAAI/bge-m3",
  "dim": 1024,
  "latency_ms": 388.42
}
```

| Field | Type | Description |
|-------|------|-------------|
| `embedding` | `List[float]` | 1024 L2-normalized floats |
| `model` | `string` | Model identifier |
| `dim` | `int` | Vector dimension (always 1024) |
| `latency_ms` | `float` | Server-side inference time |

### Usage — Qdrant Search

```python
import httpx
from qdrant_client import QdrantClient

# 1. Get vector from VM
vec = httpx.post("http://<VM>:8000/embed/semantic",
                 json={"text": "nấu phở bò"}).json()["embedding"]

# 2. Search Qdrant
client = QdrantClient(url="...", api_key="...")
results = client.query_points(
    collection_name="keyframes_v1",
    query=vec,
    using="keyframe-caption-dense",
    limit=10,
)
```

---

## `POST /embed/sparse`

Encode text → **BM25 sparse vector** (indices + values).

### Request

```json
{
  "text": "person cooking kitchen bowl"
}
```

### Response `200 OK` — `SparseResponse`

```json
{
  "indices": [1234, 5678, 9012],
  "values": [0.85, 0.72, 0.61],
  "model": "Qdrant/bm25",
  "nnz": 3,
  "latency_ms": 0.12
}
```

| Field | Type | Description |
|-------|------|-------------|
| `indices` | `List[int]` | Non-negative token indices |
| `values` | `List[float]` | Positive TF-IDF weights |
| `model` | `string` | Model identifier |
| `nnz` | `int` | Number of non-zero entries |
| `latency_ms` | `float` | Server-side inference time |

### Usage — Qdrant Search

```python
from qdrant_client.models import SparseVector

resp = httpx.post("http://<VM>:8000/embed/sparse",
                  json={"text": "person bowl"}).json()

sparse_vec = SparseVector(indices=resp["indices"], values=resp["values"])
results = client.query_points(
    collection_name="keyframes_v1",
    query=sparse_vec,
    using="keyframe-ocr-sparse",   # or "keyframe-object-sparse"
    limit=10,
)
```

---

## `POST /embed/visual`

Encode text → **SigLIP 1152d** dense vector (cross-modal text→image).

### Request

```json
{
  "text": "a person in red shirt cooking in kitchen"
}
```

### Response `200 OK` — `DenseResponse`

```json
{
  "embedding": [0.0789, -0.0234, ...],
  "model": "google/siglip-so400m-patch14-384",
  "dim": 1152,
  "latency_ms": 715.30
}
```

### Usage — Qdrant Search

```python
vec = httpx.post("http://<VM>:8000/embed/visual",
                 json={"text": "red shirt cooking"}).json()["embedding"]

results = client.query_points(
    collection_name="keyframes_v1",
    query=vec,
    using="keyframe-dense",
    limit=10,
)
```

---

## `POST /embed/query` ⭐ Unified Endpoint

**One call → all 4 vectors.** NLP-parsed query → semantic + visual + object sparse + OCR sparse.

### Pipeline

```
User Query: '5 people sitting around table with "Coca Cola" sign'
  │
  ├─→ regex: "Coca Cola" → BM25 → ocr_sparse
  ├─→ spaCy POS → [people, table, sign] (NOUNs only)
  │     └─→ WordNet filter (physical only) + Top-3 synonyms
  │           → "people person human table desk stand sign placard banner"
  │           → BM25 → object_sparse
  ├─→ full query → BGE-M3 → semantic_dense (1024d)
  └─→ full query → SigLIP → visual_dense (1152d)
```

### Request

```json
{ "text": "5 people sitting around table with \"Coca Cola\" sign" }
```

### Response `200 OK` — `QueryResponse`

| Field | Type | Description |
|-------|------|-------------|
| `semantic_dense` | `DenseResponse` | BGE-M3 1024d (L2-normalized) |
| `visual_dense` | `DenseResponse` | SigLIP 1152d |
| `object_sparse` | `SparseResponse` | BM25 from nouns + synonyms |
| `ocr_sparse` | `SparseResponse` | BM25 from quoted text |
| `nlp_analysis` | `dict` | Objects, counts, synonyms, OCR texts |
| `total_latency_ms` | `float` | End-to-end server time |

### WordNet Domain Filter

Only physical-object nouns pass (`noun.artifact`, `noun.person`, `noun.animal`, `noun.food`, `noun.plant`, `noun.body`). Abstract nouns filtered out.

---

## Error Responses

| Status | Cause | Body |
|--------|-------|------|
| `422` | Missing/empty `text`, text > 8192 chars, invalid JSON | Pydantic validation detail |
| `503` | Model not loaded (startup in progress) | `{"detail": "BGE-M3 model not loaded"}` |
| `405` | Wrong HTTP method (e.g. GET on POST endpoint) | Method Not Allowed |

### Example `422`:
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "text"],
      "msg": "String should have at least 1 character",
      "input": ""
    }
  ]
}
```

---

## Performance Benchmarks

Measured from local machine → Azure VM (`Standard_B4as_v2`, CPU-only):

| Endpoint | Server P95 | E2E P95 | Network Overhead |
|----------|------------|---------|------------------|
| `/embed/semantic` | ~388ms | ~1211ms | ~139ms |
| `/embed/sparse` | ~0.1ms | ~83ms | ~81ms |
| `/embed/visual` | ~715ms | ~715ms | — |

**Max throughput:** ~3.9 req/s at concurrency=2

---

## Hybrid Search (Colab Example)

Combine semantic + visual via Qdrant native RRF:

```python
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion

VM = "http://<VM_IP>:8000"
query = "người mặc áo đỏ nấu phở"

# Encode on VM (CPU)
bge_vec = httpx.post(f"{VM}/embed/semantic", json={"text": query}).json()["embedding"]
sig_vec = httpx.post(f"{VM}/embed/visual",   json={"text": query}).json()["embedding"]

# Hybrid search on Qdrant (RRF fusion)
client = QdrantClient(url="...", api_key="...")
results = client.query_points(
    collection_name="keyframes_v1",
    prefetch=[
        Prefetch(query=bge_vec, using="keyframe-caption-dense", limit=50),
        Prefetch(query=sig_vec, using="keyframe-dense", limit=50),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=10,
    with_payload=True,
)
```
