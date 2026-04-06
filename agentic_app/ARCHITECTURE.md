# 📖 Tài liệu Kiến trúc — Multimodal Agentic RAG

> **Dự án:** ViFoodVQA — Tìm kiếm video đa phương thức (Multimodal Video Retrieval)
> **Cập nhật:** 2026-04-05
> **Tác giả:** NLP4B Team

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Cây thư mục](#2-cây-thư-mục)
3. [Module `config/` — Cấu hình](#3-module-config--cấu-hình)
4. [Module `tools/` — Bộ công cụ tìm kiếm](#4-module-tools--bộ-công-cụ-tìm-kiếm)
5. [Module `agent/` — LangGraph Agent (Phase 2)](#5-module-agent--langgraph-agent-phase-2)
6. [Module `ui/` — Giao diện Streamlit (Phase 3)](#6-module-ui--giao-diện-streamlit-phase-3)
7. [Module `tests/` — Kiểm thử](#7-module-tests--kiểm-thử)
8. [File `demo_retrieval.py` — Demo chạy thực](#8-file-demo_retrievalpy--demo-chạy-thực)
9. [Hạ tầng Docker (Phase 4)](#9-hạ-tầng-docker-phase-4)
10. [Luồng dữ liệu end-to-end](#10-luồng-dữ-liệu-end-to-end)
11. [Bảng vector Qdrant](#11-bảng-vector-qdrant)

---

## 1. Tổng quan kiến trúc

Hệ thống sử dụng pattern **"Agent-as-Router"**: LLM không dùng để sinh văn bản mà chỉ đóng vai **bộ điều phối** — phân tích câu hỏi → quyết định gọi tool nào → thu kết quả → trả về cho UI.

```
User Input
    │
    ▼
┌──────────────────────┐
│  LLM (Router)        │  Phân tích query → chọn tool + viết lại query
│  (Phase 2 — chưa     │
│   implement)         │
└──────────┬───────────┘
     ┌─────┴──────┬──────────────┐
     ▼            ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Visual   │ │ Semantic │ │ Keyword  │   ← 4 tools (hiện tại)
│ (SigLIP) │ │ (BGE-M3) │ │ (BM25)   │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     └─────┬──────┴─────┬──────┘
           ▼            ▼
      ┌─────────────────────┐
      │   RRF Reranker      │  Gộp + xếp hạng → Top K
      └──────────┬──────────┘
                 ▼
      ┌─────────────────────┐
      │   Formatter         │  Đóng gói → JSON chuẩn cho UI
      └──────────┬──────────┘
                 ▼
          Streamlit UI
```

---

## 2. Cây thư mục

```
agentic_app/
├── config/                   # Cấu hình hệ thống
│   ├── __init__.py
│   ├── settings.py           # Pydantic Settings (đang là placeholder)
│   ├── .env.example          # Mẫu biến môi trường
│   └── .env                  # File thật (KHÔNG push lên Git)
│
├── tools/                    # ⭐ Bộ công cụ tìm kiếm — PHẦN CHÍNH
│   ├── __init__.py
│   ├── models.py             # Pydantic data models dùng chung
│   ├── embeddings.py         # Lazy-loaded embedding models
│   ├── search_visual.py      # Tool #1: SigLIP dense search
│   ├── search_semantic.py    # Tool #2: BGE-M3 dense search
│   ├── search_keyword.py     # Tool #3: BM25 sparse search (OCR + Objects)
│   ├── reranker.py           # Thuật toán Reciprocal Rank Fusion
│   └── formatter.py          # Đóng gói kết quả → JSON cho UI
│
├── agent/                    # LangGraph state machine (Phase 2 — chưa implement)
│   ├── __init__.py
│   ├── state.py              # AgentState TypedDict
│   ├── nodes.py              # Các node functions cho graph
│   └── graph.py              # StateGraph definition + compile
│
├── ui/                       # Streamlit frontend (Phase 3 — chưa implement)
│   ├── __init__.py
│   ├── app.py                # Entry point: streamlit run ui/app.py
│   └── components.py         # UI components tái sử dụng
│
├── tests/                    # pytest test files
│   ├── __init__.py
│   ├── test_tools.py         # Unit tests cho tools/
│   └── test_agent.py         # Integration tests cho agent/
│
├── demo_retrieval.py         # ⭐ File demo chạy được ngay (standalone)
├── requirements.txt          # Danh sách thư viện Python
├── Dockerfile                # Container image cho production
├── docker-compose.yml        # Orchestration cho Azure VM
├── plan.md                   # Kế hoạch phát triển 4 phases
├── ARCHITECTURE.md           # ← Bạn đang đọc file này
└── .gitignore                # Ignore Python cache, .env, logs
```

---

## 3. Module `config/` — Cấu hình

### `config/.env.example`

File mẫu chứa **tất cả biến môi trường** cần thiết. Mỗi thành viên clone repo về cần copy file này thành `.env` rồi điền giá trị thật.

| Biến | Ý nghĩa | Ví dụ |
|------|---------|-------|
| `QDRANT_URL` | URL cluster Qdrant Cloud | `https://abc-xyz.cloud.qdrant.io` |
| `QDRANT_API_KEY` | API key xác thực Qdrant | `qdr_xxx...` |
| `QDRANT_COLLECTION_NAME` | Tên collection chứa keyframes | `keyframes_v1` |
| `LLM_BASE_URL` | Endpoint OpenAI-compatible của LLM trên Azure VM | `http://10.0.0.4:8000/v1` |
| `LLM_API_KEY` | Token xác thực LLM endpoint | `token-abc123` |
| `LLM_MODEL_NAME` | Tên model đang serve | `Qwen/Qwen2.5-7B-Instruct` |
| `EMBEDDING_MODEL_NAME` | Model dùng cho query-time encoding | `BAAI/bge-m3` |
| `STREAMLIT_PORT` | Port chạy Streamlit | `8501` |
| `LOG_LEVEL` | Mức log (DEBUG/INFO/WARNING/ERROR) | `INFO` |

### `config/settings.py`

**Trạng thái: Placeholder** — sẽ implement Pydantic `BaseSettings` class ở Phase 2.

Khi hoàn thiện, file này sẽ:
- Tự đọc `.env` file
- Validate tất cả biến bắt buộc khi app khởi động
- Cung cấp object `settings` dùng chung cho mọi module

---

## 4. Module `tools/` — Bộ công cụ tìm kiếm

Đây là **phần cốt lõi** của hệ thống. Mỗi file trong `tools/` là một khối chức năng độc lập, có thể test riêng lẻ.

### 4.1 `tools/models.py` — Data Models

Định nghĩa **2 Pydantic model** dùng chung cho toàn bộ pipeline:

#### `RetrievedFrame`

Đại diện cho **1 keyframe** được trả về từ Qdrant sau khi search.

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `point_id` | `str` | ID duy nhất của point trong Qdrant (UUID5 tạo từ `video_id + frame_idx`) |
| `video_id` | `str` | Mã video gốc (VD: `video_001`) |
| `frame_idx` | `int` | Chỉ số frame trong video (VD: frame thứ 150) |
| `timestamp_sec` | `int` | Mốc thời gian tính bằng giây = `frame_idx / fps` |
| `youtube_link` | `str` | Link YouTube cắt đúng giây (VD: `https://youtube.com/watch?v=xxx&t=42s`) |
| `azure_url` | `str` | URL ảnh keyframe trên Azure Blob Storage |
| `caption` | `str` | Câu mô tả nội dung frame (từ object detection phase) |
| `ocr_text` | `str` | Chữ được nhận diện trên frame (OCR) |
| `tags` | `List[str]` | Danh sách tags vật thể phát hiện được (VD: `["person", "car", "tree"]`) |
| `score` | `float` | Điểm similarity từ Qdrant (cosine/dot-product) |
| `source_vector` | `str` | Tên vector space đã dùng để tìm ra frame này |

#### `FormattedResult`

Kết quả **cuối cùng** gửi lên UI — đã gộp điểm RRF, sẵn sàng render.

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `video_id` | `str` | Mã video |
| `timestamp_sec` | `int` | Mốc thời gian (giây) |
| `youtube_url` | `str` | Link YouTube chính xác đến giây |
| `image_url` | `str` | URL ảnh keyframe (Azure Blob) |
| `caption` | `str` | Caption mô tả |
| `ocr_text` | `str` | Chữ trên màn hình |
| `rrf_score` | `float` | Điểm sau khi gộp RRF (cao = liên quan hơn) |

---

### 4.2 `tools/embeddings.py` — Embedding Models

File này quản lý **3 model AI** dùng để **encode query** (câu hỏi) thành vector tại thời điểm tìm kiếm.

#### Thiết kế: Lazy Singleton

```
Lần gọi đầu tiên  → Tải model từ HuggingFace (~20s) → Lưu vào biến global
Lần gọi thứ 2+    → Dùng lại biến global → Tức thì (0ms)
```

Model được cache tại `~/.cache/huggingface/hub/`. Lần chạy đầu sẽ tải ~2-4GB, sau đó offline.

#### 3 Model được sử dụng

| Hàm | Model | Output | Khi nào dùng |
|-----|-------|--------|-------------|
| `encode_siglip_text(query)` | `google/siglip-so400m-patch14-384` | `list[float]` — 1152 chiều | Khi tìm frame bằng mô tả hình ảnh (màu sắc, vật thể nhìn thấy) |
| `encode_bge_m3(query)` | `BAAI/bge-m3` | `list[float]` — 1024 chiều | Khi tìm frame bằng ý nghĩa ngữ nghĩa (caption matching) |
| `encode_bm25(query)` | `Qdrant/bm25` | `SparseVector` (indices + values) | Khi tìm frame bằng từ khóa chính xác (OCR text, object tags) |

#### Tham số chi tiết

**`encode_siglip_text(query: str) → list[float]`**
- `query`: Câu mô tả hình ảnh (VD: `"người mặc áo xanh đang nấu ăn"`)
- **Cách hoạt động**: Dùng SigLIP text encoder để tạo vector 1152d. Đây là model **cross-modal** — nó hiểu mối quan hệ giữa text và image. Vector này sẽ được so khớp với vector ảnh (đã tính sẵn từ phase preprocessing)
- **Return**: Mảng 1152 số float, hoặc `[]` nếu query rỗng

**`encode_bge_m3(query: str) → list[float]`**
- `query`: Câu mô tả ngữ nghĩa (VD: `"cảnh nấu phở bò truyền thống"`)
- **Cách hoạt động**: Model multilingual (đa ngôn ngữ), encode text thành vector 1024d. So sánh với caption đã encode từ phase preprocessing
- **Return**: Mảng 1024 số float chuẩn hóa L2, hoặc `[]` nếu rỗng

**`encode_bm25(query: str) → SparseVector | None`**
- `query`: Từ khóa tìm kiếm (VD: `"person car"` hoặc `"giá: 50.000đ"`)
- **Cách hoạt động**: Tạo sparse vector theo thuật toán BM25 — chỉ lưu các token có ý nghĩa cùng trọng số TF-IDF
- **Return**: Object `SparseVector(indices=[...], values=[...])` hoặc `None` nếu không encode được

---

### 4.3 `tools/search_visual.py` — Tìm kiếm bằng hình ảnh

**Decorator `@tool`**: Đánh dấu để sau này LangGraph Agent nhận diện được hàm này là một công cụ khả dụng.

```python
@tool
def search_visual(query: str, top_k: int = 20) -> List[RetrievedFrame]
```

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `query` | `str` | *(bắt buộc)* | Mô tả cảnh muốn tìm bằng ngôn ngữ tự nhiên |
| `top_k` | `int` | `20` | Số lượng kết quả trả về tối đa |

**Cách hoạt động:**
1. Encode `query` → vector 1152d bằng SigLIP
2. Gửi vector tới Qdrant, tìm trên named vector `"keyframe-dense"`
3. Qdrant trả về top_k frame có cosine similarity cao nhất
4. Map payload → `RetrievedFrame` objects

**Khi nào dùng:** Khi query mô tả **hình dáng, màu sắc, hành động trực quan** (VD: "người mặc áo đỏ đứng trước cửa hàng")

---

### 4.4 `tools/search_semantic.py` — Tìm kiếm ngữ nghĩa

```python
@tool
def search_semantic(query: str, top_k: int = 20) -> List[RetrievedFrame]
```

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `query` | `str` | *(bắt buộc)* | Câu hỏi/mô tả mang tính ngữ nghĩa |
| `top_k` | `int` | `20` | Số kết quả tối đa |

**Cách hoạt động:**
1. Encode `query` → vector 1024d bằng BGE-M3
2. Tìm trên named vector `"keyframe-caption-dense"`
3. So khớp với caption đã encode sẵn từ phase preprocessing

**Khi nào dùng:** Khi query thiên về **ý nghĩa, ngữ cảnh, logic** (VD: "cách chế biến gỏi cuốn", "khoảnh khắc vui vẻ trong bữa ăn gia đình")

---

### 4.5 `tools/search_keyword.py` — Tìm kiếm từ khóa

```python
@tool
def search_keyword(query: str, target: str = "ocr", top_k: int = 20) -> List[RetrievedFrame]
```

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `query` | `str` | *(bắt buộc)* | Từ khóa cần tìm |
| `target` | `str` | `"ocr"` | Chọn vector space: `"ocr"` = tìm chữ trên màn hình, `"objects"` = tìm theo tên vật thể |
| `top_k` | `int` | `20` | Số kết quả tối đa |

**Mapping `target` → vector space:**
- `"ocr"` → `"keyframe-ocr-sparse"` (chữ trên màn hình — OCR text)
- `"objects"` → `"keyframe-object-sparse"` (nhãn vật thể — object tags)

**Khi nào dùng:** Khi cần tìm chính xác từ xuất hiện trên màn hình (VD: tên món ăn, giá tiền, biển hiệu) hoặc tên vật thể cụ thể (`"person"`, `"bowl"`, `"chopsticks"`)

---

### 4.6 `tools/reranker.py` — Thuật toán gộp điểm RRF

```python
def rerank(result_sets: List[List[RetrievedFrame]], k: int = 60, top_n: int = 5) -> List[RetrievedFrame]
```

> **Lưu ý:** Hàm này **KHÔNG** phải là `@tool`. Agent không gọi trực tiếp hàm này — nó được gọi bởi node trong LangGraph graph (hoặc bởi `demo_retrieval.py`).

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `result_sets` | `List[List[RetrievedFrame]]` | *(bắt buộc)* | Mảng các tập kết quả — mỗi tập từ 1 tool |
| `k` | `int` | `60` | Hằng số RRF (theo paper gốc của Cormack et al. 2009). Giá trị lớn hơn → ưu tiên cân bằng hơn |
| `top_n` | `int` | `5` | Số lượng kết quả cuối cùng trả về |

**Công thức RRF:**

```
RRF_score(document) = Σ  1 / (k + rank_i)
```

Trong đó `rank_i` là thứ hạng (1-indexed) của document trong tập kết quả `i`.

**Ví dụ:** Nếu một frame xếp hạng #1 trong visual search (1/(60+1) = 0.0164) và hạng #3 trong semantic search (1/(60+3) = 0.0159), tổng RRF = 0.0323. Frame xuất hiện ở nhiều tools sẽ có điểm cao hơn frame chỉ xuất hiện ở 1 tool → **đó chính là sức mạnh của RRF**.

---

### 4.7 `tools/formatter.py` — Đóng gói kết quả

```python
def format_results(frames: List[RetrievedFrame]) -> List[dict]
```

| Tham số | Kiểu | Mô tả |
|---------|------|-------|
| `frames` | `List[RetrievedFrame]` | Danh sách frames sau RRF (đã sắp xếp) |

**Return:** Mảng dict, mỗi dict chứa:
```json
{
  "video_id": "video_001",
  "timestamp_sec": 42,
  "youtube_url": "https://youtube.com/watch?v=xxx&t=42s",
  "image_url": "https://blob.core.windows.net/.../video_001_00150.jpg",
  "caption": "A person cooking pho in a kitchen",
  "ocr_text": "Phở Hà Nội - 50.000đ",
  "rrf_score": 0.032787
}
```

> **Quan trọng:** Hàm này **KHÔNG gọi LLM**. Nó là pure data transformation — nhận frame objects, xuất dict chuẩn cho Streamlit render.

---

## 5. Module `agent/` — LangGraph Agent (Phase 2)

**Trạng thái:** Placeholder — chưa implement. Sẽ xây dựng sau khi `tools/` và `demo_retrieval.py` được kiểm chứng hoạt động tốt.

| File | Vai trò dự kiến |
|------|----------------|
| `state.py` | Định nghĩa `AgentState` (TypedDict) — cấu trúc dữ liệu chảy qua các node trong graph |
| `nodes.py` | Các hàm xử lý: `retrieve_node`, `rerank_node`, `format_node` — mỗi hàm nhận state và trả về state cập nhật |
| `graph.py` | Chắp nối các nodes thành StateGraph: `START → LLM Router → Tools → Rerank → Format → END`, rồi compile thành graph chạy được |

---

## 6. Module `ui/` — Giao diện Streamlit (Phase 3)

**Trạng thái:** Placeholder — chưa implement.

| File | Vai trò dự kiến |
|------|----------------|
| `app.py` | Entry point chính: `streamlit run ui/app.py`. Bao gồm chat input, message loop, session state |
| `components.py` | Các component tái sử dụng: sidebar controls, source citation cards (hiển thị ảnh keyframe + metadata) |

---

## 7. Module `tests/` — Kiểm thử

| File | Mô tả |
|------|-------|
| `test_tools.py` | Unit test cho từng tool: mock Qdrant client, kiểm tra output format |
| `test_agent.py` | Integration test cho graph end-to-end: mock LLM + Qdrant, kiểm tra luồng hoàn chỉnh |

Chạy: `pytest tests/ -v`

---

## 8. File `demo_retrieval.py` — Demo chạy thực

**Đây là file runnable đầu tiên** — chạy ngay trên laptop, không cần Docker/Agent/LLM.

### Cách chạy

```bash
cp config/.env.example config/.env   # Rồi điền QDRANT_URL + QDRANT_API_KEY
pip install qdrant-client fastembed sentence-transformers transformers torch python-dotenv
cd agentic_app
python demo_retrieval.py
```

### Cấu trúc bên trong

| Section | Dòng | Chức năng |
|---------|------|-----------|
| `0. Load env` | 30-38 | Đọc `.env`, kiểm tra biến bắt buộc |
| `1. Qdrant Client` | 41-44 | Kết nối Qdrant Cloud, hiện số points |
| `2. Lazy singletons` | 48-80 | Tải model (SigLIP, BGE-M3, BM25) lần đầu dùng |
| `3. Encoding helpers` | 84-115 | Hàm encode query → vector cho mỗi model |
| `4. Bộ 4 hàm Search` | 120-186 | `search_visual`, `search_semantic`, `search_object_tags`, `search_ocr` |
| `5. RRF` | 192-234 | Thuật toán Reciprocal Rank Fusion |
| `6. Display` | 238-262 | Pretty-print kết quả ra terminal |
| `7. Tool Registry` | 266-272 | Dict `{"visual": fn, ...}` — bảng tra cứu tool theo tên |
| `8. Interactive CLI` | 276+ | Vòng lặp: nhập query → chọn tools → chạy → in Top 5 |

### Ví dụ sử dụng

```
🔍 Query: người mặc áo đỏ đang nấu ăn trong bếp
🔧 Tools [visual,semantic,objects,ocr] (hoặc 'all'): visual,semantic

   ▶ Chạy 2 tool(s): ['visual', 'semantic']
   ⏳ visual... ✅ 20 hits (3.2s)
   ⏳ semantic... ✅ 20 hits (0.8s)

══════════════════════════════════════════════════════════════════════
  🏆 Top 5 RRF — query: "người mặc áo đỏ đang nấu ăn trong bếp"
══════════════════════════════════════════════════════════════════════

  #1  RRF=0.032787
  ├─ Video:     video_023
  ├─ Frame:     150  (t=42s)
  ├─ YouTube:   https://youtube.com/watch?v=xxx&t=42s
  ├─ Image:     https://blob.core.windows.net/.../video_023_00150.jpg
  ├─ Caption:   A person in red clothing cooking in a kitchen
  └─ Matched:   [keyframe-dense, keyframe-caption-dense]
```

---

## 9. Hạ tầng Docker (Phase 4)

### `Dockerfile`

| Lệnh | Ý nghĩa |
|-------|---------|
| `FROM python:3.11-slim` | Base image nhẹ (~120MB) |
| `apt-get install build-essential curl` | Cần cho compile một số thư viện Python + health check |
| `COPY requirements.txt` → `pip install` | Cài dependencies (tách layer để cache Docker) |
| `COPY . .` | Copy toàn bộ source code |
| `EXPOSE 8501` | Mở port Streamlit |
| `HEALTHCHECK` | Docker tự kiểm tra app còn sống không (mỗi 30s) |
| `ENTRYPOINT ["streamlit", "run", ...]` | Chạy Streamlit ở chế độ headless, bind `0.0.0.0` |

### `docker-compose.yml`

```yaml
services:
  agentic-app:
    build: .                    # Build từ Dockerfile cùng thư mục
    ports: ["8501:8501"]        # Map port ra ngoài host
    env_file: config/.env       # Inject env vars từ file .env
    restart: unless-stopped     # Auto-restart nếu crash
    volumes: ["./logs:/app/logs"]  # Mount logs ra host để theo dõi
```

**Deploy:** `docker-compose up -d` trên Azure VM → truy cập `http://<vm-ip>:8501`

---

## 10. Luồng dữ liệu end-to-end

```
1. User gõ: "Tìm cảnh ông già mặc áo xanh bị bắn"

2. [Hiện tại — demo_retrieval.py]
   User chọn tools: visual, semantic

3. search_visual:
   encode_siglip("old man blue shirt shot") → [0.12, -0.34, ...] (1152d)
   → Qdrant.query("keyframe-dense", vector, top_k=20)
   → 20 RetrievedFrame objects

4. search_semantic:
   encode_bge_m3("old man blue shirt shot") → [0.56, 0.78, ...] (1024d)
   → Qdrant.query("keyframe-caption-dense", vector, top_k=20)
   → 20 RetrievedFrame objects

5. RRF Reranker:
   Input: [[20 frames from visual], [20 frames from semantic]]
   → Gộp → Deduplicate → Tính điểm RRF
   → Output: Top 5 frames (sắp xếp theo RRF score giảm dần)

6. Formatter:
   Top 5 RetrievedFrame → [{video_id, timestamp, youtube_url, image_url, ...}]

7. UI hiển thị: Ảnh keyframe + link YouTube + metadata
```

---

## 11. Bảng vector Qdrant

Collection `keyframes_v1` chứa **4 named vectors** trên mỗi point:

| Named Vector | Model | Kiểu | Kích thước | Mô tả |
|-------------|-------|------|-----------|-------|
| `keyframe-dense` | SigLIP | Dense (float32) | 1152d | Vector ảnh — encode trực tiếp từ pixel keyframe |
| `keyframe-caption-dense` | BGE-M3 | Dense (float32) | 1024d | Vector caption — encode từ `detailed_caption` |
| `keyframe-object-sparse` | BM25 | Sparse (indices+values) | Dynamic | Tags vật thể phát hiện (`unique_tags`) |
| `keyframe-ocr-sparse` | BM25 | Sparse (indices+values) | Dynamic | Chữ trên màn hình (OCR text) |

**Payload** trên mỗi point:

| Field | Kiểu | Ví dụ |
|-------|------|-------|
| `video_id` | string | `"video_023"` |
| `frame_idx` | int | `150` |
| `timestamp_sec` | int | `42` |
| `youtube_link` | string | `"https://youtube.com/watch?v=xxx&t=42s"` |
| `azure_url` | string | `"https://....blob.core.windows.net/keyframes/video_023/video_023_00150.jpg"` |
| `caption` | string | `"A person cooking in a kitchen"` |
| `detailed_caption` | string | `"A person wearing red shirt is preparing pho..."` |
| `ocr_text` | string | `"Phở Hà Nội 50.000đ"` |
| `tags` | list[string] | `["person", "bowl", "kitchen"]` |
| `object_counts` | dict | `{"person": 2, "bowl": 3}` |

---

> **Câu hỏi?** Liên hệ team qua GitHub Issues hoặc chat nhóm.
