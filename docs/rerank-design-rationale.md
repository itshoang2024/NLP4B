# Rerank Design Rationale

Tài liệu này giải thích tại sao công thức multi-signal reranking hiện tại là lựa chọn hợp lý cho hệ thống retrieval, và so sánh với các approach thay thế.

## Mục lục

- [1. Bối cảnh hệ thống](#1-bối-cảnh-hệ-thống)
- [2. Công thức hiện tại](#2-công-thức-hiện-tại)
- [3. Tại sao công thức này hợp lý](#3-tại-sao-công-thức-này-hợp-lý)
- [4. So sánh với các approach khác](#4-so-sánh-với-các-approach-khác)
- [5. Khi nào nên upgrade](#5-khi-nào-nên-upgrade)
- [6. Kết luận](#6-kết-luận)

---

## 1. Bối cảnh hệ thống

> **Phạm vi tài liệu này:** Tài liệu này mô tả **reranker nội bộ** của nhánh agentic (`backend/src/services/agentic_retrieve/nodes/rerank.py`). Hệ thống sau refactor còn có **tầng rerank thứ hai** — cross-source RRF (`backend/src/controllers/rerank.py`) — để gộp kết quả từ nhánh agentic và heuristic. Xem [backend/README.md](../backend/README.md) để hiểu tổng thể.

Pipeline agentic retrieval có 5 bước tuần tự (sau refactor, normalization đã chuyển sang middleware):

```
QueryBundle → Intent Extraction → Routing → Parallel Retrieval → Fusion → Reranking
```

Trước khi tới bước Reranking, hệ thống đã:
- **Hiểu query** qua LLM intent extraction (objects, actions, scene, text_cues, ...)
- **Phân bổ trọng số** routing cho 5 sources (keyframe, caption, object, OCR, metadata)
- **Truy vấn song song** 5 nhánh Qdrant (dense + sparse)
- **Gộp kết quả** qua weighted-sum fusion với min-max normalization

Reranker là bước cuối cùng — nhận ~30 fused candidates và quyết định thứ tự top-20 trả về.

### Ràng buộc thiết kế

| Ràng buộc | Giá trị |
|---|---|
| Latency budget tổng pipeline | ≤ 15s (hiện tại ~10s với Azure API) |
| Latency budget riêng rerank | ≤ 50ms (reranker phải gần-zero-cost) |
| Số candidates đầu vào | ~30 |
| Candidates không có ảnh gốc | Chỉ có payload metadata (video_id, frame_id, scores, evidence) |
| Không có ground-truth labels | Chưa có evaluation dataset để train |

---

## 2. Công thức hiện tại

```
rerank_score = fused_score
             + α · cross_source_agreement
             + β · intent_coverage_bonus
             - γ · missing_modality_penalty
```

### Tầng 1 — Cross-Source Agreement (α = 0.15)

```
agreement = Σ (source_score[src] × routing_weight[src])   for src in evidence
```

Mỗi source "bỏ phiếu" cho candidate, với phiếu được tính theo cả **chất lượng match** (source_score) lẫn **tầm quan trọng** (routing_weight).

### Tầng 2 — Intent Coverage (β = 0.10, γ = 0.08)

Xác định **modalities kỳ vọng** từ query intent, rồi kiểm tra candidate có đáp ứng:

| Intent signal | Modality kỳ vọng |
|---|---|
| `text_cues ≠ []` | `ocr` phải có trong evidence |
| `objects ≠ []` | `object` phải có trong evidence |
| `actions` hoặc `scene ≠ []` | `caption` phải có trong evidence |

```
coverage_bonus  = β × (|present ∩ expected| / |expected|)
missing_penalty = γ × (|expected − present| / |expected|)
```

---

## 3. Tại sao công thức này hợp lý

### 3.1. Tận dụng tối đa thông tin đã có

Pipeline đã trả giá ~10s để thu thập 5 loại evidence. Bản nháp cũ (`+0.03 × len(evidence)`) lãng phí hầu hết thông tin này. Công thức mới khai thác **3 nguồn tín hiệu** mà không cần bất kỳ lời gọi API nào thêm:

| Tín hiệu | Nguồn | Chi phí thêm |
|---|---|---|
| `source_scores` | Có sẵn từ fusion | 0 |
| `routing_weights` | Có sẵn từ routing | 0 |
| `query_intent` | Có sẵn từ intent extraction | 0 |

→ **Zero additional latency, zero additional API cost.**

### 3.2. Nguyên lý ensemble voting có cơ sở lý thuyết

Cross-source agreement dựa trên nguyên lý **Condorcet Jury Theorem** trong lý thuyết voting: Nếu nhiều hệ thống *độc lập* cùng đồng ý về một quyết định, xác suất quyết định đó đúng tăng theo cấp số nhân.

Trong hệ thống này, 5 retrieval sources sử dụng **mô hình khác nhau** (SigLIP, BGE-M3, BM25) trên **biểu diễn khác nhau** (pixel, caption text, object tags, OCR text). Chúng gần như độc lập — nên đồng thuận giữa chúng là tín hiệu mạnh.

**Quan trọng:** Công thức tính `score × weight` (weighted voting) thay vì chỉ đếm số source. Điều này tránh trường hợp 3 source match yếu thắng 1 source match rất mạnh.

### 3.3. Intent coverage mã hóa "hard constraints" ngầm

Nhiều query có yêu cầu **bắt buộc** (must-have) mà fusion score thuần túy không thể capture:

**Ví dụ:** Query "Khung hình có chữ 'Quân A.P'"
- Frame A: caption match tốt (score 0.8) nhưng **không có OCR evidence** → có thể chỉ là frame có cảnh liên quan nhưng không chứa chữ
- Frame B: OCR match (score 0.7) → chắc chắn có chữ "Quân A.P" trên hình

Không có intent coverage penalty, Frame A có thể rank cao hơn Frame B dù Frame B mới thực sự đúng.

### 3.4. Interpretability — Debug được

Mỗi candidate mang theo `rerank_signals` dict:

```json
{
  "fused_score": 0.85,
  "agreement_bonus": 0.084,
  "coverage_bonus": 0.10,
  "missing_penalty": 0.0,
  "expected_modalities": ["ocr"],
  "present_modalities": ["ocr"],
  "missing_modalities": []
}
```

Khi ranking sai, có thể nhìn breakdown để biết *tại sao* — điều mà một neural reranker không làm được. Trong giai đoạn phát triển khi chưa có evaluation dataset, khả năng debug này **quan trọng hơn accuracy marginal**.

### 3.5. Coefficients có thể tune offline

α, β, γ là keyword arguments — có thể sweep trên evaluation set mà không sửa code. Khoảng search nhỏ (chỉ 3 scalar) nên có thể grid search nhanh khi có labeled data.

---

## 4. So sánh với các approach khác

### 4.1. VLM Reranker (Vision-Language Model)

**Ý tưởng:** Gọi VLM (Gemini, GPT-4V) để verify top-K candidates — gửi query + ảnh keyframe → VLM trả về relevance score.

```
Ưu điểm:
✅ Accuracy cao nhất — VLM có thể "nhìn" ảnh và đánh giá semantic match
✅ Bắt được subtleties mà text-based scoring bỏ lỡ
✅ Xử lý tốt trường hợp mơ hồ

Nhược điểm:
❌ Latency rất cao: ~1-3s per image × 20 candidates = 20-60s thêm
❌ API cost: ~$0.01-0.05 per query (×20 images)
❌ Cần ảnh gốc — pipeline hiện tại chỉ có metadata tại thời điểm rerank
❌ Rate limiting: Gemini free tier ~15 RPM → bottleneck nghiêm trọng
❌ Black-box: không debug được tại sao VLM cho score cao/thấp
```

**Khi nào hợp lý:** Chỉ khi có budget API lớn, latency budget > 30s, VÀ cần accuracy tuyệt đối (ví dụ: production search engine có revenue implications).

**So với approach hiện tại:**

| Tiêu chí | Multi-signal (hiện tại) | VLM Reranker |
|---|---|---|
| Latency thêm | ~0ms | +20-60s |
| API cost/query | $0 | ~$0.01-0.05 |
| Cần ảnh gốc | Không | Có |
| Interpretability | Cao (signal breakdown) | Thấp |
| Accuracy tiềm năng | Tốt | Rất tốt |
| Phù hợp giai đoạn hiện tại | ✅ | ❌ |

### 4.2. Cross-Encoder / Cross-Modal Reranker

**Ý tưởng:** Dùng model chuyên biệt (VLReranker, ColBERT, cross-encoder) để tính relevance score giữa query và candidate.

**Variant A — Text-only cross-encoder (query × caption)**

```
Ưu điểm:
✅ Chính xác hơn bi-encoder khi so sánh semantic similarity
✅ Không cần ảnh — chỉ cần caption payload
✅ Latency hợp lý: ~50-200ms cho 30 candidates (batch inference)

Nhược điểm:
❌ Cần load thêm model (~500MB-2GB RAM)
❌ Chỉ xét caption — bỏ lỡ OCR, object evidence
❌ Phụ thuộc chất lượng caption gốc
❌ Giảm lợi thế multi-source diversity của pipeline
```

**Variant B — Vision-Language cross-encoder (query × image)**

```
Tương tự VLM reranker nhưng dùng model nhỏ hơn (CLIP-reranker, BLIP-2).

Ưu điểm:
✅ Accuracy tốt hơn text-only, nhẹ hơn full VLM
✅ Batch inference có thể chấp nhận được

Nhược điểm:
❌ Vẫn cần ảnh gốc
❌ Cần GPU hoặc thêm Azure API endpoint
❌ Model phụ thuộc — phải align với SigLIP ecosystem
❌ Thêm dependency nặng
```

**So với approach hiện tại:**

| Tiêu chí | Multi-signal | Text cross-encoder | VL cross-encoder |
|---|---|---|---|
| Latency thêm | ~0ms | +50-200ms | +500ms-2s |
| Bộ nhớ thêm | 0 | +500MB-2GB | +2-5GB |
| Cần ảnh gốc | Không | Không | Có |
| Dependency mới | Không | cross-encoder model | VL model + GPU |
| Tận dụng multi-source | ✅ Tất cả 5 sources | ❌ Chỉ caption | ❌ Chỉ image |
| Interpretability | Cao | Thấp | Thấp |

### 4.3. Learning-to-Rank (LTR)

**Ý tưởng:** Train một ML model (LambdaMART, neural LTR) trên labeled data để tối ưu ranking metric (nDCG, MAP).

```
Ưu điểm:
✅ Tối ưu trực tiếp ranking metric
✅ Tự học trọng số tối ưu cho từng feature
✅ Có thể kết hợp tất cả signal hiện có thành feature vector

Nhược điểm:
❌ Cần labeled data (query, relevant frame pairs) — CHƯA CÓ
❌ Risk overfitting trên dataset nhỏ
❌ Thêm training pipeline + model serving
❌ Mất interpretability (neural LTR)
```

**So với approach hiện tại:**

| Tiêu chí | Multi-signal | LTR |
|---|---|---|
| Cần labeled data | Không | Có (chưa có) |
| Interpretability | Cao | Thấp-Trung bình |
| Tối ưu metric | Heuristic | Trực tiếp |
| Training overhead | Không | Có |
| Cold-start viable | ✅ | ❌ |

### 4.4. Reciprocal Rank Fusion (RRF) thay thế

**Ý tưởng:** Thay fusion + rerank bằng RRF thuần túy: `score = Σ 1/(k + rank_in_source)`.

```
Ưu điểm:
✅ Đơn giản, proven trong IR literature
✅ Robust với score scale khác nhau giữa sources

Nhược điểm (cho agentic internal rerank):
❌ Không xét intent — mọi source đều bình đẳng
❌ Không phạt missing modality
❌ Bỏ lỡ thông tin routing_weights đã compute
❌ Không tận dụng query understanding pipeline phía trước
```

**Approach hiện tại ưu việt hơn cho agentic internal rerank vì:** Pipeline đã đầu tư vào intent extraction và routing. RRF bỏ qua hoàn toàn các output này — phí effort đã bỏ ra ở các node trước.

> **Lưu ý:** Sau refactor, hệ thống **có sử dụng RRF** — nhưng ở tầng **cross-source rerank** (`backend/src/controllers/rerank.py`) để gộp kết quả từ nhánh agentic và heuristic. Đây là use case phù hợp cho RRF: hai nhánh sử dụng methodology khác hẳn nhau nên score scale không tương thích trực tiếp, và không có intent-level signal nào áp dụng được ở tầng này.

---

## 5. Khi nào nên upgrade

### Phase hiện tại: Multi-signal formula ✅
- Không có labeled data → không thể train
- Cần nhanh, nhẹ, debug được → heuristic + interpretability
- Pipeline latency đã ~10s → không có room cho thêm model inference

### Phase tiếp theo (khi có evaluation dataset):
1. **Tune coefficients** — Grid search α, β, γ trên labeled queries
2. **Thêm payload keyword matching** (Tầng 3 — đã thiết kế nhưng chưa implement)
3. **Text cross-encoder trên Azure** — Nếu caption quality đủ tốt, host cross-encoder model trên Azure VM

### Phase xa hơn (production):
4. **VLM verify** — Gọi Gemini Vision cho top-5 candidates sau rerank (chỉ 5 ảnh thay vì 20)
5. **LTR** — Khi có đủ labeled data (~1000+ queries), train LambdaMART với features = tất cả signal hiện có

---

## 6. Kết luận

| Câu hỏi | Trả lời |
|---|---|
| Công thức có cơ sở lý thuyết không? | Có — dựa trên ensemble voting theory và intent-aware retrieval |
| Có tận dụng tốt thông tin pipeline? | Có — khai thác intent, routing weights, source scores mà 0 cost |
| Có tốt hơn bản nháp cũ? | Có — phân biệt quality vs quantity of evidence, xét missing modality |
| Có phải lựa chọn tối ưu? | Không — VLM/cross-encoder có accuracy cao hơn, nhưng chưa phù hợp |
| Có phù hợp giai đoạn hiện tại? | **Có** — zero-cost, interpretable, tunable, no labeled data needed |

**Bottom line:** Công thức multi-signal reranking là **pareto-optimal** cho ràng buộc hiện tại (không có labeled data, latency ≤ 15s, cần debug). Khi ràng buộc thay đổi (có data, có budget), nên upgrade theo roadmap ở Section 5.
