# Agentic Retrieval

> ⚠️ **This module has been migrated to `backend/src/services/agentic_retrieve/`.** This README is kept as reference documentation for understanding the pipeline design. For the active codebase, setup instructions, and API contract, see [backend/README.md](../../backend/README.md).

LangGraph-based multi-branch retrieval pipeline for multimodal video keyframe search. Takes a natural-language query (Vietnamese or English), decomposes it into structured intent, routes to multiple vector search strategies, fuses results, and returns ranked keyframe candidates.

> **Post-migration differences:** In the `backend/` version, query normalization is handled by a shared middleware (not a graph node), `raw_query` was removed from `AgentState`, and the translator is consumed by `backend/src/middlewares/search_middleware.py` instead of `nodes/normalization.py`.

## Architecture

```
User query
    │
    ▼
┌──────────────────┐
│ Query             │  Clean text, detect language, translate to English
│ Normalization     │  → QueryBundle (raw, cleaned, lang, translated_en, rewrites)
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│ Intent            │  LLM (Gemini) extracts structured intent:
│ Extraction        │  objects, attributes, actions, scene, text_cues, metadata_cues
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│ Modality          │  Compute per-source retrieval weights based on intent
│ Routing           │  → routing_weights: {keyframe, caption, object, ocr, metadata}
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│ Parallel          │  Encode via Azure Embedding API, query Qdrant:
│ Retrieval         │  keyframe (SigLIP), caption (BGE-M3), object (BM25),
│                   │  OCR (BM25), metadata (lexical fallback)
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│ Candidate         │  Weighted-sum merge across sources
│ Fusion            │  Applies routing_weights × normalized_scores
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│ Frame             │  Multi-signal reranking:
│ Reranking         │  cross-source agreement + intent coverage
│                   │  → agent_topk: List[Candidate]
└──────────────────┘
```

## Structure

```
retrieval/agentic_retrieval/
├── graph.py                # Top-level LangGraph builder
├── state.py                # Typed state schema (AgentState, Candidate, etc.)
├── run_agentic_demo.py     # CLI demo with real services
├── .env.example            # Required environment variables template
├── requirements.txt        # Python dependencies
│
├── nodes/                  # Pipeline graph nodes
│   ├── normalization.py    # Query cleaning + translation
│   ├── intent_extraction.py # LLM-based structured intent extraction
│   ├── routing.py          # Modality routing weights
│   ├── retrieval.py        # Parallel multi-source Qdrant retrieval
│   ├── fusion.py           # Weighted-sum candidate fusion
│   └── rerank.py           # Multi-signal frame reranking
│
├── services/               # External service adapters
│   ├── llm_service.py      # Gemini API client (google-genai SDK)
│   ├── qdrant_search.py    # Multi-vector Qdrant search + Azure Embedding API client
│   ├── translator.py       # Language translation
│   └── scoring.py          # Score normalization utilities
│
├── test/                   # Unit & integration tests
│   ├── test_llm_intent_extraction.py
│   ├── test_modality_routing.py
│   ├── test_qdrant_search.py
│   └── test_rerank.py      # Multi-signal rerank formula tests
│
└── utils/                  # Shared utilities
    ├── json_utils.py       # JSON extraction from LLM responses
    └── logging_utils.py    # Logging configuration helpers
```

## State Schema

The pipeline uses a typed `AgentState` (TypedDict) that flows through all nodes:

```python
class AgentState(TypedDict, total=False):
    raw_query: str                           # Original user query

    query_bundle: QueryBundle                # Cleaned, translated, rewritten
    query_intent: QueryIntent                # LLM-extracted structured intent
    routing_weights: Dict[str, float]        # Per-source weights

    retrieval_results: Dict[str, List[Candidate]]  # Results per source
    fused_candidates: List[Candidate]        # After weighted-sum fusion
    reranked_candidates: List[Candidate]     # After reranking
    agent_topk: List[Candidate]              # Final output

    trace_logs: List[TraceLog]               # Debug trace
    error: Optional[str]                     # Pipeline error message
```

Key types: `QueryBundle`, `QueryIntent`, `Candidate`, `TraceLog` — all defined in [`state.py`](state.py).

## Setup

```bash
cd retrieval/agentic_retrieval

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Qdrant and Gemini API keys
```

**Required environment variables:**
| Variable | Purpose |
|---|---|
| `QDRANT_URL` | Qdrant Cloud cluster endpoint |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `GEMINI_API_KEY` | Google Gemini API key for intent extraction |
| `EMBEDDING_API_BASE_URL` | Azure embedding API base URL (e.g. `http://<VM_IP>:8000`) |

## Running the Demo

```bash
# Single query
python run_agentic_demo.py --query "a person in red shirt cooking"

# Vietnamese query
python run_agentic_demo.py --query "Tìm video có chữ 'SALE' trên biển quảng cáo" --top_k 10

# All sample queries
python run_agentic_demo.py --all-samples --verbose

# With verbose trace logs
python run_agentic_demo.py -q "cảnh đường phố ban đêm" -v
```

**CLI args:**
| Arg | Default | Description |
|---|---|---|
| `--query, -q` | Sample query | User search query |
| `--top_k, -k` | 20 | Number of final results |
| `--top_k_per_source` | 20 | Candidates per retrieval source |
| `--verbose, -v` | false | Show source scores + trace logs |
| `--all-samples` | false | Run all built-in sample queries |

## Diagnostic Tests

These scripts are standalone tools for verifying individual pipeline components. Run them directly with `python` (pass `--query` if supported).

```bash
cd retrieval/agentic_retrieval

# Test LLM intent extraction (requires GEMINI_API_KEY)
python test/test_llm_intent_extraction.py --query "a person in red shirt cooking"

# Test modality routing
python test/test_modality_routing.py

# Test Qdrant search (requires live Qdrant connection)
python test/test_qdrant_search.py --query "example search"

# Test reranking formula (no external dependencies)
python -m pytest test/test_rerank.py -v
```

## Services

### LLMService (`services/llm_service.py`)

- **Model:** `gemini-3.1-flash-lite-preview` (configurable)
- **SDK:** `google-genai`
- **Output:** JSON string matching `QueryIntentSchema`
- **Features:** Structured JSON output, retry with backoff, force-English option
- **Reads:** `GEMINI_API_KEY`

### QdrantSearchService (`services/qdrant_search.py`)

- **Collection:** `keyframes_v1` (see [schema contract](../../docs/contracts/qdrant-collection-schema.md))
- **Embedding backend:**
  - Calls Azure Embedding API (`EMBEDDING_API_BASE_URL`)
  - `/embed/semantic` for BGE-M3 caption vectors
  - `/embed/visual` for SigLIP keyframe vectors
  - `/embed/sparse` for BM25 object/OCR sparse vectors
- **Search methods:**
  - `search_keyframe()` — SigLIP text-to-image search (dense 1152d)
  - `search_caption()` — BGE-M3 semantic text search (dense 1024d)
  - `search_object()` — BM25 sparse object tag matching
  - `search_ocr()` — BM25 sparse OCR text matching
  - `search_metadata()` — lexical title fallback (often returns empty)

### TranslatorService (`services/translator.py`)

- **Language detection:** Vietnamese fast heuristic (diacritics + markers) → `langdetect` fallback
- **Translation:** Gemini API (`gemini-3.1-flash-lite-preview`) for non-English → English
- **Retry logic:** up to 2 attempts with 1s sleep; graceful fallback to original text
- **Consumed by:** `nodes/normalization.py`

## Scoring Formulas

### Candidate Fusion (`nodes/fusion.py`)

Each candidate's fused score is a **weighted sum** across retrieval sources:

```
fused_score = Σ (routing_weight[source] × normalized_score[source])
```

- Scores within each source are **min-max normalized** to [0, 1] before fusion.
- Candidates are merged by `(video_id, frame_id)` — a frame appearing in multiple sources accumulates score from each.

### Frame Reranking (`nodes/rerank.py`)

The reranker applies a multi-signal formula on top of the fused score:

```
rerank_score = fused_score
             + α · cross_source_agreement
             + β · intent_coverage_bonus
             - γ · missing_modality_penalty
```

**Tầng 1 — Cross-Source Agreement** (α = 0.15):

```
agreement = Σ (source_score[src] × routing_weight[src])   for src in evidence
```

Rewards candidates confirmed by high-weight, high-scoring sources. Unlike a flat per-source bonus, this is quality-weighted.

**Tầng 2 — Intent Coverage** (β = 0.10, γ = 0.08):

Derives **expected modalities** from query intent:
- `text_cues ≠ []` → expects `ocr` in evidence
- `objects ≠ []` → expects `object` in evidence
- `actions` or `scene ≠ []` → expects `caption` in evidence

```
coverage_bonus  = β × (|present ∩ expected| / |expected|)
missing_penalty = γ × (|expected − present| / |expected|)
```

All coefficients (α, β, γ) are exposed as keyword arguments for tuning.

Each candidate carries a `rerank_signals` dict for debugging:
```python
{"fused_score", "agreement_bonus", "coverage_bonus", "missing_penalty",
 "expected_modalities", "present_modalities", "missing_modalities"}
```

## What to test after changes

- **State schema changes** (`state.py`): verify all nodes still read/write compatible keys
- **Node changes**: run the corresponding test in `test/` and the full demo
- **Rerank changes**: run `python -m pytest test/test_rerank.py -v`
- **Service changes**: check model alignment with `qdrant_upsert.py` models
- **LLM prompt changes**: verify `QueryIntentSchema` output still parses correctly
- **Routing weight changes**: run `test_modality_routing.py` and check edge cases
