# Agentic Retrieval

LangGraph-based multi-branch retrieval pipeline for multimodal video keyframe search. Takes a natural-language query (Vietnamese or English), decomposes it into structured intent, routes to multiple vector search strategies, fuses results, and returns ranked keyframe candidates.

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
│ Parallel          │  Query Qdrant across 5 sources simultaneously:
│ Retrieval         │  keyframe (SigLIP), caption (BGE-M3), object (sparse),
│                   │  OCR (sparse), metadata (lexical fallback)
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│ Candidate         │  RRF (Reciprocal Rank Fusion) merge across sources
│ Fusion            │  Applies routing_weights to bias fusion scores
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│ Frame             │  Final scoring, deduplication, and truncation
│ Reranking         │  → agent_topk: List[Candidate]
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
│   ├── fusion.py           # RRF candidate fusion
│   └── rerank.py           # Final frame reranking
│
├── services/               # External service adapters
│   ├── llm_service.py      # Gemini API client (google-genai SDK)
│   ├── qdrant_search.py    # Multi-vector Qdrant search + model encoders
│   ├── translator.py       # Language translation
│   └── scoring.py          # Score normalization utilities
│
├── test/                   # Integration tests
│   ├── test_llm_intent_extraction.py
│   ├── test_modality_routing.py
│   └── test_qdrant_search.py
│
└── utils/                  # Shared utilities
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
    fused_candidates: List[Candidate]        # After RRF fusion
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

## Running Tests

```bash
cd retrieval/agentic_retrieval

# Test LLM intent extraction
python -m pytest test/test_llm_intent_extraction.py -v

# Test modality routing
python -m pytest test/test_modality_routing.py -v

# Test Qdrant search (requires live Qdrant connection)
python -m pytest test/test_qdrant_search.py -v
```

## Services

### LLMService (`services/llm_service.py`)

- **Model:** `gemini-3.1-flash-lite-preview` (configurable)
- **SDK:** `google-genai`
- **Output:** JSON string matching `QueryIntentSchema`
- **Features:** Structured JSON output, retry with backoff, force-English option
- **Reads:** `GEMINI_API_KEY` or `GOOGLE_API_KEY`

### QdrantSearchService (`services/qdrant_search.py`)

- **Collection:** `keyframes_v1` (see [schema contract](../../docs/contracts/qdrant-collection-schema.md))
- **Models loaded at init:**
  - `BAAI/bge-m3` (SentenceTransformer) for caption queries
  - `google/siglip-so400m-patch14-384` (AutoModel) for visual queries
- **Search methods:**
  - `search_keyframe()` — SigLIP text→image search
  - `search_caption()` — BGE-M3 semantic text search
  - `search_object()` — sparse tag matching
  - `search_ocr()` — sparse OCR text matching
  - `search_metadata()` — lexical title fallback (often returns empty)

## What to test after changes

- **State schema changes** (`state.py`): verify all nodes still read/write compatible keys
- **Node changes**: run the corresponding test in `test/` and the full demo
- **Service changes**: check model alignment with `qdrant_upsert.py` models
- **LLM prompt changes**: verify `QueryIntentSchema` output still parses correctly
- **Routing weight changes**: run `test_modality_routing.py` and check edge cases
