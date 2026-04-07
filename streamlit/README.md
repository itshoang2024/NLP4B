# streamlit/

User-facing demo interface for LookUp.ai. Submits natural-language queries to the backend API and renders ranked keyframe results with Azure-hosted images.

## What this module is responsible for

- Search input via `st.chat_input`
- Calling `POST http://localhost:8000/search` with `raw_query` and `top_k`
- Rendering result cards with Azure keyframe images, captions, OCR text, branch badges, and score bars
- Displaying per-phase latency breakdown (agentic, heuristic, rerank, total)

## What this module is NOT responsible for

- Query processing, translation, or retrieval logic — handled by `backend/`
- Image storage or embedding generation — handled by `data-processing/` and `azure-ai-provider/`

## Structure

```
streamlit/
├── .streamlit/
│   └── config.toml          # Streamlit theme/config
├── assets/
│   └── styles/
│       └── main.css          # Full custom CSS (cards, animations, loader)
├── components/
│   └── __init__.py
├── pages/
└── app.py                    # Entry point — search UI + result rendering
```

## Prerequisites

The backend API must be running on `localhost:8000` before starting the Streamlit app:

```bash
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Run

```bash
cd streamlit
streamlit run app.py
```

## API dependency

The app calls `POST http://localhost:8000/search` and expects the response schema defined in [backend/README.md — API contract](../backend/README.md#api-contract):

```json
{
  "query": "...",
  "total_results": 5,
  "results": [
    {
      "rank": 1,
      "video_id": "abc",
      "frame_id": 42,
      "score": 0.016,
      "branch": "agentic",
      "azure_url": "https://...",
      "youtube_link": "https://...",
      "timestamp_sec": 12.5,
      "caption": "...",
      "ocr_text": null,
      "evidence": ["keyframe", "caption"]
    }
  ],
  "latency_ms": { "agentic_ms": 5898, "heuristic_ms": 0, "rerank_ms": 0, "total_ms": 5899 }
}
```

## What to test after changes

| If you change… | Then verify… |
|---|---|
| `app.py` result card rendering | Field names still match `SearchResultItem` schema from backend |
| `main.css` | Cards, loader animation, and layout render correctly |
| Backend response schema | Update `render_result_card()` and `render_latency_badge()` accordingly |
