"""
search_middleware.py — Clean and translate the raw user query.

Extracts the normalization logic that was previously in the agentic
graph's normalization node into a shared middleware used by both
agentic and heuristic branches.

Logic source: retrieval/agentic_retrieval/nodes/normalization.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas import SearchRequest
from src.services.translator import detect_language, translate_to_english


class ProcessedSearchRequest(BaseModel):
    """Request after middleware processing — carries the full query bundle."""
    raw_query: str
    top_k: int = 10
    query_bundle: dict = Field(default_factory=dict)


def _generate_safe_rewrites(cleaned: str, translated_en: str) -> list[str]:
    """
    Deterministic rewrite generation.
    Future: could use LLM for 1-2 short rewrites.
    """
    rewrites = []
    if translated_en and translated_en != cleaned:
        rewrites.append(translated_en)
    rewrites.append(cleaned)

    seen = set()
    final = []
    for x in rewrites:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            final.append(x)
    return final[:3]


def clean_and_translate_middleware(payload: SearchRequest) -> ProcessedSearchRequest:
    """
    FastAPI Depends() middleware:
    1. Clean: normalize whitespace
    2. Detect language
    3. Translate to English (if needed)
    4. Generate safe rewrites
    5. Package into query_bundle

    Usage in route:
        @router.post("/search")
        def search(request: ProcessedSearchRequest = Depends(clean_and_translate_middleware)):
            ...
    """
    raw_query = payload.raw_query

    # 1. Clean
    cleaned = " ".join(raw_query.strip().split())

    # 2. Detect language
    lang = detect_language(cleaned)

    # 3. Translate
    translated_en = translate_to_english(cleaned, lang)

    # 4. Rewrites
    rewrites = _generate_safe_rewrites(cleaned, translated_en)

    # 5. Build query bundle
    query_bundle = {
        "raw": raw_query,
        "cleaned": cleaned,
        "lang": lang,
        "translated_en": translated_en,
        "rewrites": rewrites,
    }

    return ProcessedSearchRequest(
        raw_query=raw_query,
        top_k=payload.top_k,
        query_bundle=query_bundle,
    )
