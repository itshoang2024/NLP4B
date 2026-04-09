"""
search_middleware.py — Clean and translate the raw user query.

Extracts the normalization logic that was previously in the agentic
graph's normalization node into a shared middleware used by both
agentic and heuristic branches.

Logic source: retrieval/agentic_retrieval/nodes/normalization.py
"""

from __future__ import annotations

import unicodedata

from pydantic import BaseModel, Field

from src.schemas import SearchRequest
from src.services.translator import detect_language, translate_to_english


class ProcessedSearchRequest(BaseModel):
    """Request after middleware processing — carries the full query bundle."""
    raw_query: str
    top_k: int = 10
    query_bundle: dict = Field(default_factory=dict)


# ── stopwords for keyword extraction ─────────────────────────────────────────

_EN_STOPWORDS = frozenset({
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "not", "no", "from", "by", "that", "this",
    "it", "its", "has", "have", "had", "do", "does", "did",
    "find", "show", "search", "look", "get",
})

_VI_STOPWORDS = frozenset({
    "tìm", "tìm kiếm", "cảnh", "trong", "với", "một", "ngoài",
    "đang", "không", "được", "những", "các", "cho",
    "này", "của", "phía", "trên", "dưới", "bên", "trước",
    "sau", "giữa", "hình", "ảnh", "và", "hoặc", "có",
    "bao", "nhiêu", "gì", "nào", "là", "thì", "mà", "cũng",
    "rất", "lại", "còn", "nếu", "để", "vì", "từ", "về",
    "ra", "vào", "lên", "xuống", "đến", "tại", "theo",
})

_ALL_STOPWORDS = _EN_STOPWORDS | _VI_STOPWORDS


# ── clean helpers ─────────────────────────────────────────────────────────────

def _remove_emoji(text: str) -> str:
    """Remove emoji and decorative symbol characters that hurt embedding quality."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("So", "Sk")
    )


def _clean_query(raw: str) -> str:
    """Normalize a raw user query for downstream embedding and LLM consumption.

    All operations are deterministic and O(n) — no network calls.
    """
    text = raw.strip()

    # 1. Unicode NFC — merge decomposed diacritics (critical for Vietnamese)
    text = unicodedata.normalize("NFC", text)

    # 2. Collapse whitespace
    text = " ".join(text.split())

    # 3. Strip junk punctuation at boundaries
    text = text.strip("?!.,;:…·•–—\"'""''()[]{}«»")

    # 4. Remove emoji / decorative symbols
    text = _remove_emoji(text)

    # 5. Final whitespace collapse after emoji removal
    text = " ".join(text.split())

    return text


# ── rewrite helpers ───────────────────────────────────────────────────────────

def _extract_keywords(text: str, lang: str) -> str:
    """Remove stopwords, keep content words only.

    Uses combined EN + VI stopword list since queries may be in either language.
    """
    words = text.split()
    content = [w for w in words if w.lower() not in _ALL_STOPWORDS]
    return " ".join(content) if content else ""


def _generate_safe_rewrites(
    cleaned: str,
    translated_en: str,
    lang: str,
) -> list[str]:
    """Deterministic rewrite generation — up to 3 variants.

    Strategy:
      1. translated_en (if different from cleaned)
      2. cleaned (original, normalized)
      3. keyword-condensed variant (stopwords removed)
    """
    candidates: list[str] = []

    # Variant 1: English translation
    if translated_en and translated_en.lower() != cleaned.lower():
        candidates.append(translated_en)

    # Variant 2: Original cleaned query
    candidates.append(cleaned)

    # Variant 3: Keyword-only (improves BM25 sparse matching)
    en_text = translated_en or cleaned
    keyword_only = _extract_keywords(en_text, lang)
    if keyword_only and keyword_only.lower() != en_text.lower():
        candidates.append(keyword_only)

    # Dedup (case-insensitive) + limit
    seen: set[str] = set()
    final: list[str] = []
    for x in candidates:
        x = x.strip()
        key = x.lower()
        if x and key not in seen:
            seen.add(key)
            final.append(x)
    return final[:3]


def clean_and_translate_middleware(payload: SearchRequest) -> ProcessedSearchRequest:
    """
    FastAPI Depends() middleware:
    1. Clean: normalize unicode, whitespace, punctuation, emoji
    2. Detect language
    3. Translate to English (if needed)
    4. Generate safe rewrites (original + translated + keyword-only)
    5. Package into query_bundle
    """
    raw_query = payload.raw_query

    # 1. Clean
    cleaned = _clean_query(raw_query)

    # 2. Detect language
    lang = detect_language(cleaned)

    # 3. Translate
    translated_en = translate_to_english(cleaned, lang)

    # 4. Rewrites
    rewrites = _generate_safe_rewrites(cleaned, translated_en, lang)

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
