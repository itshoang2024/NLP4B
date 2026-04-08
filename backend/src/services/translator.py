"""
Translator service - shared by the search middleware.

Provides:
- detect_language - lightweight offline language detection via heuristics + langdetect.
- translate_to_english - translation via the pluggable LLM provider layer.

The LLM backend is determined by the ``LLM_BACKEND`` env var
(see ``src.services.llm.factory``).  Gemini and any OpenAI-compatible
endpoint (llama.cpp, Ollama, etc.) are supported out of the box.
"""

from __future__ import annotations

import logging
import time
from typing import Optional


logger = logging.getLogger(__name__)

# ── Vietnamese detection heuristics ───────────────────────────────────────────

_VI_DIACRITICS = set(
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
    "ùúủũụưứừửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
    "ÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
)

_VI_MARKERS = frozenset([
    "tìm", "người", "cảnh", "trong", "với", "một", "ngoài",
    "trời", "đang", "không", "được", "những", "các", "cho",
    "này", "của", "phía", "trên", "dưới", "bên", "trước",
    "sau", "giữa", "màu", "hình", "ảnh", "chiếc", "cái",
    "và", "hoặc", "có", "bao", "nhiêu", "gì", "nào",
])


def _has_vietnamese_chars(text: str) -> bool:
    return any(ch in _VI_DIACRITICS for ch in text)


def _has_vietnamese_tokens(text: str) -> bool:
    tokens = set(text.lower().split())
    return len(tokens & _VI_MARKERS) >= 2


def detect_language(text: str) -> str:
    """
    Detect the language of *text*.
    Returns "vi" for Vietnamese, "en" for English, or an ISO-639-1 code.
    """
    if not text or not text.strip():
        return "en"

    stripped = text.strip()

    if _has_vietnamese_chars(stripped) or _has_vietnamese_tokens(stripped):
        return "vi"

    try:
        from langdetect import detect as _detect
        detected = _detect(stripped)
        logger.debug("langdetect detected '%s' for: %s", detected, stripped[:80])
        return detected
    except ImportError:
        logger.debug("langdetect not installed — falling back to heuristic-only detection.")
    except Exception as exc:
        logger.warning("langdetect raised %s — defaulting to 'en'.", exc)

    return "en"


# ── LLM-backed translation ──────────────────────────────────────────────────

_TRANSLATION_SYSTEM_INSTRUCTION = (
    "You are a precise translation module for a video-retrieval system. "
    "Translate the user's text into natural, fluent English. "
    "Preserve proper nouns, brand names, technical terms, and numbers exactly. "
    "Return ONLY the translated text - no explanations, no markdown, no quotes."
)

_MAX_RETRIES = 2
_RETRY_SLEEP = 1.0


def translate_to_english(
    text: str,
    lang: str,
    *,
    timeout: int = 30,
) -> str:
    """Translate *text* to English if it is not already in English."""
    if not text or not text.strip():
        return text

    if lang == "en":
        return text

    # Lazy import to avoid circular dependency at module load time.
    from src.services.llm import get_llm_provider

    provider = get_llm_provider()

    prompt = (
        f"Translate the following {lang.upper()} text to English.\n\n"
        f"{text}"
    )

    last_error: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            raw = provider.generate(
                prompt,
                system_instruction=_TRANSLATION_SYSTEM_INSTRUCTION,
                temperature=0.0,
                max_tokens=256,
                json_mode=False,
            )

            if raw and raw.strip():
                result = raw.strip().strip('"').strip("'")
                logger.info(
                    "Translated [%s→en] via %s: '%s' → '%s'",
                    lang, provider.model_name, text[:60], result[:60],
                )
                return result

            logger.warning(
                "LLM returned empty translation (attempt %d/%d, provider=%s).",
                attempt, _MAX_RETRIES, provider.model_name,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Translation attempt %d/%d failed (%s): %s",
                attempt, _MAX_RETRIES, provider.model_name, exc,
            )

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_SLEEP)

    logger.error(
        "All translation attempts failed (provider=%s, last error: %s). "
        "Returning original text as fallback.",
        provider.model_name, last_error,
    )
    return text
