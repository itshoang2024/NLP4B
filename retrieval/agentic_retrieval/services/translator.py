"""
Translator service for the agentic retrieval pipeline.

Provides:
- ``detect_language``  - lightweight offline language detection via *langdetect*.
- ``translate_to_english`` - Vietnamese → English translation powered by the
  Gemini Developer API (google-genai).

The module is consumed by ``nodes.normalization`` during the query-normalization
stage of the retrieval graph.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Optional



logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

# Vietnamese-specific Unicode ranges & common markers used as a fast pre-check
# before falling back to langdetect (which can be slow on very short strings).
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
    """Quick check for Vietnamese diacritical characters."""
    return any(ch in _VI_DIACRITICS for ch in text)


def _has_vietnamese_tokens(text: str) -> bool:
    """Quick check for common Vietnamese tokens."""
    tokens = set(text.lower().split())
    return len(tokens & _VI_MARKERS) >= 2


def detect_language(text: str) -> str:
    """
    Detect the language of *text*.

    Returns ``"vi"`` for Vietnamese, ``"en"`` for English, or an ISO-639-1
    code for other languages.

    Strategy (ordered by cost):
    1. Fast heuristic based on Vietnamese diacritics / markers.
    2. ``langdetect`` library (if installed).
    3. Conservative fallback to ``"en"``.
    """
    if not text or not text.strip():
        return "en"

    stripped = text.strip()

    # ---- Fast heuristic for Vietnamese ------------------------------------
    if _has_vietnamese_chars(stripped) or _has_vietnamese_tokens(stripped):
        return "vi"

    # ---- langdetect (more robust for other languages) ---------------------
    try:
        from langdetect import detect as _detect  # type: ignore[import-untyped]

        detected = _detect(stripped)
        logger.debug("langdetect detected '%s' for: %s", detected, stripped[:80])
        return detected
    except ImportError:
        logger.debug(
            "langdetect not installed - falling back to heuristic-only detection."
        )
    except Exception as exc:
        logger.warning("langdetect raised %s - defaulting to 'en'.", exc)

    return "en"


# ---------------------------------------------------------------------------
# Gemini-backed translation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_genai_client():
    """
    Lazily initialise and cache a ``genai.Client``.

    Reads ``GEMINI_API_KEY`` first, then ``GOOGLE_API_KEY`` as fallback –
    matching the convention in ``llm_service.py``.
    """
    from google import genai  # lazy import to keep module load lightweight

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing Gemini API key.  "
            "Set the GEMINI_API_KEY (preferred) or GOOGLE_API_KEY env var."
        )
    return genai.Client(api_key=api_key)


_TRANSLATION_MODEL = "gemini-3.1-flash-lite-preview"
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
    model: str = _TRANSLATION_MODEL,
    timeout: int = 30,
) -> str:
    """
    Translate *text* to English if it is not already in English.

    Parameters
    ----------
    text : str
        Source text (any language).
    lang : str
        ISO-639-1 code of the source language (``"vi"``, ``"en"``, …).
    model : str, optional
        Gemini model to use for translation.
    timeout : int, optional
        HTTP timeout in seconds.

    Returns
    -------
    str
        English translation, or the original text if *lang* is ``"en"``
        or translation fails.
    """
    if not text or not text.strip():
        return text

    # Already English → nothing to do.
    if lang == "en":
        return text

    # ----- Build prompt ----------------------------------------------------
    prompt = (
        f"Translate the following {lang.upper()} text to English.\n\n"
        f"{text}"
    )

    last_error: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            client = _get_genai_client()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 256,
                    "system_instruction": _TRANSLATION_SYSTEM_INSTRUCTION,
                },
            )

            translated = getattr(response, "text", None)
            if translated and translated.strip():
                result = translated.strip().strip('"').strip("'")
                logger.info(
                    "Translated [%s→en]: '%s' → '%s'",
                    lang, text[:60], result[:60],
                )
                return result

            # Empty response - treat as failure for retry purposes.
            logger.warning(
                "Gemini returned empty translation (attempt %d/%d).",
                attempt, _MAX_RETRIES,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Translation attempt %d/%d failed: %s",
                attempt, _MAX_RETRIES, exc,
            )

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_SLEEP)

    # ----- Graceful fallback -----------------------------------------------
    logger.error(
        "All translation attempts failed (last error: %s). "
        "Returning original text as fallback.",
        last_error,
    )
    return text