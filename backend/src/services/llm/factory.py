"""
factory.py - Singleton factory for LLM provider instances.

Reads ``LLM_BACKEND`` from the environment and returns the
appropriate provider.  The instance is cached so that only one
client connection is created per process.

Supported values for ``LLM_BACKEND``:
 - ``gemini``          (default) - Google Gemini Developer API
 - ``openai_compat``   - any OpenAI-compatible endpoint
 - ``llama_cpp``       — alias for ``openai_compat``
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from .provider import LLMProvider

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """
    Return a singleton ``LLMProvider`` based on ``LLM_BACKEND``.

    Environment variables consumed per backend:

    **gemini**
        ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``, ``LLM_MODEL_NAME`` (optional).

    **openai_compat / llama_cpp**
        ``LLM_BASE_URL`` (default ``http://localhost:8080``),
        ``LLM_API_KEY`` (optional),
        ``LLM_MODEL_NAME`` (optional, default ``"default"``).
    """
    backend = os.getenv("LLM_BACKEND", "gemini").strip().lower()

    if backend == "gemini":
        from .gemini_provider import GeminiProvider

        model = os.getenv("LLM_MODEL_NAME", "gemini-3.1-flash-lite-preview").strip() or ""
        provider = GeminiProvider(model=model)

    elif backend in ("openai_compat", "llama_cpp"):
        from .openai_compat_provider import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider()

    else:
        raise ValueError(
            f"Unknown LLM_BACKEND={backend!r}. "
            f"Supported values: gemini, openai_compat, llama_cpp."
        )

    logger.info("LLM provider ready: %r (backend=%s)", provider, backend)
    return provider
