"""
gemini_provider.py - Gemini implementation of LLMProvider.

Wraps the ``google-genai`` SDK.  Preserves existing behaviour:
 - Reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
 - json_mode=True → sets ``response_mime_type="application/json"``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from google import genai

from .provider import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """LLM provider backed by the Google Gemini Developer API."""

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite-preview",
        api_key: Optional[str] = None,
    ) -> None:
        self._model = model
        resolved_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not resolved_key:
            raise EnvironmentError(
                "GeminiProvider requires an API key. "
                "Set GEMINI_API_KEY (preferred) or GOOGLE_API_KEY."
            )
        self._client = genai.Client(api_key=resolved_key)
        logger.info("GeminiProvider initialised (model=%s)", self._model)

    # ── LLMProvider interface ────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str = "",
        temperature: float = 0.1,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        config: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config["system_instruction"] = system_instruction
        if json_mode:
            config["response_mime_type"] = "application/json"

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )

        # Gemini may expose a parsed object or raw text.
        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()

        return ""
