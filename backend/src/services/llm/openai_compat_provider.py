"""
openai_compat_provider.py - OpenAI-compatible LLM provider.

Works with any server that exposes /v1/chat/completions
in the OpenAI format:
 - llama.cpp (--api-key optional)
 - Ollama  (OLLAMA_HOST)
 - LM Studio
 - text-generation-webui (with --api)
 - vLLM, TGI, etc.

Uses the openai Python SDK which handles auth headers,
retries, and streaming internally.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import openai

from .provider import LLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible chat/completions endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._base_url = (
            base_url
            or os.getenv("LLM_BASE_URL", "http://localhost:8080").strip().rstrip("/")
        )
        # Many self-hosted servers accept any non-empty key or none at all.
        resolved_key = api_key or os.getenv("LLM_API_KEY", "").strip() or "no-key"
        self._model = (
            model
            or os.getenv("LLM_MODEL_NAME", "").strip()
            or "default"
        )

        self._client = openai.OpenAI(
            base_url=f"{self._base_url}/v1",
            api_key=resolved_key,
        )
        logger.info(
            "OpenAICompatibleProvider initialised (base_url=%s, model=%s)",
            self._base_url,
            self._model,
        )

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
        messages: list[dict] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            # Most OpenAI-compatible servers support this.
            # llama.cpp requires a grammar or this flag; Ollama supports it.
            # If the server ignores it, the prompt-level JSON instruction
            # (injected by the caller) still pushes toward valid JSON.
            try:
                kwargs["response_format"] = {"type": "json_object"}
            except Exception:
                pass  # server might not support it — rely on prompt

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return content.strip()
