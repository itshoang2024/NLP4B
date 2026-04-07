"""
llm_service.py — Gemini-backed LLM service for query intent extraction.

Migrated from: retrieval/agentic_retrieval/services/llm_service.py
Import paths updated for backend package structure.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field
from google import genai


class QueryIntentSchema(BaseModel):
    objects: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    scene: list[str] = Field(default_factory=list)
    text_cues: list[str] = Field(default_factory=list)
    metadata_cues: list[str] = Field(default_factory=list)
    query_type: str = Field(default="mixed")


@dataclass
class LLMService:
    """
    Minimal Gemini-backed LLM service for query intent extraction.

    Notes:
    - Uses google-genai / Gemini Developer API.
    - Reads GEMINI_API_KEY first, then GOOGLE_API_KEY as fallback.
    - Returns a JSON string so existing downstream code can keep using
      `extract_json_object(raw_response)` without further changes.
    """

    model_name: str = "gemini-3.1-flash-lite-preview"
    api_key: Optional[str] = None
    timeout_seconds: int = 60
    temperature: float = 0.1
    max_output_tokens: int = 512
    force_english_output: bool = True
    retry_attempts: int = 2
    retry_sleep_seconds: float = 1.5
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing Gemini API key. Set GEMINI_API_KEY (preferred) or GOOGLE_API_KEY."
            )
        self._client = genai.Client(api_key=api_key)

    def _build_system_instruction(self) -> str:
        parts = [
            "You are a strict information extraction module for a multimodal video retrieval system.",
            "Return only the requested JSON object.",
            "Stay faithful to the query and do not invent details.",
            "Light normalization is allowed, such as 'person giving a speech' -> 'speaker' and 'outside' -> 'outdoor'.",
            "Use short, retrieval-friendly phrases.",
        ]
        if self.force_english_output:
            parts.append("Return all list values and query_type in English, even if the user query is not in English.")
        else:
            parts.append("Preserve the original language when possible, but keep values concise and retrieval-friendly.")
        return " ".join(parts)

    def _empty_result_json(self) -> str:
        return QueryIntentSchema().model_dump_json(ensure_ascii=False)

    def invoke(self, prompt: str) -> str:
        """
        Execute a Gemini call and return a JSON string matching QueryIntentSchema.
        """
        last_error: Optional[Exception] = None
        system_instruction = self._build_system_instruction()

        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_output_tokens,
                        "response_mime_type": "application/json",
                        "response_schema": QueryIntentSchema,
                        "system_instruction": system_instruction,
                    },
                )

                parsed = getattr(response, "parsed", None)
                if parsed is not None:
                    if isinstance(parsed, BaseModel):
                        return parsed.model_dump_json(ensure_ascii=False)
                    if isinstance(parsed, dict):
                        return json.dumps(parsed, ensure_ascii=False)

                text = getattr(response, "text", None)
                if text and text.strip():
                    return text

                return self._empty_result_json()

            except Exception as exc:
                last_error = exc
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_sleep_seconds)
                    continue

        return self._empty_result_json()
