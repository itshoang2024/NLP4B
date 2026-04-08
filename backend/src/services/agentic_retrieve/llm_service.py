"""
llm_service.py — LLM service for query intent extraction.

Delegates to the pluggable LLM provider layer (``src.services.llm``).
The public interface (``invoke(prompt) -> str``) is unchanged so that
``nodes/intent_extraction.py`` and ``graph.py`` need zero modifications.

Handles:
 - Structured JSON output via provider's json_mode
 - Markdown fence stripping for providers that wrap JSON in ```json blocks
 - Pydantic validation against QueryIntentSchema
 - One automatic retry on malformed JSON
 - Safe fallback to empty intent on total failure
"""

from __future__ import annotations

import json
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from src.services.llm import LLMProvider, get_llm_provider

logger = logging.getLogger(__name__)


class QueryIntentSchema(BaseModel):
    objects: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    scene: list[str] = Field(default_factory=list)
    text_cues: list[str] = Field(default_factory=list)
    metadata_cues: list[str] = Field(default_factory=list)
    query_type: str = Field(default="mixed")


# ── Helpers ──────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers that some models add."""
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _try_parse_intent(raw: str) -> Optional[dict]:
    """Attempt to parse raw text into a validated QueryIntentSchema dict."""
    cleaned = _strip_markdown_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: find the first { ... } block
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    try:
        validated = QueryIntentSchema.model_validate(data)
        return validated.model_dump()
    except ValidationError:
        # JSON parsed fine but didn't match schema — still return what we got
        return data


# ── LLMService ───────────────────────────────────────────────────────────────

@dataclass
class LLMService:
    """
    LLM service for structured query intent extraction.

    Delegates to the pluggable provider layer.  Keeps the same
    ``invoke(prompt) -> str`` interface for downstream compatibility.
    """

    provider: Optional[LLMProvider] = field(default=None, repr=False)
    temperature: float = 0.1
    max_output_tokens: int = 512
    force_english_output: bool = True
    retry_attempts: int = 2
    retry_sleep_seconds: float = 1.5

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = get_llm_provider()

    @property
    def model_name(self) -> str:
        return self.provider.model_name

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
        Execute an LLM call and return a JSON string matching QueryIntentSchema.

        On malformed JSON the method retries once with a corrective prompt.
        On total failure it returns a safe empty-intent JSON string.
        """
        system_instruction = self._build_system_instruction()

        for attempt in range(1, self.retry_attempts + 1):
            try:
                raw = self.provider.generate(
                    prompt,
                    system_instruction=system_instruction,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    json_mode=True,
                )

                if not raw or not raw.strip():
                    logger.warning(
                        "LLM returned empty response (attempt %d/%d).",
                        attempt, self.retry_attempts,
                    )
                    if attempt < self.retry_attempts:
                        time.sleep(self.retry_sleep_seconds)
                    continue

                parsed = _try_parse_intent(raw)
                if parsed is not None:
                    return json.dumps(parsed, ensure_ascii=False)

                # JSON was malformed — retry with corrective prompt
                logger.warning(
                    "Malformed JSON from LLM (attempt %d/%d): %.200s",
                    attempt, self.retry_attempts, raw,
                )
                if attempt < self.retry_attempts:
                    prompt = (
                        "Your previous response was not valid JSON. "
                        "Please return ONLY a valid JSON object with these keys: "
                        "objects, attributes, actions, scene, text_cues, metadata_cues, query_type.\n\n"
                        f"Original prompt:\n{prompt}"
                    )
                    time.sleep(self.retry_sleep_seconds)

            except Exception as exc:
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt, self.retry_attempts, exc,
                )
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_sleep_seconds)

        logger.error("All LLM attempts exhausted — returning empty intent.")
        return self._empty_result_json()
