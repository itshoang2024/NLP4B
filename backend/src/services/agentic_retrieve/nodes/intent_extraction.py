"""
intent_extraction node — extract structured query intent via LLM.

Migrated from: retrieval/agentic_retrieval/nodes/intent_extraction.py
"""

from __future__ import annotations
import json
from typing import Any, Dict

from ..state import AgentState
from ..llm_service import LLMService
from ..utils.json_utils import extract_json_object


INTENT_EXTRACTION_PROMPT = """
You are a query analysis module for a multimodal video retrieval system.

Given a user query, extract:
- objects: visible entities or actors
- attributes: colors, clothing, appearance, properties
- actions: actions or activities
- scene: environment or context
- text_cues: words expected to appear in image text / OCR
- metadata_cues: title/date/channel/topic cues useful for metadata retrieval
- query_type: one of ["visual_object", "visual_event", "text_in_image", "metadata_hint", "mixed"]

Rules:
- Stay faithful to the query
- Do not invent details
- Return ONLY valid JSON

User query bundle:
{query_bundle_json}
""".strip()


def _fallback_intent(query_bundle: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = query_bundle.get("cleaned", "")
    return {
        "objects": [],
        "attributes": [],
        "actions": [],
        "scene": [],
        "text_cues": [],
        "metadata_cues": [],
        "query_type": "mixed" if cleaned else "visual_event",
    }


def query_intent_extraction_node_factory(llm: LLMService):
    def query_intent_extraction_node(state: AgentState) -> AgentState:
        query_bundle = state["query_bundle"]

        try:
            prompt = INTENT_EXTRACTION_PROMPT.format(
                query_bundle_json=json.dumps(query_bundle, ensure_ascii=False, indent=2)
            )
            raw_response = llm.invoke(prompt)
            intent = extract_json_object(raw_response)
        except Exception:
            intent = _fallback_intent(query_bundle)

        state["query_intent"] = intent
        state.setdefault("trace_logs", []).append({
            "node": "query_intent_extraction",
            "payload": {"query_intent": intent},
        })
        return state

    return query_intent_extraction_node
