from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _add_project_paths() -> None:
    """Add the agentic_retrieval root to sys.path so 'services.*' is importable
    regardless of where the script is invoked from."""
    here = Path(__file__).resolve()
    # here = .../agentic_retrieval/test/test_llm_intent_extraction.py
    # parent = .../agentic_retrieval/test/
    # parent.parent = .../agentic_retrieval/   <-- this is what we need
    root = here.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_add_project_paths()

from services.llm_service import LLMService  # noqa: E402

PROMPT_TEMPLATE = """
You are a query analysis module for a multimodal video retrieval system.

Given a user query, extract:
- objects: visible entities or actors
- attributes: colors, clothing, appearance, properties
- actions: actions or activities
- scene: environment or context
- text_cues: words expected to appear in image text / OCR
- metadata_cues: title/date/channel/topic cues useful for metadata retrieval
- query_type: one of [\"visual_object\", \"visual_event\", \"text_in_image\", \"metadata_hint\", \"mixed\"]

Rules:
- Stay faithful to the query
- Do not invent details
- Return ONLY valid JSON

User query bundle:
{query_bundle_json}
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--keep-original-language", action="store_true")
    args = parser.parse_args()

    llm = LLMService(
        model_name=args.model,
        force_english_output=not args.keep_original_language,
    )

    query_bundle = {
        "raw": args.query,
        "cleaned": args.query,
        "lang": "vi",
        "translated_en": "",
        "rewrites": [],
    }

    prompt = PROMPT_TEMPLATE.format(
        query_bundle_json=json.dumps(query_bundle, ensure_ascii=False, indent=2)
    )
    response = llm.invoke(prompt)

    print("=" * 80)
    print("RAW JSON RESPONSE")
    print("=" * 80)
    print(response)
    print()

    print("=" * 80)
    print("PARSED")
    print("=" * 80)
    print(json.dumps(json.loads(response), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
