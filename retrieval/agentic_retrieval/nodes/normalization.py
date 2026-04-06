from __future__ import annotations
from state import AgentState
from services.translator import detect_language, translate_to_english


def generate_safe_rewrites(cleaned: str, translated_en: str) -> list[str]:
    """
    Hiện tại để deterministic.
    Sau này có thể dùng LLM tạo 1-2 rewrite ngắn gọn hơn.
    """
    rewrites = []
    if translated_en and translated_en != cleaned:
        rewrites.append(translated_en)
    rewrites.append(cleaned)
    # unique, giữ thứ tự
    seen = set()
    final = []
    for x in rewrites:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            final.append(x)
    return final[:3]


def query_normalization_node(state: AgentState) -> AgentState:
    raw_query = state["raw_query"]
    cleaned = " ".join(raw_query.strip().split())
    lang = detect_language(cleaned)
    translated_en = translate_to_english(cleaned, lang)
    rewrites = generate_safe_rewrites(cleaned, translated_en)

    state["query_bundle"] = {
        "raw": raw_query,
        "cleaned": cleaned,
        "lang": lang,
        "translated_en": translated_en,
        "rewrites": rewrites,
    }

    state.setdefault("trace_logs", []).append({
        "node": "query_normalization",
        "payload": {"query_bundle": state["query_bundle"]},
    })
    return state