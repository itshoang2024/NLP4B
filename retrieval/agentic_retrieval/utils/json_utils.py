import json
import re
from typing import Any, Dict


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Trích JSON object đầu tiên từ text.
    Hữu ích khi LLM lỡ sinh thêm text xung quanh.
    """
    text = text.strip()

    # thử parse trực tiếp trước
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # fallback: tìm block {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")

    return json.loads(match.group(0))