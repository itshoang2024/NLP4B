from __future__ import annotations


def detect_language(text: str) -> str:
    """
    Stub đơn giản. Có thể thay bằng langdetect / fasttext sau.
    """
    lowered = text.lower()
    vi_markers = ["tìm", "video", "người", "cảnh", "trong", "với", "một", "ngoài trời"]
    if any(tok in lowered for tok in vi_markers):
        return "vi"
    return "en"


def translate_to_english(text: str, lang: str) -> str:
    """
    Stub. Tạm thời nếu là tiếng Anh thì giữ nguyên.
    Nếu là tiếng Việt thì có thể:
    - gọi LLM
    - gọi translator service
    - hoặc dùng bản rewrite tiếng Anh từ LLM
    """
    if lang == "en":
        return text
    # fallback rất đơn giản
    return text