import streamlit as st
import requests
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LookUp.ai",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global stylesheet ─────────────────────────────────────────────────────────
with open("assets/styles/main.css", "r", encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"
DEFAULT_TOP_K = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def call_search_api(query: str, top_k: int = DEFAULT_TOP_K) -> dict | None:
    """POST /search to the FastAPI backend. Returns parsed JSON or None on error."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/search",
            json={"query": query, "top_k": top_k, "prefetch_limit": 50},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Không thể kết nối tới backend. Hãy chắc chắn FastAPI đang chạy tại `localhost:8000`.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timeout — server mất quá nhiều thời gian phản hồi.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"⚠️ API lỗi: {e.response.status_code} — {e.response.text[:200]}")
        return None


def render_result_card(result: dict, idx: int) -> str:
    """Build an HTML card string for one search result."""
    p = result.get("payload", {})

    video_id      = p.get("video_id") or "—"
    frame_idx     = p.get("frame_idx", "—")
    timestamp_sec = p.get("timestamp_sec", 0) or 0
    caption       = p.get("caption") or p.get("detailed_caption") or ""
    ocr_text      = p.get("ocr_text") or ""
    azure_url     = p.get("azure_url") or ""
    youtube_link  = p.get("youtube_link") or ""
    rrf_score     = result.get("rrf_score", 0)

    # Score bar width (RRF scores are typically 0.01–0.07 range)
    bar_pct = min(int(rrf_score * 2000), 100)

    img_html = (
        f'<img src="{azure_url}" class="card-img" onerror="this.style.display=\'none\'">'
        if azure_url else
        '<div class="card-img-placeholder"><span>No Image</span></div>'
    )

    yt_html = (
        f'<a href="{youtube_link}" target="_blank" class="yt-btn">▶ YouTube (t={int(timestamp_sec)}s)</a>'
        if youtube_link else ""
    )

    caption_html = (
        f'<p class="card-caption">"{caption[:150]}{"..." if len(caption) > 150 else ""}"</p>'
        if caption else ""
    )

    ocr_html = (
        f'<div class="card-ocr"><span class="ocr-label">OCR</span>{ocr_text[:100]}</div>'
        if ocr_text else ""
    )

    return f"""
    <div class="result-card" style="animation-delay: {idx * 0.06}s">
      <div class="card-rank">#{idx + 1}</div>
      <div class="card-img-wrap">{img_html}</div>
      <div class="card-body">
        <div class="card-meta-row">
          <span class="meta-pill video">🎬 {video_id}</span>
          <span class="meta-pill frame">🖼 Frame {frame_idx}</span>
          <span class="meta-pill time">⏱ {timestamp_sec:.1f}s</span>
        </div>
        <div class="score-row">
          <span class="score-label">RRF</span>
          <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{bar_pct}%"></div>
          </div>
          <span class="score-val">{rrf_score:.5f}</span>
        </div>
        {caption_html}
        {ocr_html}
        {yt_html}
      </div>
    </div>
    """


def render_latency_badge(latency: dict) -> str:
    enc  = latency.get("encode_ms", 0)
    srch = latency.get("search_ms", 0)
    tot  = latency.get("total_ms", 0)
    return f"""
    <div class="latency-bar">
      <span class="lat-chip enc">⚡ Encode {enc:.0f}ms</span>
      <span class="lat-chip srch">🔍 Search {srch:.0f}ms</span>
      <span class="lat-chip tot">🕐 Total {tot:.0f}ms</span>
    </div>
    """


# ── Fixed brand logo ──────────────────────────────────────────────────────────
st.markdown('<div class="brand-logo">LookUp.ai</div>', unsafe_allow_html=True)

# ── Idle hero (shown only when no results) ────────────────────────────────────
if "results_data" not in st.session_state:
    st.markdown("""
    <div class="hero-center">
      <div class="hero-sparkle">✨</div>
      <h1 class="hero-title">Tìm kiếm bằng ngôn ngữ tự nhiên</h1>
      <p class="hero-sub">Nhập mô tả cảnh quay bạn muốn tìm — hệ thống sẽ truy vấn qua SigLIP + BGE-M3</p>
      <div class="chip-row">
        <span class="chip">🏃 người đang chạy ngoài đường</span>
        <span class="chip">🍜 cảnh nấu ăn ban ngày</span>
        <span class="chip">🎤 người đứng nói chuyện trước đám đông</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Results section ───────────────────────────────────────────────────────────
if "results_data" in st.session_state:
    data = st.session_state["results_data"]
    query_shown = st.session_state.get("last_query", "")

    # Header row
    st.markdown(f"""
    <div class="results-header">
      <div class="results-query">"{query_shown}"</div>
      <div class="results-count">{data['total_results']} kết quả</div>
    </div>
    {render_latency_badge(data.get("latency_ms", {}))}
    """, unsafe_allow_html=True)

    # Cards grid
    cards_html = '<div class="results-grid">'
    for i, result in enumerate(data["results"]):
        cards_html += render_result_card(result, i)
    cards_html += "</div>"

    st.markdown(cards_html, unsafe_allow_html=True)

    # Clear results button
    col_center = st.columns([3, 1, 3])[1]
    with col_center:
        if st.button("🔄 Tìm kiếm mới", use_container_width=True):
            del st.session_state["results_data"]
            del st.session_state["last_query"]
            st.rerun()

# ── Chat input (fixed bottom) ─────────────────────────────────────────────────
search_query = st.chat_input("Mô tả cảnh bạn muốn tìm...")

if search_query:
    # Show spinner while calling API
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown("""
    <div class="custom-loader-container">
      <div class="custom-loader">
        <svg class="custom-loader-icon" width="28" height="28" viewBox="0 0 24 24" fill="none">
          <defs>
            <linearGradient id="sparkle-grad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#ffffff"/>
              <stop offset="100%" stop-color="#5a8dec"/>
            </linearGradient>
          </defs>
          <path d="M10.8 2.4c.4-1.2 2-1.2 2.4 0l1.4 4c.2.6.7 1.1 1.3 1.3l4 1.4c1.2.4 1.2 2 0 2.4l-4 1.4c-.6.2-1.1.7-1.3 1.3l-1.4 4c-.4 1.2-2 1.2-2.4 0l-1.4-4c-.2-.6-.7-1.1-1.3-1.3l-4-1.4c-1.2-.4-1.2-2 0-2.4l4-1.4c.6-.2 1.1-.7 1.3-1.3l1.4-4z" fill="url(#sparkle-grad)"/>
          <path d="M19 16.5c.2-.6 1-.6 1.2 0l.4 1.3c.1.2.3.4.5.5l1.3.4c.6.2.6 1 0 1.2l-1.3.4c-.2.1-.4.3-.5.5l-.4 1.3c-.2.6-1 .6-1.2 0l-.4-1.3c-.1-.2-.3-.4-.5-.5l-1.3-.4c-.6-.2-.6-1 0-1.2l1.3-.4c.2-.1.4-.3.5-.5l.4-1.3z" fill="url(#sparkle-grad)"/>
        </svg>
      </div>
      <div class="custom-loader-text">Đang mã hóa & tìm kiếm...</div>
    </div>
    """, unsafe_allow_html=True)

    data = call_search_api(search_query, top_k=DEFAULT_TOP_K)

    spinner_placeholder.empty()

    if data:
        st.session_state["results_data"] = data
        st.session_state["last_query"] = search_query
        st.rerun()
