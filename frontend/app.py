import streamlit as st
import streamlit.components.v1 as components
import requests
import html
import re

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LookUp.ai",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global stylesheet ─────────────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "assets", "styles", "main.css"), "r", encoding="utf-8") as css_file:
    _APP_CSS = css_file.read()

st.markdown(f"<style>{_APP_CSS}</style>", unsafe_allow_html=True)

# Absolute Logo (Fixed at Top-Left)
st.markdown(
    '<a href="/" target="_self" class="absolute-brand-logo" title="Về trang chủ">LookUp.ai</a>',
    unsafe_allow_html=True
)

# Fix nút toggle sidebar: thay icon lỗi (keyboard_double_arrow text) bằng SVG chevron
st.markdown("""
<script>
(function fixSidebarToggle() {
    const CHEVRON_RIGHT = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>';
    const CHEVRON_LEFT  = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>';

    function patchBtn() {
        // Nút mở sidebar khi đang đóng
        const expandBtn = document.querySelector('[data-testid="stSidebarCollapsedControl"] button');
        if (expandBtn) {
            const svg = expandBtn.querySelector('svg');
            if (!svg) expandBtn.innerHTML = CHEVRON_RIGHT;
        }
        // Nút đóng sidebar khi đang mở (nằm trong header)
        const collapseBtn = document.querySelector('[data-testid="stExpandSidebarButton"]');
        if (collapseBtn) {
            const svg = collapseBtn.querySelector('svg');
            if (!svg) collapseBtn.innerHTML = CHEVRON_LEFT;
        }
    }

    patchBtn();
    setInterval(patchBtn, 800);
})();
</script>
""", unsafe_allow_html=True)

import os

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DEFAULT_TOP_K = 10
SCORE_BAR_SCALE = 2000
SCORE_BAR_MAX_PCT = 100
CARD_HEIGHT_PX = 320
MIN_IFRAME_HEIGHT_PX = 400
CARDS_PER_ROW = 2

STRATEGY_OPTIONS = {
    "🔀 Cả hai (RRF Fusion)": "both",
    "🤖 Agentic": "agentic",
    "📊 Heuristic (Normal)": "heuristic",
}

BRANCH_COLORS = {
    "agentic": "#6366f1",
    "heuristic": "#10b981",
    "fused": "#f59e0b",
}
DEFAULT_BRANCH_COLOR = "#6b7280"

OCR_PREVIEW_MAX_CHARS = 100

SPINNER_HTML = """\
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
  <div class="custom-loader-text">Đang tìm kiếm</div>
</div>"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def escape_html(value) -> str:
    """Escape a value for safe HTML rendering."""
    return html.escape(str(value or ""))


def call_search_api(query: str, top_k: int = DEFAULT_TOP_K, strategy: str = "both") -> dict | None:
    """POST /search to the unified backend API."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/search",
            json={"raw_query": query, "top_k": top_k, "strategy": strategy},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Không thể kết nối tới backend. Hãy chắc chắn FastAPI đang chạy tại `localhost:8000`.")
    except requests.exceptions.Timeout:
        st.error("Request timeout — server mất quá nhiều thời gian phản hồi.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"API lỗi: {exc.response.status_code} — {exc.response.text[:200]}")
    return None


# ── Card sub-builders ─────────────────────────────────────────────────────────

def _build_image_html(azure_url: str, youtube_link: str) -> str:
    embed_url = ""
    if youtube_link:
        vid_match = re.search(r"(?:v=|youtu\.be/)([^&?]+)", youtube_link)
        t_match = re.search(r"[?&]t=(\d+)s?", youtube_link)
        if vid_match:
            vid = vid_match.group(1)
            t = t_match.group(1) if t_match else "0"
            embed_url = f"https://www.youtube.com/embed/{vid}?start={t}&autoplay=1&mute=1&controls=0&showinfo=0&rel=0&loop=1&playlist={vid}"

    if not azure_url:
        return '<div class="card-img-placeholder"><span>No Image</span></div>'
    safe_url = html.escape(azure_url, quote=True)
    fallback = '<div class=&quot;card-img-placeholder&quot;><span>Image load failed</span></div>'
    img_tag = (
        f'<img src="{safe_url}" class="card-img" '
        f"onerror=\"this.style.display='none'; "
        f"this.insertAdjacentHTML('afterend', '{fallback}');\">"
    )
    
    if youtube_link and embed_url:
        safe_yt = html.escape(youtube_link, quote=True)
        safe_embed = html.escape(embed_url, quote=True)
        return (
            f'<a href="{safe_yt}" target="_blank" class="card-img-link" data-embed="{safe_embed}">'
            f'{img_tag}'
            f'<div class="yt-embed-container"></div>'
            f'</a>'
        )
    return img_tag


def _build_youtube_html(youtube_link: str) -> str:
    if not youtube_link:
        return ""
    safe_url = html.escape(youtube_link, quote=True)
    return f'<a href="{safe_url}" target="_blank" class="yt-btn">▶ YouTube</a>'


def _build_ocr_html(ocr_text: str) -> str:
    if not ocr_text:
        return ""
    preview = escape_html(ocr_text[:OCR_PREVIEW_MAX_CHARS])
    return f'<div class="card-ocr"><span class="ocr-label">OCR</span>{preview}</div>'


def _build_evidence_html(evidence: list[str]) -> str:
    if not evidence:
        return ""
    escaped = [escape_html(item) for item in evidence]
    return (
        '<div style="font-size:0.7rem;color:#9ca3af;margin-top:4px;">'
        f'Sources: {", ".join(escaped)}</div>'
    )


# ── Card renderer ─────────────────────────────────────────────────────────────

def render_result_card(result: dict, idx: int) -> str:
    video_id = escape_html(result.get("video_id") or "—")
    frame_id = escape_html(result.get("frame_id", "—"))
    score = float(result.get("score", 0) or 0)
    branch = escape_html(result.get("branch", ""))
    branch_color = BRANCH_COLORS.get(result.get("branch", ""), DEFAULT_BRANCH_COLOR)
    bar_pct = min(int(score * SCORE_BAR_SCALE), SCORE_BAR_MAX_PCT)

    img_html = _build_image_html(result.get("azure_url") or "", result.get("youtube_link") or "")
    yt_html = _build_youtube_html(result.get("youtube_link") or "")
    ocr_html = _build_ocr_html(result.get("ocr_text") or "")
    evidence_html = _build_evidence_html(result.get("evidence") or [])

    return f"""\
<div class="result-card" style="animation-delay: {idx * 0.06}s">
  <div class="card-rank">#{idx + 1}</div>
  <div class="card-img-wrap">{img_html}</div>
  <div class="card-body">
    <div class="card-meta-row">
      <span class="meta-pill video">🎬 {video_id}</span>
      <span class="meta-pill frame">🖼 Frame {frame_id}</span>
      <span class="meta-pill" style="background:{branch_color};color:white;font-size:0.65rem;">{branch}</span>
    </div>
    <div class="score-row">
      <span class="score-label">Score</span>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{bar_pct}%"></div>
      </div>
      <span class="score-val">{score:.5f}</span>
    </div>
    {ocr_html}
    {evidence_html}
    {yt_html}
  </div>
</div>"""


def render_latency_badge(latency: dict) -> str:
    agentic = latency.get("agentic_ms", 0)
    heuristic = latency.get("heuristic_ms", 0)
    rerank = latency.get("rerank_ms", 0)
    total = latency.get("total_ms", 0)
    
    parts = []
    if agentic > 0:
        parts.append(f'<span class="lat-chip enc">🤖 Agentic {agentic:.0f}ms</span>')
    if heuristic > 0:
        parts.append(f'<span class="lat-chip srch">📊 Heuristic {heuristic:.0f}ms</span>')
    if rerank > 0:
        parts.append(f'<span class="lat-chip" style="background:rgba(245,158,11,0.15);color:#f59e0b;">🔀 Rerank {rerank:.0f}ms</span>')
    parts.append(f'<span class="lat-chip tot">🕐 Total {total:.0f}ms</span>')
    
    inner_html = "".join(parts)
    return f'<div class="latency-bar">{inner_html}</div>'


# ── Iframe builder ────────────────────────────────────────────────────────────

def build_cards_iframe(results: list[dict]) -> tuple[str, int]:
    """Build full HTML doc for the results iframe and estimate its height."""
    cards = "".join(render_result_card(r, i) for i, r in enumerate(results))
    full_html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<style>body {{ margin:0; padding:0; background:transparent; overflow:hidden; "
        f"font-family:'Inter','Segoe UI',sans-serif; }} {_APP_CSS}</style>"
        f"</head><body><div class=\"results-grid\">{cards}</div>"
        f"<script>\n"
        f"document.querySelectorAll('.card-img-link').forEach(link => {{\n"
        f"  let hoverTimer;\n"
        f"  link.addEventListener('mouseenter', () => {{\n"
        f"    const embed = link.getAttribute('data-embed');\n"
        f"    if(!embed) return;\n"
        f"    hoverTimer = setTimeout(() => {{\n"
        f"      const container = link.querySelector('.yt-embed-container');\n"
        f"      if(!container.innerHTML) {{\n"
        f"        container.innerHTML = `<iframe src=\"${{embed}}\" allow=\"autoplay\" style=\"position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;border:none;\"></iframe>`;\n"
        f"      }}\n"
        f"      container.style.opacity = 1;\n"
        f"    }}, 400);\n"
        f"  }});\n"
        f"  link.addEventListener('mouseleave', () => {{\n"
        f"    clearTimeout(hoverTimer);\n"
        f"    const container = link.querySelector('.yt-embed-container');\n"
        f"    if(container) {{\n"
        f"      container.style.opacity = 0;\n"
        f"      setTimeout(() => container.innerHTML = '', 300);\n"
        f"    }}\n"
        f"  }});\n"
        f"}});\n"
        f"</script></body></html>"
    )
    rows = (len(results) + CARDS_PER_ROW - 1) // CARDS_PER_ROW
    gap_px = 20
    padding_px = 20
    calculated_height = (rows * CARD_HEIGHT_PX) + max(0, rows - 1) * gap_px + padding_px
    height = max(calculated_height, MIN_IFRAME_HEIGHT_PX)
    return full_html, height


# ── UI: Brand logo ────────────────────────────────────────────────────────────
st.markdown('<a href="/" target="_self" class="brand-logo" title="Về trang chủ">LookUp.ai</a>', unsafe_allow_html=True)

# ── UI: Idle hero (shown only when no results) ────────────────────────────────
hero_placeholder = st.empty()

if "results_data" not in st.session_state:
    with hero_placeholder.container():
        st.markdown("""\
<div class="hero-center">
  <div class="hero-sparkle">✨</div>
  <h1 class="hero-title">Tìm kiếm bằng ngôn ngữ tự nhiên</h1>
  <p class="hero-sub">Nhập mô tả cảnh quay bạn muốn tìm — hệ thống sẽ truy vấn qua Agentic + Heuristic retrieval</p>
  <div class="chip-row">
    <span class="chip" data-query="Người đang chạy ngoài đường">🏃 Người đang chạy ngoài đường</span>
    <span class="chip" data-query="Cảnh nấu ăn ban ngày">🍜 Cảnh nấu ăn ban ngày</span>
    <span class="chip" data-query="Người đứng nói chuyện trước đám đông">🎤 Người đứng nói chuyện trước đám đông</span>
  </div>
</div>""", unsafe_allow_html=True)

    components.html("""
    <script>
    // Access the parent Streamlit DOM
    const parentWindow = window.parent.document;
    const chips = parentWindow.querySelectorAll('.chip');
    
    chips.forEach(chip => {
        if (chip.hasAttribute('data-hooked')) return;
        chip.setAttribute('data-hooked', 'true');
        
        chip.addEventListener('click', () => {
            const query = chip.getAttribute('data-query') || chip.innerText;
            const textarea = parentWindow.querySelector('[data-testid="stChatInput"] textarea');
            
            if (textarea) {
                // Must use native setter to bypass React 16+ event suppression
                const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                nativeSetter.call(textarea, query);
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                textarea.focus();
            }
        });
    });
    </script>
    """, height=0, width=0)

# ── UI: Results section ──────────────────────────────────────────────────────
if "results_data" in st.session_state:
    data = st.session_state["results_data"]
    query_shown = st.session_state.get("last_query", "")

    st.markdown(
        f'<div class="results-header-wrapper">'
        f'  <div class="results-stats-row">'
        f'    <div class="results-count">{data["total_results"]} kết quả</div>'
        f'    {render_latency_badge(data.get("latency_ms", {}))}'
        f'  </div>'
        f'  <div class="centered-query-container">'
        f'    <div class="results-query">"{query_shown}"</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    iframe_html, iframe_height = build_cards_iframe(data["results"])
    components.html(iframe_html, height=iframe_height, scrolling=False)

# ── UI: Search input (fixed bottom) ──────────────────────────────────────────
search_query = st.chat_input("Mô tả cảnh bạn muốn tìm...")

if search_query:
    hero_placeholder.empty()
    
    spinner = st.empty()
    spinner.markdown(SPINNER_HTML, unsafe_allow_html=True)

    data = call_search_api(query_to_run, top_k=current_top_k, strategy=current_strategy)
    spinner.empty()

    if data:
        st.session_state["results_data"] = data
        st.session_state["last_query"] = query_to_run
        st.session_state["last_strategy"] = current_strategy
        st.session_state["last_top_k"] = current_top_k
        st.rerun()
