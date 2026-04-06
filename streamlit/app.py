import streamlit as st
import time
st.set_page_config(
    page_title="LookUp.ai",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load global styles.
with open("assets/styles/main.css", "r", encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# Hiển thị Tên Team
st.markdown('<div class="brand-logo">LookUp.ai</div>', unsafe_allow_html=True)

# Keep the first screen intentionally blank for incremental UI building.
st.markdown('<div class="blank-canvas"></div>', unsafe_allow_html=True)

# Thanh search nằm dưới cùng
search_query = st.chat_input("Bạn đang tìm gì?")
if search_query:
    # Div rỗng để giữ chỗ cho spinner
    spinner_placeholder = st.empty()
    
    spinner_html = """
    <div class="custom-loader-container">
        <div class="custom-loader">
            <svg class="custom-loader-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
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
        <div class="custom-loader-text">Đang xử lý dữ liệu...</div>
    </div>
    """
    spinner_placeholder.markdown(spinner_html, unsafe_allow_html=True)
    
    time.sleep(3) # Cố ý cho chờ một nhịp ngắn (không quá dài như 15s) để show spinner đẹp
    
    # Xoá loading spinner sau khi chạy xong
    spinner_placeholder.empty()
    
    st.success(f"Kết quả cho: **{search_query}**")
