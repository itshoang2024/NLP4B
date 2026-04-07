"""
mock_api.py — Lightweight mock server to test the Streamlit UI
without loading SigLIP / BGE-M3 / Qdrant.

Run:
  cd retrieval/heuristic_retrieval
  python mock_api.py

Then open http://localhost:8501 (Streamlit) — UI will call this server normally.
"""

import time
import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="LookUp.ai Mock API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MOCK_VIDEOS   = ["L01_V001", "L02_V034", "L03_V012", "L04_V088", "L05_V007"]
MOCK_CAPTIONS = [
    "Một người đang đứng trước bảng trắng, giải thích điều gì đó cho khán giả.",
    "Cảnh quay ngoài trời, hai người đang đi bộ dọc theo con đường có cây xanh.",
    "Một đầu bếp đang chuẩn bị món ăn trong bếp, có nhiều nguyên liệu trên bàn.",
    "Cảnh toàn cảnh thành phố buổi đêm nhìn từ trên cao, đèn đường rực sáng.",
    "Một nhóm người đang ngồi quanh bàn họp, thảo luận với nhau.",
    "Trận đấu thể thao đang diễn ra ngoài sân vận động, khán giả đông đảo.",
    "Người phụ nữ đang đọc sách trong thư viện yên tĩnh.",
    "Cảnh hoàng hôn tuyệt đẹp bên bờ biển, sóng nhẹ vỗ vào.",
    "Em bé đang chơi với đồ chơi trong phòng khách ấm áp.",
    "Một ca sĩ đang trình diễn trên sân khấu lớn, ánh đèn rực rỡ.",
]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    prefetch_limit: int = 50


@app.get("/health")
def health():
    return {"status": "ok", "collection": "keyframes_v1 (mock)", "device": "cpu (mock)"}


@app.post("/search")
def search(req: SearchRequest):
    time.sleep(0.4)  # simulate latency

    results = []
    for i in range(req.top_k):
        vid = random.choice(MOCK_VIDEOS)
        frame = random.randint(0, 500)
        ts = round(frame / 30, 1)
        caption = MOCK_CAPTIONS[i % len(MOCK_CAPTIONS)]
        score = round(0.065 / (i + 1) + random.uniform(0, 0.003), 6)

        results.append({
            "rank": i + 1,
            "point_id": f"mock-{i:04d}",
            "rrf_score": score,
            "payload": {
                "video_id": vid,
                "frame_idx": frame,
                "timestamp_sec": ts,
                "caption": caption,
                "detailed_caption": None,
                "ocr_text": f"OCR sample text #{i}" if i % 3 == 0 else None,
                "azure_url": f"https://picsum.photos/seed/{i+10}/640/360",
                "youtube_link": f"https://www.youtube.com/watch?v=dQw4w9WgXcQ&t={int(ts)}s",
            },
        })

    return {
        "query": req.query,
        "total_results": req.top_k,
        "results": results,
        "latency_ms": {
            "encode_ms": round(random.uniform(180, 320), 2),
            "search_ms": round(random.uniform(30, 80), 2),
            "total_ms":  round(random.uniform(220, 400), 2),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mock_api:app", host="0.0.0.0", port=8000, reload=False)
