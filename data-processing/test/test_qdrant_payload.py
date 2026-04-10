import os
import random
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv(".env")
qdrant_url = os.environ.get('QDRANT_URL')
qdrant_key = os.environ.get('QDRANT_API_KEY')

client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=30)
col = "keyframes_v1"

print(f"Connecting to Qdrant at {col}...")
# Scroll some points to verify
res, next_page = client.scroll(
    collection_name=col,
    limit=100,
    with_payload=True
)

print(f"Scanned {len(res)} points. Let's look for ocr_text (excluding '0015ef2b-ad46-5c80-92da-9772dc1face7'):\n")

found_ocr = 0
for point in res:
    if str(point.id) == "0015ef2b-ad46-5c80-92da-9772dc1face7":
        continue
        
    if "ocr_text" in point.payload and point.payload["ocr_text"]:
        ocr_text = point.payload["ocr_text"]
        print(f"Point {point.id} (Video: {point.payload.get('video_id')} | Frame: {point.payload.get('frame_idx')}):")
        print(f"  --> ocr_text: {ocr_text.encode('utf-8')}\n")
        found_ocr += 1
        
    # We just want to print 3 random examples
    if found_ocr >= 5:
        break

if found_ocr == 0:
    print("Could not find any new points with ocr_text in the first 100 points.")
