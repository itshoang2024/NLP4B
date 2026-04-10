import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import SetPayloadOperation, SetPayload
from dotenv import load_dotenv

load_dotenv(".env")

qdrant_url = os.environ.get('QDRANT_URL')
qdrant_key = os.environ.get('QDRANT_API_KEY')

print(f"Connecting to Qdrant...")
client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=30)
col = "keyframes_v1"
pid = "0015ef2b-ad46-5c80-92da-9772dc1face7"

res = client.retrieve(collection_name=col, ids=[pid], with_payload=True)
print("Before keys:", list(res[0].payload.keys()))

payload = {"ocr_text": "chờ đợi..."}
payload_batch = [SetPayloadOperation(set_payload=SetPayload(payload=payload, points=[pid]))]

try:
    print("Updating using batch_update_points...")
    client.batch_update_points(collection_name=col, update_operations=payload_batch)
    print("Update successful.")
except Exception as e:
    print("Update failed:", e)

res = client.retrieve(collection_name=col, ids=[pid], with_payload=True)
print("After keys:", list(res[0].payload.keys()))
print("ocr_text is:", res[0].payload.get("ocr_text", "NOT FOUND").encode('utf-8'))
