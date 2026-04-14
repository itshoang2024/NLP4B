import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter

def main():
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    
    if not qdrant_url:
        qdrant_url = input("Nhập QDRANT_URL của bạn: ")
        qdrant_key = input("Nhập QDRANT_API_KEY của bạn: ")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    collection_name = "keyframes_v1"

    print(f"Đang tiến hành xoá toàn bộ trường 'timestamp_sec' khỏi collection {collection_name}...")
    
    try:
        # Filter() rỗng tương đương việc match tất cả các Point trong Collection
        client.delete_payload(
            collection_name=collection_name,
            keys=["timestamp_sec"],
            points_selector=Filter()
        )
        print("✅ Xoá thành công trường 'timestamp_sec' khỏi toàn bộ DB!")
    except Exception as e:
        print(f"❌ Xoá thất bại: {e}")

if __name__ == "__main__":
    main()
