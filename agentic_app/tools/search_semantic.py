from typing import List
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from .models import RetrievedFrame
from .embeddings import encode_bge_m3
from ..config.settings import settings

@tool
def search_semantic(query: str, top_k: int = 20) -> List[RetrievedFrame]:
    """
    Search the vector database using semantic text meaning. 
    Use this when the query is about abstract concepts, logic, or detailed actions that might be described in a caption.
    """
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    
    query_vector = encode_bge_m3(query)
    if not query_vector:
        return []

    hits = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_vector,
        using="keyframe-caption-dense",
        limit=top_k,
        with_payload=True
    ).points

    results = []
    for hit in hits:
        results.append(RetrievedFrame(
            point_id=hit.id if isinstance(hit.id, str) else str(hit.id),
            video_id=hit.payload.get("video_id", ""),
            frame_idx=hit.payload.get("frame_idx", 0),
            timestamp_sec=hit.payload.get("timestamp_sec", 0),
            youtube_link=hit.payload.get("youtube_link", ""),
            azure_url=hit.payload.get("azure_url", ""),
            caption=hit.payload.get("caption", ""),
            ocr_text=hit.payload.get("ocr_text", ""),
            tags=hit.payload.get("tags", []),
            score=hit.score,
            source_vector="keyframe-caption-dense"
        ))
    return results
