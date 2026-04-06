from typing import List, Optional
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from .models import RetrievedFrame
from .embeddings import encode_siglip_text
from ..config.settings import settings

@tool
def search_visual(query: str, top_k: int = 20) -> List[RetrievedFrame]:
    """
    Search the vector database using visual/image vibes. 
    Use this when the query describes what a scene LOOKS like (e.g., colors, objects, actions visible in a video).
    """
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    
    query_vector = encode_siglip_text(query)
    if not query_vector:
        return []

    hits = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_vector,
        using="keyframe-dense",
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
            source_vector="keyframe-dense"
        ))
    return results
