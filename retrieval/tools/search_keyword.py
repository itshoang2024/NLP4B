from typing import List
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from .models import RetrievedFrame
from .embeddings import encode_bm25
from ..config.settings import settings

@tool
def search_keyword(query: str, target: str = "ocr", top_k: int = 20) -> List[RetrievedFrame]:
    """
    Search the vector database for exact keyword matches.
    Use this when the query asks about exact text written on screen (target="ocr") or exact object tags (target="objects").
    """
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    
    query_vector = encode_bm25(query)
    if not query_vector:
        return []

    using_vector = "keyframe-ocr-sparse" if target == "ocr" else "keyframe-object-sparse"

    hits = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_vector,
        using=using_vector,
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
            source_vector=using_vector
        ))
    return results
