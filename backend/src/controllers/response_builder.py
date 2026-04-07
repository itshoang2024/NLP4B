"""
response_builder.py — Build the final SearchResponse from ranked candidates.

Extracts azure_url, youtube_link, and other metadata from raw_payload
to produce flat SearchResultItem objects for the frontend.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.schemas import SearchResponse, SearchResultItem


def build_response(
    query: str,
    candidates: List[Dict[str, Any]],
    latency_ms: Dict[str, float],
) -> SearchResponse:
    """
    Convert internal Candidate dicts into the API response schema.

    Pulls azure_url, youtube_link, timestamp_sec, caption, ocr_text
    from raw_payload and flattens them into SearchResultItem.
    """
    results: List[SearchResultItem] = []

    for rank, item in enumerate(candidates, start=1):
        payload = item.get("raw_payload", {}) or {}

        results.append(SearchResultItem(
            rank=rank,
            video_id=item.get("video_id", "unknown"),
            frame_id=int(item.get("frame_id", 0)),
            score=round(float(item.get("score", 0.0)), 6),
            branch=item.get("branch", "unknown"),
            azure_url=payload.get("azure_url"),
            youtube_link=payload.get("youtube_link"),
            timestamp_sec=payload.get("timestamp_sec"),
            caption=payload.get("caption") or payload.get("detailed_caption"),
            ocr_text=payload.get("ocr_text"),
            evidence=item.get("evidence", []),
        ))

    return SearchResponse(
        query=query,
        total_results=len(results),
        results=results,
        latency_ms=latency_ms,
    )
