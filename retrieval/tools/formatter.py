from typing import List
from .models import RetrievedFrame, FormattedResult

def format_results(frames: List[RetrievedFrame]) -> List[dict]:
    """
    Pure data packaging function. Packages the reranked output into a clean JSON/Dict 
    for the Streamlit UI to render natively. Contains no LLM calls.
    """
    formatted = []
    
    for frame in frames:
        # Guarantee exact-second YouTube matching if provided
        yt_url = frame.youtube_link
        if not yt_url and frame.video_id:
            # Fallback generation if missing in payload but we have logic to build it
            # Depends on your URL scheme, leaving as-is from payload string
            pass

        formatted.append(FormattedResult(
            video_id=frame.video_id,
            timestamp_sec=frame.timestamp_sec,
            youtube_url=frame.youtube_link,
            image_url=frame.azure_url,
            caption=frame.caption,
            ocr_text=frame.ocr_text,
            rrf_score=frame.score  # Using the mutated rank/score from RRF
        ).model_dump())
        
    return formatted
