from typing import List, Optional
from pydantic import BaseModel, Field

class RetrievedFrame(BaseModel):
    """
    A single frame retrieved from Qdrant by one of the search tools.
    """
    point_id: str
    video_id: str
    frame_idx: int
    timestamp_sec: int
    youtube_link: str
    azure_url: str
    caption: str = ""
    ocr_text: str = ""
    tags: List[str] = Field(default_factory=list)
    score: float
    source_vector: str

class FormattedResult(BaseModel):
    """
    The final packaged result sent to the Streamlit UI.
    """
    video_id: str
    timestamp_sec: int
    youtube_url: str
    image_url: str
    caption: str
    ocr_text: str
    rrf_score: float
