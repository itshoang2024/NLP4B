"""
service.py — Mock heuristic retrieval service.

This is a STUB implementation. The real logic (encode_visual + encode_semantic
→ hybrid_rrf_search via Qdrant) will be implemented by the team member
responsible for the heuristic branch.

Interface contract:
    service = HeuristicRetrieveService()
    candidates = service.retrieve(query_bundle, top_k=10)
    # Returns list[dict] with keys: video_id, frame_id, score, branch, source
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


MOCK_VIDEOS = ["L01_V001", "L02_V034", "L03_V012", "L04_V088", "L05_V007"]


class HeuristicRetrieveService:
    """
    Mock heuristic retrieval service.

    TODO (team member): Replace this mock with real implementation.
    Reference files from the old codebase:
      - retrieval/heuristic_retrieval/embedder.py   (encode_visual, encode_semantic)
      - retrieval/heuristic_retrieval/searcher.py   (hybrid_rrf_search)
      - retrieval/heuristic_retrieval/config.py     (Qdrant connection)

    The real implementation should:
      1. Encode query_bundle["translated_en"] (or cleaned) via Azure Embedding API
      2. Run hybrid_rrf_search with SigLIP + BGE-M3 dense vectors
      3. Return candidates in the same dict format as this mock
    """

    def __init__(self):
        logger.info("HeuristicRetrieveService initialized (MOCK)")

    def retrieve(self, query_bundle: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        MOCK: return dummy candidates.

        Parameters
        ----------
        query_bundle : dict
            Pre-processed query bundle from middleware.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[dict]
            List of candidate dicts.
        """
        logger.info("HeuristicRetrieveService.retrieve (MOCK) — top_k=%d", top_k)

        candidates = []
        for i in range(top_k):
            candidates.append({
                "video_id": random.choice(MOCK_VIDEOS),
                "frame_id": random.randint(0, 500),
                "score": round(0.5 / (i + 1), 5),
                "source": "rrf",
                "branch": "heuristic",
                "evidence": ["keyframe-dense", "caption-dense"],
                "raw_payload": {
                    "azure_url": f"https://picsum.photos/seed/{i + 100}/640/360",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "timestamp_sec": round(random.uniform(0, 300), 1),
                    "caption": f"Mock heuristic result #{i + 1}",
                },
            })
        return candidates
