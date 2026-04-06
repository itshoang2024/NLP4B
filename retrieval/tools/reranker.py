from typing import List, Dict
from collections import defaultdict
from .models import RetrievedFrame

def rerank(result_sets: List[List[RetrievedFrame]], k: int = 60, top_n: int = 5) -> List[RetrievedFrame]:
    """
    Reciprocal Rank Fusion (RRF) algorithm to merge multiple lists of RetrievedFrames.
    Pure function called by the LangGraph "rerank" node.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    frame_lookup: Dict[str, RetrievedFrame] = {}

    for results in result_sets:
        for rank, frame in enumerate(results):
            # RRF formula: 1 / (k + rank)
            # rank is 0-indexed here, so we do +1 for standard RRF
            score = 1.0 / (k + rank + 1)
            rrf_scores[frame.point_id] += score
            
            # Keep the frame data, update with max score
            if frame.point_id not in frame_lookup:
                frame_lookup[frame.point_id] = frame
            else:
                # Merge logic if needed, currently we just keep the first one we saw
                # We could append `source_vector` to show it came from multiple tools
                pass

    # Sort by RRF score descending
    sorted_points = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for pid, score in sorted_points[:top_n]:
        frame = frame_lookup[pid]
        # Mutate the score to be the RRF score
        frame.score = score
        final_results.append(frame)

    return final_results
