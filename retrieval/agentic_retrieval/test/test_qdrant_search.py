from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()


def _add_project_paths() -> None:
    """
    Make the script runnable from either:
    - project root
    - tests/ folder
    - arbitrary working directory (as long as script is copied into repo)
    """
    here = Path(__file__).resolve()
    candidates = [here.parent, Path.cwd(), here.parent.parent]
    for base in candidates:
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))


_add_project_paths()


try:
    from services.qdrant_search import QdrantSearchService
except ImportError as exc:
    raise SystemExit(
        "Could not import services.qdrant_search.QdrantSearchService. "
        "Place this script in your project root (or adjust sys.path) so that the 'services/' package is importable.\n"
        f"Original error: {exc}"
    )


try:
    from retrieval import parallel_retrieval_node_factory
except ImportError:
    parallel_retrieval_node_factory = None  # optional integration test


def _safe_preview(items: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    preview = []
    for item in items[:limit]:
        payload = item.get("raw_payload", {}) or {}
        preview.append(
            {
                "video_id": item.get("video_id"),
                "frame_id": item.get("frame_id"),
                "score": round(float(item.get("score", 0.0)), 6),
                "caption": payload.get("caption", ""),
                "tags": payload.get("tags", []),
                "timestamp_sec": payload.get("timestamp_sec"),
                "youtube_link": payload.get("youtube_link"),
            }
        )
    return preview


def _print_block(title: str, data: Any) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def test_service_methods(service: QdrantSearchService, query_texts: List[str], top_k: int) -> Dict[str, List[Dict[str, Any]]]:
    results = {
        "keyframe": service.search_keyframe(query_texts, top_k=top_k),
        "ocr": service.search_ocr(query_texts, top_k=top_k),
        "object": service.search_object(query_texts, top_k=top_k),
        "metadata": service.search_metadata(query_texts, top_k=top_k),
        "caption": service.search_caption(query_texts, top_k=top_k),
    }

    counts = {k: len(v) for k, v in results.items()}
    _print_block("Raw QdrantSearchService result counts", counts)

    for source, items in results.items():
        _print_block(f"Top {min(3, len(items))} preview for source='{source}'", _safe_preview(items, limit=3))

    return results


def test_parallel_node(service: QdrantSearchService, query_texts: List[str], top_k: int) -> None:
    if parallel_retrieval_node_factory is None:
        print("\n[WARN] retrieval.parallel_retrieval_node_factory could not be imported. Skipping integration test.")
        return

    # Build a minimal state shaped like your current AgentState usage
    query_bundle = {
        "raw": query_texts[0],
        "cleaned": query_texts[0],
        "lang": "en",
        "translated_en": query_texts[0],
        "rewrites": query_texts[1:3],
    }
    query_intent = {
        "objects": [],
        "attributes": [],
        "actions": [],
        "scene": [],
        "text_cues": [],
        "metadata_cues": [],
        "query_type": "mixed",
    }
    state = {
        "query_bundle": query_bundle,
        "query_intent": query_intent,
        "trace_logs": [],
    }

    node = parallel_retrieval_node_factory(service, top_k_per_source=top_k)
    out_state = node(state)

    counts = {k: len(v) for k, v in out_state.get("retrieval_results", {}).items()}
    _print_block("parallel_retrieval_node_factory output counts", counts)
    _print_block("parallel_retrieval trace logs", out_state.get("trace_logs", []))

    for source, items in out_state.get("retrieval_results", {}).items():
        _print_block(f"Node preview for source='{source}'", _safe_preview(items, limit=3))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Qdrant-backed multimodal retrieval.")
    parser.add_argument(
        "--query",
        type=str,
        default="speaker in red speaking outdoors",
        help="Main query text used for all retrievers.",
    )
    parser.add_argument(
        "--rewrite",
        action="append",
        default=[],
        help="Optional additional query variant. Can be passed multiple times.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k results per source.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="keyframes_v1",
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Qdrant request timeout in seconds.",
    )
    parser.add_argument(
        "--skip-node-test",
        action="store_true",
        help="Only test raw QdrantSearchService methods, skip parallel_retrieval_node_factory.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    QDRANT_URL = os.getenv('QDRANT_URL') or os.environ.get("QDRANT_URL")
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY') or os.environ.get("QDRANT_API_KEY")

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise SystemExit("Missing QDRANT_URL or QDRANT_API_KEY in environment.")

    query_texts = [args.query] + [rw for rw in args.rewrite if rw.strip()]

    _print_block(
        "Test configuration",
        {
            "collection": args.collection,
            "top_k": args.top_k,
            "query_texts": query_texts,
            "qdrant_url_present": bool(QDRANT_URL),
            "qdrant_api_key_present": bool(QDRANT_API_KEY),
        },
    )

    service = QdrantSearchService(
        collection_name=args.collection,
        timeout=args.timeout,
    )

    test_service_methods(service, query_texts=query_texts, top_k=args.top_k)

    if not args.skip_node_test:
        test_parallel_node(service, query_texts=query_texts, top_k=args.top_k)


if __name__ == "__main__":
    main()
