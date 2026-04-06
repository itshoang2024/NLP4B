from __future__ import annotations

import json
from pathlib import Path
import sys

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

from nodes.routing import compute_modality_weights

TEST_INTENTS = {
    "visual_event": {
        "objects": ["speaker"],
        "attributes": ["red shirt"],
        "actions": ["speaking"],
        "scene": ["outdoor"],
        "text_cues": [],
        "metadata_cues": [],
        "query_type": "visual_event",
    },
    "text_in_image": {
        "objects": [],
        "attributes": [],
        "actions": [],
        "scene": [],
        "text_cues": ["final round", "championship"],
        "metadata_cues": [],
        "query_type": "text_in_image",
    },
    "visual_object": {
        "objects": ["microphone", "camera", "person"],
        "attributes": ["black suit"],
        "actions": [],
        "scene": [],
        "text_cues": [],
        "metadata_cues": [],
        "query_type": "visual_object",
    },
    "empty_fallback": {
        "objects": [],
        "attributes": [],
        "actions": [],
        "scene": [],
        "text_cues": [],
        "metadata_cues": [],
        "query_type": "mixed",
    },
}

if __name__ == "__main__":
    for name, intent in TEST_INTENTS.items():
        print("=" * 80)
        print(name)
        print(json.dumps(intent, ensure_ascii=False, indent=2))
        print("weights:")
        print(json.dumps(compute_modality_weights(intent), ensure_ascii=False, indent=2))
