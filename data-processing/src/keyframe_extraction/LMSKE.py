"""
LMSKE.py — Large Model based Sequential Keyframe Extraction
============================================================
Full end-to-end pipeline:
  1. Resolve video source → local path, Google Drive link, or any URL
  2. Shot segmentation (TransNetV2)
  3. Feature extraction (OpenAI CLIP, with per-shot frame sampling)
  4. Adaptive clustering + redundancy elimination (src/extraction)
  5. Save keyframes with naming: <video_id>_00001.jpg, ...

Usage (Google Colab — video already on Drive):
  !python LMSKE.py --video "/content/drive/MyDrive/myvideo.mp4" --output_dir "/content/output"

Usage (Google Drive share link):
  !python LMSKE.py --video "https://drive.google.com/file/d/FILE_ID/view" --output_dir "/content/output"

Usage (YouTube):
  !python LMSKE.py --video "https://youtu.be/..." --output_dir "/content/output"

Optional performance tuning:
  --max_frames_per_shot 30   (default: 30, lower = faster CLIP step)
  --redundancy_threshold 0.94
"""

# ── 0. Auto-install dependencies (Colab-friendly) ─────────────────────────────
import subprocess, sys

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

try:
    import transnetv2  # noqa: F401
except ImportError:
    _pip("transnetv2")

try:
    from transformers import CLIPModel, CLIPProcessor  # noqa: F401
except ImportError:
    _pip("transformers", "torch", "torchvision")

try:
    import yt_dlp  # noqa: F401
except ImportError:
    _pip("yt-dlp")

try:
    import gdown  # noqa: F401
except ImportError:
    _pip("gdown")

# ── 1. Standard imports ────────────────────────────────────────────────────────
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from PIL import Image

# Make src modules importable
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "extraction"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "scripts"))

from Kmeans_improvment import kmeans_silhouette  # noqa: E402
from Redundancy import redundancy                # noqa: E402
from save_keyframe import save_frames            # noqa: E402

# ── 2. CLI Arguments ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="LMSKE: Large Model Sequential Keyframe Extraction"
    )
    parser.add_argument(
        "--video",
        required=True,
        help=(
            "Video source. Accepts: "
            "(1) local file path (e.g. /content/drive/MyDrive/video.mp4), "
            "(2) Google Drive share link, "
            "(3) YouTube or any URL supported by yt-dlp."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        help="Root folder to save keyframe images (default: ./output)",
    )
    # ── Tunable performance knob ──────────────────────────────────────────────
    parser.add_argument(
        "--max_frames_per_shot",
        type=int,
        default=30,
        help=(
            "Max frames CLIP reads per shot. "
            "Lower = faster but coarser features. (default: 30)"
        ),
    )
    parser.add_argument(
        "--redundancy_threshold",
        type=float,
        default=0.94,
        help="Cosine-similarity threshold for redundancy removal. (default: 0.94)",
    )
    return parser.parse_args()


# ── 3. Step 1: Resolve video source ──────────────────────────────────────────
def _is_gdrive_url(url: str) -> bool:
    return "drive.google.com" in url


def resolve_video(video_source: str, save_dir: str) -> str:
    """
    Resolve video from one of three sources:
      - Local file path  → return as-is (no download)
      - Google Drive URL → download with gdown
      - Any other URL    → download with yt-dlp
    Returns the local path to the video file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Case 1: already a local file ─────────────────────────────────────────
    if os.path.isfile(video_source):
        print(f"[LMSKE] Using local video file → {video_source}")
        return video_source

    # ── Case 2: Google Drive link ─────────────────────────────────────────────
    if _is_gdrive_url(video_source):
        import gdown  # pylint: disable=import-outside-toplevel

        # gdown can handle share URLs; output file name is derived from Drive metadata
        output_path = os.path.join(save_dir, "gdrive_video.mp4")
        print(f"[LMSKE] Downloading from Google Drive → {output_path}")
        gdown.download(video_source, output_path, quiet=False, fuzzy=True)
        if not os.path.isfile(output_path):
            raise FileNotFoundError(
                f"gdown did not produce the expected file: {output_path}. "
                "Make sure the file is shared publicly (Anyone with the link)."
            )
        print(f"[LMSKE] Google Drive download complete → {output_path}")
        return output_path

    # ── Case 3: YouTube / generic URL (yt-dlp) ────────────────────────────────
    import yt_dlp  # pylint: disable=import-outside-toplevel

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(save_dir, "%(id)s.%(ext)s"),
        "quiet": False,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_source, download=True)
        ext = info.get("ext", "mp4")
        video_path = os.path.join(save_dir, f"{info.get('id', 'video')}.{ext}")

    print(f"[LMSKE] yt-dlp download complete → {video_path}")
    return video_path


# ── 4. Step 2: Shot segmentation via TransNetV2 ───────────────────────────────
def run_shot_segmentation(video_path: str, scenes_txt_path: str) -> list[tuple[int, int]]:
    """
    Run TransNetV2 on video_path.
    Returns list of (start_frame, end_frame) tuples for each shot.
    Also writes results to scenes_txt_path in the original format expected by
    Keyframe_extraction.py (pairs of ints, one pair per line).
    """
    from transnetv2 import TransNetV2  # pylint: disable=import-outside-toplevel

    model = TransNetV2()
    video_frames, single_frame_predictions, all_frame_predictions = (
        model.predict_video(video_path)
    )
    scenes = model.predictions_to_scenes(single_frame_predictions)
    # scenes is a list of (start, end) numpy tuples

    # Write to txt in the format the existing code expects: "start end\n"
    with open(scenes_txt_path, "w") as f:
        for start, end in scenes:
            f.write(f"{int(start)} {int(end)}\n")

    shots = [(int(s), int(e)) for s, e in scenes]
    print(f"[LMSKE] Shot segmentation done → {len(shots)} shots, saved to {scenes_txt_path}")
    return shots


# ── 5. Step 3: Feature extraction via CLIP ────────────────────────────────────
def extract_clip_features(
    video_path: str,
    shots: list[tuple[int, int]],
    features_pkl_path: str,
    max_frames_per_shot: int,
) -> np.ndarray:
    """
    For each shot, sample up to max_frames_per_shot frames evenly,
    encode with CLIP (openai/clip-vit-large-patch14, 768-dim).
    Non-sampled frames inside the shot get the nearest sampled frame's vector.
    Returns features array of shape (total_frames, 768).
    """
    from transformers import CLIPModel, CLIPProcessor  # pylint: disable=import-outside-toplevel

    HF_MODEL_ID = "openai/clip-vit-large-patch14"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LMSKE] Loading CLIP ({HF_MODEL_ID}) from Hugging Face on {device} ...")
    model = CLIPModel.from_pretrained(HF_MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(HF_MODEL_ID)
    model.eval()

    # Count total frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"[LMSKE] Total frames in video: {total_frames}")

    features = np.zeros((total_frames, 768), dtype=np.float32)

    cap = cv2.VideoCapture(video_path)

    for shot_idx, (start, end) in enumerate(shots):
        shot_len = end - start + 1
        if shot_len <= 0:
            continue

        # Build evenly-spaced sample indices within the shot
        n_samples = min(shot_len, max_frames_per_shot)
        sample_offsets = np.linspace(0, shot_len - 1, n_samples, dtype=int)
        sample_frame_ids = sorted(set(start + o for o in sample_offsets))

        # Read only sampled frames
        sampled_vectors: dict[int, np.ndarray] = {}
        for fid in sample_frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret:
                continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model.get_image_features(**inputs)  # (1, 768)
                feat = feat.float().cpu().numpy().flatten()  # (768,)
            sampled_vectors[fid] = feat

        if not sampled_vectors:
            continue

        # Fill all frames in the shot using nearest sampled frame (O(shot_len))
        sorted_sampled = sorted(sampled_vectors.keys())
        for fid in range(start, min(end + 1, total_frames)):
            # Binary-search-like nearest: find closest sampled key
            idx = np.searchsorted(sorted_sampled, fid)
            if idx == 0:
                nearest = sorted_sampled[0]
            elif idx >= len(sorted_sampled):
                nearest = sorted_sampled[-1]
            else:
                left = sorted_sampled[idx - 1]
                right = sorted_sampled[idx]
                nearest = left if abs(fid - left) <= abs(fid - right) else right
            features[fid] = sampled_vectors[nearest]

        print(
            f"[LMSKE] Shot {shot_idx + 1}/{len(shots)} "
            f"(frames {start}-{end}, sampled {len(sampled_vectors)}) ✓",
            flush=True,
        )

    cap.release()

    # Persist features
    with open(features_pkl_path, "wb") as f:
        pickle.dump(features, f)
    print(f"[LMSKE] Features saved → {features_pkl_path}  shape={features.shape}")
    return features


# ── 6. Step 4+5: Clustering + Redundancy (reuse existing code) ───────────────
def extract_keyframes(
    shots: list[tuple[int, int]],
    features: np.ndarray,
    video_path: str,
    redundancy_threshold: float,
) -> list[int]:
    """
    Mirrors the logic of Keyframe_extraction.scen_keyframe_extraction but
    accepts pre-loaded shots & features directly (avoids re-reading files).
    Uses kmeans_silhouette and redundancy from src/extraction as-is.
    """
    keyframe_index = []
    for shot_idx, (start, end) in enumerate(shots):
        sub_features = features[start : end + 1]
        if len(sub_features) < 3:
            # Too few frames to cluster — keep the middle frame
            mid = start + len(sub_features) // 2
            keyframe_index.append(mid)
            continue

        best_labels, best_centers, k, index = kmeans_silhouette(sub_features)
        final_index = [x + start for x in index]
        final_index = redundancy(video_path, final_index, redundancy_threshold)
        keyframe_index.extend(final_index)

    keyframe_index.sort()
    print(f"[LMSKE] Keyframe extraction complete → {len(keyframe_index)} keyframes")
    return keyframe_index


# ── 7. Step 6: Save keyframes ─────────────────────────────────────────────────
def save_keyframes(
    keyframe_index: list[int],
    video_path: str,
    output_dir: str,
    video_id: str,
) -> None:
    """
    Save each keyframe as <output_dir>/<video_id>/<video_id>_00001.jpg, ...
    Calls save_frames from src/scripts/save_keyframe.py with a custom prefix.
    """
    save_frames(
        keyframe_indexes=keyframe_index,
        video_path=video_path,
        save_path=output_dir,
        folder_name=video_id,
        prefix=video_id,  # new kwarg added to save_keyframe.py
    )
    out_folder = os.path.join(output_dir, video_id)
    print(f"[LMSKE] Keyframes saved → {out_folder}/")


# ── 8. Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Paths setup ───────────────────────────────────────────────────────────
    tmp_dir = os.path.join(args.output_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # ── Step 1: Resolve video source ─────────────────────────────────────────
    print("\n[LMSKE] ══ Step 1/5: Resolving video source ══")
    video_path = resolve_video(args.video, tmp_dir)

    # video_id = name of the original input (file/URL), without extension
    raw_name = os.path.basename(args.video.rstrip("/"))  # works for paths & URLs
    video_id = os.path.splitext(raw_name)[0] or "video"

    # Output folder for this video (frames + scenes file go here)
    video_out_dir = os.path.join(args.output_dir, video_id)
    os.makedirs(video_out_dir, exist_ok=True)

    # Intermediate file paths
    scenes_txt = os.path.join(video_out_dir, f"{video_id}_scenes.txt")  # saved with frames
    features_pkl = os.path.join(tmp_dir, f"{video_id}_features.pkl")

    # ── Step 2: Shot segmentation ─────────────────────────────────────────────
    print("\n[LMSKE] ══ Step 2/5: Shot segmentation (TransNetV2) ══")
    shots = run_shot_segmentation(video_path, scenes_txt)

    # ── Step 3: CLIP feature extraction ──────────────────────────────────────
    print(
        f"\n[LMSKE] ══ Step 3/5: Feature extraction (CLIP, "
        f"max {args.max_frames_per_shot} frames/shot) ══"
    )
    features = extract_clip_features(
        video_path=video_path,
        shots=shots,
        features_pkl_path=features_pkl,
        max_frames_per_shot=args.max_frames_per_shot,
    )

    # ── Step 4+5: Clustering + redundancy ────────────────────────────────────
    print("\n[LMSKE] ══ Step 4/5: Clustering + Redundancy elimination ══")
    keyframe_index = extract_keyframes(
        shots=shots,
        features=features,
        video_path=video_path,
        redundancy_threshold=args.redundancy_threshold,
    )

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    print("\n[LMSKE] ══ Step 5/5: Saving keyframes ══")
    save_keyframes(
        keyframe_index=keyframe_index,
        video_path=video_path,
        output_dir=args.output_dir,
        video_id=video_id,
    )

    print(
        f"\n[LMSKE] ✅ Done! {len(keyframe_index)} keyframes → "
        f"{os.path.join(args.output_dir, video_id)}/"
    )


if __name__ == "__main__":
    main()
