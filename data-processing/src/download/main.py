from __future__ import annotations

"""
main.py — Download videos and extract metadata in a single pipeline step.

Logic:
  - Step 1: manifest
  - Step 2: download
  - Step 3: ffprobe metadata extraction

Usage example:
    python -m src.download.main \
        --input-excel data/manifests/templates/link_videos_template.xlsx \
        --output-root downloaded_videos/ \
        --download-mode 720p_mp4

Run from the data-processing/ directory so that relative imports resolve correctly.
"""

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the data-processing/ root is on sys.path so that
# `from src.download...` works when the file is executed directly.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from src.download.excel_manifest import load_manifest_from_excel, save_normalized_manifest
from src.download.youtube_download import download_from_manifest
from src.download.ffprobe_extract import build_video_metadata_from_manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline Step 1: Load an Excel file of YouTube URLs, "
            "download the videos, then extract technical metadata with ffprobe."
        )
    )

    # --- Download args ---
    download_group = parser.add_argument_group("Download options")
    download_group.add_argument(
        "--input-excel",
        type=Path,
        required=True,
        help="Path to the input Excel file. The first column must be named 'url'.",
    )
    download_group.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Output directory that will contain videos/, info_json/, "
            "normalized_manifest.csv, download_log.csv, and metadata files."
        ),
    )
    download_group.add_argument(
        "--sheet-name",
        default=0,
        help="Excel sheet name or sheet index. Default: 0",
    )
    download_group.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="Optional limit on the number of videos to download (useful for testing).",
    )
    download_group.add_argument(
        "--download-mode",
        type=str,
        default="720p_mp4",
        choices=["best", "720p_mp4", "480p_mp4"],
        help="yt-dlp download format preset. Default: 720p_mp4",
    )
    download_group.add_argument(
        "--sleep-between-downloads",
        type=float,
        default=1.0,
        help="Seconds to sleep between individual downloads. Default: 1.0",
    )
    download_group.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing video files instead of skipping them.",
    )
    download_group.add_argument(
        "--no-info-json",
        action="store_true",
        help="Do not save yt-dlp .info.json sidecar files.",
    )
    download_group.add_argument(
        "--deduplicate-video-ids",
        action="store_true",
        help="Drop duplicate video_id rows before downloading.",
    )

    # --- Metadata args ---
    meta_group = parser.add_argument_group("Metadata options")
    meta_group.add_argument(
        "--video-metadata-output",
        type=Path,
        default=None,
        help=(
            "Explicit output path for video_metadata.csv. "
            "Defaults to <output-root>/video_metadata.csv"
        ),
    )
    meta_group.add_argument(
        "--metadata-log-output",
        type=Path,
        default=None,
        help=(
            "Explicit output path for metadata_log.csv. "
            "Defaults to <output-root>/metadata_log.csv"
        ),
    )
    meta_group.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip the ffprobe metadata extraction step (download only).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_sheet_name(value):
    """Convert CLI --sheet-name string to int when it looks like a digit."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _check_ffprobe() -> str:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        raise RuntimeError(
            "ffprobe was not found in PATH. "
            "Please install FFmpeg (which includes ffprobe) and make sure it is accessible."
        )
    return ffprobe_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    normalized_manifest_path = output_root / "normalized_manifest.csv"
    download_log_path         = output_root / "download_log.csv"
    video_metadata_output     = args.video_metadata_output or output_root / "video_metadata.csv"
    metadata_log_output       = args.metadata_log_output   or output_root / "metadata_log.csv"

    # ------------------------------------------------------------------
    # Step 1 — Load Excel → normalized_manifest.csv
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Load Excel and create normalized manifest")
    print("=" * 60)
    print("Input Excel           :", args.input_excel)
    print("Output root           :", output_root)
    print("Normalized manifest   :", normalized_manifest_path)
    print()

    manifest_df = load_manifest_from_excel(
        excel_path=args.input_excel,
        sheet_name=_normalize_sheet_name(args.sheet_name),
        deduplicate_video_ids=args.deduplicate_video_ids,
    )

    if args.max_downloads is not None:
        manifest_df = manifest_df.head(args.max_downloads).copy()

    save_normalized_manifest(manifest_df, normalized_manifest_path)

    print(f"Rows in normalized manifest: {len(manifest_df)}")
    print(manifest_df.head(10).to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # Step 2 — Download videos
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Download videos")
    print("=" * 60)

    download_log_df = download_from_manifest(
        manifest_df=manifest_df,
        output_root=output_root,
        max_downloads=None,
        write_info_json=not args.no_info_json,
        overwrite_existing=args.overwrite_existing,
        sleep_between_downloads=args.sleep_between_downloads,
        download_mode=args.download_mode,
    )

    print()
    print("Download log          :", download_log_path)
    print("Download archive      :", output_root / "download_archive.txt")
    print()

    if not download_log_df.empty and "status" in download_log_df.columns:
        ok_count    = int((download_log_df["status"] == "ok").sum())
        error_count = int((download_log_df["status"] == "error").sum())
        print(f"Total : {len(download_log_df)}")
        print(f"OK    : {ok_count}")
        print(f"Error : {error_count}")
        print()

        if error_count > 0:
            print("Failed downloads:")
            failed = download_log_df.loc[
                download_log_df["status"] == "error",
                ["row_number", "video_id", "url", "error"],
            ]
            print(failed.head(20).to_string(index=False))
            print()

    # ------------------------------------------------------------------
    # Step 3 — Extract video metadata with ffprobe
    # ------------------------------------------------------------------
    if args.skip_metadata:
        print("Skipping metadata extraction (--skip-metadata flag is set).")
        return

    print("=" * 60)
    print("Step 3: Extract video metadata (ffprobe)")
    print("=" * 60)

    ffprobe_path = _check_ffprobe()
    print("ffprobe               :", ffprobe_path)
    print("video_metadata.csv    :", video_metadata_output)
    print("metadata_log.csv      :", metadata_log_output)
    print()

    # Re-read the manifest from disk so previously downloaded (but
    # not in this run's manifest_df) videos are also included.
    manifest_for_meta = pd.read_csv(normalized_manifest_path)
    if args.max_downloads is not None:
        manifest_for_meta = manifest_for_meta.head(args.max_downloads).copy()

    video_metadata_df, metadata_log_df = build_video_metadata_from_manifest(
        manifest_df=manifest_for_meta,
        base_dir=output_root,
        video_metadata_output_path=video_metadata_output,
        metadata_log_output_path=metadata_log_output,
    )

    ok_count    = int((metadata_log_df["status"] == "ok").sum())   if "status" in metadata_log_df.columns else 0
    error_count = int((metadata_log_df["status"] == "error").sum()) if "status" in metadata_log_df.columns else 0

    print(f"Processed : {len(metadata_log_df)}")
    print(f"OK        : {ok_count}")
    print(f"Error     : {error_count}")
    print()

    if not video_metadata_df.empty:
        preview_cols = [
            col for col in [
                "video_id", "title", "duration_sec", "fps",
                "width", "height", "codec_name", "has_audio", "video_path",
            ]
            if col in video_metadata_df.columns
        ]
        print("video_metadata.csv preview:")
        print(video_metadata_df[preview_cols].head(10).to_string(index=False))
        print()

    if error_count > 0:
        failed_meta = metadata_log_df.loc[
            metadata_log_df["status"] == "error",
            [col for col in ["video_id", "video_path", "error"] if col in metadata_log_df.columns],
        ]
        print("Failed metadata rows:")
        print(failed_meta.head(20).to_string(index=False))
        print()

    print("=" * 60)
    print("Pipeline finished successfully.")
    print("=" * 60)
    print("Outputs:")
    print("  normalized_manifest.csv :", normalized_manifest_path)
    print("  download_log.csv        :", download_log_path)
    print("  video_metadata.csv      :", video_metadata_output)
    print("  metadata_log.csv        :", metadata_log_output)


if __name__ == "__main__":
    main()
