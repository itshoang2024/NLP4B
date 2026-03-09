from __future__ import annotations

import argparse
from pathlib import Path

from src.ingest.excel_manifest import load_manifest_from_excel, save_normalized_manifest
from src.ingest.youtube_download import download_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a one-column Excel file of YouTube URLs, create normalized_manifest.csv, and download videos."
    )

    parser.add_argument(
        "--input-excel",
        type=Path,
        required=True,
        help="Path to the input Excel file. The first column must be named 'url'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory containing videos/, info_json/, normalized_manifest.csv, and logs.",
    )
    parser.add_argument(
        "--sheet-name",
        default=0,
        help="Excel sheet name or sheet index. Default: 0",
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    parser.add_argument(
        "--download-mode",
        type=str,
        default="720p_mp4",
        choices=["best", "720p_mp4", "480p_mp4"],
        help="yt-dlp download format preset.",
    )
    parser.add_argument(
        "--sleep-between-downloads",
        type=float,
        default=1.0,
        help="Seconds to sleep between downloads.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing files if yt-dlp would otherwise skip them.",
    )
    parser.add_argument(
        "--no-info-json",
        action="store_true",
        help="Do not save yt-dlp .info.json files.",
    )
    parser.add_argument(
        "--deduplicate-video-ids",
        action="store_true",
        help="Drop duplicate video_id rows before downloading.",
    )

    return parser.parse_args()


def _normalize_sheet_name(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def main() -> None:
    args = parse_args()

    input_excel = args.input_excel
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    normalized_manifest_path = output_root / "normalized_manifest.csv"

    print("=== Step 1: Load Excel and create normalized manifest ===")
    print("Input Excel           :", input_excel)
    print("Output root           :", output_root)
    print("Normalized manifest   :", normalized_manifest_path)
    print()

    manifest_df = load_manifest_from_excel(
        excel_path=input_excel,
        sheet_name=_normalize_sheet_name(args.sheet_name),
        deduplicate_video_ids=args.deduplicate_video_ids,
    )

    if args.max_downloads is not None:
        manifest_df = manifest_df.head(args.max_downloads).copy()

    save_normalized_manifest(manifest_df, normalized_manifest_path)

    print(f"Rows in normalized manifest: {len(manifest_df)}")
    print(manifest_df.head(10).to_string(index=False))
    print()

    print("=== Step 2: Download videos ===")
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
    print("=== Finished ===")
    print("Download log path     :", output_root / "download_log.csv")
    print("Download archive path :", output_root / "download_archive.txt")
    print()

    if not download_log_df.empty and "status" in download_log_df.columns:
        ok_count = int((download_log_df["status"] == "ok").sum())
        error_count = int((download_log_df["status"] == "error").sum())
        print(f"Total rows: {len(download_log_df)}")
        print(f"OK       : {ok_count}")
        print(f"Error    : {error_count}")
        print()

        if error_count > 0:
            print("Failed rows:")
            failed = download_log_df.loc[
                download_log_df["status"] == "error",
                ["row_number", "video_id", "url", "error"],
            ]
            print(failed.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
