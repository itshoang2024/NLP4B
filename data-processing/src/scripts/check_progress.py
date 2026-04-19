'''
Cách chạy:
    Kiểm tra 1 cột:
        python check_progress.py --csv hoang.csv --column keyframe_extraction
    Kiểm tra tất cả các cột:
        python check_progress.py --csv hoang.csv
    Kiểm tra tất cả và hiện luôn các video_id chưa xong:
        python check_progress.py --csv hoang.csv --show-missing
    Kiểm tra riêng embedding:
        python check_progress.py --csv lam.csv --column embedding --show-missing
    Lưu danh sách video_id chưa xong ra file CSV:
        python check_progress.py --csv hoang.csv --column embedding --save_failed
    Kiểm tra tất cả các cột và lưu toàn bộ file failed:
        python check_progress.py --csv hoang.csv --save_failed
'''
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client


TABLE_NAME = "video_processing_progress"
PROGRESS_COLUMNS = [
    "keyframe_extraction",
    "keyframe_upload",
    "embedding",
    "object_detection",
    "ocr",
]
VALID_RUNNERS = {"hoang", "nanh", "lam", "binh"}


def get_supabase_client() -> Client:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment.")
    return create_client(url, key)


def normalize_video_id(raw: object) -> str:
    if pd.isna(raw):
        return ""
    value = str(raw).strip()
    if value.lower().endswith(".mp4"):
        value = value[:-4]
    return value.strip()


def infer_runner_from_filename(csv_path: Path) -> str:
    stem = csv_path.stem.lower().strip()
    runner = stem.split("_")[0]
    if runner not in VALID_RUNNERS:
        raise ValueError(
            f"Cannot infer runner from filename '{csv_path.name}'. "
            f"Expected filename starting with one of: {sorted(VALID_RUNNERS)}"
        )
    return runner


def chunked(items: list[str], size: int = 500) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_progress_rows(
    client: Client,
    video_ids: list[str],
    columns: list[str],
) -> list[dict]:
    all_rows: list[dict] = []
    select_cols = ["video_id"] + columns

    for batch_ids in chunked(video_ids, size=500):
        resp = (
            client.table(TABLE_NAME)
            .select(",".join(select_cols))
            .in_("video_id", batch_ids)
            .execute()
        )
        if resp.data:
            all_rows.extend(resp.data)

    return all_rows


def summarize_column(rows: list[dict], csv_video_ids: list[str], column: str) -> dict:
    row_map = {row["video_id"]: row for row in rows}

    total = len(csv_video_ids)
    done = 0

    for video_id in csv_video_ids:
        value = row_map.get(video_id, {}).get(column, False)
        if value is True:
            done += 1

    not_done = total - done
    pct = (done / total * 100) if total > 0 else 0.0

    return {
        "column": column,
        "total": total,
        "done": done,
        "not_done": not_done,
        "pct": pct,
    }


def get_missing_ids(row_map: dict[str, dict], csv_video_ids: list[str], column: str) -> list[str]:
    return [
        vid for vid in csv_video_ids
        if row_map.get(vid, {}).get(column, False) is not True
    ]


def save_failed_csv(
    runner: str,
    column: str,
    failed_video_ids: list[str],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{runner}_{column}_failed.csv"
    pd.DataFrame({"video_id": failed_video_ids}).to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check progress status in Supabase for video_ids from a CSV file."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV containing video_id")
    parser.add_argument(
        "--column",
        default=None,
        help="Specific progress column to check. If omitted, check all progress columns.",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Show video_ids not marked true for the checked column(s).",
    )
    parser.add_argument(
        "--save_failed",
        action="store_true",
        help="Save video_ids not marked true to {RUNNER}_{PHASE}_failed.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save failed CSV file(s) when using --save_failed",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "video_id" not in df.columns:
        raise ValueError(f"CSV must contain a 'video_id' column. Found: {list(df.columns)}")

    df["video_id"] = df["video_id"].map(normalize_video_id)
    csv_video_ids = (
        df[df["video_id"] != ""]["video_id"]
        .drop_duplicates()
        .tolist()
    )

    if not csv_video_ids:
        raise ValueError("No valid video_id found in CSV after cleaning.")

    if args.column:
        if args.column not in PROGRESS_COLUMNS:
            raise ValueError(
                f"Invalid column '{args.column}'. Must be one of: {PROGRESS_COLUMNS}"
            )
        columns_to_check = [args.column]
    else:
        columns_to_check = PROGRESS_COLUMNS

    runner = infer_runner_from_filename(csv_path) if args.save_failed else None

    client = get_supabase_client()
    rows = fetch_progress_rows(client, csv_video_ids, columns_to_check)

    print(f"CSV file      : {csv_path}")
    print(f"Total video_id: {len(csv_video_ids)}")
    print(f"Checked table : {TABLE_NAME}")
    if runner:
        print(f"Runner        : {runner}")
    print("-" * 60)

    row_map = {row["video_id"]: row for row in rows}
    output_dir = Path(args.output_dir)

    for column in columns_to_check:
        summary = summarize_column(rows, csv_video_ids, column)
        print(f"[{summary['column']}]")
        print(f"  Done     : {summary['done']}/{summary['total']}")
        print(f"  Not done : {summary['not_done']}/{summary['total']}")
        print(f"  Progress : {summary['pct']:.2f}%")
        print()

        missing_ids = get_missing_ids(row_map, csv_video_ids, column)

        if args.show_missing:
            print(f"  Missing video_ids for '{column}':")
            for vid in missing_ids:
                print(f"    - {vid}")
            print()

        if args.save_failed:
            output_path = save_failed_csv(runner, column, missing_ids, output_dir)
            print(f"  Saved failed CSV: {output_path}")
            print()

    found_ids = set(row_map.keys())
    not_found_in_table = [vid for vid in csv_video_ids if vid not in found_ids]
    if not_found_in_table:
        print("-" * 60)
        print(f"video_id not found in Supabase table: {len(not_found_in_table)}")
        for vid in not_found_in_table[:50]:
            print(f"  - {vid}")
        if len(not_found_in_table) > 50:
            print(f"  ... and {len(not_found_in_table) - 50} more")


if __name__ == "__main__":
    main()