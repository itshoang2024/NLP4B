from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client


TABLE_NAME = "video_processing_progress"
PROGRESS_COLUMNS = {
    "keyframe_extraction",
    "keyframe_upload",
    "embedding",
    "object_detection",
    "ocr",
}


def get_supabase_client() -> Client:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment.")
    return create_client(url, key)


def normalize_video_id(raw: str) -> str:
    value = str(raw).strip()
    if value.endswith(".mp4"):
        value = value[:-4]
    return value


def chunked(items: list[str], size: int = 500):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV file containing video_id column")
    parser.add_argument("--column", required=True, choices=sorted(PROGRESS_COLUMNS))
    parser.add_argument("--value", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "video_id" not in df.columns:
        raise ValueError(f"CSV must contain a 'video_id' column. Found: {list(df.columns)}")

    df["video_id"] = df["video_id"].map(normalize_video_id)
    video_ids = (
        df[df["video_id"].astype(str).str.len() > 0]["video_id"]
        .drop_duplicates()
        .tolist()
    )

    client = get_supabase_client()
    total_updated = 0
    value = bool(args.value)

    for batch_ids in chunked(video_ids, size=500):
        resp = (
            client.table(TABLE_NAME)
            .update({args.column: value})
            .in_("video_id", batch_ids)
            .execute()
        )
        updated_count = len(resp.data) if resp.data else 0
        total_updated += updated_count

    print(
        f"Done. Updated column '{args.column}' = {value} "
        f"for ~{total_updated} rows from {csv_path.name}."
    )


if __name__ == "__main__":
    main()