from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client


VALID_RUNNERS = {"hoang", "nanh", "lam", "binh"}
TABLE_NAME = "video_processing_progress"


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


def infer_runner_from_filename(csv_path: Path) -> str:
    runner = csv_path.stem.lower().strip()
    if runner not in VALID_RUNNERS:
        raise ValueError(
            f"Cannot infer runner from filename '{csv_path.name}'. "
            f"Expected one of: {sorted(VALID_RUNNERS)}"
        )
    return runner


def chunked(items: list[dict], size: int = 500):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to runner csv, e.g. hoang.csv")
    parser.add_argument(
        "--runner",
        default=None,
        help="Optional override runner. If omitted, infer from filename.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    runner = args.runner.lower().strip() if args.runner else infer_runner_from_filename(csv_path)
    if runner not in VALID_RUNNERS:
        raise ValueError(f"Invalid runner: {runner}")

    df = pd.read_csv(csv_path)
    if "video_id" not in df.columns:
        raise ValueError(f"CSV must contain a 'video_id' column. Found: {list(df.columns)}")

    df["video_id"] = df["video_id"].map(normalize_video_id)
    df = df[df["video_id"].astype(str).str.len() > 0].drop_duplicates(subset=["video_id"])

    rows = [
        {
            "video_id": row["video_id"],
            "runner": runner,
        }
        for _, row in df.iterrows()
    ]

    client = get_supabase_client()

    total = 0
    for batch in chunked(rows, size=500):
        client.table(TABLE_NAME).upsert(batch).execute()
        total += len(batch)

    print(f"Done. Upserted {total} rows into {TABLE_NAME} for runner='{runner}'.")


if __name__ == "__main__":
    main()