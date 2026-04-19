from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def clean_video_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower().endswith(".mp4"):
        s = s[:-4]
    return s.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Clean .mp4 suffix from video_id column in a CSV file."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output CSV. If omitted, overwrite input file.",
    )
    parser.add_argument(
        "--column",
        default="video_id",
        help="Column name to clean (default: video_id)",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Drop duplicate video_id values after cleaning",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path

    df = pd.read_csv(input_path)

    if args.column not in df.columns:
        raise ValueError(
            f"Column '{args.column}' not found. Available columns: {list(df.columns)}"
        )

    original_rows = len(df)

    df[args.column] = df[args.column].map(clean_video_id)
    df = df[df[args.column] != ""]

    if args.drop_duplicates:
        df = df.drop_duplicates(subset=[args.column])

    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Done: {input_path} -> {output_path}")
    print(f"Rows before: {original_rows}")
    print(f"Rows after : {len(df)}")


if __name__ == "__main__":
    main()