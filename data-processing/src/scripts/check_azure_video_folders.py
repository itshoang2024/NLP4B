'''
Cách dùng: python check_azure_video_folders.py --csv hoang.csv --container_name keyframes --output_dir reports
'''

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


def normalize_video_id(raw: object) -> str:
    if pd.isna(raw):
        return ""
    value = str(raw).strip()
    if value.lower().endswith(".mp4"):
        value = value[:-4]
    return value.strip()


def folder_exists(container_client, video_id: str) -> bool:
    prefix = f"{video_id}/"
    blobs = container_client.list_blobs(name_starts_with=prefix)
    return next(blobs, None) is not None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether Azure Blob container has a folder/prefix for each video_id in a CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file containing video_id column")
    parser.add_argument("--container_name", required=True, help="Azure Blob container name")
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to save result CSV files",
    )
    args = parser.parse_args()

    load_dotenv()
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("Missing AZURE_STORAGE_CONNECTION_STRING in environment.")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "video_id" not in df.columns:
        raise ValueError(f"CSV must contain a 'video_id' column. Found: {list(df.columns)}")

    df["video_id"] = df["video_id"].map(normalize_video_id)
    video_ids = (
        df[df["video_id"] != ""]["video_id"]
        .drop_duplicates()
        .tolist()
    )

    if not video_ids:
        raise ValueError("No valid video_id found in CSV after cleaning.")

    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service_client.get_container_client(args.container_name)

    exists_ids: list[str] = []
    missing_ids: list[str] = []

    total = len(video_ids)
    for idx, video_id in enumerate(video_ids, start=1):
        ok = folder_exists(container_client, video_id)
        if ok:
            exists_ids.append(video_id)
        else:
            missing_ids.append(video_id)

        if idx % 20 == 0 or idx == total:
            print(f"Checked {idx}/{total}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = csv_path.stem
    exists_path = output_dir / f"{stem}_{args.container_name}_success.csv"
    missing_path = output_dir / f"{stem}_{args.container_name}_failed.csv"

    pd.DataFrame({"video_id": exists_ids}).to_csv(exists_path, index=False, encoding="utf-8")
    pd.DataFrame({"video_id": missing_ids}).to_csv(missing_path, index=False, encoding="utf-8")

    print("-" * 60)
    print(f"CSV file       : {csv_path}")
    print(f"Container      : {args.container_name}")
    print(f"Total video_id : {total}")
    print(f"Exists         : {len(exists_ids)}")
    print(f"Missing        : {len(missing_ids)}")
    print(f"Exists file    : {exists_path}")
    print(f"Missing file   : {missing_path}")


if __name__ == "__main__":
    main()