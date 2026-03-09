from __future__ import annotations

import time
import shutil
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm.auto import tqdm
from yt_dlp import YoutubeDL


PathLike = Union[str, Path]


def build_format_selector(download_mode: str = "720p_mp4") -> str:
    """
    Return a yt-dlp format selector string.
    """
    if download_mode == "best":
        return "bv*+ba/b"
    if download_mode == "720p_mp4":
        return "bv*[ext=mp4][height<=720]+ba[ext=m4a]/b[ext=mp4][height<=720]/b"
    if download_mode == "480p_mp4":
        return "bv*[ext=mp4][height<=480]+ba[ext=m4a]/b[ext=mp4][height<=480]/b"
    raise ValueError(f"Unsupported download_mode: {download_mode}")


def move_info_json_to_separate_folder(video_id: str, videos_dir: Path, info_json_dir: Path) -> Optional[str]:
    """
    Move {video_id}.info.json from the videos folder into the dedicated info_json folder.
    """
    src = videos_dir / f"{video_id}.info.json"
    if not src.exists():
        return None

    info_json_dir.mkdir(parents=True, exist_ok=True)
    dst = info_json_dir / f"{video_id}.info.json"
    shutil.move(str(src), str(dst))
    return str(dst)


def ensure_download_dirs(output_root: PathLike) -> dict[str, Path]:
    """
    Create and return the standard download directory structure.
    """
    output_root = Path(output_root)
    videos_dir = output_root / "videos"
    info_json_dir = output_root / "info_json"

    output_root.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    info_json_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_root": output_root,
        "videos_dir": videos_dir,
        "info_json_dir": info_json_dir,
        "download_log_path": output_root / "download_log.csv",
        "download_archive_path": output_root / "download_archive.txt",
    }


def download_one_video(
    row: pd.Series,
    videos_dir: PathLike,
    info_json_dir: PathLike,
    download_archive_path: PathLike,
    write_info_json: bool = True,
    overwrite_existing: bool = False,
    download_mode: str = "720p_mp4",
) -> dict:
    """
    Download a single video described by one row of the normalized manifest.
    """
    videos_dir = Path(videos_dir)
    info_json_dir = Path(info_json_dir)
    download_archive_path = Path(download_archive_path)

    video_id = str(row["video_id"])
    outtmpl = str(videos_dir / f"{video_id}.%(ext)s")

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": build_format_selector(download_mode),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 10,
        "fragment_retries": 10,
        "continuedl": True,
        "overwrites": overwrite_existing,
        "download_archive": str(download_archive_path),
        "writeinfojson": write_info_json,
    }

    start_time = time.time()
    result = {
        "row_number": row.get("row_number"),
        "video_id": video_id,
        "title": row.get("title"),
        "url": row.get("url"),
        "status": "unknown",
        "downloaded_path": None,
        "info_json_path": None,
        "error": None,
        "elapsed_sec": None,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(str(row["url"]), download=True)
            ext = info.get("ext", "mp4")

        result["status"] = "ok"
        result["downloaded_path"] = str(videos_dir / f"{video_id}.{ext}")

        if write_info_json:
            result["info_json_path"] = move_info_json_to_separate_folder(
                video_id=video_id,
                videos_dir=videos_dir,
                info_json_dir=info_json_dir,
            )

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    result["elapsed_sec"] = round(time.time() - start_time, 2)
    return result


def download_from_manifest(
    manifest_df: pd.DataFrame,
    output_root: PathLike,
    max_downloads: Optional[int] = None,
    write_info_json: bool = True,
    overwrite_existing: bool = False,
    sleep_between_downloads: float = 1.0,
    download_mode: str = "720p_mp4",
    download_log_path: Optional[PathLike] = None,
    download_archive_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """
    Download all videos listed in a normalized manifest DataFrame.

    Returns the download log as a DataFrame.
    """
    paths = ensure_download_dirs(output_root)
    videos_dir = paths["videos_dir"]
    info_json_dir = paths["info_json_dir"]

    if download_log_path is None:
        download_log_path = paths["download_log_path"]
    if download_archive_path is None:
        download_archive_path = paths["download_archive_path"]

    download_log_path = Path(download_log_path)
    download_archive_path = Path(download_archive_path)

    df = manifest_df.copy()
    if max_downloads is not None:
        df = df.head(max_downloads).copy()

    results: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        result = download_one_video(
            row=row,
            videos_dir=videos_dir,
            info_json_dir=info_json_dir,
            download_archive_path=download_archive_path,
            write_info_json=write_info_json,
            overwrite_existing=overwrite_existing,
            download_mode=download_mode,
        )
        results.append(result)

        pd.DataFrame(results).to_csv(download_log_path, index=False)

        if sleep_between_downloads:
            time.sleep(sleep_between_downloads)

    return pd.DataFrame(results)
