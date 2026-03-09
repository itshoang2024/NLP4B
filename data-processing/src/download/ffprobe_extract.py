from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm.auto import tqdm

PathLike = Union[str, Path]


def run_ffprobe(video_path: PathLike) -> dict:
    video_path = Path(video_path)
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _safe_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def parse_fraction(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        if "/" in value:
            num, den = value.split("/")
            num = float(num)
            den = float(den)
            if den == 0:
                return None
            return num / den
        return float(value)
    except Exception:
        return None


def extract_video_stream(ffprobe_data: dict) -> Optional[dict]:
    for stream in ffprobe_data.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    return None


def has_audio_stream(ffprobe_data: dict) -> bool:
    return any(s.get("codec_type") == "audio" for s in ffprobe_data.get("streams", []))


def extract_basic_metadata(video_path: PathLike, info_json_path: Optional[PathLike] = None) -> dict:
    video_path = Path(video_path)
    ffprobe_data = run_ffprobe(video_path)

    format_info = ffprobe_data.get("format", {})
    video_stream = extract_video_stream(ffprobe_data)
    if video_stream is None:
        raise ValueError(f"No video stream found in: {video_path}")

    duration_sec = _safe_float(format_info.get("duration")) or _safe_float(video_stream.get("duration"))
    fps = parse_fraction(video_stream.get("avg_frame_rate")) or parse_fraction(video_stream.get("r_frame_rate"))
    width = _safe_int(video_stream.get("width"))
    height = _safe_int(video_stream.get("height"))
    codec_name = video_stream.get("codec_name")
    file_size_bytes = _safe_int(format_info.get("size"))
    audio_flag = has_audio_stream(ffprobe_data)

    info_title = None
    source_url = None

    if info_json_path is not None:
        info_json_path = Path(info_json_path)
        if info_json_path.exists():
            info_data = json.loads(info_json_path.read_text(encoding="utf-8"))
            info_title = info_data.get("title")
            source_url = info_data.get("webpage_url") or info_data.get("original_url")

    return {
        "video_path": str(video_path),
        "info_json_path": str(info_json_path) if info_json_path else None,
        "duration_sec": duration_sec,
        "fps": fps,
        "width": width,
        "height": height,
        "codec_name": codec_name,
        "file_size_bytes": file_size_bytes,
        "has_audio": audio_flag,
        "title_from_info_json": info_title,
        "source_url": source_url,
        "status": "ok",
    }


def resolve_video_path(base_dir: PathLike, relative_path: str) -> Path:
    return Path(base_dir) / relative_path


def build_video_metadata_from_manifest(
    manifest_df: pd.DataFrame,
    base_dir: PathLike,
    video_metadata_output_path: Optional[PathLike] = None,
    metadata_log_output_path: Optional[PathLike] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_dir = Path(base_dir)

    records = []
    logs = []

    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extracting metadata"):
        video_id = row["video_id"]
        rel_video_path = row["local_video_path"]
        rel_info_json_path = row.get("info_json_path")

        video_path = resolve_video_path(base_dir, rel_video_path)
        info_json_path = resolve_video_path(base_dir, rel_info_json_path) if pd.notna(rel_info_json_path) else None

        log_item = {
            "video_id": video_id,
            "video_path": str(video_path),
            "status": "unknown",
            "error": None,
        }

        try:
            meta = extract_basic_metadata(video_path=video_path, info_json_path=info_json_path)
            meta["video_id"] = video_id
            meta["title"] = row.get("title") if pd.notna(row.get("title")) else meta.get("title_from_info_json")
            records.append(meta)
            log_item["status"] = "ok"
        except Exception as exc:
            log_item["status"] = "error"
            log_item["error"] = str(exc)

        logs.append(log_item)

    video_metadata_df = pd.DataFrame(records)
    metadata_log_df = pd.DataFrame(logs)

    if video_metadata_output_path is not None:
        video_metadata_output_path = Path(video_metadata_output_path)
        video_metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
        video_metadata_df.to_csv(video_metadata_output_path, index=False)

    if metadata_log_output_path is not None:
        metadata_log_output_path = Path(metadata_log_output_path)
        metadata_log_output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_log_df.to_csv(metadata_log_output_path, index=False)

    return video_metadata_df, metadata_log_df
