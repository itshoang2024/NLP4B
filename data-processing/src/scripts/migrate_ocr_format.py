"""
migrate_ocr_format.py — Migrate old OCR JSON format to new canonical format
============================================================================
Converts: {"file_name": "...", "ocr_result": "..."} → {"image": "...", "ocr_text": "..."}

Usage:
  # Single file
  python migrate_ocr_format.py -i path/to/video_id_ocr.json

  # Entire directory of OCR JSONs
  python migrate_ocr_format.py -i path/to/ocr_folder/

  # Dry-run (preview without writing)
  python migrate_ocr_format.py -i path/to/ocr_folder/ --dry-run
"""

import argparse
import json
from pathlib import Path


def migrate_file(filepath: Path, dry_run: bool = False) -> dict:
    """Migrate a single OCR JSON file. Returns stats."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return {"file": str(filepath), "status": "skipped", "reason": "not a JSON array"}

    migrated = 0
    already_new = 0

    for item in data:
        if not isinstance(item, dict):
            continue

        # Migrate "file_name" → "image"
        if "file_name" in item and "image" not in item:
            item["image"] = item.pop("file_name")
            migrated += 1
        elif "image" in item:
            if "file_name" in item:
                item.pop("file_name")  # remove duplicate
            already_new += 1

        # Migrate "ocr_result" → "ocr_text"
        if "ocr_result" in item and "ocr_text" not in item:
            item["ocr_text"] = item.pop("ocr_result")
        elif "ocr_text" in item:
            if "ocr_result" in item:
                item.pop("ocr_result")  # remove duplicate

    stats = {
        "file": filepath.name,
        "total": len(data),
        "migrated": migrated,
        "already_new": already_new,
    }

    if migrated == 0:
        stats["status"] = "no_change"
    elif dry_run:
        stats["status"] = "dry_run"
    else:
        tmp = filepath.with_suffix(filepath.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(filepath)
        stats["status"] = "migrated"

    return stats


def main():
    parser = argparse.ArgumentParser(description="Migrate OCR JSON: file_name/ocr_result → image/ocr_text")
    parser.add_argument("-i", "--input", required=True, help="Path to a single OCR JSON file or a directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")
    args = parser.parse_args()

    target = Path(args.input)

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob("*_ocr.json"))
    else:
        print(f"Error: {target} not found")
        return

    if not files:
        print(f"No *_ocr.json files found in {target}")
        return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(files)} file(s)...\n")

    total_migrated = 0
    for f in files:
        stats = migrate_file(f, dry_run=args.dry_run)
        status_icon = {"migrated": "✅", "dry_run": "👁️", "no_change": "⏭️", "skipped": "⚠️"}.get(stats["status"], "❓")
        print(f"  {status_icon} {stats['file']} — {stats.get('migrated', 0)}/{stats.get('total', '?')} entries migrated [{stats['status']}]")
        total_migrated += stats.get("migrated", 0)

    print(f"\nDone. {total_migrated} entries migrated across {len(files)} file(s).")


if __name__ == "__main__":
    main()
