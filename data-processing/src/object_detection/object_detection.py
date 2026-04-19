
import argparse
import gc
import json
import os
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from ultralytics import YOLO

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


# ==========================================
# 0. LOGGING / SYSTEM HELPERS
# ==========================================
def log(message: str) -> None:
    print(f"[object_detection] {message}", flush=True)


def log_system_state(prefix: str = "") -> None:
    parts = []
    if prefix:
        parts.append(prefix)

    parts.append(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current)
        parts.append(f"gpu_count={device_count}")
        parts.append(f"current_gpu={current}:{gpu_name}")
        try:
            allocated = torch.cuda.memory_allocated(current) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(current) / (1024 ** 3)
            parts.append(f"cuda_allocated_gb={allocated:.2f}")
            parts.append(f"cuda_reserved_gb={reserved:.2f}")
        except Exception:
            pass

    if psutil is not None:
        vm = psutil.virtual_memory()
        parts.append(f"ram_used_pct={vm.percent}")
        parts.append(f"ram_available_gb={vm.available / (1024 ** 3):.2f}")

    log(" | ".join(parts))


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def calculate_iou(box1, box2):
    """Calculates Intersection over Union for two bounding boxes [x1, y1, x2, y2]."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if box1_area + box2_area - intersection_area == 0:
        return 0.0

    return intersection_area / (box1_area + box2_area - intersection_area)


def merge_yolo_and_florence(yolo_objects, florence_objects, iou_threshold=0.75):
    """Merges YOLO and Florence detections, prioritizing YOLO for tight bounding boxes."""
    merged_objects = []

    for y_obj in yolo_objects:
        merged_objects.append(y_obj)

    for f_obj in florence_objects:
        is_duplicate = False
        for m_obj in merged_objects:
            iou = calculate_iou(f_obj["bbox"], m_obj["bbox"])

            if iou > iou_threshold:
                if f_obj["label"].lower() in m_obj["label"].lower() or m_obj["label"].lower() in f_obj["label"].lower():
                    is_duplicate = True
                    break
                elif m_obj["source"] == "yolo" and f_obj["source"] == "dense_region":
                    m_obj["rich_label"] = f_obj["label"]
                    is_duplicate = True
                    break

        if not is_duplicate:
            merged_objects.append(f_obj)

    return merged_objects


# ==========================================
# 2. MODEL PREP / LOAD
# ==========================================
def prepare_hf_cache(
    florence_model_id: str,
    cache_dir: str | Path,
    revision: str = "main",
) -> str:
    """Pre-download Florence model/code to a local cache directory."""
    from huggingface_hub import snapshot_download

    cache_dir = str(cache_dir)
    log(f"Preparing Florence cache | model={florence_model_id} | revision={revision} | cache_dir={cache_dir}")
    local_model_dir = snapshot_download(
        repo_id=florence_model_id,
        revision=revision,
        cache_dir=cache_dir,
    )
    log(f"Florence cache ready at: {local_model_dir}")
    return local_model_dir


def resolve_model_source(
    florence_model: str,
    hf_cache_dir: str | Path | None = None,
    florence_revision: str = "main",
    prepare_only: bool = False,
) -> str:
    """Return local model path when cache dir is provided, otherwise return original model id/path."""
    is_local_path = Path(florence_model).exists()
    if is_local_path:
        log(f"Using local Florence path: {florence_model}")
        return str(Path(florence_model))

    if hf_cache_dir:
        local_model_dir = prepare_hf_cache(
            florence_model_id=florence_model,
            cache_dir=hf_cache_dir,
            revision=florence_revision,
        )
        return local_model_dir

    if prepare_only:
        raise ValueError("--prepare_only requires either a local --florence_model path or --hf_cache_dir.")

    return florence_model


def get_default_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


def load_models(
    yolo_model_path: str = "yolov8m-worldv2.pt",
    florence_model: str = "microsoft/Florence-2-large-ft",
    hf_cache_dir: str | Path | None = None,
    florence_revision: str = "main",
    device: str | None = None,
    torch_dtype: str = "auto",
) -> dict[str, Any]:
    """Load YOLO + Florence exactly once and return reusable runtime bundle."""
    runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_default_dtype(runtime_device) if torch_dtype == "auto" else getattr(torch, torch_dtype)

    log_system_state("Before model load")
    log(f"Loading YOLO model from: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    log("YOLO model loaded")

    florence_source = resolve_model_source(
        florence_model=florence_model,
        hf_cache_dir=hf_cache_dir,
        florence_revision=florence_revision,
    )

    local_files_only = Path(florence_source).exists()

    log(f"Loading Florence processor from: {florence_source} | local_files_only={local_files_only}")
    florence_processor = AutoProcessor.from_pretrained(
        florence_source,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    log("Florence processor loaded")

    log(f"Loading Florence model from: {florence_source} | dtype={dtype} | device={runtime_device}")
    florence_model_obj = AutoModelForCausalLM.from_pretrained(
        florence_source,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=dtype,
    ).to(runtime_device)
    florence_model_obj.eval()
    log("Florence model loaded")
    log_system_state("After model load")

    return {
        "yolo_model": yolo_model,
        "florence_model": florence_model_obj,
        "florence_processor": florence_processor,
        "device": runtime_device,
        "dtype": dtype,
        "florence_source": florence_source,
    }


# ==========================================
# 3. CORE PIPELINE FUNCTION
# ==========================================
def process_image_pipeline(image_path_str, yolo_model, florence_model, florence_processor, device, iou_thresh):
    image_path = Path(image_path_str)
    pil_image = Image.open(image_path).convert("RGB")

    yolo_results = yolo_model(image_path_str, verbose=False)
    yolo_objects = []
    yolo_tags = set()

    for result in yolo_results:
        for i in range(len(result.boxes)):
            class_id = int(result.boxes.cls[i])
            label = result.names[class_id]
            confidence = float(result.boxes.conf[i])
            bbox_xyxy = result.boxes.xyxy[i].tolist()

            yolo_tags.add(label)
            yolo_objects.append({
                "label": label,
                "confidence": round(confidence, 2),
                "bbox": [round(c, 2) for c in bbox_xyxy],
                "source": "yolo"
            })

    florence_tasks = ["<OD>", "<DENSE_REGION_CAPTION>", "<MORE_DETAILED_CAPTION>", "<CAPTION>"]
    florence_raw_results = {}

    for task in florence_tasks:
        inputs = florence_processor(text=task, images=pil_image, return_tensors="pt")
        inputs["input_ids"] = inputs["input_ids"].to(device).long()
        inputs["pixel_values"] = inputs["pixel_values"].to(device, florence_model.dtype)

        with torch.no_grad():
            generated_ids = florence_model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=3,
                pad_token_id=florence_processor.tokenizer.eos_token_id,
                do_sample=False,
            )
        raw_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = florence_processor.post_process_generation(
            raw_text,
            task=task,
            image_size=(pil_image.width, pil_image.height),
        )
        florence_raw_results[task] = parsed.get(task, {})

    florence_objects = []
    if "labels" in florence_raw_results.get("<OD>", {}):
        for label, bbox in zip(florence_raw_results["<OD>"]["labels"], florence_raw_results["<OD>"]["bboxes"]):
            florence_objects.append({"label": label, "bbox": [round(c, 2) for c in bbox], "source": "object_detection"})

    if "labels" in florence_raw_results.get("<DENSE_REGION_CAPTION>", {}):
        for label, bbox in zip(
            florence_raw_results["<DENSE_REGION_CAPTION>"]["labels"],
            florence_raw_results["<DENSE_REGION_CAPTION>"]["bboxes"],
        ):
            florence_objects.append({"label": label, "bbox": [round(c, 2) for c in bbox], "source": "dense_region"})

    final_merged_objects = merge_yolo_and_florence(yolo_objects, florence_objects, iou_threshold=iou_thresh)
    all_tags = list(yolo_tags.union(set(florence_raw_results.get("<OD>", {}).get("labels", []))))

    final_json = {
        "image_id": image_path.stem,
        "metadata": {
            "file_path": str(image_path),
            "width": pil_image.width,
            "height": pil_image.height,
        },
        "global_descriptions": {
            "tags": sorted(all_tags),
            "caption": florence_raw_results.get("<CAPTION>", ""),
            "detailed_caption": florence_raw_results.get("<MORE_DETAILED_CAPTION>", "")
        },
        "objects": final_merged_objects
    }
    return final_json


def save_combined_payload(combined_file, video_id, input_folder, all_results):
    """Writes combined results atomically to avoid partially written JSON files."""
    combined_payload = {
        "video_id": video_id,
        "input_dir": str(input_folder),
        "total_images": len(all_results),
        "results": all_results,
    }
    tmp_file = combined_file.with_suffix(combined_file.suffix + ".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(combined_payload, f, indent=2, ensure_ascii=False)
    tmp_file.replace(combined_file)


def load_existing_results(combined_file):
    """Loads existing output JSON and returns a deduplicated list of previous results."""
    if not combined_file.exists():
        return []

    try:
        with open(combined_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        log(f"Could not load existing output file {combined_file}: {e}")
        return []

    if not isinstance(payload, dict):
        log(f"Existing output file {combined_file} is not a valid JSON object. Starting fresh.")
        return []

    previous_results = payload.get("results", [])
    if not isinstance(previous_results, list):
        log(f"Existing output file {combined_file} has invalid 'results'. Starting fresh.")
        return []

    deduped_results = []
    seen_image_ids = set()
    for item in previous_results:
        if not isinstance(item, dict):
            continue
        image_id = item.get("image_id")
        if not image_id or image_id in seen_image_ids:
            continue
        seen_image_ids.add(image_id)
        deduped_results.append(item)

    if len(deduped_results) != len(previous_results):
        log(
            f"Existing output contained {len(previous_results) - len(deduped_results)} duplicated or invalid entries; they were ignored."
        )

    return deduped_results


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    runtime: dict[str, Any],
    iou: float = 0.75,
    limit: int | None = None,
    save_every: int = 50,
) -> Path:
    """Process one video directory using preloaded runtime bundle."""
    input_folder = Path(input_dir)
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_files = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.jpeg")) + list(input_folder.glob("*.png"))
    image_files = sorted(image_files)
    if limit:
        image_files = image_files[:limit]

    video_id = input_folder.name or input_folder.resolve().name or "unknown_video"
    combined_file = output_folder / f"{video_id}_object_detection.json"

    all_results = load_existing_results(combined_file)
    processed_image_ids = {
        item.get("image_id")
        for item in all_results
        if isinstance(item, dict) and item.get("image_id")
    }
    remaining_images = [img_path for img_path in image_files if img_path.stem not in processed_image_ids]

    log(f"Video {video_id} | total_images={len(image_files)} | already_processed={len(processed_image_ids)} | remaining={len(remaining_images)}")

    if not remaining_images:
        save_combined_payload(combined_file, video_id, input_folder, all_results)
        log(f"All images already processed for {video_id}. Output is up to date at {combined_file}")
        return combined_file

    processed_since_last_save = 0

    for i, img_path in enumerate(remaining_images, start=1):
        log(f"Video {video_id} | [{i}/{len(remaining_images)}] Processing {img_path.name}")
        try:
            result_json = process_image_pipeline(
                str(img_path),
                runtime["yolo_model"],
                runtime["florence_model"],
                runtime["florence_processor"],
                runtime["device"],
                iou,
            )

            all_results.append(result_json)
            processed_since_last_save += 1

            if save_every > 0 and processed_since_last_save >= save_every:
                save_combined_payload(combined_file, video_id, input_folder, all_results)
                log(f"Checkpoint saved for {video_id} after {processed_since_last_save} new images: {combined_file}")
                processed_since_last_save = 0
        except Exception as e:
            log(f"Error on {img_path.name}: {e}")
            traceback.print_exc()
        finally:
            cleanup_memory()

    save_combined_payload(combined_file, video_id, input_folder, all_results)

    if all_results:
        log(f"Batch complete for {video_id}! Saved {len(all_results)} results to {combined_file}")
    else:
        log(f"Batch complete for {video_id} with no successful results. Created empty output: {combined_file}")

    return combined_file


# ==========================================
# 4. CLI
# ==========================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch Process Images with YOLO and Florence-2")
    parser.add_argument("-i", "--input_dir", type=str, help="Path to input directory containing images")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Path to output directory to save JSONs")
    parser.add_argument("--yolo_model", type=str, default="yolov8m-worldv2.pt", help="Path to YOLO model")
    parser.add_argument("--florence_model", type=str, default="microsoft/Florence-2-large-ft", help="Florence model ID or local path")
    parser.add_argument("--florence_revision", type=str, default="main", help="Pinned Florence revision / commit hash")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="HF cache dir for pre-downloading Florence")
    parser.add_argument("--prepare_only", action="store_true", help="Only pre-download Florence model/code into cache, then exit")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", None], help="Execution device")
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "float32", "bfloat16"], help="Florence model dtype")
    parser.add_argument("--iou", type=float, default=0.75, help="IoU threshold for removing duplicates")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process (default: all)")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint JSON every N successful images (0 disables periodic save)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    florence_source = resolve_model_source(
        florence_model=args.florence_model,
        hf_cache_dir=args.hf_cache_dir,
        florence_revision=args.florence_revision,
        prepare_only=args.prepare_only,
    )

    if args.prepare_only:
        log(f"Prepare-only completed. Florence source is ready at: {florence_source}")
        return

    if not args.input_dir:
        parser.error("--input_dir is required unless --prepare_only is used.")

    runtime = load_models(
        yolo_model_path=args.yolo_model,
        florence_model=florence_source,
        hf_cache_dir=None,
        florence_revision=args.florence_revision,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        runtime=runtime,
        iou=args.iou,
        limit=args.limit,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
