import argparse
import json
import traceback
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from ultralytics import YOLO

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
    
    # Step A: Add all YOLO objects first
    for y_obj in yolo_objects:
        merged_objects.append(y_obj)

    # Step B: Add Florence objects only if they don't overlap with YOLO
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
# 2. CORE PIPELINE FUNCTION
# ==========================================
def process_image_pipeline(image_path_str, yolo_model, florence_model, florence_processor, device, iou_thresh):
    image_path = Path(image_path_str)
    pil_image = Image.open(image_path).convert("RGB")
    
    # --- YOLO INFERENCE ---
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

    # --- FLORENCE-2 INFERENCE ---
    florence_tasks = ["<OD>", "<DENSE_REGION_CAPTION>", "<MORE_DETAILED_CAPTION>", "<CAPTION>"]
    florence_raw_results = {}
    
    for task in florence_tasks:
        inputs = florence_processor(text=task, images=pil_image, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].to(device).long()
        inputs['pixel_values'] = inputs['pixel_values'].to(device, torch.float32)
        
        with torch.no_grad():
            generated_ids = florence_model.generate(
                **inputs, max_new_tokens=200, num_beams=3,
                pad_token_id=florence_processor.tokenizer.eos_token_id, do_sample=False
            )
        raw_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = florence_processor.post_process_generation(raw_text, task=task, image_size=(pil_image.width, pil_image.height))
        florence_raw_results[task] = parsed.get(task, {})

    florence_objects = []
    if "labels" in florence_raw_results.get("<OD>", {}):
        for label, bbox in zip(florence_raw_results["<OD>"]["labels"], florence_raw_results["<OD>"]["bboxes"]):
            florence_objects.append({"label": label, "bbox": [round(c, 2) for c in bbox], "source": "object_detection"})
            
    if "labels" in florence_raw_results.get("<DENSE_REGION_CAPTION>", {}):
        for label, bbox in zip(florence_raw_results["<DENSE_REGION_CAPTION>"]["labels"], florence_raw_results["<DENSE_REGION_CAPTION>"]["bboxes"]):
            florence_objects.append({"label": label, "bbox": [round(c, 2) for c in bbox], "source": "dense_region"})

    # --- MERGE & BUILD JSON ---
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

# ==========================================
# 3. BATCH RUNNER
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Batch Process Images with YOLO and Florence-2")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to input directory containing images")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to output directory to save JSONs")
    parser.add_argument("--yolo_model", type=str, default="yolov8s-world.pt", help="Path to YOLO model")
    parser.add_argument("--florence_model", type=str, default="microsoft/Florence-2-large-ft", help="Florence model ID")
    parser.add_argument("--iou", type=float, default=0.75, help="IoU threshold for removing duplicates")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process (default: all)")
    args = parser.parse_args()

    # Model Setup
    print("Loading models into memory...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    yolo_model = YOLO(args.yolo_model)
    florence_processor = AutoProcessor.from_pretrained(args.florence_model, trust_remote_code=True)
    florence_model = AutoModelForCausalLM.from_pretrained(
        args.florence_model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)

    # Directory Setup
    input_folder = Path(args.input_dir)
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.jpeg")) + list(input_folder.glob("*.png"))
    if args.limit:
        image_files = image_files[:args.limit]
        
    print(f"Found {len(image_files)} images. Starting batch process...")
    all_results = []
    
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Processing: {img_path.name}")
        try:
            result_json = process_image_pipeline(
                str(img_path), yolo_model, florence_model, florence_processor, device, args.iou
            )
            
            out_file = output_folder / f"{img_path.stem}_merged.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)
                
            all_results.append(result_json)
        except Exception as e:
            print(f"❌ Error on {img_path.name}: {e}")
            traceback.print_exc()
            
    if all_results:
        combined_file = output_folder / "all_images_merged.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Batch complete! Saved {len(all_results)} results to {combined_file}")

if __name__ == "__main__":
    main()
