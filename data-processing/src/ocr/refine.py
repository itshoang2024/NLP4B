import json
import os
import torch
import numpy as np
import argparse
from PIL import Image
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: LayoutLMv3 Text Refinement")
    parser.add_argument('--input-json', default='../stage_2_ViT/ocr_out/ocr_stage2_text.json',
                        help='Path to input JSON from Stage 2')
    parser.add_argument('--output-json', default='./ocr_out/ocr_stage3_refined.json',
                        help='Path to save refined text JSON')
    parser.add_argument('--keyframes-dir', default='../keyframes',
                        help='Directory containing the original keyframes')
    parser.add_argument('--model-name', default='microsoft/layoutlmv3-base',
                        help='HuggingFace model name or local path')
    parser.add_argument('--num-labels', type=int, default=2,
                        help='Number of labels used by the LayoutLMv3 model')
    return parser.parse_args()

def normalize_box(box, width, height):
    """Normalize box coords to 0–1000 scale as required by LayoutLMv3."""
    pts = np.array(box, dtype=np.float32)
    x0 = int(min(pts[:, 0]) / width * 1000)
    y0 = int(min(pts[:, 1]) / height * 1000)
    x1 = int(max(pts[:, 0]) / width * 1000)
    y1 = int(max(pts[:, 1]) / height * 1000)
    return [max(0, min(x0, 1000)), max(0, min(y0, 1000)),
            max(0, min(x1, 1000)), max(0, min(y1, 1000))]

def main():
    args = parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🔄 Loading LayoutLMv3 ({args.model_name}) on {DEVICE.upper()}...")
    processor = AutoProcessor.from_pretrained(args.model_name, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    ).to(DEVICE)
    model.eval()

    if not os.path.exists(args.input_json):
        print(f"❌ Error: Input JSON not found at {args.input_json}")
        return

    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    total_images = len(data)

    print(f"🔍 Refining layout for {total_images} images...\n{'─'*50}")

    for i, (img_name, regions) in enumerate(data.items(), 1):
        img_path = os.path.join(args.keyframes_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"⚠️ [{i}/{total_images}] Skipping {img_name} (Image not found)")
            continue

        W, H = image.size
        words, boxes, meta = [], [], []
        
        for region in regions:
            box = region["box"]
            for rec in region.get("recognized", []):
                text = rec["text"].strip()
                if not text:
                    continue
                words.append(text)
                boxes.append(normalize_box(box, W, H))
                meta.append({"box": box, "text": text, "confidence": rec["confidence"]})

        if not words:
            results[img_name] = []
            print(f"✅ [{i}/{total_images}] {img_name} → 0 tokens (No text found)")
            continue

        # LayoutLMv3 inference
        encoding = processor(
            image, words, boxes=boxes,
            return_tensors="pt", truncation=True,
            padding="max_length", max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**encoding)

        logits = outputs.logits[0] 
        scores = torch.softmax(logits, dim=-1).cpu().numpy()

        # Align tokens back to words
        word_ids = encoding.word_ids(batch_index=0)
        word_scores = {}
        for idx, wid in enumerate(word_ids):
            if wid is None or wid >= len(meta):
                continue
            s = float(scores[idx].max())
            if wid not in word_scores or s > word_scores[wid]:
                word_scores[wid] = s

        # Build refined output
        refined = []
        for j, m in enumerate(meta):
            refined.append({
                "box": m["box"],
                "text": m["text"],
                "ocr_conf": m["confidence"],
                "layout_score": round(word_scores.get(j, 0.0), 4)
            })

        # Sort spatially: top-to-bottom, left-to-right
        refined.sort(key=lambda r: (
            np.array(r["box"])[:, 1].min(), # y (top)
            np.array(r["box"])[:, 0].min()  # x (left)
        ))

        results[img_name] = refined
        print(f"✅ [{i}/{total_images}] {img_name} → {len(refined)} tokens refined")

    # Save output JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'─'*50}")
    print(f"📄 Stage 3 done! Output saved to → {args.output_json}")

if __name__ == "__main__":
    main()


#python refine-2.py 
    #--input-json "./my_stage2_output.json" 
    #--output-json "./final_results.json" 
    #--keyframes-dir "../my_images_folder"
