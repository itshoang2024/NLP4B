import json
import cv2
import os
import numpy as np
import easyocr
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: EasyOCR Text Recognition")
    parser.add_argument('--input-json', default='../stage_1_craft/ocr_out/ocr_stage1_boxes.json',
                        help='Path to input JSON from Stage 1')
    parser.add_argument('--output-json', default='./ocr_out/ocr_stage2_text.json',
                        help='Path to save recognized text JSON')
    parser.add_argument('--keyframes-dir', default='../keyframes',
                        help='Directory containing the original keyframes')
    parser.add_argument('--langs', nargs='+', default=['en', 'vi'],
                        help='Languages for EasyOCR (e.g. --langs en vi)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage instead of GPU')
    return parser.parse_args()

def main():
    args = parse_args()
    
    use_gpu = not args.cpu
    print(f"🔄 Loading EasyOCR... (Languages: {args.langs}, GPU={use_gpu})")
    reader = easyocr.Reader(args.langs, gpu=use_gpu)

    if not os.path.exists(args.input_json):
        print(f"❌ Error: Input JSON not found at {args.input_json}")
        return

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {}
    total_images = len(data)

    print(f"🔍 Processing {total_images} images from Stage 1...\n{'─'*50}")

    for i, (img_name, boxes) in enumerate(data.items(), 1):
        img_path = os.path.join(args.keyframes_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ [{i}/{total_images}] Skipping {img_name} (Image not found at {img_path})")
            continue

        frame_results = []
        for box in boxes:
            pts = np.array(box, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            x, y = max(0, x), max(0, y)
            crop = img[y:y+h, x:x+w]

            if crop.size == 0:
                continue

            ocr_result = reader.readtext(crop)

            texts = [
                {"text": text, "confidence": round(conf, 4)}
                for (_, text, conf) in ocr_result if text.strip()
            ]

            frame_results.append({"box": box, "recognized": texts})

        results[img_name] = frame_results
        total_texts = sum(len(r["recognized"]) for r in frame_results)
        print(f"✅ [{i}/{total_images}] {img_name} → {len(boxes)} boxes → {total_texts} text(s) recognized")

    # Save output JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'─'*50}")
    print(f"📄 Stage 2 done! Output saved to → {args.output_json}")

if __name__ == "__main__":
    main()

#python recognize.py 
    #--input-json '../stage_1_craft/ocr_out/ocr_stage1_boxes.json' 
    #--output-json './ocr_out/ocr_stage2_text.json' 
    #--keyframes-dir '../keyframes'
    #--langs en vi
    #--cpu --> to use cpu instead of gpu
