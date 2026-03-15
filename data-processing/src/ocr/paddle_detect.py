import os, glob, json, argparse
import easyocr

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: EasyOCR Text Detection")
    parser.add_argument('--keyframes-dir', default='./keyframes')
    parser.add_argument('--output-json', default='./ocr_stage1_boxes.json')
    parser.add_argument('--max-images', type=int, default=9999)
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("🔄 Loading EasyOCR Detector...")
    reader = easyocr.Reader(['en'], gpu=True)
    
    paths = sorted(
        glob.glob(os.path.join(args.keyframes_dir, "*.jpg")) +
        glob.glob(os.path.join(args.keyframes_dir, "*.png"))
    )[:args.max_images]
    
    all_boxes = {}
    
    for i, p in enumerate(paths, 1):
        fname = os.path.basename(p)
        
        # Chỉ gọi detect() thay vì readtext()
        result = reader.detect(p)
        
        boxes = []
        # Result của EasyOCR detect trả về (h_boxes, f_boxes)
        # h_boxes format: [x_min, x_max, y_min, y_max]
        if result and len(result[0]) > 0:
            for bbox in result[0][0]:
                xmin, xmax, ymin, ymax = bbox
                # Convert sang format 4 góc đa giác
                poly_box = [
                    [float(xmin), float(ymin)],
                    [float(xmax), float(ymin)],
                    [float(xmax), float(ymax)],
                    [float(xmin), float(ymax)]
                ]
                boxes.append(poly_box)
                
        all_boxes[fname] = boxes
        print(f"[{i}/{len(paths)}] {fname} → {len(boxes)} text region(s) found")
        
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(all_boxes, f, indent=2)
        
    print(f"✅ Xong! Lưu tại {args.output_json}")

if __name__ == "__main__":
    main()
