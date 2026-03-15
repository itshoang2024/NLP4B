import os, glob, json, argparse
from paddleocr import TextDetection 

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: PaddleOCR Text Detection 3.0")
    parser.add_argument('--keyframes-dir', default='./keyframes')
    parser.add_argument('--output-json', default='./ocr_stage1_boxes.json')
    parser.add_argument('--max-images', type=int, default=9999)
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("🔄 Loading PaddleOCR 3.0 Detector...")
    model = TextDetection(model_name="PP-OCRv4_server_det")
    
    paths = sorted(
        glob.glob(os.path.join(args.keyframes_dir, "*.jpg")) +
        glob.glob(os.path.join(args.keyframes_dir, "*.png"))
    )[:args.max_images]
    
    all_boxes = {}
    
    for i, p in enumerate(paths, 1):
        fname = os.path.basename(p)
        result_gen = model.predict(p)
        
        boxes = []
        for r in result_gen:
            # 1. Chuyển kết quả sang dict
            res_dict = r if isinstance(r, dict) else getattr(r, 'json', {})
            
            # 2. Xử lý bóc tách chính xác dựa trên bài test
            # Lấy data thực sự nằm trong key 'res'
            core_data = res_dict.get('res', res_dict) 
            
            # Lấy tọa độ từ 'dt_polys'
            polys = core_data.get('dt_polys', [])
            
            # 3. Ép kiểu và đưa vào mảng
            for box in polys:
                try:
                    # Convert từng điểm sang float
                    poly_box = [[float(pt[0]), float(pt[1])] for pt in box]
                    boxes.append(poly_box)
                except Exception as e:
                    pass
                    
        all_boxes[fname] = boxes
        print(f"[{i}/{len(paths)}] {fname} → {len(boxes)} text region(s) found")
        
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    
    out_file = args.output_json if args.output_json.endswith('.json') else os.path.join(args.output_json, "boxes.json")
    with open(out_file, 'w') as f:
        json.dump(all_boxes, f, indent=2)
        
    print(f"✅ Đã lưu JSON tại: {out_file}")

if __name__ == "__main__":
    main()
