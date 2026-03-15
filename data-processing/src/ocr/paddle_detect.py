import os, glob, json, argparse
# Module chuyên trị Detection của bản 3.0
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
    # Khởi tạo mô hình Detection
    # Nếu muốn nhanh hơn nhưng bớt chính xác một chút, đổi thành model_name="PP-OCRv4_mobile_det"
    model = TextDetection(model_name="PP-OCRv4_server_det")
    
    paths = sorted(
        glob.glob(os.path.join(args.keyframes_dir, "*.jpg")) +
        glob.glob(os.path.join(args.keyframes_dir, "*.png"))
    )[:args.max_images]
    
    all_boxes = {}
    
    for i, p in enumerate(paths, 1):
        fname = os.path.basename(p)
        
        # Hàm predict trả về 1 generator, ta lấy các kết quả bên trong
        result_gen = model.predict(p)
        
        boxes = []
        for res in result_gen:
            # Ở bản 3.0, kết quả trích xuất qua thuộc tính .json (trả về kiểu Dictionary)
            res_dict = res.json 
            
            # Tọa độ các khung chữ được lưu trong key 'dt_polys'
            if 'dt_polys' in res_dict:
                for box in res_dict['dt_polys']:
                    # Convert về float cho an toàn khi lưu JSON
                    poly_box = [[float(pt[0]), float(pt[1])] for pt in box]
                    boxes.append(poly_box)
                    
        all_boxes[fname] = boxes
        print(f"[{i}/{len(paths)}] {fname} → {len(boxes)} text region(s) found")
        
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(all_boxes, f, indent=2)
        
    print(f"✅ Xong! Lưu tại {args.output_json}")

if __name__ == "__main__":
    main()
