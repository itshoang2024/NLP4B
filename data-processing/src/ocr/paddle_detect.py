import os, glob, json, argparse
from paddleocr import PaddleOCR

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: PaddleOCR Text Detection")
    parser.add_argument('--keyframes-dir', default='./keyframes')
    parser.add_argument('--output-json', default='./ocr_stage1_boxes.json')
    parser.add_argument('--max-images', type=int, default=9999)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Khởi tạo model (Bỏ tham số rec=False đi để không bị lỗi)
    print("🔄 Loading PaddleOCR Detector...")
    ocr = PaddleOCR(use_textline_orientation=False, lang='en')
    
    paths = sorted(
        glob.glob(os.path.join(args.keyframes_dir, "*.jpg")) +
        glob.glob(os.path.join(args.keyframes_dir, "*.png"))
    )[:args.max_images]
    
    all_boxes = {}
    
    for i, p in enumerate(paths, 1):
        fname = os.path.basename(p)
        
        # 2. Chạy dự đoán: Chỉ truyền rec=False vào lúc chạy hàm ocr()
        # Điều này sẽ khiến mô hình chỉ trả về Bounding Box (Tọa độ chữ)
        result = ocr.ocr(p, cls=False, det=True, rec=False)
        
        boxes = []
        if result and result[0]:
            for line in result[0]:
                # Trong chế độ rec=False, result trả về trực tiếp tọa độ
                # Format mặc định là: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                box = line 
                
                # Convert thành kiểu float để đưa vào file JSON cho đồng nhất
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
