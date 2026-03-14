import json, cv2, os, numpy as np
import easyocr

JSON_IN    = "../stage_1_craft/ocr_out/ocr_stage1_boxes.json"
JSON_OUT   = "ocr_out/ocr_stage2_text.json"
FRAMES_DIR = "../keyframes"

# Add more languages if needed e.g. ['en', 'vi'] for Vietnamese
reader = easyocr.Reader(['en', 'vi'], gpu=True)

with open(JSON_IN) as f:
    data = json.load(f)

results = {}

for img_name, boxes in data.items():
    img_path = os.path.join(FRAMES_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  Skipping {img_name}")
        continue

    frame_results = []
    for box in boxes:
        pts        = np.array(box, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        x, y       = max(0, x), max(0, y)
        crop       = img[y:y+h, x:x+w]

        if crop.size == 0:
            continue

        ocr_result = reader.readtext(crop)

        texts = [
            {"text": text, "confidence": round(conf, 4)}
            for (_, text, conf) in ocr_result
            if text.strip()
        ]

        frame_results.append({"box": box, "recognized": texts})

    results[img_name] = frame_results
    total = sum(len(r["recognized"]) for r in frame_results)
    print(f"✅ {img_name} → {len(boxes)} boxes → {total} text(s) recognized")

os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n📄 Stage 2 done! → {JSON_OUT}  ← feed this to Stage 3")
