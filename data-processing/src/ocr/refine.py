import json, os, torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

JSON_IN    = "../stage_2_ViT/ocr_out/ocr_stage2_text.json"
JSON_OUT   = "ocr_out/ocr_stage3_refined.json"
FRAMES_DIR = "../keyframes"
MODEL_NAME = "microsoft/layoutlmv3-base"   # swap with fine-tuned path if available
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False)
model     = LayoutLMv3ForTokenClassification.from_pretrained(
                MODEL_NAME, num_labels=2   # adjust if fine-tuned
            ).to(DEVICE)
model.eval()

with open(JSON_IN, encoding="utf-8") as f:
    data = json.load(f)

def normalize_box(box, width, height):
    """Normalize box coords to 0–1000 scale as required by LayoutLMv3."""
    pts  = np.array(box, dtype=np.float32)
    x0   = int(min(pts[:, 0]) / width  * 1000)
    y0   = int(min(pts[:, 1]) / height * 1000)
    x1   = int(max(pts[:, 0]) / width  * 1000)
    y1   = int(max(pts[:, 1]) / height * 1000)
    # clamp to [0, 1000]
    return [max(0, min(x0, 1000)), max(0, min(y0, 1000)),
            max(0, min(x1, 1000)), max(0, min(y1, 1000))]

results = {}

for img_name, regions in data.items():
    img_path = os.path.join(FRAMES_DIR, img_name)
    image    = Image.open(img_path).convert("RGB")
    W, H     = image.size

    # Flatten all recognized texts + boxes for this image
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
        continue

    # LayoutLMv3 inference
    encoding = processor(
        image, words, boxes=boxes,
        return_tensors="pt", truncation=True,
        padding="max_length", max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**encoding)

    logits    = outputs.logits[0]                      # (seq_len, num_labels)
    token_ids = encoding["input_ids"][0].tolist()
    scores    = torch.softmax(logits, dim=-1).cpu().numpy()

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
    for i, m in enumerate(meta):
        refined.append({
            "box":        m["box"],
            "text":       m["text"],
            "ocr_conf":   m["confidence"],
            "layout_score": round(word_scores.get(i, 0.0), 4)
        })

    # Sort spatially: top-to-bottom, left-to-right
    refined.sort(key=lambda r: (
        np.array(r["box"])[:, 1].min(),   # y (top)
        np.array(r["box"])[:, 0].min()    # x (left)
    ))

    results[img_name] = refined
    print(f"✅ {img_name} → {len(refined)} tokens refined")

os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n📄 Stage 3 done! → {JSON_OUT}")
