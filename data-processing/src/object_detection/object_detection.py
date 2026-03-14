# ============================================================
#  YOLO Module — Film Retrieval Project
#  Run: python yolo_pipeline.py
#  Input:  ./keyframes/   (your JPG/PNG keyframes)
#  Output: ./yolo_out/    (annotated images + metadata.json)
# ============================================================

import os, json, glob, cv2, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# ── CONFIG ───────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='YOLO Object Detection — Film Retrieval Project'
    )
    parser.add_argument('--keyframes-dir', default='./keyframes',
                        help='Directory containing input keyframe images (default: ./keyframes)')
    parser.add_argument('--output-dir',    default='./yolo_out',
                        help='Directory to save annotated images and metadata (default: ./yolo_out)')
    parser.add_argument('--conf',          type=float, default=0.25,
                        help='YOLO confidence threshold (default: 0.25)')
    parser.add_argument('--grid-size',     type=int,   default=7,
                        help='Grid size for spatial encoding (default: 7)')
    parser.add_argument('--max-images',    type=int,   default=9999,
                        help='Max number of images to process (default: all)')
    parser.add_argument('--model-coco',    default='yolo11n.pt',
                        help='YOLO COCO model weights file (default: yolo11n.pt)')
    parser.add_argument('--model-open',    default='yolov8n.pt',
                        help='YOLO open-vocabulary model weights file (default: yolov8n.pt)')
    return parser.parse_args()

args = parse_args()

KEYFRAMES_DIR  = args.keyframes_dir
OUTPUT_DIR     = args.output_dir
CONF_THRESHOLD = args.conf
GRID_SIZE      = args.grid_size
MAX_IMAGES     = args.max_images

os.makedirs(KEYFRAMES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,    exist_ok=True)

# ── LOAD MODELS ──────────────────────────────────────────────
print("🔄 Loading YOLO models...")
model_coco = YOLO(args.model_coco)
model_open = YOLO(args.model_open)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {DEVICE}\n")

# ── ENCODING ─────────────────────────────────────────────────
def get_color(cls_name):
    h = hash(cls_name)
    return (h % 200 + 30, (h*3) % 200 + 30, (h*7) % 200 + 30)

def encode_counts(detections):
    return ' '.join(f"{k}{v}" for k, v in sorted(detections.items()))

def encode_tags(detections):
    return sorted(detections.keys())

def encode_bbox_string(bboxes):
    return ' '.join(
        f"{cls}({nx1:.2f},{ny1:.2f},{nx2:.2f},{ny2:.2f}:{cf:.2f})"
        for cls, cf, nx1, ny1, nx2, ny2 in bboxes
    )

def encode_7x7_grid(bboxes, img_w, img_h):
    ZONE = {
        (r,c): f"{'top' if r<2 else 'mid' if r<5 else 'bot'}-"
               f"{'left' if c<2 else 'center' if c<5 else 'right'}"
        for r in range(GRID_SIZE) for c in range(GRID_SIZE)
    }
    grid_cls  = [['empty']*GRID_SIZE for _ in range(GRID_SIZE)]
    grid_conf = [[0.0]*GRID_SIZE     for _ in range(GRID_SIZE)]
    cw, ch = img_w/GRID_SIZE, img_h/GRID_SIZE

    for cls, conf, nx1, ny1, nx2, ny2 in bboxes:
        cx = ((nx1+nx2)/2) * img_w
        cy = ((ny1+ny2)/2) * img_h
        gx = min(int(cx/cw), GRID_SIZE-1)
        gy = min(int(cy/ch), GRID_SIZE-1)
        if conf > grid_conf[gy][gx]:
            grid_cls[gy][gx]  = cls
            grid_conf[gy][gx] = conf

    codloc = ' '.join(
        f"{grid_cls[r][c]}@{ZONE[(r,c)]}"
        for r in range(GRID_SIZE) for c in range(GRID_SIZE)
        if grid_cls[r][c] != 'empty'
    )
    flat = ' '.join(grid_cls[r][c] for r in range(GRID_SIZE) for c in range(GRID_SIZE))
    return {'codloc_string': codloc, 'flat_string': flat, 'grid_2d': grid_cls}

def encode_region_map(bboxes):
    zones = {}
    for cls, _, nx1, ny1, nx2, ny2 in bboxes:
        cx, cy = (nx1+nx2)/2, (ny1+ny2)/2
        row = 'top' if cy<0.33 else ('mid' if cy<0.67 else 'bot')
        col = 'left' if cx<0.33 else ('center' if cx<0.67 else 'right')
        zones.setdefault(f"{row}-{col}", []).append(cls)
    return zones

# ── DETECTION ────────────────────────────────────────────────
def detect_and_annotate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ⚠️  Cannot read: {image_path} — skipping")
        return None, None

    orig_h, orig_w = img.shape[:2]
    annotated  = img.copy()
    detections = {}
    bboxes     = []

    for model in [model_coco, model_open]:
        for r in model(image_path, conf=CONF_THRESHOLD,
                       device=DEVICE, verbose=False):
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_name = r.names[cls_id]

                detections[cls_name] = detections.get(cls_name, 0) + 1
                bboxes.append((cls_name, conf,
                               x1/orig_w, y1/orig_h,
                               x2/orig_w, y2/orig_h))

                color = get_color(cls_name)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)

                label = f"{cls_name} {conf:.2f}"
                (lw, lh), base = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                ly = max(y1-lh-base-4, lh+base+4)
                cv2.rectangle(annotated,
                              (x1, ly-lh-base-2), (x1+lw+4, ly+base),
                              color, -1)
                cv2.putText(annotated, label, (x1+2, ly-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255,255,255), 2)

        torch.cuda.empty_cache()   # free VRAM between models

    grid = encode_7x7_grid(bboxes, orig_w, orig_h)
    metadata = {
        'frame_id':       os.path.basename(image_path),
        'image_path':     image_path,
        'timestamp':      None,
        'obj_counts':     encode_counts(detections),
        'obj_tags':       encode_tags(detections),
        'bbox_string':    encode_bbox_string(bboxes),
        'codloc_string':  grid['codloc_string'],
        'flat_grid':      grid['flat_string'],
        'grid_2d':        grid['grid_2d'],
        'region_map':     encode_region_map(bboxes),
        'total_objects':  sum(detections.values()),
        'raw_detections': detections,
    }
    return annotated, metadata

# ── DISPLAY ──────────────────────────────────────────────────
def show_grid(imgs, titles, cols=3):
    n    = len(imgs)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows*5))
    axes = np.array(axes).flatten()
    for i, (img, title) in enumerate(zip(imgs, titles)):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')
    for j in range(len(imgs), len(axes)):
        axes[j].axis('off')
    plt.suptitle("🎬 YOLO Detection — Film Keyframes",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'results_grid.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"🖼️  Grid saved: {OUTPUT_DIR}/results_grid.png")

# ── MAIN ─────────────────────────────────────────────────────
def run():
    paths = sorted(
        glob.glob(os.path.join(KEYFRAMES_DIR, '*.jpg')) +
        glob.glob(os.path.join(KEYFRAMES_DIR, '*.jpeg')) +
        glob.glob(os.path.join(KEYFRAMES_DIR, '*.png'))
    )[:MAX_IMAGES]

    if not paths:
        print(f"❌ No images in '{KEYFRAMES_DIR}'")
        print("   Put your keyframes there and re-run.")
        return

    print(f"🎬 Processing {len(paths)} keyframes...\n{'─'*50}")

    all_metadata  = {}
    all_annotated = []
    all_titles    = []

    for i, img_path in enumerate(paths, 1):
        fname = os.path.basename(img_path)
        print(f"[{i}/{len(paths)}] {fname}")

        annotated, meta = detect_and_annotate(img_path)
        if annotated is None:
            continue

        print(f"  Counts  : {meta['obj_counts']}")
        print(f"  CodLoc  : {meta['codloc_string']}")
        print(f"  Regions : {meta['region_map']}")

        out_path = os.path.join(OUTPUT_DIR, f"annotated_{fname}")
        cv2.imwrite(out_path, annotated)

        all_metadata[fname]  = meta
        all_annotated.append(annotated)
        all_titles.append(f"{fname}\n{meta['obj_counts'] or 'no detections'}")

    json_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    json.dump(all_metadata, open(json_path, 'w'), indent=2)

    print(f"\n{'─'*50}")
    print(f"✅ Done! {len(all_metadata)} frames processed")
    print(f"📁 Annotated images → {OUTPUT_DIR}/")
    print(f"📄 metadata.json    → {json_path}  ← hand this to teammates")

    if all_annotated:
        show_grid(all_annotated, all_titles)

if __name__ == "__main__":
    run()


