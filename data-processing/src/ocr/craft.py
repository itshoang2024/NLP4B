import os, glob, json, argparse
import numpy as np
from craft_text_detector import Craft

# ── CONFIG ───────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: CRAFT Text Detection — Film Retrieval OCR Pipeline"
    )
    parser.add_argument('--keyframes-dir', default='./keyframes',
                        help='Directory containing input keyframe images (default: ./keyframes)')
    parser.add_argument('--output-json',   default='./ocr_stage1_boxes.json',
                        help='Path to save detected text boxes JSON (default: ./ocr_stage1_boxes.json)')
    parser.add_argument('--cuda',          action='store_true',
                        help='Use GPU if available (default: CPU)')
    parser.add_argument('--text-threshold', type=float, default=0.7,
                        help='CRAFT text confidence threshold (default: 0.7)')
    parser.add_argument('--link-threshold', type=float, default=0.4,
                        help='CRAFT link confidence threshold (default: 0.4)')
    parser.add_argument('--low-text',      type=float, default=0.4,
                        help='CRAFT low-bound text score (default: 0.4)')
    parser.add_argument('--max-images',    type=int,   default=9999,
                        help='Max number of images to process (default: all)')
    return parser.parse_args()

args = parse_args()

# ── INIT CRAFT ───────────────────────────────────────────────
print(f"🔄 Loading CRAFT (cuda={args.cuda})...")
craft = Craft(
    output_dir      = None,
    crop_type       = "poly",
    cuda            = args.cuda,
    text_threshold  = args.text_threshold,
    link_threshold  = args.link_threshold,
    low_text        = args.low_text,
)
print("✅ CRAFT ready!\n")

# ── MAIN ─────────────────────────────────────────────────────
def main():
    paths = sorted(
        glob.glob(os.path.join(args.keyframes_dir, "*.jpg")) +
        glob.glob(os.path.join(args.keyframes_dir, "*.jpeg")) +
        glob.glob(os.path.join(args.keyframes_dir, "*.png"))
    )[:args.max_images]

    if not paths:
        print(f"❌ No images in '{args.keyframes_dir}'")
        print("   Put your keyframes there and re-run.")
        return

    print(f"🔍 CRAFT: Processing {len(paths)} keyframes...\n{'─'*50}")
    all_boxes = {}

    for i, p in enumerate(paths, 1):
        fname = os.path.basename(p)
        result = craft.detect_text(p)

        # Convert numpy arrays → plain lists for JSON
        boxes = [
            [[float(pt[0]), float(pt[1])] for pt in box]
            for box in result["boxes"]
        ]
        all_boxes[fname] = boxes
        print(f"  [{i}/{len(paths)}] {fname} → {len(boxes)} text region(s) found")

    craft.unload_craftnet_model()
    craft.unload_refinenet_model()

    # Save output JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(all_boxes, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"✅ Done! {len(all_boxes)} frames processed")
    print(f"📄 Boxes saved → {args.output_json}  ← feed this to Stage 2")

if __name__ == "__main__":
    main()

#python detect_text.py \
#  --keyframes-dir ./keyframes \
#  --output-json   ./ocr_out/ocr_stage1_boxes.json \
#  --max-images    20 \
#  --cuda
