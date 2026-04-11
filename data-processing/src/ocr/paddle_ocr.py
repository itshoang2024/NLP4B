"""
paddle_ocr.py — PaddleOCR-VL-1.5 Script Pipeline
===================================================
Run PaddleOCR-VL on keyframes extracted by LMSKE.

Usage (Colab):
  !python paddle_ocr.py --input_dir "/content/keyframes/video_id" --output_dir "/content/output" --batch_size 6
"""

# ── 0. Auto-install dependencies (Colab-friendly) ─────────────────────────────
import subprocess
import sys

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

try:
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
except ImportError:
    _pip("transformers>=5.0.0", "torch", "torchvision")

try:
    from PIL import Image
except ImportError:
    _pip("Pillow")

try:
    from tqdm import tqdm
except ImportError:
    _pip("tqdm")

# ── 1. Standard imports ────────────────────────────────────────────────────────
import argparse
import logging
import os
import json
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from PIL import Image
from tqdm import tqdm

# ── 2. Setup logging & Environment ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[PaddleOCR] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# ── 3. CLI Arguments ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Extract OCR from keyframes using PaddleOCR-VL")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing keyframe images from a SINGLE video",
    )
    parser.add_argument(
        "--output_dir",
        default="./ocr_output",
        help="Directory to save OCR JSON output",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help="Batch size for processing (default: 6 for T4 4-bit/bfloat16)",
    )
    return parser.parse_args()

# ── 4. Helper functions ────────────────────────────────────────────────────────
def discover_images(input_dir: Path):
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    )

    if not images:
        raise ValueError(f"No image files found in {input_dir}")

    logger.info(f"Found {len(images)} images in {input_dir}")
    return images

def load_model(device: str):
    model_path = "PaddlePaddle/PaddleOCR-VL-1.5"
    logger.info(f"Loading {model_path} on {device}...")

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if not hasattr(config, 'text_config'):
            config.text_config = config

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        logger.info("Model loaded successfully.")
        return model, processor
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# ── 5. Main Processing ─────────────────────────────────────────────────────────
def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    batch_size = args.batch_size

    logger.info("═" * 60)
    logger.info("PaddleOCR-VL-1.5 Script Pipeline")
    logger.info("═" * 60)
    logger.info(f"Input directory:  {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size:       {batch_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Optimize Torch for GPU
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    try:
        images = discover_images(input_dir)
    except Exception as e:
        logger.error(str(e))
        return

    model, processor = load_model(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = input_dir.name
    output_file = output_dir / f"{video_id}_ocr.json"

    final_results = []
    start_time = time.time()

    logger.info(f"Starting Turbo OCR generation with max token length 512...")
    with torch.inference_mode():
        for i in tqdm(range(0, len(images), batch_size), desc="Processing Batches"):
            batch_paths = images[i : i + batch_size]
            batch_filenames = [p.name for p in batch_paths]
            batch_images = []

            for path in batch_paths:
                try:
                    with Image.open(path) as img:
                        batch_images.append(img.convert("RGB").copy())
                except Exception as e:
                    logger.warning(f"Failed to open {path.name}: {e}")

            if not batch_images:
                continue

            max_pixels = 1280 * 28 * 28
            batch_messages = [
                [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "OCR:"}]}]
                for img in batch_images
            ]

            inputs = processor.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                images_kwargs={"size": {"shortest_edge": 448, "longest_edge": max_pixels}},
            ).to(device, dtype=torch.bfloat16)

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id
            )

            prompt_len = inputs["input_ids"].shape[-1]
            for idx, output_tensor in enumerate(outputs):
                text = processor.decode(output_tensor[prompt_len:-1], skip_special_tokens=True)
                
                # ADHERE STRICTLY to AGENTS.md schema for OCR outputs
                # OCR JSON schema: {"image": str, "ocr_text": str}
                final_results.append({
                    "image": batch_filenames[idx],
                    "ocr_text": text.strip()
                })

            del inputs, outputs
            if i % 20 == 0 and device == "cuda":
                torch.cuda.empty_cache()

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
        
    elapsed = time.time() - start_time

    # Summary
    logger.info("\n" + "═" * 60)
    logger.info("OCR Processing Complete!")
    logger.info("═" * 60)
    logger.info(f"Total keyframes:     {len(images)}")
    logger.info(f"Batch Size:          {batch_size}")
    logger.info(f"Processing time:     {elapsed:.2f}s ({elapsed/len(images):.3f}s per frame)")
    logger.info(f"Output saved to:     {output_file}")


if __name__ == "__main__":
    main()
