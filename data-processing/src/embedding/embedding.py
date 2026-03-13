"""
embedding.py — SigLIP Keyframe Embedding Pipeline
===================================================
Encode keyframes extracted by LMSKE.py using Google's SigLIP model.

Full end-to-end pipeline:
  1. Auto-install dependencies
  2. Load SigLIP model (google/siglip-so400m-patch14-384, 1152-dim embeddings)
  3. Discover keyframe images from input directory
  4. Encode each keyframe to embeddings
  5. Save embeddings as individual .npy files

Usage (Google Colab — keyframes already extracted):
  !python embedding.py --input_dir "/content/keyframes_output/video_id" --output_dir "/content/embeddings_output"

Usage (Local):
  python embedding.py --input_dir "./keyframes_output/video_id" --output_dir "./embeddings_output"

Optional performance tuning:
  --batch_size 32 (default: 1, increase for GPU to speed up processing)
"""

# ── 0. Auto-install dependencies (Colab-friendly) ─────────────────────────────
import subprocess
import sys


def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


try:
    from transformers import AutoImageProcessor, AutoModel  # noqa: F401
except ImportError:
    _pip("transformers", "torch", "torchvision")

try:
    from PIL import Image  # noqa: F401
except ImportError:
    _pip("Pillow")

# ── 1. Standard imports ────────────────────────────────────────────────────────
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

# ── 2. Setup logging ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[Embedding] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Type alias for flexible path handling
PathLike = Union[str, Path]


# ── 3. CLI Arguments ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode keyframes using Google SigLIP model"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing keyframe images (output from LMSKE.py)",
    )
    parser.add_argument(
        "--output_dir",
        default="./embeddings",
        help="Directory to save embedding .npy files (default: ./embeddings)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (increase for GPU speedup, default: 1)",
    )
    return parser.parse_args()


# ── 4. Helper functions ────────────────────────────────────────────────────────
def discover_images(input_dir: PathLike) -> list[Path]:
    """
    Discover all image files in the input directory.
    Supports: .jpg, .jpeg, .png, .bmp, .gif, .webp
    Returns sorted list of image paths.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    images = sorted(
        [
            f
            for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    )

    if not images:
        raise ValueError(f"No image files found in {input_dir}")

    logger.info(f"Found {len(images)} keyframe images in {input_dir}")
    return images


def load_model(device: str):
    """
    Load SigLIP model and image processor from Hugging Face.
    Returns (model, processor).
    """
    model_id = "google/siglip-so400m-patch14-384"
    logger.info(f"Loading SigLIP model ({model_id}) on {device} ...")

    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
        logger.info(f"Model loaded successfully. Output dim: 1152")
        return model, processor
    except Exception as e:
        raise RuntimeError(f"Failed to load SigLIP model: {e}")


def embed_image(
    image_path: Path,
    model,
    processor,
    device: str,
) -> Optional[np.ndarray]:
    """
    Load image and generate SigLIP embedding (1152 dims).
    Returns numpy array of shape (1152,) or None if failed.
    """
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")

        # Preprocess
        inputs = processor(images=img, return_tensors="pt").to(device)

        # Generate embedding
        with torch.no_grad():
            outputs = model(**inputs)
            # SigLIP returns image_embeds
            embedding = outputs.image_embeds[0].cpu().numpy().flatten()  # (1152,)

        return embedding
    except Exception as e:
        logger.warning(f"Failed to embed {image_path.name}: {e}")
        return None


def save_embedding(embedding: np.ndarray, output_path: Path) -> bool:
    """
    Save embedding as .npy file.
    Returns True if successful, False otherwise.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embedding)
        return True
    except Exception as e:
        logger.warning(f"Failed to save embedding to {output_path}: {e}")
        return False


# ── 5. Main embedding pipeline ─────────────────────────────────────────────────
def main():
    args = parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logger.info("═" * 60)
    logger.info("SigLIP Keyframe Embedding Pipeline")
    logger.info("═" * 60)
    logger.info(f"Input directory:  {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size:       {args.batch_size}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Step 1: Discover images
    logger.info("\n[Step 1/3] Discovering keyframe images...")
    images = discover_images(input_dir)

    # Step 2: Load model
    logger.info("\n[Step 2/3] Loading SigLIP model...")
    model, processor = load_model(device)

    # Step 3: Embed keyframes
    logger.info("\n[Step 3/3] Encoding keyframes...")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    successful = 0
    failed = 0
    failed_images = []

    for image_path in tqdm(images, desc="Embedding keyframes"):
        # Generate embedding
        embedding = embed_image(image_path, model, processor, device)

        if embedding is None:
            failed += 1
            failed_images.append(image_path.name)
            continue

        # Save embedding
        output_filename = f"{image_path.stem}_embedding.npy"
        output_path = output_dir / output_filename

        if save_embedding(embedding, output_path):
            successful += 1
        else:
            failed += 1
            failed_images.append(image_path.name)

    elapsed = time.time() - start_time

    # Summary
    logger.info("\n" + "═" * 60)
    logger.info("Embedding Complete!")
    logger.info("═" * 60)
    logger.info(f"Total keyframes:     {len(images)}")
    logger.info(f"Successfully saved:  {successful}")
    logger.info(f"Failed:              {failed}")
    logger.info(
        f"Processing time:     {elapsed:.2f}s ({elapsed/len(images):.3f}s per frame)"
    )
    logger.info(f"Output directory:    {output_dir}/")

    if failed > 0:
        logger.warning(f"\nFailed to process {len(failed_images)} image(s):")
        for fname in failed_images[:5]:  # Show first 5
            logger.warning(f"  - {fname}")
        if len(failed_images) > 5:
            logger.warning(f"  ... and {len(failed_images) - 5} more")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
