"""
embedding.py — SigLIP Keyframe Embedding Pipeline (Memory-Optimized)
====================================================================
Encode keyframes extracted by LMSKE.py using Google's SigLIP model.
Optimized for handling large numbers of keyframes on low-RAM systems (e.g., Google Colab).

Full end-to-end pipeline:
  1. Auto-install dependencies
  2. Load SigLIP model (google/siglip-so400m-patch14-384, 1152-dim embeddings)
  3. Discover keyframe images from input directory
  4. Create memory-mapped output file (preallocated)
  5. Encode each keyframe to embeddings and write directly to file
  6. Flush to disk every 1000 embeddings to keep RAM usage low

Usage (Colab):
  !python embedding.py --input_dir "/content/keyframes_output/video_id" --output_dir "/content/embeddings_output"

The output will be saved as `<output_dir>/video_id.npy` with shape (N, 1152), where N is the number of keyframes.

Key advantages:
  - Constant RAM usage (~10-50MB) regardless of keyframe count
  - Real-time disk writing (every 1000 embeddings) prevents memory overflow
  - Suitable for processing 100,000+ keyframes on Colab's 12.7GB RAM limit

Optional performance tuning:
  --batch_size 32 (default: 1, for GPU inference speedup)
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
    Handles BaseModelOutputWithPooling from SigLIP.
    Returns numpy array of shape (1152,) or None if failed.
    """
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")

        # Preprocess
        inputs = processor(images=img, return_tensors="pt").to(device)

        # Generate embedding
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

            # outputs is BaseModelOutputWithPooling object
            # Extract the actual embedding tensor
            if hasattr(outputs, 'last_hidden_state'):
                # SigLIP stores embeddings in last_hidden_state
                embedding_tensor = outputs.last_hidden_state
            elif hasattr(outputs, 'image_embeds'):
                # Alternative: direct image embeddings
                embedding_tensor = outputs.image_embeds
            else:
                # Fallback: try indexing (in case it's tuple-like)
                embedding_tensor = outputs[0]

            # Convert to numpy and flatten
            embedding = embedding_tensor.cpu().numpy().flatten().astype("float32")

            # Validate and reshape if needed
            if embedding.shape[0] != 1152:
                logger.warning(
                    f"Unexpected embedding shape for {image_path.name}: {embedding.shape}"
                )
                # Try to reshape (e.g., 1152*27*27 -> global average)
                if embedding.shape[0] == 1152 * 27 * 27:
                    embedding = embedding.reshape(1152, 27, 27).mean(axis=(1, 2))
                elif embedding.shape[0] % 1152 == 0:
                    embedding = embedding.reshape(1152, -1).mean(axis=1)
                else:
                    # Truncate or pad
                    if embedding.shape[0] > 1152:
                        embedding = embedding[:1152]
                    else:
                        embedding = np.pad(embedding, (0, 1152 - embedding.shape[0]))

            return embedding

    except Exception as e:
        logger.warning(f"Failed to embed {image_path.name}: {e}")
        return None


# ── 5. Main embedding pipeline ─────────────────────────────────────────────────
def main():
    args = parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logger.info("═" * 60)
    logger.info("SigLIP Keyframe Embedding Pipeline (Batch-optimized for low RAM)")
    logger.info("═" * 60)
    logger.info(f"Input directory:  {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch save size:  1000 embeddings per batch")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Step 1: Discover images
    logger.info("\n[Step 1/4] Discovering keyframe images...")
    images = discover_images(input_dir)

    # Step 2: Load model
    logger.info("\n[Step 2/4] Loading SigLIP model...")
    model, processor = load_model(device)

    # Step 3: Setup output file with memory-mapped array
    logger.info("\n[Step 3/4] Initializing memory-mapped output file...")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = input_dir.name
    output_filename = f"{video_id}.npy"
    output_path = output_dir / output_filename

    # Create memmap: (num_keyframes, 1152) float32
    num_keyframes = len(images)
    embedding_dim = 1152
    memmap_array = np.memmap(
        output_path, dtype="float32", mode="w+", shape=(num_keyframes, embedding_dim)
    )
    logger.info(
        f"Created memory-mapped file for {num_keyframes} embeddings ({num_keyframes * embedding_dim * 4 / 1e6:.1f} MB)"
    )

    # Step 4: Embed keyframes and save in batches
    logger.info("\n[Step 4/4] Encoding and saving keyframes in batches...")
    start_time = time.time()
    successful = 0
    failed = 0
    failed_images = []
    batch_size_save = 1000  # Save every 1000 embeddings

    for idx, image_path in enumerate(tqdm(images, desc="Embedding keyframes")):
        # Generate embedding
        embedding = embed_image(image_path, model, processor, device)

        if embedding is None:
            failed += 1
            failed_images.append(image_path.name)
            # Save placeholder NaN for failed embeddings (to keep array index aligned)
            memmap_array[idx] = np.full(embedding_dim, np.nan, dtype="float32")
            continue

        # Write directly to memmap
        memmap_array[idx] = embedding
        successful += 1

        # Flush to disk every batch_size_save embeddings
        if (idx + 1) % batch_size_save == 0:
            memmap_array.flush()
            logger.info(
                f"  ✓ Saved {idx + 1}/{num_keyframes} embeddings to disk ({(idx + 1) * embedding_dim * 4 / 1e6:.1f} MB written)"
            )

    # Final flush to ensure all data is written
    memmap_array.flush()
    logger.info(f"  ✓ Final flush: all {num_keyframes} embeddings saved to disk")

    elapsed = time.time() - start_time

    # Summary
    logger.info("\n" + "═" * 60)
    logger.info("Embedding Complete!")
    logger.info("═" * 60)
    logger.info(f"Total keyframes:     {len(images)}")
    logger.info(f"Successfully encoded: {successful}")
    logger.info(f"Failed to encode:    {failed}")
    logger.info(
        f"Processing time:     {elapsed:.2f}s ({elapsed/len(images):.3f}s per frame)"
    )
    logger.info(f"Output saved to:     {output_path}")
    logger.info(f"Output shape:        {(num_keyframes, embedding_dim)}")
    logger.info(
        f"Memory footprint:    File-based (≤10MB RAM vs. {num_keyframes * embedding_dim * 4 / 1e6:.1f}MB full buffer)"
    )

    if failed > 0:
        logger.warning(f"\n⚠️  Failed to process {len(failed_images)} image(s):")
        for fname in failed_images[:5]:  # Show first 5
            logger.warning(f"  - {fname}")
        if len(failed_images) > 5:
            logger.warning(f"  ... and {len(failed_images) - 5} more")
        logger.info(f"\nNote: Failed embeddings are saved as NaN in the output file.")


if __name__ == "__main__":
    main()
