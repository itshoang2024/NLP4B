
"""
embedding.py — SigLIP Keyframe Embedding Pipeline
================================================
Kaggle/Colab-friendly embedding pipeline with:
- pinned revision support
- Hugging Face cache predownload support
- local_files_only loading
- reusable model runtime across many video folders
- detailed logging for debugging slow model initialization
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

PathLike = Union[str, Path]

DEFAULT_MODEL_ID = "google/siglip-so400m-patch14-384"
DEFAULT_CACHE_ROOT = "/kaggle/working/hf_home"
DEFAULT_REVISION = "main"

logging.basicConfig(level=logging.INFO, format="[Embedding] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Encode keyframes using Google SigLIP")
    parser.add_argument("--input_dir", required=True, help="Directory containing keyframe images")
    parser.add_argument("--output_dir", default="./embeddings", help="Directory to save embedding files")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding inference")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Pinned model revision")
    parser.add_argument("--hf_cache_root", default=DEFAULT_CACHE_ROOT, help="HF cache root")
    parser.add_argument(
        "--local_model_dir",
        default=None,
        help="Local snapshot directory to load from. If provided, skip remote model id loading.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load only from local cache / local_model_dir without network access.",
    )
    return parser.parse_args()


def discover_images(input_dir: PathLike) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    images = sorted(
        f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions
    )
    if not images:
        raise ValueError(f"No image files found in {input_dir}")

    logger.info("Found %d keyframe images in %s", len(images), input_dir)
    return images


def _resolve_model_source(
    model_id: str,
    revision: str,
    local_model_dir: Optional[PathLike],
) -> str:
    if local_model_dir:
        local_model_dir = Path(local_model_dir)
        if not local_model_dir.exists():
            raise FileNotFoundError(f"Local model directory not found: {local_model_dir}")
        logger.info("Using local SigLIP model dir: %s", local_model_dir)
        return str(local_model_dir)

    logger.info("Using model id: %s (revision=%s)", model_id, revision)
    return model_id


def load_model(
    device: str,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    local_model_dir: Optional[PathLike] = None,
    local_files_only: bool = False,
):
    model_source = _resolve_model_source(model_id, revision, local_model_dir)
    load_kwargs = {
        "local_files_only": local_files_only,
    }
    if local_model_dir is None:
        load_kwargs["revision"] = revision

    logger.info("Loading SigLIP processor from %s ...", model_source)
    processor = AutoImageProcessor.from_pretrained(model_source, **load_kwargs)
    logger.info("SigLIP processor loaded successfully.")

    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info("Loading SigLIP model from %s on %s with dtype=%s ...", model_source, device, dtype)
    model = AutoModel.from_pretrained(model_source, torch_dtype=dtype, **load_kwargs).to(device)
    model.eval()
    logger.info("SigLIP model loaded successfully. Output dim expected: 1152")
    return model, processor


def _load_images(batch_paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in batch_paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def embed_images_batch(
    image_paths: list[Path],
    model,
    processor,
    device: str,
) -> tuple[list[np.ndarray], list[str]]:
    failed_images: list[str] = []
    try:
        images = _load_images(image_paths)
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                features = model.get_image_features(**inputs)
            else:
                outputs = model(**inputs)
                if isinstance(outputs, torch.Tensor):
                    features = outputs
                elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                    features = outputs.image_embeds
                elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    features = outputs.last_hidden_state.mean(dim=1)
                else:
                    features = outputs[0]
                    if getattr(features, "ndim", 0) == 3:
                        features = features.mean(dim=1)
                    elif getattr(features, "ndim", 0) == 4:
                        features = features.mean(dim=(2, 3))

        embeddings = features.detach().float().cpu().numpy().astype("float32")
        output_embeddings: list[np.ndarray] = []
        for path, embedding in zip(image_paths, embeddings):
            flat = embedding.flatten()
            if len(flat) != 1152:
                logger.warning("Unexpected feature shape for %s: %s", path.name, flat.shape)
            output_embeddings.append(flat)
        return output_embeddings, failed_images
    except Exception as exc:
        logger.warning("Batch failed for %d image(s): %s", len(image_paths), exc)
        failed_images.extend([p.name for p in image_paths])
        return [], failed_images


def save_embedding(embedding: np.ndarray, output_path: Path) -> bool:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embedding)
        return True
    except Exception as e:
        logger.warning("Failed to save embeddings to %s: %s", output_path, e)
        return False


def process_directory(
    input_dir: PathLike,
    output_dir: PathLike,
    model,
    processor,
    device: str,
    batch_size: int = 8,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    images = discover_images(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    successful = 0
    failed = 0
    failed_images: list[str] = []
    video_embeddings: list[np.ndarray] = []
    frame_indices: list[Union[int, str]] = []

    logger.info("Encoding %d keyframes from %s ...", len(images), input_dir.name)

    for start in tqdm(range(0, len(images), batch_size), desc=f"Embedding {input_dir.name}"):
        batch_paths = images[start : start + batch_size]
        batch_embeddings, batch_failed = embed_images_batch(batch_paths, model, processor, device)

        if batch_failed:
            failed += len(batch_failed)
            failed_images.extend(batch_failed)

        for image_path, embedding in zip(batch_paths[: len(batch_embeddings)], batch_embeddings):
            try:
                frame_idx = int(image_path.stem.split("_")[-1])
            except ValueError:
                frame_idx = image_path.stem
            video_embeddings.append(embedding)
            frame_indices.append(frame_idx)
            successful += 1

        if device == "cuda":
            torch.cuda.empty_cache()

    output_path = None
    save_success = False
    indices_path = None
    if video_embeddings:
        video_id = input_dir.name
        output_path = output_dir / f"{video_id}.npy"
        final_embedding = np.stack(video_embeddings)
        save_success = save_embedding(final_embedding, output_path)
        if save_success:
            indices_path = output_dir / f"{video_id}_frames.json"
            with open(indices_path, "w", encoding="utf-8") as f:
                json.dump(frame_indices, f)
            logger.info(
                "Saved %d embeddings to %s with shape %s",
                len(video_embeddings),
                output_path.name,
                final_embedding.shape,
            )
            logger.info("Saved frame indices mapping to %s", indices_path.name)
    else:
        logger.warning("No embeddings generated for %s.", input_dir.name)

    elapsed = time.time() - start_time
    result = {
        "video_id": input_dir.name,
        "total_images": len(images),
        "successful": successful,
        "failed": failed,
        "failed_images": failed_images,
        "elapsed_sec": elapsed,
        "output_path": str(output_path) if output_path else None,
        "indices_path": str(indices_path) if indices_path else None,
        "save_success": save_success,
    }
    logger.info(
        "Done %s | ok=%d failed=%d elapsed=%.2fs",
        input_dir.name,
        successful,
        failed,
        elapsed,
    )
    return result


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("═" * 60)
    logger.info("SigLIP Keyframe Embedding Pipeline")
    logger.info("═" * 60)
    logger.info("Input directory:  %s", args.input_dir)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Batch size:       %s", args.batch_size)
    logger.info("Device:           %s", device)
    logger.info("HF cache root:    %s", args.hf_cache_root)
    if torch.cuda.is_available():
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))

    model, processor = load_model(
        device=device,
        model_id=args.model_id,
        revision=args.revision,
        local_model_dir=args.local_model_dir,
        local_files_only=args.local_files_only,
    )
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=model,
        processor=processor,
        device=device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
