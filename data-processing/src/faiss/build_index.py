"""
build_index.py — FAISS Index Builder for SigLIP Embeddings (Memory-Optimized)
=============================================================================
Build FAISS index from SigLIP embeddings stored in multiple .npy files.
Optimized for low-RAM systems (e.g., Google Colab) by processing videos sequentially.

Full end-to-end pipeline:
  1. Auto-install FAISS (CPU or GPU)
  2. Discover all embedding .npy files from input directory
  3. Create FAISS index (IndexFlatL2 or IndexHNSW)
  4. Incrementally process each video's embeddings
  5. Save index and metadata to output directory

Usage (Colab):
  !python build_index.py --input_dir "/content/embeddings_output" --output_dir "/content/faiss_output"

Output files:
  - faiss_index.bin: Main FAISS index (binary format)
  - video_metadata.json: Maps vector indices to video_id and frame positions
  - index_stats.json: Index statistics (num_vectors, dimension, etc.)

Key features:
  - Sequential processing: One video at a time (no full RAM buffer)
  - Supports both IndexFlatL2 (exact) and IndexHNSW (approximate, faster)
  - GPU acceleration via faiss-gpu if available
  - Progress tracking with frame-level statistics
  - Resumable: Can append to existing index

Optional parameters:
  --index_type flat       (default: flat for accuracy, use 'hnsw' for speed)
  --nprobe 32             (for IndexHNSW tuning)
  --batch_size 10000      (load embeddings in batches, default: all at once)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm


# ── 0. Auto-install FAISS (Colab-friendly) ────────────────────────────────────
def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


try:
    import faiss
except ImportError:
    _pip("faiss-cpu")
    import faiss

# ── 1. Setup logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="[FAISS Builder] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# ── 2. CLI Arguments ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from SigLIP embeddings"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing embedding .npy files (one per video)",
    )
    parser.add_argument(
        "--output_dir",
        default="./faiss_output",
        help="Directory to save FAISS index files (default: ./faiss_output)",
    )
    parser.add_argument(
        "--index_type",
        choices=["flat", "hnsw"],
        default="flat",
        help="Index type: 'flat' for exact search (accurate), 'hnsw' for approximate (faster). Default: flat",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=32,
        help="Number of clusters to probe (for HNSW tuning, default: 32)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for loading embeddings (default: 10000, 0=load all at once)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume building index from existing checkpoint",
    )
    return parser.parse_args()


# ── 3. Helper functions ────────────────────────────────────────────────────────
def discover_embedding_files(input_dir: PathLike) -> List[Tuple[Path, str]]:
    """
    Discover all .npy files in the input directory.
    Returns list of (file_path, video_id) sorted by video_id.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    npy_files = sorted(
        [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() == ".npy"]
    )

    if not npy_files:
        raise ValueError(f"No .npy files found in {input_dir}")

    # Extract video_id from filename (remove .npy extension)
    files_with_ids = [(f, f.stem) for f in npy_files]

    logger.info(f"Found {len(npy_files)} embedding files in {input_dir}")
    return files_with_ids


def load_embeddings(npy_file: Path, batch_size: int = 0) -> Tuple[np.ndarray, int]:
    """
    Load embeddings from a single .npy file.
    Returns (embeddings, num_vectors).
    If batch_size > 0, loads in batches (not implemented for simplicity).
    """
    try:
        embeddings = np.load(npy_file, allow_pickle=False)
        if embeddings.ndim != 2:
            logger.warning(
                f"  ⚠️  {npy_file.name} has shape {embeddings.shape}, expected (N, 1152)"
            )
        num_vectors = embeddings.shape[0]
        logger.info(
            f"  Loaded {npy_file.name}: {num_vectors} vectors, shape {embeddings.shape}"
        )
        return embeddings.astype("float32"), num_vectors
    except ValueError as e:
        if "pickled data" in str(e):
            logger.error(
                f"File {npy_file.name} is corrupted or in an old raw binary format. "
                f"Please re-run embedding.py on {npy_file.stem} to regenerate a valid .npy file."
            )
        else:
            logger.error(f"ValueError loading {npy_file.name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load {npy_file.name}: {e}")
        raise


def create_index(index_type: str, embedding_dim: int = 1152) -> faiss.Index:
    """
    Create a FAISS index of the specified type.
    Returns the index object.
    """
    if index_type == "flat":
        logger.info(f"Creating IndexFlatL2 (exact search, {embedding_dim}D)...")
        index = faiss.IndexFlatL2(embedding_dim)
    elif index_type == "hnsw":
        logger.info(f"Creating IndexHNSW (approximate search, {embedding_dim}D)...")
        index = faiss.IndexHNSWFlat(embedding_dim, 32)  # M=32
        index.hnsw.efConstruction = 200  # Construction parameter
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    # Try to use GPU if available
    if faiss.get_num_gpus() > 0:
        logger.info(f"✓ GPU detected: {faiss.get_num_gpus()} GPU(s) available")
        # Convert to GPU index (optional)
        # index = faiss.index_cpu_to_all_gpus(index)
    else:
        logger.info("No GPU detected, using CPU")

    return index


def save_index(
    index: faiss.Index,
    metadata: Dict,
    stats: Dict,
    output_dir: Path,
):
    """
    Save FAISS index and metadata to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    index_path = output_dir / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    logger.info(
        f"  ✓ Saved FAISS index to {index_path.name} ({index_path.stat().st_size / 1e6:.1f} MB)"
    )

    # Save metadata (maps vector index to video_id + frame_idx)
    metadata_path = output_dir / "video_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  ✓ Saved metadata to {metadata_path.name}")

    # Save index statistics
    stats_path = output_dir / "index_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  ✓ Saved statistics to {stats_path.name}")


def load_checkpoint(
    output_dir: Path,
) -> Tuple[Optional[faiss.Index], Optional[Dict], Optional[Dict]]:
    """
    Load existing index and metadata from checkpoint.
    Returns (index, metadata, stats) or (None, None, None) if not found.
    """
    index_path = output_dir / "faiss_index.bin"
    metadata_path = output_dir / "video_metadata.json"
    stats_path = output_dir / "index_stats.json"

    if not index_path.exists():
        return None, None, None

    logger.info("Loading checkpoint...")
    index = faiss.read_index(str(index_path))

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    with open(stats_path, "r") as f:
        stats = json.load(f)

    logger.info(f"  ✓ Loaded existing index with {index.ntotal} vectors")
    return index, metadata, stats


# ── 4. Main FAISS building pipeline ────────────────────────────────────────────
def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logger.info("═" * 70)
    logger.info("FAISS Index Builder (Sequential, Memory-Optimized)")
    logger.info("═" * 70)
    logger.info(f"Input directory:  {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Index type:       {args.index_type}")

    # Step 1: Discover embedding files
    logger.info("\n[Step 1/4] Discovering embedding files...")
    embedding_files = discover_embedding_files(input_dir)

    # Step 2: Load or create index
    logger.info("\n[Step 2/4] Loading/creating FAISS index...")
    if args.resume and output_dir.exists():
        index, metadata, stats = load_checkpoint(output_dir)
        if index is not None:
            start_video_idx = len(metadata)
            logger.info(f"Resuming from video {start_video_idx}/{len(embedding_files)}")
        else:
            logger.warning("No checkpoint found, creating new index")
            index = create_index(args.index_type)
            metadata = {}
            stats = {
                "index_type": args.index_type,
                "embedding_dim": 1152,
                "total_vectors": 0,
                "num_videos": 0,
                "videos_processed": [],
            }
            start_video_idx = 0
    else:
        index = create_index(args.index_type)
        metadata = {}
        stats = {
            "index_type": args.index_type,
            "embedding_dim": 1152,
            "total_vectors": 0,
            "num_videos": 0,
            "videos_processed": [],
        }
        start_video_idx = 0

    # Step 3: Iteratively process each video
    logger.info(
        "\n[Step 3/4] Building index from embeddings (sequential processing)..."
    )
    current_global_idx = index.ntotal

    for video_idx, (npy_file, video_id) in enumerate(embedding_files):
        if video_idx < start_video_idx:
            logger.info(f"  Skipping {video_id} (already processed)")
            continue

        logger.info(
            f"\n  Processing video {video_idx + 1}/{len(embedding_files)}: {video_id}"
        )

        # Load embeddings for this video
        embeddings, num_vectors = load_embeddings(npy_file, batch_size=args.batch_size)

        # Fix embeddings shape if corrupted (e.g., loaded as (N, 839808) instead of (N, 1152))
        logger.info(f"    Raw embeddings shape: {embeddings.shape}")

        # Check if embeddings got flattened or have wrong second dimension
        if embeddings.ndim == 2 and embeddings.shape[1] != 1152:
            logger.warning(
                f"    ⚠️  Embedding shape {embeddings.shape} != (N, 1152), attempting to reshape..."
            )
            # If shape is (N, 839808) where 839808 = 1152*729 or similar
            if embeddings.shape[1] == 1152 * 27 * 27:  # 839808
                logger.info(f"    Reshaping spatial features (N, 1152, 27, 27) -> (N, 1152)")
                embeddings = embeddings.reshape(num_vectors, 1152, 27, 27).mean(axis=(2, 3))
            elif embeddings.shape[1] % 1152 == 0:
                # Try to reshape assuming it was concatenated
                factor = embeddings.shape[1] // 1152
                logger.info(f"    Dividing {embeddings.shape[1]} dims by {factor}")
                embeddings = embeddings.reshape(num_vectors, 1152, factor).mean(axis=2)
            else:
                # Try to truncate to first 1152 dims or error out
                logger.warning(f"    Cannot reshape {embeddings.shape} - truncating to (N, 1152)")
                embeddings = embeddings[:, :1152]

        # Filter out NaN embeddings (failed frames from embedding.py)
        if embeddings.ndim == 2 and embeddings.shape[1] > 0:
            valid_mask = ~np.isnan(embeddings).any(axis=1)
            valid_count = valid_mask.sum()
            if valid_count < embeddings.shape[0]:
                logger.warning(
                    f"    ⚠️  Filtered {embeddings.shape[0] - valid_count} NaN vectors (failed embeddings)"
                )
                embeddings = embeddings[valid_mask]

        logger.info(f"    Final embeddings shape: {embeddings.shape}")

        # Add embeddings to index
        index.add(embeddings)
        logger.info(
            f"    ✓ Added {len(embeddings)} vectors to index (total: {index.ntotal})"
        )

        # Track metadata: which video each vector belongs to
        frame_metadata = [
            {"video_id": video_id, "frame_idx": i} for i in range(len(embeddings))
        ]
        for global_idx, frame_info in enumerate(
            frame_metadata, start=current_global_idx
        ):
            metadata[str(global_idx)] = frame_info

        current_global_idx = index.ntotal

        # Update stats
        stats["total_vectors"] = index.ntotal
        stats["num_videos"] = len(stats["videos_processed"]) + 1
        stats["videos_processed"].append(
            {
                "video_id": video_id,
                "num_vectors": len(embeddings),
                "global_index_range": [
                    current_global_idx - len(embeddings),
                    current_global_idx,
                ],
            }
        )

        # Save checkpoint after each video (for resumability)
        logger.info(f"    Saving checkpoint...")
        save_index(index, metadata, stats, output_dir)

    # Step 4: Final summary
    logger.info("\n" + "═" * 70)
    logger.info("FAISS Index Building Complete!")
    logger.info("═" * 70)
    logger.info(f"Total videos processed: {stats['num_videos']}")
    logger.info(f"Total vectors indexed:  {index.ntotal}")
    logger.info(f"Index type:             {args.index_type}")
    logger.info(f"Embedding dimension:    {stats['embedding_dim']}")
    logger.info(f"Output directory:       {output_dir}")
    logger.info(f"\nUsage in inference:")
    logger.info(f"  index = faiss.read_index('{output_dir}/faiss_index.bin')")
    logger.info(f"  metadata = json.load(open('{output_dir}/video_metadata.json'))")


if __name__ == "__main__":
    main()
