"""
faiss.py — FAISS Vector Indexing Pipeline
===================================================
Load SigLIP embeddings (.npy) and build a FAISS index.

Usage (Google Colab — embeddings already extracted):
  !python faiss.py --input_dir "/content/embeddings" --output_dir "/content/faiss_index"

Usage (Local):
  python faiss.py --input_dir "./embeddings_output" --output_dir "./faiss_index"
"""

# ── 0. Auto-install dependencies (Colab-friendly) ─────────────────────────────
import subprocess
import sys

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

try:
    import faiss  # noqa: F401
except ImportError:
    # First try faiss-gpu normally faster on colab
    try:
        _pip("faiss-gpu")
        import faiss  # noqa: F401
    except Exception:
        # Fallback to cpu if not running on GPU system
        _pip("faiss-cpu")
        import faiss  # noqa: F401

# ── 1. Standard imports ────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Union

import numpy as np

# ── 2. Setup logging ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[FAISS] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# ── 3. CLI Arguments ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index from .npy embeddings")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing embedding .npy files (output from embedding.py)",
    )
    parser.add_argument(
        "--output_dir",
        default="./faiss_index",
        help="Directory to save the FAISS index and mapping (default: ./faiss_index)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2-normalize vectors before indexing (recommended for SigLIP/CLIP with cosine similarity)",
    )
    return parser.parse_args()


# ── 4. Helper functions ────────────────────────────────────────────────────────
def discover_npy_files(input_dir: PathLike) -> list[Path]:
    """
    Discover all .npy files in the input directory.
    Returns sorted list of file paths.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() == ".npy"])
    
    if not files:
        raise ValueError(f"No .npy files found in {input_dir}")
        
    logger.info(f"Found {len(files)} .npy files in {input_dir}")
    return files


# ── 5. Main Indexing Pipeline ─────────────────────────────────────────────────
def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("═" * 60)
    logger.info("FAISS Vector Indexing Pipeline")
    logger.info("═" * 60)
    logger.info(f"Input directory:  {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Normalize vecs:   {args.normalize}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Discover numpy files
    try:
        npy_files = discover_npy_files(input_dir)
    except Exception as e:
        logger.error(f"Failed to discover mapping files: {e}")
        sys.exit(1)
    
    all_embeddings = []
    mapping = []  # FAISS row index -> metadata mapping
    global_id = 0
    start_time = time.time()
    
    logger.info("\n[Step 1/3] Loading embeddings and building mapping...")
    for npy_file in npy_files:
        try:
            video_id = npy_file.stem  # the file name without .npy
            emb = np.load(npy_file)
            
            if emb.ndim != 2:
                logger.warning(f"Skipping {npy_file.name}: invalid shape {emb.shape}, expected 2D array.")
                continue
                
            num_frames = emb.shape[0]
            
            # Try to load corresponding frame indices mapping
            indices_file = npy_file.with_name(f"{video_id}_frames.json")
            if indices_file.exists():
                with open(indices_file, "r", encoding="utf-8") as f:
                    frame_indices = json.load(f)
                if len(frame_indices) != num_frames:
                    logger.warning(f"Mismatch in frame indices length for {video_id}, falling back to sequential.")
                    frame_indices = list(range(num_frames))
            else:
                frame_indices = list(range(num_frames))
                
            # Build id-to-metadata mapping
            for i in range(num_frames):
                mapping.append({
                    "faiss_id": global_id + i,
                    "video_id": video_id, 
                    "frame_idx": frame_indices[i]
                })
                
            all_embeddings.append(emb)
            global_id += num_frames
        except Exception as e:
            logger.warning(f"Failed to process {npy_file.name}: {e}")
            
    if not all_embeddings:
        logger.error("No valid embeddings loaded. Exiting.")
        sys.exit(1)
        
    # Stack all embeddings together into one giant matrix
    embeddings_matrix = np.vstack(all_embeddings).astype("float32")
    dim = embeddings_matrix.shape[1]
    
    logger.info(f"Total vectors loaded: {embeddings_matrix.shape[0]}, Dimension: {dim}")
    
    # Step 2: Build FAISS index
    logger.info("\n[Step 2/3] Building FAISS Index...")
    if args.normalize:
        logger.info("Normalizing vectors for Cosine Similarity (using IndexFlatIP)...")
        faiss.normalize_L2(embeddings_matrix)
        index = faiss.IndexFlatIP(dim)
    else:
        logger.info("Using L2 Distance (using IndexFlatL2)...")
        index = faiss.IndexFlatL2(dim)
        
    # Add vectors to index
    index.add(embeddings_matrix)
    logger.info(f"Index built successfully with {index.ntotal} vectors.")
    
    # Step 3: Save outputs
    logger.info("\n[Step 3/3] Saving outputs...")
    index_filename = "keyframe_index.bin"
    index_path = output_dir / index_filename
    
    # Save the FAISS index structure to file
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS index saved to {index_path}")
    
    # Save the mapping (to associate a FAISS search result back to a video and frame)
    mapping_filename = "index_mapping.json"
    mapping_path = output_dir / mapping_filename
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Index mapping saved to {mapping_path}")
        
    elapsed = time.time() - start_time
    
    # Summary
    logger.info("\n" + "═" * 60)
    logger.info("FAISS Indexing Complete!")
    logger.info("═" * 60)
    logger.info(f"Total vectors:     {index.ntotal}")
    logger.info(f"Processing time:   {elapsed:.2f}s")
    logger.info(f"Index file:        {index_path}")
    logger.info(f"Mapping file:      {mapping_path}")


if __name__ == "__main__":
    main()