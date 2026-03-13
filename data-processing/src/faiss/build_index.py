"""
build_index.py — FAISS Vector Indexing Pipeline
===================================================
Load SigLIP embeddings (.npy) and build a FAISS index.

Usage (Google Colab):
  !python build_index.py --input_dir "/content/embeddings" --output_dir "/content/faiss_index"

Usage (Local):
  python build_index.py --input_dir "./embeddings_output" --output_dir "./faiss_index"
"""

import subprocess
import sys

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

try:
    import faiss
except ImportError:
    try:
        _pip("faiss-gpu")
        import faiss
    except Exception:
        _pip("faiss-cpu")
        import faiss

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="[FAISS] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from .npy embeddings")
    parser.add_argument("--input_dir", required=True, help="Directory containing embedding .npy files")
    parser.add_argument("--output_dir", default="./faiss_index", help="Directory to save FAISS index")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Loading embeddings from .npy files...")
    logger.info("=" * 60)

    # Load all .npy files
    npy_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])

    if not npy_files:
        logger.error(f"No .npy files found in {input_dir}")
        return

    logger.info(f"Found {len(npy_files)} .npy files")

    start_time = time.time()

    # Load and concatenate all embeddings
    embeddings_list = []
    mapping_data = []
    global_id = 0

    for npy_file in tqdm(npy_files, desc="Loading embeddings"):
        emb = np.load(input_dir / npy_file)
        video_name = os.path.splitext(npy_file)[0]

        num_frames = emb.shape[0]

        # Create mapping entries
        for i in range(num_frames):
            mapping_data.append({
                "faiss_id": global_id + i,
                "video_id": video_name,
                "frame_idx": i
            })

        embeddings_list.append(emb)
        global_id += num_frames

    # Stack embeddings
    embeddings = np.concatenate(embeddings_list, axis=0).astype("float32")
    logger.info(f"\nTotal embeddings: {embeddings.shape[0]}, Dimension: {embeddings.shape[1]}")

    # Build FAISS index
    logger.info("\nBuilding FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"Index built with {index.ntotal} vectors")

    # Save index
    index_path = output_dir / "keyframe_index.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved index to {index_path}")

    # Save mapping
    mapping_df = pd.DataFrame(mapping_data)
    mapping_path = output_dir / "keyframe_index_mapping.csv"
    mapping_df.to_csv(str(mapping_path), index=False)
    logger.info(f"Saved mapping to {mapping_path}")

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info(f"Total time: {elapsed:.2f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
