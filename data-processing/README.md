# Data Processing Pipeline

Offline preprocessing module that transforms raw YouTube videos into searchable artifacts for the multimodal retrieval system.

## What this module is responsible for

- Downloading YouTube videos and extracting technical metadata (fps, resolution, codec)
- Extracting semantic keyframes via shot segmentation + adaptive clustering
- Generating SigLIP visual embeddings for keyframes
- Running hybrid object detection (YOLO + Florence-2) and OCR
- Uploading artifacts to Azure Blob Storage
- Upserting 4-vector points into Qdrant Cloud

## What this module is NOT responsible for

- Query-time retrieval (see `retrieval/agentic_retrieval/`)
- UI rendering (see `streamlit/`)
- Live embedding inference (see `azure-ai-provider/`)

## Structure

```
data-processing/
├── src/
│   ├── download/                  # Step 1: Video download + metadata
│   │   ├── main.py                # Entry point (3-step pipeline)
│   │   ├── excel_manifest.py      # Excel → normalized manifest CSV
│   │   ├── youtube_download.py    # yt-dlp wrapper
│   │   └── ffprobe_extract.py     # ffprobe metadata extraction
│   │
│   ├── keyframe_extraction/       # Step 2: Keyframe extraction
│   │   ├── LMSKE.py               # Entry point (5-step pipeline)
│   │   ├── Kmeans_improvment.py   # Adaptive K-means + silhouette
│   │   ├── Redundancy.py          # Cosine similarity deduplication
│   │   ├── save_keyframe.py       # Frame saving utility
│   │   └── transnetv2_pytorch/    # Bundled TransNetV2 model
│   │
│   ├── embedding/                 # Step 3: SigLIP embeddings
│   │   └── embedding.py           # Entry point
│   │
│   ├── object_detection/          # Step 4a: Object detection
│   │   ├── object_detection.py    # Entry point (YOLO + Florence-2)
│   │   └── object_detection_info.md  # Feature documentation
│   │
│   ├── ocr/                       # Step 4b: Text recognition
│   │   ├── paddle_ocr.py          # Entry point (PaddleOCR-VL 1.5)
│   │   └── PaddleOCR.ipynb        # Original Colab notebook
│   │
│   ├── qdrant/                    # Step 5: Vector indexing
│   │   └── qdrant_upsert.py       # 4-vector Qdrant upsert (streams from Azure)
│   │
│   ├── faiss/                     # Alternative: local FAISS index
│   │   └── build_index.py         # FAISS index builder
│   │
│   ├── scripts/                   # Auxiliary scripts
│   │
│   └── azure_migrator.py          # Upload keyframes/embeddings/OCR to Azure
│
├── notebook/                      # Step-by-step Colab/Kaggle notebooks
│   ├── 01_download_videos.ipynb
│   ├── 02_keyframe_extraction.ipynb
│   ├── 03_embedding_keyframe.ipynb
│   ├── 04_faiss_indexing.ipynb
│   ├── object_detection.ipynb
│   ├── object_detection_kaggle.ipynb
│   ├── azure_migration.ipynb
│   └── up_google_drive_to_kaggle_fixed.ipynb
│
├── output/                        # Generated manifests and metadata
├── templates/                     # Input Excel templates
│   └── link_videos_template.xlsx
└── requirements.txt
```

## Pipeline Steps

### Step 1: Download Videos + Metadata

```bash
cd data-processing
python -m src.download.main \
    --input-excel templates/link_videos_template.xlsx \
    --output-root ./output \
    --download-mode 720p_mp4
```

**Input:** Excel file with a `url` column containing YouTube URLs.

**Output:**
| File | Description |
|---|---|
| `normalized_manifest.csv` | Cleaned URL list with `video_id`, `url`, `title` columns |
| `download_log.csv` | Download status per video (`ok` / `error`) |
| `download_archive.txt` | yt-dlp archive for skipping already-downloaded videos |
| `video_metadata.csv` | ffprobe-derived: `video_id`, `fps`, `duration_sec`, `width`, `height`, `codec_name` |
| `metadata_log.csv` | Metadata extraction status per video |

**Key CLI args:** `--max-downloads`, `--download-mode {best,720p_mp4,480p_mp4}`, `--skip-metadata`, `--overwrite-existing`

---

### Step 2: Keyframe Extraction (LMSKE)

```bash
python src/keyframe_extraction/LMSKE.py \
    --video ./output/videos/<video_id>.mp4 \
    --output_dir ./output/keyframes
```

**Input:** Video file path, Google Drive link, or YouTube URL.

**Output:**
| File | Description |
|---|---|
| `<output_dir>/<video_id>/<video_id>_XXXXX.jpg` | Keyframe images (zero-padded frame index) |
| `<output_dir>/<video_id>/<video_id>_scenes.txt` | Cached shot boundaries (reused on re-run) |
| `_tmp/<video_id>_features.pkl` | Cached CLIP features (reused on re-run) |

**Pipeline:** Resolve video → TransNetV2 shot segmentation → CLIP feature extraction → K-means clustering + silhouette → redundancy elimination → save frames

**Key CLI args:** `--max_frames_per_shot 30`, `--redundancy_threshold 0.94`

---

### Step 3: Embedding Generation (SigLIP)

```bash
python src/embedding/embedding.py \
    --input_dir ./output/keyframes/<video_id> \
    --output_dir ./output/embeddings \
    --batch_size 1
```

**Input:** Directory containing keyframe `.jpg` images.

**Output:**
| File | Description |
|---|---|
| `<video_id>.npy` | Stacked embeddings, shape `(N, 1152)`, float32 |
| `<video_id>_frames.json` | JSON array of frame indices mapping row order → original frame index |

**Model:** `google/siglip-so400m-patch14-384` (1152-dimensional image embeddings)

---

### Step 4a: Object Detection (YOLO + Florence-2)

```bash
python src/object_detection/object_detection.py \
    -i ./output/keyframes/<video_id> \
    -o ./output/detections \
    --save_every 50
```

**Input:** Directory containing keyframe images.

**Output:** `<video_id>_object_detection.json` — see [docs/contracts/object-detection-output.md](../docs/contracts/object-detection-output.md) for full schema.

**Key CLI args:** `--yolo_model`, `--florence_model`, `--iou 0.75`, `--limit`, `--save_every 50`

**Features:** Checkpoint saving, resume support, IoU deduplication between YOLO and Florence-2.

---

### Step 4b: OCR (PaddleOCR-VL 1.5)

```bash
python -m src.ocr.paddle_ocr \
    -i ./output/keyframes/<video_id> \
    -o ./output/ocr \
    --batch_size 6
```

**Input:** Directory containing keyframe `.jpg` images.

**Output:** `<video_id>_ocr.json` — array of `{"image": "<filename>", "ocr_text": "<text>"}` entries.

**Model:** `PaddlePaddle/PaddleOCR-VL-1.5` (VLM-based OCR, requires `transformers>=5.0.0`, GPU recommended)

**Key CLI args:** `--model`, `--batch_size 6`, `--max_new_tokens 512`, `--limit`, `--save_every 50`, `--prepare_only`, `--hf_cache_dir`

**Features:** Checkpoint saving, resume support, atomic writes, `--prepare_only` for model pre-download on Kaggle/Colab.

---

### Step 5: Azure Migration

```bash
python src/azure_migrator.py \
    --frames_dir ./output/keyframes \
    --embeddings_dir ./output/embeddings \
    --ocr_dir ./output/ocr \
    --workers 10
```

**Requires:** `AZURE_STORAGE_CONNECTION_STRING` environment variable.

**Uploads to Azure containers:**
| Container | Content | Blob path |
|---|---|---|
| `keyframes` | Keyframe images | `{video_id}/{video_id}_XXXXX.jpg` |
| `embeddings` | `.npy` + `_frames.json` | `{video_id}/{filename}` |
| `ocr` | OCR JSON | `{video_id}/{video_id}_ocr.json` |

---

### Step 6: Qdrant Upsert

```bash
python src/qdrant/qdrant_upsert.py \
    --collection_name keyframes_v1 \
    --batch_size 64
```

**Requires:** `AZURE_STORAGE_CONNECTION_STRING`, `QDRANT_URL`, `QDRANT_API_KEY`, `AZURE_BLOB_BASE_URL`

Streams artifacts from Azure Blob and creates 4-vector points in Qdrant. See [docs/contracts/qdrant-collection-schema.md](../docs/contracts/qdrant-collection-schema.md) for the full schema.

**Modes:** `--mode upsert` (full point creation) or `--mode update` (patch payload/vectors on existing points)

## Notebooks

The `notebook/` directory provides Colab-friendly step-by-step execution:

| Notebook | Maps to |
|---|---|
| `01_download_videos.ipynb` | Step 1 |
| `02_keyframe_extraction.ipynb` | Step 2 |
| `03_embedding_keyframe.ipynb` | Step 3 |
| `04_faiss_indexing.ipynb` | FAISS alternative to Qdrant |
| `object_detection.ipynb` | Step 4a (local) |
| `object_detection_kaggle.ipynb` | Step 4a (Kaggle GPU) |
| `PaddleOCR.ipynb` (in `src/ocr/`) | Step 4b (original notebook) |
| `azure_migration.ipynb` | Step 5 |

## What to test after changes

- If you change **download/manifest logic**: verify `normalized_manifest.csv` column names haven't changed
- If you change **keyframe extraction**: verify frame filename pattern `{video_id}_{frame_idx:05d}.jpg` is preserved
- If you change **embedding generation**: verify `.npy` shape is `(N, 1152)` and `_frames.json` ordering matches
- If you change **object detection**: verify JSON schema matches [the contract](../docs/contracts/object-detection-output.md)
- If you change **azure_migrator**: verify blob path conventions match what `qdrant_upsert.py` expects
- If you change **OCR** (`paddle_ocr.py`): verify JSON schema `{"image": str, "ocr_text": str}` is preserved; check `build_ocr_lookup()` in `qdrant_upsert.py`
- If you change **qdrant_upsert**: verify against [the Qdrant schema contract](../docs/contracts/qdrant-collection-schema.md)
