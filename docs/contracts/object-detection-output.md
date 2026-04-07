# Object Detection Output Contract

> **This is a schema contract.** The object detection JSON format is consumed by `qdrant_upsert.py` to populate Qdrant payload fields and sparse/dense vectors. Changes to this schema must be coordinated with downstream consumers.

## Purpose

Document the JSON output schema produced by the hybrid YOLO + Florence-2 object detection pipeline.

## Producer

- **Script:** [`data-processing/src/object_detection/object_detection.py`](../../data-processing/src/object_detection/object_detection.py)
- **Models:** YOLOv8m-WorldV2 (bounding boxes) + Microsoft Florence-2-large-ft (captions, dense region labels, OD)

## Consumers

- [`data-processing/src/qdrant/qdrant_upsert.py`](../../data-processing/src/qdrant/qdrant_upsert.py) — reads detection JSON to populate Qdrant payload fields (`tags`, `caption`, `detailed_caption`, `object_counts`) and to encode `keyframe-object-sparse` and `keyframe-caption-dense` vectors

## Output File

- **Path:** `<output_dir>/<video_id>_object_detection.json`
- **Encoding:** UTF-8
- **Video ID:** Derived from the input directory name (`input_dir.name`)

## Top-Level Schema

```json
{
  "video_id": "L01_V001",
  "input_dir": "/content/keyframes/L01_V001",
  "total_images": 42,
  "results": [ ... ]
}
```

| Field | Type | Description |
|---|---|---|
| `video_id` | `str` | Name of the input directory (e.g., YouTube video ID) |
| `input_dir` | `str` | Absolute path to the input image directory |
| `total_images` | `int` | Number of successfully processed images |
| `results` | `list[FrameResult]` | Per-frame detection results |

## Per-Frame Result Schema (`FrameResult`)

```json
{
  "image_id": "L01_V001_00142",
  "metadata": {
    "file_path": "/content/keyframes/L01_V001/L01_V001_00142.jpg",
    "width": 1280,
    "height": 720
  },
  "global_descriptions": {
    "tags": ["person", "car", "building"],
    "caption": "A city street with cars and people.",
    "detailed_caption": "A busy city street with several cars parked on the side, people walking on the sidewalk, and tall buildings in the background."
  },
  "objects": [
    {
      "label": "person",
      "confidence": 0.92,
      "bbox": [100.5, 200.3, 250.1, 450.7],
      "source": "yolo"
    },
    {
      "label": "a red car parked on the street",
      "bbox": [300.0, 350.0, 600.0, 500.0],
      "source": "dense_region",
      "rich_label": "a red sedan with license plate visible"
    }
  ]
}
```

### `metadata`

| Field | Type | Description |
|---|---|---|
| `file_path` | `str` | Absolute path to the source image |
| `width` | `int` | Image width in pixels |
| `height` | `int` | Image height in pixels |

### `global_descriptions`

| Field | Type | Source | Downstream use |
|---|---|---|---|
| `tags` | `list[str]` | Union of YOLO labels + Florence-2 `<OD>` labels, sorted | → Qdrant `tags` payload |
| `caption` | `str` | Florence-2 `<CAPTION>` task | → Qdrant `caption` payload |
| `detailed_caption` | `str` | Florence-2 `<MORE_DETAILED_CAPTION>` task | → Qdrant `detailed_caption` payload; encoded with BGE-M3 → `keyframe-caption-dense` vector |

### `objects[]`

| Field | Type | Required | Description |
|---|---|---|---|
| `label` | `str` | ✅ | Object class label |
| `confidence` | `float` | ❌ | Detection confidence (YOLO only; Florence-2 does not produce this) |
| `bbox` | `list[float]` | ✅ | Bounding box `[x1, y1, x2, y2]` in pixel coordinates |
| `source` | `str` | ✅ | One of: `"yolo"`, `"object_detection"` (Florence OD), `"dense_region"` (Florence dense region caption) |
| `rich_label` | `str` | ❌ | Florence-2 dense region label that was merged with a YOLO detection via IoU overlap |

## Downstream Processing by `qdrant_upsert.py`

The upsert script reads each frame result via the `build_det_lookup()` and `extract_frame_metadata()` functions:

1. **Lookup key:** `image_id` field (e.g., `"L01_V001_00142"`) — must match the pattern `{video_id}_{frame_idx:05d}`
2. **Tags extraction:** `global_descriptions.tags` → stored as Qdrant payload
3. **Caption extraction:** `global_descriptions.caption` and `detailed_caption` → stored as payload
4. **Object counts:** Counts of all `objects[].label` values → stored as `object_counts` payload
5. **Unique tags for sparse vector:** All unique `objects[].label` values are joined with spaces and encoded with BM25 → `keyframe-object-sparse` vector
6. **Caption dense vector:** `detailed_caption` is encoded with BGE-M3 → `keyframe-caption-dense` vector

## Failure Modes

| Failure | Cause | Symptom |
|---|---|---|
| Missing frame in detection JSON | Image processing error (caught gracefully) | Frame gets no `tags`, `caption`, or `object_counts` in Qdrant |
| Empty `tags` list | No objects detected in frame | `keyframe-object-sparse` vector not created for this point |
| Empty `detailed_caption` | Florence-2 returned empty string | `keyframe-caption-dense` vector not created for this point |
| Corrupt JSON file | Partial write / disk full | `qdrant_upsert.py` skips entire video |

## Compatibility Risks

1. **Renaming `global_descriptions` keys** (e.g., `tags` → `labels`) will break `extract_frame_metadata()` in `qdrant_upsert.py`
2. **Changing `image_id` format** will break the lookup by `{video_id}_{frame_idx:05d}` in the upsert script
3. **Removing `source` field** from objects will not break upsert (not used), but may break future analysis
4. **Adding new fields** is safe — `qdrant_upsert.py` only reads known keys

## Validation Checklist

Before changing the detection output format:

- [ ] Verify `image_id` follows pattern `{video_id}_{frame_idx:05d}`
- [ ] Verify `global_descriptions` contains `tags`, `caption`, `detailed_caption`
- [ ] Verify `objects[].label` and `objects[].bbox` are present
- [ ] Test with `qdrant_upsert.py` — confirm `build_det_lookup()` correctly indexes frames
- [ ] Confirm checkpoint/resume behavior still works with `--save_every`
