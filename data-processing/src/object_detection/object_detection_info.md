# Hybrid Object Detection Pipeline (YOLO + Florence-2)

This script processes a directory of images using a dual-model approach. It leverages the speed and bounding-box accuracy of **YOLO (Ultralytics)** alongside the rich vocabulary and spatial reasoning of **Microsoft's Florence-2**. 

To prevent redundant outputs, the script uses Intersection over Union (IoU) to automatically deduplicate overlapping bounding boxes, prioritizing YOLO's precise coordinates while retaining Florence-2's descriptive labels.

## Features
- **Batch Processing:** Automatically reads all `.jpg`, `.jpeg`, and `.png` files in a target directory.
- **IoU Deduplication:** Merges overlapping bounding boxes automatically.
- **Rich JSON Output:** Generates standardized JSONs containing flat tag lists, global descriptions, and precise coordinates for every detected object/region.
- **Graceful Error Handling:** If one image fails (e.g., due to file corruption), the script catches the error and continues to the next file.

## Prerequisites

Ensure your Python environment (preferably Python 3.10) has the following libraries installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.48.3 ultralytics==8.4.21 accelerate==1.13.0 pillow==11.3.0
