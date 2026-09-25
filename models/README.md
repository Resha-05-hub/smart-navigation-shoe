# Models Directory

Holds the object-detection weights used by the Smart Navigation Shoe.

## Default Model

| File | Model | Size | Classes |
|---|---|---|---|
| `yolov8n.pt` | Ultralytics YOLOv8 Nano (pretrained on COCO) | ~6.2 MB | 80 COCO classes |

The weights are **not stored in git** (`.gitignore` excludes `models/*.pt`, `*.onnx`, `*.engine`).
After a fresh clone, `yolov8n.pt` is **downloaded automatically into this folder on the first run**,
so that first run needs internet access. Later runs use the local file and work offline.

> Before presenting somewhere without internet, run the project once so `models/yolov8n.pt` exists.

## Configuration (`config/config.yaml`, `detection:` section)

| Setting | Current value | Meaning |
|---|---|---|
| `model_path` | `models/yolov8n.pt` | Weights file to load |
| `device` | `cpu` | `cpu`, or `cuda` with an NVIDIA GPU and a CUDA build of PyTorch (the installed one is CPU-only) |
| `confidence_threshold` | `0.5` | Minimum YOLO confidence for a detection |
| `target_classes` | 14 classes | Only these COCO classes are reported (empty list = all 80) |

Camera-linked distance estimates also need a typical real size for each detected class
(`sensors.simulation.camera_linked.object_sizes_m` in the config, plus built-in defaults in
`src/sensors/vision_distance_estimator.py`).

## Using Different Weights

1. Place the weights file in this folder.
2. Set `detection.model_path` to it.
3. Check `detection.target_classes`: names must match the model's class names exactly. Names the model
   does not know are ignored with a warning in `logs/system.log`.

Larger YOLO variants are more accurate but slower; on this project's laptop CPU the nano model already
takes about 145 ms per frame (see *Limitations* in the main README).
