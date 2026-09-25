"""Ultralytics YOLO object detector implementation."""

from typing import Optional, List, Any
import time

from .detector_interface import DetectorInterface
from .detection_result import DetectionResult
from ..core.models import DetectionItem, BoundingBox, FusedObstacle
from ..utils.helpers import determine_zone
from ..utils.logger import get_logger

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

logger = get_logger("yolo_detector")


class YoloDetector(DetectorInterface):
    """YOLOv8 implementation of DetectorInterface using Ultralytics library."""

    SUPPORTED_CLASSES: List[str] = [
        "person", "bicycle", "car", "motorcycle", "bus", "truck",
        "dog", "cat", "chair", "couch", "potted plant", "dining table",
        "stairs", "door", "bench", "fire hydrant", "stop sign"
    ]

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None
        self._is_loaded = False

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Loads Ultralytics YOLO model weights. If model file does not exist, Ultralytics downloads it automatically."""
        path_to_load = model_path or self.model_path
        logger.info(f"Loading YOLO model weights from: {path_to_load}")

        try:
            from ultralytics import YOLO
            self._model = YOLO(path_to_load)
            self._is_loaded = True
            logger.info("YOLO model initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model weights at '{path_to_load}': {e}")
            self._is_loaded = False
            return False

    def detect(self, frame: Any) -> DetectionResult:
        """Runs real-time object detection on frame. Returns DetectionResult."""
        start_time = time.time()

        height, width = 480, 640
        if hasattr(frame, "shape"):
            height, width = frame.shape[0], frame.shape[1]

        if not self._is_loaded or self._model is None or frame is None:
            return DetectionResult(
                detections=[],
                frame_width=width,
                frame_height=height,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        detections: List[DetectionItem] = []

        try:
            results = self._model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            if results and len(results) > 0:
                res = results[0]
                boxes = res.boxes
                names = res.names if hasattr(res, 'names') else getattr(self._model, 'names', {})

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                        conf = float(box.conf[0].item())
                        cls_id = int(box.cls[0].item())

                        if isinstance(names, dict):
                            label_name = names.get(cls_id, f"object_{cls_id}")
                        elif isinstance(names, list) and cls_id < len(names):
                            label_name = names[cls_id]
                        else:
                            label_name = str(cls_id)

                        xmin, ymin, xmax, ymax = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                        bbox = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

                        center_x_norm = bbox.center_x / width if width > 0 else 0.5
                        zone = determine_zone(center_x_norm)

                        detection_item = DetectionItem(
                            label=label_name,
                            confidence=round(conf, 4),
                            bbox=bbox,
                            zone=zone,
                            estimated_distance_m=0.0,
                            track_id=-1,
                        )
                        detections.append(detection_item)
        except Exception as e:
            logger.error(f"Error executing YOLO inference: {e}")

        processing_time = (time.time() - start_time) * 1000

        return DetectionResult(
            detections=detections,
            frame_width=width,
            frame_height=height,
            processing_time_ms=processing_time,
        )

    def draw_detections(
        self,
        frame: Any,
        result: DetectionResult,
        show_confidence: bool = True,
        show_box: bool = True,
        box_color: tuple = (0, 255, 0),
    ) -> Any:
        """Annotates frame with bounding boxes, object labels, and confidence percentages."""
        if frame is None or not hasattr(frame, "copy") or cv2 is None:
            return frame

        annotated_frame = frame.copy()

        for item in result.detections:
            bbox = item.bbox
            if bbox is None:
                continue

            xmin, ymin, xmax, ymax = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)

            if show_box:
                cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), box_color, 2)

            if show_confidence:
                label_str = f"{item.label} {int(item.confidence * 100)}%"
            else:
                label_str = item.label

            (text_w, text_h), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_ymin = max(ymin, text_h + 10)
            cv2.rectangle(
                annotated_frame,
                (xmin, text_ymin - text_h - 4),
                (xmin + text_w + 6, text_ymin + baseline - 2),
                box_color,
                -1,
            )
            cv2.putText(
                annotated_frame,
                label_str,
                (xmin + 3, text_ymin - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated_frame

    def draw_fused_obstacles(
        self,
        frame: Any,
        fused_obstacles: List[FusedObstacle],
        box_color: tuple = (0, 255, 0),
    ) -> Any:
        """Annotates frame with bounding boxes, object names, confidence %, zone, sensor distance cm, and risk level."""
        if frame is None or not hasattr(frame, "copy") or cv2 is None:
            return frame

        annotated_frame = frame.copy()

        for obs in fused_obstacles:
            bbox = obs.bbox
            if bbox is None:
                continue

            xmin, ymin, xmax, ymax = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
            cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), box_color, 2)

            conf_str = f"{int(obs.confidence * 100)}%" if obs.confidence <= 1.0 else f"{obs.confidence}%"
            label_str = f"{obs.label} {conf_str} | {obs.zone.value} | {int(obs.sensor_distance_cm)}cm | {obs.risk_level.value}"

            (text_w, text_h), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_ymin = max(ymin, text_h + 10)
            cv2.rectangle(
                annotated_frame,
                (xmin, text_ymin - text_h - 4),
                (xmin + text_w + 6, text_ymin + baseline - 2),
                box_color,
                -1,
            )
            cv2.putText(
                annotated_frame,
                label_str,
                (xmin + 3, text_ymin - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated_frame

    def is_loaded(self) -> bool:
        """Returns True if YOLO model is initialized."""
        return self._is_loaded
