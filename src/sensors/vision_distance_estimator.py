"""Monocular distance estimation used to drive SIMULATED distance sensors from the camera view.

Disclaimer:
This is an approximation for the software-only demonstration. It estimates distance from the
pinhole camera model using typical real-world object sizes, which is NOT a substitute for the
physical HC-SR04 ultrasonic sensors of the real shoe. YOLO itself still provides no distance;
this estimator only feeds the simulated sensors in camera-linked mode.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import DetectionItem
from ..core.enums import ObstacleZone

# Typical real-world (height_m, width_m) of COCO classes relevant to navigation
DEFAULT_OBJECT_SIZES_M: Dict[str, Tuple[float, float]] = {
    "person": (1.70, 0.45),
    "bicycle": (1.00, 1.70),
    "car": (1.50, 1.80),
    "motorcycle": (1.10, 2.00),
    "bus": (3.00, 2.50),
    "truck": (3.00, 2.50),
    "dog": (0.55, 0.70),
    "cat": (0.30, 0.45),
    "chair": (0.90, 0.50),
    "couch": (0.85, 2.00),
    "potted plant": (0.60, 0.40),
    "dining table": (0.75, 1.20),
    "bench": (0.85, 1.50),
    "suitcase": (0.65, 0.45),
    "backpack": (0.50, 0.35),
}


class VisionDistanceEstimator:
    """Estimates object distance from bounding-box size with the pinhole camera model.

    distance = real_size_m * focal_length_px / size_px

    Height and width each give an estimate. A box cut off by the frame edge looks smaller than
    the object really is, which overestimates distance, so the smaller (closer, safer) estimate is used.
    """

    def __init__(
        self,
        horizontal_fov_deg: float = 60.0,
        object_sizes_m: Optional[Dict[str, Tuple[float, float]]] = None,
        default_size_m: Tuple[float, float] = (1.0, 0.5),
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.horizontal_fov_deg = horizontal_fov_deg
        self.object_sizes_m: Dict[str, Tuple[float, float]] = dict(DEFAULT_OBJECT_SIZES_M)
        self.default_size_m = default_size_m

        if config:
            cam_cfg = config.get("sensors", {}).get("simulation", {}).get("camera_linked", {})
            self.horizontal_fov_deg = cam_cfg.get("horizontal_fov_deg", horizontal_fov_deg)
            if "default_size_m" in cam_cfg:
                self.default_size_m = tuple(cam_cfg["default_size_m"])  # type: ignore[assignment]
            for label, size in (cam_cfg.get("object_sizes_m") or {}).items():
                self.object_sizes_m[label] = (float(size[0]), float(size[1]))

        if object_sizes_m:
            self.object_sizes_m.update(object_sizes_m)

    def focal_length_px(self, frame_width: int) -> float:
        """Focal length in pixels from the horizontal field of view (square pixels assumed)."""
        return (frame_width / 2.0) / math.tan(math.radians(self.horizontal_fov_deg) / 2.0)

    def estimate_distance_m(self, detection: DetectionItem, frame_width: int, frame_height: int) -> Optional[float]:
        """Estimated distance to one detected object, or None if its box is unusable."""
        bbox = detection.bbox
        if bbox is None or frame_width <= 0 or frame_height <= 0:
            return None

        width_px, height_px = bbox.width, bbox.height
        if bbox.xmax <= 1.0 and bbox.ymax <= 1.0:  # Normalized coordinates
            width_px *= frame_width
            height_px *= frame_height
        if width_px < 1.0 or height_px < 1.0:
            return None

        real_height_m, real_width_m = self.object_sizes_m.get(detection.label, self.default_size_m)
        focal_px = self.focal_length_px(frame_width)
        return min(real_height_m * focal_px / height_px, real_width_m * focal_px / width_px)

    def zone_distances_cm(
        self,
        detections: List[DetectionItem],
        frame_width: int,
        frame_height: int,
    ) -> Dict[ObstacleZone, Optional[float]]:
        """Nearest estimated distance (cm) per zone, like an ultrasonic sensor; None = nothing seen."""
        nearest: Dict[ObstacleZone, Optional[float]] = {
            ObstacleZone.LEFT: None,
            ObstacleZone.CENTER: None,
            ObstacleZone.RIGHT: None,
        }
        for detection in detections:
            if detection.zone not in nearest:
                continue
            distance_m = self.estimate_distance_m(detection, frame_width, frame_height)
            if distance_m is None:
                continue
            current = nearest[detection.zone]
            distance_cm = distance_m * 100.0
            nearest[detection.zone] = distance_cm if current is None else min(current, distance_cm)
        return nearest
