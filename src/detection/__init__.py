"""Detection interfaces and models module for computer vision."""

from .detection_result import DetectionResult
from .detector_interface import DetectorInterface
from .yolo_detector import YoloDetector

__all__ = ["DetectionResult", "DetectorInterface", "YoloDetector"]
