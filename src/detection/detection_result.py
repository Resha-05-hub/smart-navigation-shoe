"""Encapsulation of detection results for a single image frame."""

from dataclasses import dataclass, field
from typing import List
import time

from ..core.models import DetectionItem


@dataclass
class DetectionResult:
    """Holds bounding boxes, class labels, and metadata from object detection."""
    detections: List[DetectionItem] = field(default_factory=list)
    frame_width: int = 640
    frame_height: int = 480
    processing_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def count(self) -> int:
        """Return the number of detected objects."""
        return len(self.detections)
