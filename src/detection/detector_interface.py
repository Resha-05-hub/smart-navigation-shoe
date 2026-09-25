"""Abstract Base Class for object detectors."""

from abc import ABC, abstractmethod
from typing import Optional, Any

from .detection_result import DetectionResult


class DetectorInterface(ABC):
    """Interface defining object detection operations for Smart Navigation Shoe."""

    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Loads object detection model weights.

        Args:
            model_path: Path to model file.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        pass

    @abstractmethod
    def detect(self, frame: Any) -> DetectionResult:
        """Performs object detection on an image frame.

        Args:
            frame: OpenCV BGR image array (height x width x 3) or frame object.

        Returns:
            DetectionResult container with detected items.
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Checks if the detection model is initialized and ready.

        Returns:
            True if ready, False otherwise.
        """
        pass
