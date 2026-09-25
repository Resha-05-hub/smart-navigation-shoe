"""Abstract Base Class for camera inputs."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any

try:
    import numpy as np
except ImportError:
    np = None

from ..core.enums import CameraStatus


class CameraInterface(ABC):
    """Interface defining standard camera operations for Smart Navigation Shoe."""

    def __init__(self) -> None:
        self.status: CameraStatus = CameraStatus.UNINITIALIZED

    @abstractmethod
    def connect(self) -> bool:
        """Initializes connection to the camera hardware or stream.

        Returns:
            True if connection succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[Any]]:
        """Captures a single frame from the camera.

        Returns:
            Tuple of (success_flag, image_frame_numpy_array).
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Releases the camera resource and cleans up."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Checks if the camera stream is actively open.

        Returns:
            True if active, False otherwise.
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Checks if the camera stream is connected and active.

        Returns:
            True if connected and open, False otherwise.
        """
        return self.status == CameraStatus.CONNECTED and self.is_opened()

