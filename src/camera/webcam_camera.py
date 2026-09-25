"""Webcam camera implementation for laptop/PC execution."""

import os
from typing import Tuple, Optional, Any
from .camera_interface import CameraInterface
from ..core.enums import CameraStatus
from ..utils.logger import get_logger

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

logger = get_logger("webcam_camera")


class WebcamCamera(CameraInterface):
    """Webcam implementation of CameraInterface for laptop/simulation execution."""

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        simulation_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.simulation_fallback = simulation_fallback
        self._cap: Any = None

    def connect(self) -> bool:
        """Connects to physical webcam device via OpenCV."""
        logger.info(f"Connecting to webcam device ID {self.device_id}...")
        if cv2 is not None:
            try:
                # Try opening camera with DirectShow backend on Windows if available, or default backend
                if os.name == 'nt' and hasattr(cv2, 'CAP_DSHOW'):
                    self._cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
                else:
                    self._cap = cv2.VideoCapture(self.device_id)

                if self._cap is not None and self._cap.isOpened():
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self._cap.set(cv2.CAP_PROP_FPS, self.fps)
                    self.status = CameraStatus.CONNECTED
                    logger.info("Webcam connected successfully.")
                    return True
                else:
                    logger.warning(f"Webcam device ID {self.device_id} could not be opened.")
            except Exception as e:
                logger.error(f"Error connecting to webcam device ID {self.device_id}: {e}")

        if self.simulation_fallback:
            logger.info("Enabling synthetic simulation frame fallback for testing.")
            self.status = CameraStatus.CONNECTED
            return True

        self.status = CameraStatus.ERROR
        return False

    def read_frame(self) -> Tuple[bool, Optional[Any]]:
        """Reads a live frame from the webcam. If unavailable, generates synthetic frame if simulation_fallback is enabled."""
        if self._cap is not None and cv2 is not None and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                return True, frame

        if self.simulation_fallback and self.status == CameraStatus.CONNECTED:
            if np is not None:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if cv2 is not None:
                    cv2.putText(
                        frame,
                        "SIMULATION FRAME (No Physical Camera)",
                        (40, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                return True, frame
            else:
                return True, f"SIMULATION_FRAME_{self.width}x{self.height}"

        return False, None

    def release(self) -> None:
        """Releases the camera device hardware handle."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning(f"Error releasing camera resource: {e}")
            self._cap = None
        self.status = CameraStatus.DISCONNECTED
        logger.info("Webcam released.")

    def is_opened(self) -> bool:
        """Checks if webcam handle is active or operating in simulation fallback mode."""
        if self._cap is not None and hasattr(self._cap, 'isOpened') and self._cap.isOpened():
            return True
        return self.simulation_fallback and self.status == CameraStatus.CONNECTED
