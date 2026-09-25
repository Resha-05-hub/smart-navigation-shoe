"""Video file / image input implementing CameraInterface, used as a webcam alternative or fallback."""

import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from .camera_interface import CameraInterface
from ..core.enums import CameraStatus
from ..utils.logger import get_logger

try:
    import cv2
except ImportError:
    cv2 = None

logger = get_logger("video_file_camera")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class VideoFileCamera(CameraInterface):
    """Plays a video file, a single image, or a folder of images as if it were a live camera.

    - realtime=True keeps playback at the source frame rate: when processing (e.g. YOLO on CPU)
      is slower than the video, frames are skipped instead of the video running in slow motion.
    - Images are shown for image_duration_s each (a folder plays in filename order).
    - loop=True restarts the source when it ends; otherwise read_frame() fails and is_opened() turns False.
    """

    def __init__(
        self,
        source: str,
        loop: bool = True,
        realtime: bool = True,
        image_duration_s: float = 3.0,
        max_width: Optional[int] = 640,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.source = Path(source)
        self.loop = loop
        self.realtime = realtime
        self.image_duration_s = image_duration_s
        self.max_width = max_width
        self._clock = clock
        self._cap: Any = None
        self._images: List[Path] = []
        self._image_cache: Tuple[int, Any] = (-1, None)
        self._last_frame: Any = None
        self.fps: float = 0.0
        self._frame_index = 0
        self._start_time: Optional[float] = None

    @property
    def source_description(self) -> str:
        kind = "IMAGES" if self._images else "VIDEO"
        return f"{kind}: {self.source.name}"

    def connect(self) -> bool:
        """Opens the video file or collects the image files."""
        logger.info(f"Opening video/image source '{self.source}'...")
        if cv2 is None:
            logger.error("OpenCV is not installed; cannot open video/image sources.")
            self.status = CameraStatus.ERROR
            return False

        if self.source.is_dir():
            self._images = sorted(p for p in self.source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        elif self.source.is_file() and self.source.suffix.lower() in IMAGE_EXTENSIONS:
            self._images = [self.source]
        elif self.source.is_file():
            self._cap = cv2.VideoCapture(str(self.source))
            if not self._cap.isOpened():
                logger.error(f"Could not open video file '{self.source}'.")
                self._cap = None
                self.status = CameraStatus.ERROR
                return False
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            logger.error(f"Video/image source '{self.source}' does not exist.")
            self.status = CameraStatus.ERROR
            return False

        if self.source.is_dir() and not self._images:
            logger.error(f"No images ({', '.join(sorted(IMAGE_EXTENSIONS))}) found in '{self.source}'.")
            self.status = CameraStatus.ERROR
            return False

        self._frame_index = 0
        self._last_frame = None
        self._start_time = None  # Playback starts at the first read, not while models are still loading
        self.status = CameraStatus.CONNECTED
        logger.info(f"Source opened: {self.source_description}")
        return True

    def read_frame(self) -> Tuple[bool, Optional[Any]]:
        """Returns the frame due at the current playback time."""
        if self.status != CameraStatus.CONNECTED:
            return False, None
        frame = self._read_image() if self._images else self._read_video()
        if frame is None:
            return False, None
        return True, self._fit_width(frame)

    def _elapsed_s(self) -> float:
        """Seconds since playback started (the first read)."""
        now = self._clock()
        if self._start_time is None:
            self._start_time = now
        return now - self._start_time

    def _read_image(self) -> Optional[Any]:
        index = int(self._elapsed_s() / self.image_duration_s)
        if index >= len(self._images):
            if not self.loop:
                self._end_of_source()
                return None
            index %= len(self._images)

        cached_index, cached_frame = self._image_cache
        if cached_index != index:
            cached_frame = cv2.imread(str(self._images[index]))
            if cached_frame is None:
                logger.warning(f"Could not read image '{self._images[index]}'; skipping.")
            self._image_cache = (index, cached_frame)
        return cached_frame

    def _read_video(self) -> Optional[Any]:
        if self.realtime:
            due_index = int(self._elapsed_s() * self.fps)
            if due_index < self._frame_index and self._last_frame is not None:
                return self._last_frame  # Ahead of the source frame rate: repeat the current frame
            behind = due_index - self._frame_index
            if behind > self.fps:  # Far behind: seek instead of decoding every skipped frame
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, due_index)
                self._frame_index = due_index
            else:
                for _ in range(max(0, behind)):
                    if not self._cap.grab():
                        break
                    self._frame_index += 1

        ret, frame = self._cap.read()
        if not ret or frame is None:
            if not self.loop:
                self._end_of_source()
                return None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_index = 0
            self._last_frame = None
            self._start_time = self._clock()
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._end_of_source()
                return None

        self._frame_index += 1
        self._last_frame = frame
        return frame

    def _fit_width(self, frame: Any) -> Any:
        """Downscales wide sources to max_width (keeping aspect ratio) so overlays stay legible."""
        if self.max_width and frame.shape[1] > self.max_width:
            scale = self.max_width / frame.shape[1]
            frame = cv2.resize(frame, (self.max_width, int(round(frame.shape[0] * scale))))
        return frame

    def _end_of_source(self) -> None:
        logger.info(f"End of source reached: {self.source_description}")
        self.status = CameraStatus.DISCONNECTED

    def release(self) -> None:
        """Closes the video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.status = CameraStatus.DISCONNECTED

    def is_opened(self) -> bool:
        """True while the source is playing."""
        return self.status == CameraStatus.CONNECTED
