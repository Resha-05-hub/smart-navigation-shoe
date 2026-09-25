"""Selects and connects the frame source: webcam, video/image file, or Raspberry Pi camera."""

from pathlib import Path
from typing import Any, Dict, Optional

from .camera_interface import CameraInterface
from .webcam_camera import WebcamCamera
from .video_file_camera import VideoFileCamera
from .raspberry_pi_camera import RaspberryPiCamera
from ..utils.logger import get_logger

logger = get_logger("camera_factory")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_source_path(source: str) -> Path:
    """Resolves a relative source path against the working directory, then the project root."""
    path = Path(source)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def create_video_camera(source: str, cam_cfg: Dict[str, Any]) -> VideoFileCamera:
    """Builds a VideoFileCamera using the camera section of the config."""
    return VideoFileCamera(
        source=str(resolve_source_path(source)),
        loop=cam_cfg.get("loop", True),
        realtime=cam_cfg.get("realtime", True),
        image_duration_s=cam_cfg.get("image_duration_s", 3.0),
        max_width=cam_cfg.get("width", 640),
    )


def open_camera(config: Dict[str, Any], source: Optional[str] = None) -> Optional[CameraInterface]:
    """Returns a connected frame source, or None if nothing could be opened.

    Order:
    1. An explicit source (e.g. --source) or camera.type "video" -> camera.video_source.
    2. The webcam (or Raspberry Pi camera when camera.type is "raspberry_pi").
    3. camera.fallback_source, if the live camera could not be opened.
    """
    cam_cfg = config.get("camera", {})
    cam_type = cam_cfg.get("type", "webcam")

    if source is None and cam_type == "video":
        source = cam_cfg.get("video_source") or None
        if source is None:
            logger.error("camera.type is 'video' but camera.video_source is empty.")
            return None

    if source:
        camera: CameraInterface = create_video_camera(source, cam_cfg)
        return camera if camera.connect() else None

    if cam_type == "raspberry_pi":
        camera = RaspberryPiCamera(
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30),
        )
    else:
        camera = WebcamCamera(
            device_id=cam_cfg.get("device_id", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30),
            simulation_fallback=False,
        )
    if camera.connect():
        return camera

    fallback = cam_cfg.get("fallback_source") or None
    if fallback:
        logger.warning(f"Live camera unavailable; falling back to '{fallback}'.")
        fallback_camera = create_video_camera(fallback, cam_cfg)
        if fallback_camera.connect():
            return fallback_camera
    return None
