"""Unit tests for video/image file input and camera source selection (webcam fallback)."""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.camera.video_file_camera import VideoFileCamera
from src.camera.camera_factory import open_camera, resolve_source_path, PROJECT_ROOT
from src.camera.webcam_camera import WebcamCamera
from src.core.enums import CameraStatus


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def write_video(path, frames: int = 10, fps: float = 10.0, size=(320, 240)) -> None:
    """Writes a video whose frame i is filled with gray level i * 20 (frame index recoverable)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), i * 20, dtype=np.uint8))
    writer.release()


def frame_number(frame) -> int:
    return int(round(float(frame.mean()) / 20.0))


@pytest.fixture
def video_path(tmp_path):
    path = tmp_path / "clip.avi"
    write_video(path)
    return path


def test_video_plays_in_order_without_realtime(video_path):
    camera = VideoFileCamera(str(video_path), loop=False, realtime=False)
    assert camera.connect()
    assert camera.is_connected
    assert camera.source_description == "VIDEO: clip.avi"

    numbers = []
    while True:
        ok, frame = camera.read_frame()
        if not ok:
            break
        numbers.append(frame_number(frame))

    assert numbers == list(range(10))
    assert not camera.is_opened()  # Non-looping source ends cleanly


def test_video_loops(video_path):
    camera = VideoFileCamera(str(video_path), loop=True, realtime=False)
    camera.connect()
    numbers = [frame_number(camera.read_frame()[1]) for _ in range(13)]
    assert numbers == list(range(10)) + [0, 1, 2]
    assert camera.is_opened()


def test_realtime_skips_frames_when_processing_is_slow(video_path):
    clock = FakeClock()
    camera = VideoFileCamera(str(video_path), loop=False, realtime=True, clock=clock)
    camera.connect()

    assert frame_number(camera.read_frame()[1]) == 0
    clock.now = 0.55  # 10 fps source: frame 5 is due
    assert frame_number(camera.read_frame()[1]) == 5


def test_realtime_repeats_frame_when_processing_is_fast(video_path):
    clock = FakeClock()
    camera = VideoFileCamera(str(video_path), loop=False, realtime=True, clock=clock)
    camera.connect()

    assert frame_number(camera.read_frame()[1]) == 0
    clock.now = 0.05  # Next frame not due until 0.1 s
    assert frame_number(camera.read_frame()[1]) == 0
    clock.now = 0.1
    assert frame_number(camera.read_frame()[1]) == 1


def test_playback_starts_at_first_read_not_connect(video_path):
    """Slow startup work (e.g. loading YOLO) between connect() and the first read must not skip frames."""
    clock = FakeClock()
    camera = VideoFileCamera(str(video_path), loop=False, realtime=True, clock=clock)
    camera.connect()

    clock.now = 11.0  # Model loading took 11 s
    assert frame_number(camera.read_frame()[1]) == 0
    clock.now = 11.3
    assert frame_number(camera.read_frame()[1]) == 3


def test_wide_video_downscaled_to_max_width(tmp_path):
    path = tmp_path / "wide.avi"
    write_video(path, frames=2, size=(1280, 720))
    camera = VideoFileCamera(str(path), realtime=False, max_width=640)
    camera.connect()
    _, frame = camera.read_frame()
    assert frame.shape == (360, 640, 3)


def test_image_folder_cycles_by_duration(tmp_path):
    for i, name in enumerate(["b.png", "a.png", "notes.txt"]):
        path = tmp_path / name
        if name.endswith(".png"):
            cv2.imwrite(str(path), np.full((48, 64, 3), (i + 1) * 20, dtype=np.uint8))
        else:
            path.write_text("ignored")

    clock = FakeClock()
    camera = VideoFileCamera(str(tmp_path), loop=True, image_duration_s=2.0, clock=clock)
    assert camera.connect()
    assert camera.source_description.startswith("IMAGES")

    # Sorted by name: a.png (value 40) then b.png (value 20)
    assert frame_number(camera.read_frame()[1]) == 2
    clock.now = 2.5
    assert frame_number(camera.read_frame()[1]) == 1
    clock.now = 4.5  # Loops back to a.png
    assert frame_number(camera.read_frame()[1]) == 2


def test_single_image_ends_when_not_looping(tmp_path):
    path = tmp_path / "still.jpg"
    cv2.imwrite(str(path), np.zeros((48, 64, 3), dtype=np.uint8))
    clock = FakeClock()
    camera = VideoFileCamera(str(path), loop=False, image_duration_s=1.0, clock=clock)
    camera.connect()

    assert camera.read_frame()[0]
    clock.now = 1.5
    assert camera.read_frame() == (False, None)
    assert camera.status == CameraStatus.DISCONNECTED


def test_missing_or_empty_sources_fail(tmp_path):
    assert not VideoFileCamera(str(tmp_path / "missing.mp4")).connect()
    assert not VideoFileCamera(str(tmp_path)).connect()  # Folder without images


# --------------------------------------------------------------------- factory


def test_explicit_source_overrides_webcam(video_path):
    with patch.object(WebcamCamera, "connect") as webcam_connect:
        camera = open_camera({"camera": {"type": "webcam"}}, source=str(video_path))
    webcam_connect.assert_not_called()
    assert isinstance(camera, VideoFileCamera)
    assert camera.is_connected


def test_video_type_uses_configured_source(video_path):
    camera = open_camera({"camera": {"type": "video", "video_source": str(video_path)}})
    assert isinstance(camera, VideoFileCamera)
    assert open_camera({"camera": {"type": "video", "video_source": ""}}) is None


def test_webcam_failure_falls_back_to_video(video_path):
    config = {"camera": {"type": "webcam", "fallback_source": str(video_path)}}
    with patch.object(WebcamCamera, "connect", return_value=False):
        camera = open_camera(config)
    assert isinstance(camera, VideoFileCamera)
    assert camera.source_description == "VIDEO: clip.avi"


def test_webcam_success_ignores_fallback(video_path):
    config = {"camera": {"type": "webcam", "fallback_source": str(video_path)}}
    with patch.object(WebcamCamera, "connect", return_value=True):
        camera = open_camera(config)
    assert isinstance(camera, WebcamCamera)


def test_no_fallback_returns_none():
    with patch.object(WebcamCamera, "connect", return_value=False):
        assert open_camera({"camera": {"type": "webcam", "fallback_source": ""}}) is None


def test_relative_source_resolves_to_project_root(tmp_path, monkeypatch):
    """Launched from another directory, config-relative paths still point into the project."""
    monkeypatch.chdir(tmp_path)
    assert resolve_source_path("demo/approach_demo.mp4") == PROJECT_ROOT / "demo" / "approach_demo.mp4"

    (tmp_path / "local.mp4").write_bytes(b"")
    assert resolve_source_path("local.mp4") == Path("local.mp4")  # Existing relative path kept as-is
