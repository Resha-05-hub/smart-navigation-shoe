"""Shared visualization helpers: risk colors and frame-rate measurement."""

import time
from typing import Callable, Dict, Optional, Tuple

BGR = Tuple[int, int, int]

# OpenCV BGR colors per risk level (legacy levels map onto the same scale)
RISK_COLORS_BGR: Dict[str, BGR] = {
    "SAFE": (0, 200, 0),
    "LOW": (0, 220, 220),
    "CAUTION": (0, 220, 220),
    "MEDIUM": (0, 165, 255),
    "WARNING": (0, 165, 255),
    "HIGH": (0, 165, 255),
    "DANGER": (0, 0, 255),
    "CRITICAL": (0, 0, 255),
}
FAULT_COLOR_BGR: BGR = (128, 128, 128)


def risk_color(risk_level: str) -> BGR:
    """BGR color for a risk level name (white if unknown)."""
    return RISK_COLORS_BGR.get(str(risk_level), (255, 255, 255))


def label_text_color(risk_level: str) -> BGR:
    """Readable text color on top of a risk-colored label background."""
    return (255, 255, 255) if str(risk_level) in ("DANGER", "CRITICAL") else (0, 0, 0)


class FrameRateMeter:
    """Smoothed frames-per-second of a processing loop (exponential moving average)."""

    def __init__(self, smoothing: float = 0.1, clock: Callable[[], float] = time.monotonic) -> None:
        self.smoothing = smoothing
        self._clock = clock
        self._last: Optional[float] = None
        self.fps: float = 0.0

    def tick(self) -> float:
        """Call once per processed frame; returns the current smoothed FPS."""
        now = self._clock()
        if self._last is not None:
            interval = now - self._last
            if interval > 0:
                instant = 1.0 / interval
                self.fps = instant if self.fps == 0.0 else self.fps + self.smoothing * (instant - self.fps)
        self._last = now
        return self.fps
