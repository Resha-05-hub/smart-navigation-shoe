"""Creates demo/approach_demo.mp4: a simulated walk toward pedestrians, for demos without a webcam.

The clip digitally zooms into Ultralytics' bundled sample street photo (bus.jpg). Apparent size is
inversely proportional to distance, so the zoom is driven by a distance that decreases linearly
over time, which looks like walking toward the person at a steady pace. It is labelled on-screen
as a simulation.

Usage:
    venvv\\Scripts\\python.exe scripts/make_demo_video.py [--output demo/approach_demo.mp4]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import ASSETS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WIDTH, HEIGHT, FPS = 640, 480, 15
FOCUS_XY = (282.0, 630.0)  # Centre of the middle pedestrian in bus.jpg (810 x 1080)
CLOSE_FOCUS_Y = 470.0      # Head and shoulders: the zoom drifts here so YOLO still sees a person up close
START_M, END_M = 2.6, 0.35  # Apparent walking distance at zoom 1.0 and at the closest point

# (duration_s, from_m, to_m): hold, approach, hold close, back away
TIMELINE = [
    (2.0, START_M, START_M),
    (9.0, START_M, END_M),
    (2.5, END_M, END_M),
    (2.5, END_M, START_M),
]


def crop_for_distance(image: np.ndarray, distance_m: float) -> np.ndarray:
    """Crops a 4:3 window around the pedestrian with zoom = START_M / distance, resized to WIDTH x HEIGHT."""
    img_h, img_w = image.shape[:2]
    zoom = START_M / distance_m
    crop_w = img_w / zoom
    crop_h = crop_w * HEIGHT / WIDTH

    closeness = (zoom - 1.0) / (START_M / END_M - 1.0)  # 0 at the start distance, 1 at the closest
    focus_y = FOCUS_XY[1] + (CLOSE_FOCUS_Y - FOCUS_XY[1]) * closeness
    cx = min(max(FOCUS_XY[0], crop_w / 2), img_w - crop_w / 2)
    cy = min(max(focus_y, crop_h / 2), img_h - crop_h / 2)
    x0, y0 = int(round(cx - crop_w / 2)), int(round(cy - crop_h / 2))
    crop = image[y0:y0 + int(round(crop_h)), x0:x0 + int(round(crop_w))]
    return cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="demo/approach_demo.mp4", help="Output video path")
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    if image is None:
        raise SystemExit(f"Could not read sample image {ASSETS / 'bus.jpg'}")

    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise SystemExit(f"Could not create video writer for {output}")

    frames = 0
    for duration_s, from_m, to_m in TIMELINE:
        steps = int(round(duration_s * FPS))
        for i in range(steps):
            distance = from_m + (to_m - from_m) * (i / max(steps - 1, 1))
            frame = crop_for_distance(image, distance)
            cv2.putText(frame, "SIMULATED APPROACH (zoomed sample photo)", (10, HEIGHT - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            frames += 1

    writer.release()
    print(f"Wrote {output} ({frames} frames, {frames / FPS:.1f} s at {FPS} fps, {WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
