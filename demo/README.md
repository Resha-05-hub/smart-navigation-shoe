# Demo Media

Video and image inputs for demonstrating the Smart Navigation Shoe without a live webcam.

## Generated demo clip

`approach_demo.mp4` is created locally (it is not stored in git):

```bash
venvv\Scripts\python.exe scripts/make_demo_video.py
```

It digitally zooms into the Ultralytics sample street photo (`bus.jpg`) so a pedestrian appears to
walk toward the user. With camera-linked sensors, the pipeline goes
CAUTION → WARNING → DANGER → WARNING → CAUTION. Every frame is labelled
"SIMULATED APPROACH (zoomed sample photo)".

## Using a source

```bash
venvv\Scripts\python.exe -m src.main --mode dashboard --source demo/approach_demo.mp4
venvv\Scripts\python.exe -m src.main --mode dashboard --source path/to/your_recording.mp4
venvv\Scripts\python.exe -m src.main --mode dashboard --source path/to/photo_folder
```

`--source` accepts a video file, a single image, or a folder of images (played in filename order,
`camera.image_duration_s` seconds each). Without `--source`, the webcam is used, and
`camera.fallback_source` in `config/config.yaml` plays automatically if the webcam cannot be opened.

## Recording your own clip

For the most convincing fallback, record a short phone or webcam video of someone walking toward
the camera in the demo room (hold the camera at shoe/knee height, landscape, 640x480 or larger)
and place it here. Video files in this folder are ignored by git.
