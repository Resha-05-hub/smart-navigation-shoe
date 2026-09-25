# AI-Powered Smart Navigation Shoe for Visually Impaired People Using Computer Vision and Sensor Fusion

## Project Overview

The **AI-Powered Smart Navigation Shoe** is an assistive-device prototype that gives real-time obstacle
detection, collision-risk warnings, and directional guidance to visually impaired users. It combines
**computer vision (YOLOv8)** with **distance sensing** through **sensor fusion**, and responds with
**voice instructions** and **directional haptic vibration**.

This repository is a **software-only demonstration**, approved by our faculty: there is no Raspberry Pi,
no ultrasonic sensors, and no vibration motors. The camera and object detection are real; the shoe's
hardware is simulated in software and clearly labelled as such everywhere it appears.

> [!WARNING]
> **Research prototype disclaimer.** This is an educational project. It is **not** a safety-certified,
> medical-grade, or life-critical navigation system and must not be relied on for real navigation.
> Distances in this demonstration come from **simulated** sensors, including distances **estimated** from
> the camera image; they are approximations, not measurements.

---

## Problem Statement

Visually impaired individuals face significant daily mobility challenges:
* Standard white canes cannot detect elevated or hanging obstacles (open cabinet doors, branches, truck beds).
* Guide dogs require extensive training and maintenance and are not universally accessible.
* Many wearable sensors signal distance but cannot tell *what* an obstacle is — a curtain or a person or a vehicle.

## Proposed Solution

A shoe-mounted system that:
1. **Camera + YOLOv8** identifies obstacles and which zone they are in (LEFT / CENTER / RIGHT).
2. **Three ultrasonic sensors** on the toe measure distance per zone — including low obstacles the camera misses.
3. **Sensor fusion** attaches each zone's distance to what the camera sees.
4. **Risk and direction analysis** grades each hazard and chooses a safe way to go.
5. **Feedback**: directional vibration on the side of the obstacle, and voice instructions such as
   *"Warning. Obstacle ahead. Slow down. Move slightly left."*
6. **Dashboard and event log** show and record everything for evaluation.

---

## What Is Real vs. Simulated

| Component | In this demo | Notes |
|---|---|---|
| Camera | **Real** (laptop webcam) | Or a video file / image folder via `--source`; automatic fallback if the webcam fails |
| Object detection | **Real** YOLOv8n (Ultralytics) | 80 COCO classes, filtered to navigation-relevant ones |
| Object tracking | **Real** | IoU tracker with stable IDs (`person #3`) |
| Ultrasonic sensors ×3 | **Simulated** | Driven by camera-linked estimates, keyboard, or scripted scenarios |
| Sensor fusion, risk, direction | **Real logic** | Same code the hardware version would run |
| Voice alerts | **Real** (offline Windows TTS via pyttsx3) | |
| Vibration motors ×2 | **Simulated** | Drawn on screen, blinking in the real pulse rhythm |
| Raspberry Pi GPIO | **Placeholder** | Interfaces and pin maps are ready for Phase 8 |

---

## Quick Start (Windows)

```powershell
# 1. Install dependencies into the project virtual environment (Python 3.11)
py -3.11 -m venv venvv                      # only if venvv does not exist yet
venvv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Create the demo video used as a webcam fallback (about 5 MB, not stored in git)
venvv\Scripts\python.exe scripts\make_demo_video.py

# 3. Run the full demonstration (dashboard mode is the default)
venvv\Scripts\python.exe -m src.main
```

The first launch loads YOLO (about 5–10 s). Press **`q`** or **`ESC`** in the video window to exit.

---

## Execution Modes

| Command | What it shows |
|---|---|
| `venvv\Scripts\python.exe -m src.main` | **Dashboard** (default): video with risk-coloured boxes and zone lines, simulated shoe panel, terminal dashboard, voice, CSV logging |
| `... -m src.main --source demo\approach_demo.mp4` | Same, from a video file instead of the webcam (also accepts an image or a folder of images) |
| `... -m src.main --scenario approaching_obstacle` | Dashboard with a scripted sensor scenario running |
| `... -m src.main --mode scenario` | **No camera needed**: plays every scripted scenario through the full pipeline, with the shoe panel in a window |
| `... -m src.main --mode scenario --scenario sensor_fault` | One scenario only |
| `... -m src.main --mode fusion` | Full pipeline with console output instead of the dashboard |
| `... -m src.main --mode detection` | Raw YOLO detection only |
| `... -m src.main --mode alerts` | Vibration + voice alert demonstration (no camera) |
| `... -m src.main --mode sensors` | Simulated sensor readings and validation (no camera) |
| `... -m src.main --mode architecture` | Three steps of the whole architecture, then exit |

Scenarios: `approaching_obstacle`, `passing_left_then_right`, `narrow_corridor`, `mixed_zones`, `sensor_fault`
(defined in `config/config.yaml`; add your own as keyframes).

### Keyboard Controls (dashboard and fusion windows)

Click the video window first so it receives key presses.

| Key | Action |
|---|---|
| `a` / `z` | LEFT obstacle closer / farther (10 cm per press) |
| `s` / `x` | CENTER obstacle closer / farther |
| `d` / `c` | RIGHT obstacle closer / farther |
| `v` | Toggle camera-linked sensors (distances from the camera) |
| `0` | Clear path (all sensors at maximum range) |
| `r` | Reset to the startup mode |
| `n` | Start the next scripted scenario |
| `p` | Pause / resume the scenario |
| `q` / `ESC` | Quit |

---

## Faculty Demonstration Script (about 6 minutes)

1. **Explain the scope (30 s).** Show the *What Is Real vs. Simulated* table. The side panel is titled
   "SIMULATED SHOE – software demo, no hardware".
2. **Live camera, camera-linked sensors (2 min).** Run `venvv\Scripts\python.exe -m src.main`.
   Start about 3 m from the webcam and walk toward it. Point out:
   the box turning yellow → orange → red, the CENTER cone shrinking, the motors blinking faster,
   the arrow changing to **STOP**, and the voice: *"Caution…", "Warning… Move slightly right", "Danger… Stop."*
   Step to one side: the guidance switches to steering away from that side.
3. **Obstacles the camera cannot see (1 min).** Press `s` repeatedly: the CENTER sensor reports an
   unidentified `obstacle` although YOLO sees nothing — this is why the shoe has distance sensors.
   Press `0` to clear, `v` to return to camera-linked mode.
4. **Sensor failure (1 min).** Press `q`, then run `venvv\Scripts\python.exe -m src.main --mode scenario --scenario sensor_fault`.
   The CENTER cone turns grey (FAULT) and the voice says *"Caution. Center sensor not responding."* —
   the system never claims the path is clear when it cannot know.
5. **Evidence (1 min).** Open `logs\detection_events.csv`: every spoken alert with time, object, zone,
   distance, risk, vibration, and direction. Run the test suite (below) to show 180 passing tests.

**If the webcam fails at the venue**, the demo video plays automatically (`camera.fallback_source`), or run
`--source demo\approach_demo.mp4`. `--mode scenario` needs no camera at all.

---

## How It Works

```text
Camera / video file ──► YOLOv8 detection ──► IoU object tracker
                                                   │
                  ┌────────────────────────────────┤
                  ▼                                ▼
   Simulated ultrasonic sensors ◄── camera-linked estimates / keyboard / scripted scenarios
          (LEFT, CENTER, RIGHT)
                  │
                  ▼
           Sensor fusion  ──►  Risk analysis  ──►  Direction guidance  ──►  Alert manager
     (zone distance per object;   (thresholds +       (steer / stop;          ├─ Simulated vibration
      sensor-only obstacles)       hysteresis;         hysteresis)             ├─ Voice (pyttsx3)
                                   sensor faults)                              └─ Dashboard + CSV / log
```

| Stage | Behaviour |
|---|---|
| **Detection** (`src/detection/yolo_detector.py`) | YOLOv8n on CPU (~80 ms/frame after warm-up), filtered to `detection.target_classes`, zone from the box centre |
| **Tracking** (`src/detection/object_tracker.py`) | Class-aware IoU matching; objects confirmed after 2 sightings, kept through 3 missed frames, boxes smoothed |
| **Simulated sensors** (`src/sensors/scenario_engine.py`) | *Camera-linked* (default): each zone reads the nearest tracked object's distance, estimated as `real_size × focal_px ÷ box_px` (`vision_distance_estimator.py`). *Manual*: keyboard. *Scenario*: timed keyframes, including sensor faults. ±1 cm noise mimics real ultrasonic readings |
| **Fusion** (`src/fusion/sensor_fusion.py`) | Each detection takes its zone's distance; a close reading in a zone with no detection becomes an unidentified `obstacle` |
| **Risk** (`src/decision/risk_analyzer.py`) | ≤ 50 cm DANGER, ≤ 100 cm WARNING, ≤ 200 cm CAUTION, else SAFE. Rises immediately, falls only 10 cm past a threshold (no flicker). A failed sensor raises risk to at least CAUTION |
| **Direction** (`src/decision/direction_analyzer.py`) | STOP for danger ahead; otherwise keep straight, move slightly left/right, or turn. Never steers into a zone with a failed sensor; keeps its chosen side unless the other is 30 cm clearer |
| **Alerts** (`src/alerts/alert_manager.py`) | Describes the most severe obstacle. Speaks immediately on change, repeats every 2 s while a hazard persists, suppresses A→B→A flapping, and says "Path clear" only after 1 s clear |
| **Dashboard** (`src/dashboard/`) | Video with risk-coloured boxes, zone lines, and status strip; simulated shoe panel (sensor cones, motors, direction, FPS); terminal dashboard |
| **Logging** (`src/event_logging/`) | `logs/detection_events.csv` (debounced events, spoken text, alert status, direction); `logs/system.log` (component logs) |

The per-frame wiring is in `src/pipeline.py` (`NavigationPipeline`), shared by every mode.

---

## Configuration (`config/config.yaml`)

| Setting | Default | Purpose |
|---|---|---|
| `system.mode` | `dashboard` | Mode used when `--mode` is not given |
| `camera.device_id` / `fallback_source` | `0` / `demo/approach_demo.mp4` | Webcam, and video played if it cannot be opened |
| `detection.target_classes` | person, car, chair, … | COCO classes that count as obstacles |
| `sensors.simulation.startup` | `camera` | `camera`, `manual`, or a scenario name |
| `sensors.simulation.camera_linked.horizontal_fov_deg` | `60` | Webcam field of view — tune if estimated distances read consistently off |
| `risk_analysis.thresholds_cm` | 50 / 100 / 200 | DANGER / WARNING / CAUTION limits |
| `risk_analysis.hysteresis_cm` | `10` | Margin before risk is lowered |
| `alerts.voice.cooldown_seconds` | `2.0` | Reminder interval while a hazard persists |
| `alerts.voice.enabled` | `true` | Turn speech off (e.g. in a quiet room) |
| `tracking.min_hits` | `2` | Sightings before an object is reported |
| `dashboard.show_side_panel` | `true` | Simulated shoe panel beside the video (window ≈ 900 px wide) |

---

## Running Tests

```powershell
venvv\Scripts\python.exe -m pytest
```

180 tests cover detection parsing, tracking, simulated sensors and scenarios, distance estimation, fusion,
risk hysteresis, sensor faults, direction guidance, alert debouncing, dashboard rendering, event logging,
video input, and configuration consistency. Tests use mocks and generated frames; no webcam is needed.

---

## Project Structure

```text
smart-navigation-shoe/
├── config/config.yaml            # All settings, scenarios, and thresholds
├── demo/                         # Demo media (README; generated/recorded videos are not in git)
├── logs/                         # detection_events.csv and system.log (created at runtime)
├── models/yolov8n.pt             # YOLOv8 nano weights
├── scripts/make_demo_video.py    # Builds demo/approach_demo.mp4
├── src/
│   ├── main.py                   # CLI entry point and execution modes
│   ├── pipeline.py               # NavigationPipeline shared by all modes
│   ├── camera/                   # Webcam, video/image file input, camera factory, Pi camera placeholder
│   ├── detection/                # YOLO detector and IoU object tracker
│   ├── sensors/                  # Simulated + ultrasonic sensors, scenario engine, vision distance estimator
│   ├── fusion/                   # Vision + distance sensor fusion
│   ├── decision/                 # Risk analyzer and direction analyzer
│   ├── alerts/                   # Alert manager, voice (TTS), simulated/Pi vibration
│   ├── dashboard/                # Terminal dashboard, video overlay, simulated shoe panel
│   ├── event_logging/            # CSV event log and system log
│   ├── core/                     # Shared data models and enums
│   └── utils/                    # Config loading, logging setup, visual helpers
└── tests/                        # Unit tests (pytest)
```

---

## Limitations

* **Distances are simulated.** Camera-linked distances are estimates from box size and typical object
  sizes; accuracy depends on the webcam's field of view. Wide poses (arms outstretched) read closer
  than reality — the error is on the safe side.
* **Only COCO classes are recognised.** Stairs, doors, and potholes are not; on the real shoe the
  ultrasonic sensors cover these, which the demo shows with manual and scripted sensor distances.
* **CPU only.** About 10 frames per second on a laptop; a Raspberry Pi would need a smaller input size
  or an accelerator.
* **The demo video** is a zoomed still photo, labelled as a simulation on every frame; a real recording
  from the demo room is more convincing (see `demo/README.md`).

## Development Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Base architecture and abstractions | ✅ |
| 2 | Real-time webcam + YOLOv8 detection | ✅ |
| 3 | Simulated LEFT / CENTER / RIGHT distance sensors | ✅ |
| 4 | Sensor fusion and risk analysis | ✅ |
| 5 | Vibration and voice alerts | ✅ |
| 6 | Dashboard and event logging | ✅ |
| 7 | Software-only demonstration: scenario engine, camera-linked sensors, video input, direction guidance, simulated shoe panel, tracking and flicker control | ✅ |
| 8 | Raspberry Pi hardware (HC-SR04 sensors, vibration motors, Pi camera) | ⏳ Future — interfaces and GPIO pin map ready |
