# AI-Powered Smart Navigation Shoe for Visually Impaired People Using Computer Vision and Sensor Fusion

## Project Overview

The **AI-Powered Smart Navigation Shoe** is an assistive device designed to provide real-time spatial awareness, obstacle detection, collision hazard warning, and directional navigation guidance for visually impaired individuals.

By combining low-latency **Computer Vision (YOLOv8)** with **Distance Sensors (Ultrasonic Rangefinders)** through **Sensor Fusion**, the system detects both elevated and low-level ground obstacles, evaluates collision risks, and delivers non-intrusive feedback via **Voice Audio Announcements** and **Haptic Vibration Actuators**.

> [!WARNING]
> **Assistive-Device Research Prototype Disclaimer**:
> This software is an experimental assistive-device research prototype developed for educational and academic project purposes. It is **NOT** a safety-certified, medical-grade, or life-critical navigation system. Computer vision object detection alone does **NOT** provide 3D distance calculation—distances in current laptop simulation phases are provided by simulated range sensors. Users must not rely solely on this prototype for primary navigation in hazards or outdoor traffic.

---

## Problem Statement

Visually impaired individuals face significant daily mobility challenges:
* Standard white canes cannot detect elevated or hanging obstacles (e.g., open cabinet doors, low-hanging tree branches, truck beds).
* Guide dogs require extensive training, high maintenance costs, and are not universally accessible.
* Existing wearable sensors often lack object classification abilities—they signal distance but cannot distinguish between a harmless curtain and a dangerous flight of stairs or approaching vehicle.

---

## Proposed Solution

The Smart Navigation Shoe integrates multi-modal sensing on a wearable footgear platform:
1. **Camera Module**: Captures forward environmental imagery for deep-learning-based object detection.
2. **Ultrasonic Range Sensors**: Provides distance measurements across three spatial zones (`LEFT`, `CENTER`, `RIGHT`).
3. **Sensor Fusion Engine**: Correlates visual bounding boxes with physical range readings to establish true object spatial positioning and distance.
4. **Risk & Direction Analyzer**: Calculates real-time collision threats and evaluates safe detour trajectories.
5. **Multi-Modal Feedback**: Delivers immediate directional vibration pulses to the shoe and clear voice instructions via earphones.
6. **Real-Time Dashboard & Persistent Event Logger**: Monitors system status, live multi-object detections, risk assessments, alert triggers, and records events to persistent CSV and log files without duplicate frame spam.

---

## Development Phases & System Status

| Phase | Description | Implementation Status |
|---|---|---|
| **Phase 1** | Base Architecture & System Abstractions | ✅ Completed (12/12 Tests Passing) |
| **Phase 2** | Real-Time Laptop Webcam + YOLO Object Detection | ✅ Completed (Real Camera + YOLOv8) |
| **Phase 3** | Simulated LEFT / CENTER / RIGHT Distance Sensors | ✅ Completed (Sensor Manager + Status Logic) |
| **Phase 4** | Sensor Fusion & Risk Analysis Pipeline | ✅ Completed (YOLO Zone + Distance Mapping + Risk) |
| **Phase 5** | Multi-Modal Alert System (Vibration & Voice) | ✅ Completed (Thread-Safe TTS + Cooldown + Haptic Simulation) |
| **Phase 6** | Real-Time Dashboard & Persistent Event Logging | ✅ Completed (Terminal + OpenCV Overlay + CSV Logging, 50/50 Tests Passing) |
| **Phase 7** | System Performance & Low-Latency Optimization | ⏳ Upcoming |
| **Phase 8** | Raspberry Pi Hardware & GPIO Integration | ⏳ Upcoming |

---

## Data Flow & System Pipeline Architecture

### Current Laptop Prototype Pipeline (Phases 1–6)
```text
Laptop Webcam
    │
    ▼
YOLO Object Detection (Ultralytics YOLOv8)
    │
    ▼
Object Spatial Position / Zone Determination (LEFT, CENTER, RIGHT)
    │
    ▼
Simulated Distance Sensor Association (SensorManager)
    │
    ▼
Sensor Fusion Engine (FusedObstacle Data Structure)
    │
    ▼
Risk Analysis Engine (DANGER, WARNING, CAUTION, SAFE Classification)
    │
    ▼
Alert Manager (Debouncing & Alert Cooldown Evaluation)
    │
    ├─────────────────────────────────────────┼─────────────────────────────────────────┐
    ▼                                         ▼                                         ▼
Simulated Directional Vibration           Offline pyttsx3 Voice Speech              Real-Time Dashboard & Event Logger
(Console Log: LEFT / BOTH / RIGHT)       (Thread-Safe Queue Worker)               (CLI + Video Overlay + CSV Logs)
```

---

## PHASE 6 — DASHBOARD & EVENT LOGGING

### 1. Real-Time System Dashboard
The system dashboard displays real-time telemetry from camera, vision model, simulated distance sensors, detected objects, confidence, spatial zones, distances, risk levels, and alert states:

```
--------------------------------------------------
    SMART NAVIGATION SHOE
--------------------------------------------------
SYSTEM STATUS

Camera          : CONNECTED
YOLO            : ACTIVE
Sensors         : ACTIVE (SIMULATED)

CURRENT DETECTION

Object       Confidence    Zone       Distance    Risk
person       87%           CENTER     80 cm       WARNING
chair        65%           LEFT       150 cm      CAUTION

ALERT STATUS

Vibration       : BOTH - MEDIUM PULSE
Voice           : Warning. Obstacle ahead. Slow down.

SENSOR STATUS

LEFT            : 150 cm     HEALTHY
CENTER          : 80 cm      HEALTHY
RIGHT           : 220 cm     HEALTHY

RECENT EVENTS

Time       Object      Zone      Distance    Risk
10:21:04   person      CENTER    80 cm       WARNING
10:21:08   chair       LEFT      150 cm      CAUTION
10:21:12   suitcase    RIGHT     220 cm      SAFE
--------------------------------------------------
```

### 2. Persistent Event Logging (`logs/detection_events.csv`)
Events are logged to structured CSV format for analysis and demonstration:

* **File**: `logs/detection_events.csv`
* **Columns**: `timestamp`, `object_name`, `confidence`, `zone`, `distance_cm`, `risk_level`, `vibration_action`, `voice_message`
* **Deduplication Rule**: Identical consecutive frame events are debounced to prevent duplicate log spam. New logs are triggered only when a new detection occurs, risk changes, zone changes, alert triggers, or heartbeat interval expires.

### 3. System Logging (`logs/system.log`)
Maintains operational system logs including camera connection, YOLO model loading, sensor status, alert triggers, and clean shutdown events.

---

## Command-Line Execution Modes

### 1. Run Phase 6 Real-Time Dashboard & Event Logging Mode
Launches live webcam stream, YOLO detection, simulated distance sensors, sensor fusion, risk assessment, alert debouncing, live ASCII terminal dashboard, OpenCV overlay, and persistent CSV event logging:
```bash
py -3.11 -m src.main --mode dashboard
```
*(Press **`q`** or **`ESC`** on the camera display window to exit).*

### 2. Run Real-Time Webcam Sensor Fusion & Alert Pipeline (Phase 4 & 5)
```bash
py -3.11 -m src.main --mode fusion
```

### 3. Run Phase 5 Alert System Demonstration
```bash
py -3.11 -m src.main --mode alerts
```

### 4. Run Real-Time Webcam YOLO Detection Only (Phase 2)
```bash
py -3.11 -m src.main --mode detection
```

### 5. Run Distance Sensor Demonstration (Phase 3)
```bash
py -3.11 -m src.main --mode sensors
```

### 6. Run System Architecture Simulation Loop (Phase 1)
```bash
py -3.11 -m src.main --mode simulation
```

---

## Running Unit Tests

Run `pytest` to execute all 50 unit tests across Phases 1–6:
```bash
py -3.11 -m pytest
```

---

## Project Structure

```
smart-navigation-shoe/
│
├── README.md                  # Project documentation & Phase 6 guide
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Version control exclusion rules
│
├── config/
│   └── config.yaml            # Configurable parameters, dashboard & logging settings
│
├── logs/                      # Log directory (auto-created)
│   ├── detection_events.csv   # Persistent CSV event log
│   └── system.log             # Application operational log
│
├── models/
│   └── README.md              # Machine learning model weights directory
│
├── src/
│   ├── __init__.py
│   ├── main.py                # Main application entry point (--mode dashboard | fusion | alerts | detection)
│   │
│   ├── dashboard/             # Real-time dashboard module
│   │   ├── __init__.py
│   │   ├── dashboard.py       # ASCII terminal & OpenCV video overlay renderer
│   │   └── dashboard_data.py  # Snapshot dataclasses & pipeline formatting
│   │
│   ├── event_logging/         # Persistent event & system logging module
│   │   ├── __init__.py
│   │   └── event_logger.py    # Debounced CSV detection logger & system log writer
│   │
│   ├── camera/                # Camera capture abstraction layer
│   │   ├── __init__.py
│   │   ├── camera_interface.py       # ABC for camera inputs
│   │   ├── webcam_camera.py          # Real OpenCV webcam driver
│   │   └── raspberry_pi_camera.py    # Future Pi CSI camera driver shell
│   │
│   ├── detection/             # Vision detection layer
│   │   ├── __init__.py
│   │   ├── detector_interface.py     # ABC for object detectors
│   │   ├── yolo_detector.py          # Real Ultralytics YOLOv8 detector & visualizer
│   │   └── detection_result.py       # Dataclass encapsulating detection outputs
│   │
│   ├── sensors/               # Range sensor abstraction layer
│   │   ├── __init__.py
│   │   ├── distance_sensor_interface.py # ABC for range sensors
│   │   ├── simulated_sensor.py          # Simulated sensor for LEFT, CENTER, RIGHT
│   │   ├── sensor_manager.py            # SensorManager coordinating 3 spatial distance sensors
│   │   └── ultrasonic_sensor.py         # Future HC-SR04 GPIO driver shell
│   │
│   ├── fusion/                # Multi-sensor fusion engine
│   │   ├── __init__.py
│   │   └── sensor_fusion.py          # Zone-based vision + distance sensor fusion
│   │
│   ├── decision/              # Risk evaluation and navigation logic
│   │   ├── __init__.py
│   │   ├── risk_analyzer.py          # Configurable distance-based risk assessment
│   │   └── direction_analyzer.py     # Spatial zone path recommendation
│   │
│   ├── alerts/                # Multi-modal feedback layer
│   │   ├── __init__.py
│   │   ├── alert_manager.py          # Central alert dispatcher with debouncing
235: │   │   ├── vibration_interface.py    # ABC for haptic motors
│   │   ├── simulated_vibration.py    # Console vibration simulator (LEFT / BOTH / RIGHT)
│   │   ├── raspberry_pi_vibration.py # Future Pi GPIO motor driver shell
│   │   ├── voice_alert.py            # Thread-safe pyttsx3 Text-to-Speech manager
│   │   └── audio_manager.py          # Sound cue & tone manager
│   │
│   ├── core/                  # Shared data models and enums
│   │   ├── __init__.py
│   │   ├── models.py                 # Dataclasses (FusedObstacle, SensorReading, BoundingBox)
│   │   └── enums.py                  # System Enums (RiskLevel, SensorStatus, ObstacleZone)
│   │
│   └── utils/                 # Utility helpers
│       ├── __init__.py
│       ├── logger.py                 # System logging setup
│       └── helpers.py                # YAML configuration loader & spatial helpers
│
└── tests/                     # Unit test suite
    ├── __init__.py
    ├── test_alerts.py                # Phase 5 alert system & TTS worker tests
    ├── test_detection.py             # Detector and camera interface tests
    ├── test_real_time_detection.py   # Phase 2 YOLO visualizer tests
    ├── test_sensor_fusion.py         # Sensor fusion engine tests
    ├── test_simulated_sensors.py     # Phase 3 simulated sensor & manager tests
    ├── test_phase4_fusion.py         # Phase 4 sensor fusion & risk tests
    ├── test_phase6_dashboard_logging.py # Phase 6 dashboard & CSV event logger tests
    ├── test_risk_analysis.py         # Risk classification tests
    └── test_direction.py             # Directional guidance tests
```
