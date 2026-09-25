"""Main application entry point for Smart Navigation Shoe."""

import sys
import time
import argparse
from pathlib import Path
from typing import Any, Optional

# Add project root directory to sys.path to support execution as a script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.helpers import load_config
from src.utils.logger import setup_logging, get_logger
from src.camera.camera_interface import CameraInterface
from src.camera.camera_factory import open_camera
from src.sensors.sensor_manager import SensorManager
from src.sensors.scenario_engine import CONTROLS_HELP
from src.detection.yolo_detector import YoloDetector
from src.core.models import FusedObstacle, RiskAssessment, BoundingBox
from src.core.enums import RiskLevel, ObstacleZone
from src.dashboard.dashboard_data import DashboardSnapshot
from src.dashboard.dashboard import Dashboard
from src.event_logging.event_logger import EventLogger
from src.pipeline import NavigationPipeline, PipelineResult, create_alert_system, create_detector
from src.utils.visual import FrameRateMeter

MODES = ["dashboard", "fusion", "scenario", "detection", "alerts", "sensors", "simulation", "architecture"]
DEFAULT_WINDOW_NAME = "Smart Navigation Shoe"


# --------------------------------------------------------------------------- shared helpers


def load_detector(config: dict) -> Optional[YoloDetector]:
    """Creates the YOLO detector and loads its weights, printing progress; None on failure."""
    detector = create_detector(config)
    print(f"[INFO] Loading YOLO model weights from '{detector.model_path}'...")
    if not detector.load_model():
        print(f"[ERROR] Failed to load YOLO model weights from '{detector.model_path}'.")
        return None
    return detector


def next_frame(camera: CameraInterface) -> Optional[Any]:
    """Next frame from the source; None once a finite (video/image) source has ended."""
    while True:
        success, frame = camera.read_frame()
        if success and frame is not None:
            return frame
        if not camera.is_opened():
            return None
        time.sleep(0.03)  # Transient read failure: retry


def is_quit_key(key: int) -> bool:
    return key == ord("q") or key == 27


def window_name(config: dict) -> str:
    return config.get("detection", {}).get("display", {}).get("window_name", DEFAULT_WINDOW_NAME)


def build_snapshot(result: PipelineResult, event_logger: EventLogger, **status: Any) -> DashboardSnapshot:
    """Dashboard snapshot of a pipeline step; status carries camera/YOLO/simulation/speed fields."""
    return DashboardSnapshot.from_pipeline_results(
        fused_obstacles=result.fused_obstacles,
        sensor_readings=result.sensor_readings,
        overall_risk=result.risk_assessment.overall_risk_level,
        vibration_action=result.vibration_text,
        voice_message=result.last_spoken or "Path clear.",  # The last message the user heard
        recent_events=event_logger.recent_events,
        direction=result.direction,
        **status,
    )


def report_camera_failure(config: dict, source: Optional[str]) -> str:
    """Prints why no frame source could be opened and returns the message for logging."""
    if source:
        message = f"Could not open video/image source '{source}'."
    else:
        cam_cfg = config.get("camera", {})
        message = f"Could not connect to webcam device ID {cam_cfg.get('device_id', 0)}"
        fallback = cam_cfg.get("fallback_source")
        message += f" or fallback source '{fallback}'." if fallback else "."
    print(f"[ERROR] {message}")
    print("[HINT] Close other apps using the webcam, or play a recording with --source path/to/video.mp4 "
          "(run 'python scripts/make_demo_video.py' to create the demo clip).")
    return message


def close_windows() -> None:
    try:
        import cv2
        cv2.destroyAllWindows()
    except Exception:
        pass


# --------------------------------------------------------------------------- architecture mode


class SmartNavigationShoeApp:
    """Main application orchestrator for Smart Navigation Shoe (architecture / simulation mode)."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config = load_config(config_path)
        self.logger = get_logger("smart_shoe")
        self.logger.info("Initializing Smart Navigation Shoe System...")
        self.camera: Optional[CameraInterface] = None
        self.detector = create_detector(self.config)
        self.detector.load_model()
        self.pipeline = NavigationPipeline(self.config)

    def initialize(self) -> bool:
        """Connects the frame source (webcam, or the configured fallback video)."""
        self.logger.info("Connecting hardware/simulated interfaces...")
        self.camera = open_camera(self.config)
        readings = self.pipeline.sensor_manager.read_all_sensors()
        sensors_ok = any(r.is_valid for r in readings.values())
        camera_desc = self.camera.source_description if self.camera else "UNAVAILABLE"
        self.logger.info(f"Camera: {camera_desc}, Distance sensors active: {sensors_ok}")
        return self.camera is not None

    def run_step(self) -> None:
        """Executes a single processing loop step."""
        success, frame = self.camera.read_frame() if self.camera else (False, None)
        if not success or frame is None:
            self.logger.warning("Failed to capture frame from camera source.")
            return

        detection_result = self.detector.detect(frame)
        result = self.pipeline.process(
            detection_result.detections, (detection_result.frame_width, detection_result.frame_height)
        )
        center = next((r for r in result.sensor_readings if r.position == ObstacleZone.CENTER), None)
        center_str = f"{center.distance_m:.2f}m" if center and center.is_valid else "N/A"
        message = (
            f"Step Completed | Center distance: {center_str} | "
            f"Risk: {result.risk_assessment.overall_risk_level.value} | "
            f"Direction: {result.direction.recommended_direction.value} | "
            f"VIBRATION: {result.vibration_text} | {result.alert_status}"
        )
        print(message)
        self.logger.info(message)

    def shutdown(self) -> None:
        """Cleans up system resources."""
        self.logger.info("Shutting down Smart Navigation Shoe application...")
        if self.camera is not None:
            self.camera.release()
        self.pipeline.shutdown()
        self.logger.info("Shutdown complete.")


def run_architecture_demo(config_path: str = "config/config.yaml") -> None:
    """Phase 1: runs three steps of the full architecture and exits."""
    print("=" * 70)
    print("  AI-Powered Smart Navigation Shoe — Simulation Architecture Run")
    print("=" * 70)
    app = SmartNavigationShoeApp(config_path=config_path)
    if not app.initialize():
        print("[ERROR] Application initialization failed (no camera or fallback source).")
        app.shutdown()
        return
    print("\n[INFO] Running 3 demonstration architecture steps...\n")
    for i in range(3):
        print(f"--- Iteration {i + 1} ---")
        app.run_step()
        time.sleep(0.5)
    app.shutdown()
    print("\n[SUCCESS] Phase 1 architecture loop executed successfully.")


# --------------------------------------------------------------------------- component demos


def run_alert_demo(config_path: str = "config/config.yaml") -> None:
    """Phase 5: Demonstrates directional vibration simulation and voice debouncing."""
    config = load_config(config_path)
    _vibration, voice, alert_mgr = create_alert_system(config)  # Honors alerts.voice / alerts.vibration settings

    print("\n" + "=" * 70)
    print("  PHASE 5 ALERT DEMONSTRATION (SOFTWARE SIMULATION)")
    print("=" * 70)

    demo_scenarios = [
        ("Scenario 1: SAFE", RiskAssessment(overall_risk_level=RiskLevel.SAFE), []),
        (
            "Scenario 2: CAUTION (LEFT)",
            RiskAssessment(overall_risk_level=RiskLevel.CAUTION),
            [FusedObstacle("1", "chair", 0.8, BoundingBox(10, 10, 50, 50), 1.5, ObstacleZone.LEFT)],
        ),
        (
            "Scenario 3: WARNING (CENTER)",
            RiskAssessment(overall_risk_level=RiskLevel.WARNING),
            [FusedObstacle("2", "person", 0.91, BoundingBox(250, 10, 350, 50), 0.8, ObstacleZone.CENTER)],
        ),
        (
            "Scenario 4: DANGER (RIGHT)",
            RiskAssessment(overall_risk_level=RiskLevel.DANGER),
            [FusedObstacle("3", "box", 0.95, BoundingBox(500, 10, 600, 50), 0.4, ObstacleZone.RIGHT)],
        ),
    ]

    for label, risk, obstacles in demo_scenarios:
        print(f"\n{label}")
        vib_out, status_out = alert_mgr.evaluate_and_trigger(risk, obstacles)

        direction_str = obstacles[0].zone.value if obstacles else "CENTER"
        print(f"  Risk      : {risk.overall_risk_level.value}")
        print(f"  Direction : {direction_str}")
        print(f"  Vibration : {vib_out.replace('VIBRATION: ', '')}")
        print(f"  Voice     : {status_out}")
        time.sleep(0.4)

    voice.stop()
    print("\n" + "=" * 70)
    print("[SUCCESS] Phase 5 Alert System Demonstration Completed Cleanly.\n")


def run_sensor_demo(config_path: str = "config/config.yaml") -> None:
    """Phase 3: Demonstrates simulated LEFT, CENTER, and RIGHT distance sensors."""
    config = load_config(config_path)
    sensor_manager = SensorManager(config=config)

    print("\n" + "=" * 70)
    print("  AI-Powered Smart Navigation Shoe — Phase 3 Sensor Demonstration")
    print("=" * 70)
    print("\n1. Current Simulated Distance Readings:")
    print(sensor_manager.format_sensor_display(display_unit="cm"))

    print("\n2. Testing Sensor Safety & Invalid Status Handling:")
    print("  Setting LEFT sensor to -50 cm (Negative Invalid Reading)...")
    left_sensor: Any = sensor_manager.get_sensor(ObstacleZone.LEFT)
    if left_sensor and hasattr(left_sensor, "set_simulated_distance_cm"):
        left_sensor.set_simulated_distance_cm(-50.0)

    print("  Setting RIGHT sensor to 500 cm (Out Of Bounds > 400 cm)...")
    right_sensor: Any = sensor_manager.get_sensor(ObstacleZone.RIGHT)
    if right_sensor and hasattr(right_sensor, "set_simulated_distance_cm"):
        right_sensor.set_simulated_distance_cm(500.0)

    print(sensor_manager.format_sensor_display(display_unit="cm"))

    print("\n3. Setting Example Distances (LEFT=150cm, CENTER=80cm, RIGHT=220cm)...")
    sensor_manager.set_simulated_distances_cm(150.0, 80.0, 220.0)
    print(sensor_manager.format_sensor_display(display_unit="cm"))

    print("\n[SUCCESS] Phase 3 Distance Sensor Demonstration Completed Cleanly.\n")


def run_realtime_detection(config_path: str = "config/config.yaml", source: Optional[str] = None) -> None:
    """Phase 2: Raw real-time YOLO object detection (no sensors, tracking, or alerts)."""
    config = load_config(config_path)
    disp_cfg = config.get("detection", {}).get("display", {})

    print("=" * 70)
    print("  AI-Powered Smart Navigation Shoe — Phase 2 Real-Time YOLO Detection")
    print("=" * 70)
    print("Press 'q' or 'ESC' on the camera display window to exit.\n")

    camera = open_camera(config, source)
    if camera is None:
        report_camera_failure(config, source)
        return
    print(f"[INFO] Frame source: {camera.source_description}")

    detector = load_detector(config)
    if detector is None:
        camera.release()
        return

    print("[INFO] Camera & YOLO initialized. Starting live video feed...")
    try:
        import cv2
        while True:
            frame = next_frame(camera)
            if frame is None:
                print("\n[INFO] Video/image source finished.")
                break

            detection_result = detector.detect(frame)
            annotated_frame = detector.draw_detections(
                frame=frame,
                result=detection_result,
                show_confidence=disp_cfg.get("show_confidence", True),
                show_box=disp_cfg.get("show_box", True),
            )
            cv2.imshow(window_name(config), annotated_frame)

            if is_quit_key(cv2.waitKey(1) & 0xFF):
                print("[INFO] User requested exit ('q' / ESC pressed). Stopping camera stream...")
                break
    except Exception as e:
        print(f"[ERROR] Error during real-time webcam detection loop: {e}")
    finally:
        camera.release()
        close_windows()
        print("[INFO] Camera released and OpenCV display windows closed cleanly.")


# --------------------------------------------------------------------------- full pipeline modes


def run_realtime_fusion(
    config_path: str = "config/config.yaml",
    scenario: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    """Phase 4 & 5: Camera + YOLO + simulated sensors + fusion + risk + direction + alerts (console output)."""
    config = load_config(config_path)

    print("=" * 70)
    print("  AI-Powered Smart Navigation Shoe — Phase 4/5 Sensor Fusion & Alerts")
    print("=" * 70)
    print("Press 'q' or 'ESC' on the camera display window to exit.\n")

    camera = open_camera(config, source)
    if camera is None:
        report_camera_failure(config, source)
        return
    print(f"[INFO] Frame source: {camera.source_description}")

    pipeline = NavigationPipeline(config, scenario=scenario)
    print("[INFO] Sensor Manager initialized with simulated spatial distance sensors:")
    print(pipeline.sensor_manager.format_sensor_display(display_unit="cm"))
    print(f"[INFO] Simulation: {pipeline.scenario_engine.status_text}")
    print(f"[INFO] {CONTROLS_HELP}\n")

    detector = load_detector(config)
    if detector is None:
        camera.release()
        pipeline.shutdown()
        return

    print("[INFO] Camera, YOLO, Sensor Fusion, Risk Analyzer & Alert Manager ready. Starting stream...\n")
    try:
        import cv2
        last_logged_time = 0.0

        while True:
            frame = next_frame(camera)
            if frame is None:
                print("\n[INFO] Video/image source finished.")
                break

            detection_result = detector.detect(frame)
            result = pipeline.process(
                detection_result.detections, (detection_result.frame_width, detection_result.frame_height)
            )

            current_time = time.time()
            if result.fused_obstacles and (current_time - last_logged_time > 1.5):
                for obs in result.fused_obstacles:
                    print(obs.format_display())
                print(
                    f"Overall Risk: {result.risk_assessment.overall_risk_level.value} | "
                    f"VIBRATION: {result.vibration_text} | Status: {result.alert_status}\n"
                )
                last_logged_time = current_time

            cv2.imshow(window_name(config), detector.draw_fused_obstacles(frame, result.fused_obstacles))

            key = cv2.waitKey(1) & 0xFF
            if is_quit_key(key):
                print("\n[INFO] User requested exit ('q' / ESC pressed). Stopping live video feed...")
                break
            if pipeline.scenario_engine.handle_key(key):
                engine = pipeline.scenario_engine
                print(f"[SIMULATION] {engine.status_text} | {engine.distances_text}")
    except Exception as e:
        print(f"[ERROR] Error during real-time sensor fusion loop: {e}")
    finally:
        camera.release()
        pipeline.shutdown()
        close_windows()
        print("[INFO] Camera released and OpenCV display windows closed cleanly.")


def run_dashboard_mode(
    config_path: str = "config/config.yaml",
    scenario: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    """Phase 6+: full demo view — video with risk-colored boxes, simulated shoe panel, CLI dashboard, logging."""
    config = load_config(config_path)

    event_logger = EventLogger(config=config)
    event_logger.log_system_message("Dashboard mode started")

    print("=" * 70)
    print("  AI-Powered Smart Navigation Shoe — Dashboard & Event Logging")
    print("=" * 70)
    print("Press 'q' or 'ESC' on the camera display window to exit.\n")

    camera = open_camera(config, source)
    if camera is None:
        event_logger.log_system_message(report_camera_failure(config, source), level="ERROR")
        return
    event_logger.log_system_message(f"Camera connected: {camera.source_description}")

    pipeline = NavigationPipeline(config, scenario=scenario)
    event_logger.log_system_message(f"Sensors initialized (SIMULATED, {pipeline.scenario_engine.status_text})")

    detector = load_detector(config)
    if detector is None:
        event_logger.log_system_message(f"Failed to load YOLO model weights from '{config.get('detection', {}).get('model_path')}'", level="ERROR")
        camera.release()
        pipeline.shutdown()
        return
    event_logger.log_system_message("YOLO model loaded")

    dashboard = Dashboard(config=config)
    event_logger.log_system_message("Dashboard started")
    print("[INFO] Dashboard, Camera, YOLO, Sensor Fusion & Event Logger active.\n")

    try:
        import cv2
        last_cli_update_time = 0.0
        frame_rate = FrameRateMeter()

        while True:
            frame = next_frame(camera)
            if frame is None:
                event_logger.log_system_message("Video/image source finished")
                print("\n[INFO] Video/image source finished.")
                break

            detection_result = detector.detect(frame)
            result = pipeline.process(
                detection_result.detections, (detection_result.frame_width, detection_result.frame_height)
            )
            NavigationPipeline.log_event(event_logger, result)

            snapshot = build_snapshot(
                result,
                event_logger,
                camera_status=f"CONNECTED ({camera.source_description})" if camera.is_connected else "DISCONNECTED",
                yolo_status="ACTIVE" if detector.is_loaded() else "INACTIVE",
                sensors_status="ACTIVE (SIMULATED)",
                sensor_mode=pipeline.scenario_engine.status_text,
                controls_hint=CONTROLS_HELP,
                fps=frame_rate.tick(),
                inference_ms=detection_result.processing_time_ms,
            )

            now = time.time()
            if now - last_cli_update_time >= dashboard.refresh_interval:
                dashboard.display_cli(snapshot, clear_screen=True)
                last_cli_update_time = now

            annotated_frame = detector.draw_fused_obstacles(frame, result.fused_obstacles)
            cv2.imshow(window_name(config), dashboard.render_frame(annotated_frame, snapshot))

            key = cv2.waitKey(1) & 0xFF
            if is_quit_key(key):
                event_logger.log_system_message("User requested exit from dashboard mode")
                print("\n[INFO] User requested exit ('q' / ESC pressed). Stopping dashboard stream...")
                break
            if pipeline.scenario_engine.handle_key(key):
                engine = pipeline.scenario_engine
                event_logger.log_system_message(f"Simulation control: {engine.status_text} | {engine.distances_text}")
    except Exception as e:
        event_logger.log_system_message(f"Error during real-time dashboard loop: {e}", level="ERROR")
        print(f"[ERROR] Error during real-time dashboard loop: {e}")
    finally:
        camera.release()
        event_logger.log_system_message("Camera released")
        pipeline.shutdown()
        close_windows()
        event_logger.log_system_message("Dashboard closed cleanly")
        print("[INFO] Camera released and dashboard display window closed cleanly.")


def scenario_placeholder_frame(name: str, description: str, width: int = 640, height: int = 480):
    """Dark stand-in for the camera view in camera-free scenario mode."""
    import cv2
    import numpy as np

    frame = np.full((height, width, 3), 32, dtype=np.uint8)
    lines = [
        ("NO CAMERA - SENSOR-ONLY SCENARIO", 0.7, (0, 220, 220), 2),
        (f"Scenario: {name}", 0.55, (230, 230, 230), 1),
        (description, 0.45, (170, 170, 170), 1),
        ("Obstacles come from the simulated distance sensors", 0.45, (170, 170, 170), 1),
        ("Press q or ESC to stop", 0.45, (170, 170, 170), 1),
    ]
    y = height // 2 - 70
    for text, scale, color, thickness in lines:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(frame, text, (max(10, (width - tw) // 2), y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
        y += 34
    return frame


def run_scenario_mode(config_path: str = "config/config.yaml", scenario: Optional[str] = None) -> None:
    """Software-only demo without a camera: plays scripted sensor scenarios through the full pipeline.

    Detections are empty, so every obstacle comes from the SIMULATED distance sensors
    (sensor-only fusion), exercising risk analysis, direction, vibration, voice, dashboard, and logging.
    """
    config = load_config(config_path)
    tick_s = config.get("sensors", {}).get("distance_sensor", {}).get("update_interval_sec", 0.1)
    min_scenario_s = 5.0  # Static (single keyframe) scenarios still get time to be heard
    tail_s = 1.5          # Hold the final state so its alert is announced

    pipeline = NavigationPipeline(config, use_tracker=False)
    engine = pipeline.scenario_engine
    names = [scenario] if scenario else engine.scenario_names
    if not names:
        print("[ERROR] No scenarios defined under sensors.simulation.scenarios in the config.")
        pipeline.shutdown()
        return

    event_logger = EventLogger(config=config)
    dashboard = Dashboard(config=config)
    frame_rate = FrameRateMeter()

    # Show the simulated shoe panel in a window unless disabled (terminal dashboard always runs)
    cv2 = None
    if config.get("dashboard", {}).get("scenario_window", True):
        try:
            import cv2
        except ImportError:
            cv2 = None
    scenario_window = f"{window_name(config)} - Sensor Scenario (no camera)"

    event_logger.log_system_message(f"Scenario mode started: {', '.join(names)}")

    try:
        for index, name in enumerate(names, start=1):
            engine.start_scenario(name, loop=False)
            description = engine.scenarios[name].description
            end_time = time.monotonic() + max(engine.scenarios[name].duration_s, min_scenario_s) + tail_s

            while time.monotonic() < end_time:
                result = pipeline.process()
                NavigationPipeline.log_event(event_logger, result)
                snapshot = build_snapshot(
                    result,
                    event_logger,
                    camera_status="NOT USED (scenario mode)",
                    yolo_status="NOT USED (sensor-only)",
                    sensors_status="ACTIVE (SIMULATED)",
                    sensor_mode=f"{engine.status_text} [{index}/{len(names)}]",
                    controls_hint=f"{description} | Ctrl+C to stop",
                    fps=frame_rate.tick(),
                )
                dashboard.display_cli(snapshot, clear_screen=True)

                if cv2 is None:
                    time.sleep(tick_s)
                    continue
                placeholder = scenario_placeholder_frame(name, description)
                cv2.imshow(scenario_window, dashboard.render_frame(placeholder, snapshot))
                if is_quit_key(cv2.waitKey(max(1, int(tick_s * 1000))) & 0xFF):
                    raise KeyboardInterrupt

        print("\n[SUCCESS] Scenario playback completed.")
    except KeyboardInterrupt:
        print("\n[INFO] Scenario playback interrupted.")
    finally:
        pipeline.shutdown()
        if cv2 is not None:
            cv2.destroyAllWindows()
        event_logger.log_system_message("Scenario mode closed cleanly")


# --------------------------------------------------------------------------- entry point


def main() -> None:
    """Main execution function supporting CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Smart Navigation Shoe - AI Assistive Navigation System (software-only demonstration)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=MODES,
        help="dashboard: full demo view (default, see system.mode in the config); "
             "fusion: pipeline with console output; scenario: camera-free scripted sensor scenarios; "
             "detection: raw YOLO only; alerts / sensors: component demos; "
             "simulation / architecture: 3-step architecture run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Simulated sensor scenario to start (see sensors.simulation.scenarios in the config). "
             "In 'scenario' mode, omit it to play every scenario once.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Use a video file, image, or folder of images instead of the webcam "
             "(detection, fusion, and dashboard modes).",
    )

    args, _ = parser.parse_known_args()
    config = load_config(args.config)
    setup_logging(config)

    mode = args.mode or config.get("system", {}).get("mode", "dashboard")
    if mode not in MODES:
        print(f"[ERROR] Unknown mode '{mode}' in config system.mode. Choose from: {', '.join(MODES)}")
        return

    if args.scenario:
        available = list(config.get("sensors", {}).get("simulation", {}).get("scenarios", {}) or {})
        if args.scenario not in available:
            print(f"[ERROR] Unknown scenario '{args.scenario}'. Available: {', '.join(available) or 'none'}")
            return

    if mode == "dashboard":
        run_dashboard_mode(config_path=args.config, scenario=args.scenario, source=args.source)
    elif mode == "scenario":
        run_scenario_mode(config_path=args.config, scenario=args.scenario)
    elif mode == "fusion":
        run_realtime_fusion(config_path=args.config, scenario=args.scenario, source=args.source)
    elif mode == "alerts":
        run_alert_demo(config_path=args.config)
    elif mode == "detection":
        run_realtime_detection(config_path=args.config, source=args.source)
    elif mode == "sensors":
        run_sensor_demo(config_path=args.config)
    else:
        run_architecture_demo(config_path=args.config)


if __name__ == "__main__":
    main()
