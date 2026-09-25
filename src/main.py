"""Main application entry point for Smart Navigation Shoe."""

import sys
import time
import argparse
from pathlib import Path

# Add project root directory to sys.path to support execution as a script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.helpers import load_config
from src.utils.logger import setup_logger
from src.camera.webcam_camera import WebcamCamera
from src.camera.raspberry_pi_camera import RaspberryPiCamera
from src.sensors.simulated_sensor import SimulatedDistanceSensor
from src.sensors.ultrasonic_sensor import UltrasonicSensor
from src.sensors.sensor_manager import SensorManager
from src.detection.yolo_detector import YoloDetector
from src.fusion.sensor_fusion import SensorFusionEngine
from src.decision.risk_analyzer import RiskAnalyzer
from src.decision.direction_analyzer import DirectionAnalyzer
from src.alerts.simulated_vibration import SimulatedVibration
from src.alerts.raspberry_pi_vibration import RaspberryPiVibration
from src.alerts.voice_alert import VoiceAlertManager
from src.alerts.alert_manager import AlertManager
from src.core.models import FusedObstacle, RiskAssessment, BoundingBox
from src.core.enums import RiskLevel, ObstacleZone
from src.dashboard.dashboard_data import DashboardSnapshot
from src.dashboard.dashboard import Dashboard
from src.event_logging.event_logger import EventLogger


class SmartNavigationShoeApp:
    """Main application orchestrator for Smart Navigation Shoe."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config = load_config(config_path)
        log_cfg = self.config.get("logging", {})
        self.logger = setup_logger(
            name="smart_shoe",
            level=log_cfg.get("level", "INFO"),
            log_file=log_cfg.get("log_file"),
            log_format=log_cfg.get("format"),
        )
        self.logger.info("Initializing Smart Navigation Shoe System...")

        # 1. Initialize Camera Module
        cam_cfg = self.config.get("camera", {})
        cam_type = cam_cfg.get("type", "webcam")
        if cam_type == "webcam":
            self.camera = WebcamCamera(
                device_id=cam_cfg.get("device_id", 0),
                width=cam_cfg.get("width", 640),
                height=cam_cfg.get("height", 480),
                fps=cam_cfg.get("fps", 30),
                simulation_fallback=cam_cfg.get("simulation_fallback", True),
            )
        else:
            self.camera = RaspberryPiCamera(
                width=cam_cfg.get("width", 640),
                height=cam_cfg.get("height", 480),
                fps=cam_cfg.get("fps", 30),
            )

        # 2. Initialize Sensor Manager (LEFT, CENTER, RIGHT distance sensors)
        self.sensor_manager = SensorManager(config=self.config)

        # 3. Initialize YOLO Detector Interface
        det_cfg = self.config.get("detection", {})
        self.detector = YoloDetector(
            model_path=det_cfg.get("model_path", "models/yolov8n.pt"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.5),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            device=det_cfg.get("device", "cpu"),
        )
        self.detector.load_model()

        # 4. Initialize Sensor Fusion Engine
        sensor_cfg = self.config.get("sensors", {}).get("distance_sensor", {})
        self.fusion_engine = SensorFusionEngine(
            default_range_m=sensor_cfg.get("simulated_default_m", 2.5)
        )

        # 5. Initialize Decision Engine (Risk & Direction Analyzers)
        self.risk_analyzer = RiskAnalyzer(config=self.config)

        dir_cfg = self.config.get("direction", {})
        self.direction_analyzer = DirectionAnalyzer(
            clear_path_threshold_m=dir_cfg.get("clear_path_threshold", 2.0)
        )

        # 6. Initialize Alert System
        alert_cfg = self.config.get("alerts", {})
        vib_type = alert_cfg.get("vibration", {}).get("type", "simulated")
        if vib_type == "simulated":
            self.vibration = SimulatedVibration(
                enabled=alert_cfg.get("vibration", {}).get("enabled", True)
            )
        else:
            vib_pins = alert_cfg.get("vibration", {}).get("gpio_pins", {})
            self.vibration = RaspberryPiVibration(
                left_pin=vib_pins.get("left", 17),
                center_pin=vib_pins.get("center", 27),
                right_pin=vib_pins.get("right", 22),
            )

        voice_cfg = alert_cfg.get("voice", {})
        self.voice = VoiceAlertManager(
            enabled=voice_cfg.get("enabled", True),
            speech_rate=voice_cfg.get("speech_rate", 160),
            volume=voice_cfg.get("volume", 0.9),
        )

        self.alert_manager = AlertManager(
            vibration=self.vibration,
            voice=self.voice,
            config=self.config,
        )

    def initialize(self) -> bool:
        """Connects camera and sensor peripherals."""
        self.logger.info("Connecting hardware/simulated interfaces...")
        cam_ok = self.camera.connect()
        readings = self.sensor_manager.read_all_sensors()
        all_valid = any(r.is_valid for r in readings.values())
        self.logger.info(
            f"Camera status: {self.camera.status.value}, Distance sensors active: {all_valid}"
        )
        return cam_ok

    def run_step(self) -> None:
        """Executes a single processing loop step."""
        success, frame = self.camera.read_frame()
        if not success or frame is None:
            self.logger.warning("Failed to capture frame from camera source.")
            return

        detection_result = self.detector.detect(frame)
        sensor_readings = self.sensor_manager.get_readings_list()
        fused_obstacles = self.fusion_engine.fuse(
            detections=detection_result.detections,
            sensor_readings=sensor_readings,
        )
        risk_assessment = self.risk_analyzer.evaluate(fused_obstacles)
        vib_text, alert_status = self.alert_manager.evaluate_and_trigger(risk_assessment, fused_obstacles)

        primary_dist = sensor_readings[0].distance_m if sensor_readings else 0.0
        self.logger.info(
            f"Step Completed | Distance: {primary_dist:.2f}m | Risk: {risk_assessment.overall_risk_level.value} | {vib_text} | {alert_status}"
        )

    def shutdown(self) -> None:
        """Cleans up system resources."""
        self.logger.info("Shutting down Smart Navigation Shoe application...")
        self.camera.release()
        self.vibration.stop_all()
        self.voice.stop()
        self.logger.info("Shutdown complete.")


def run_alert_demo(config_path: str = "config/config.yaml") -> None:
    """Phase 5: Demonstrates directional vibration simulation and voice debouncing."""
    config = load_config(config_path)
    vib = SimulatedVibration(enabled=True)
    voice = VoiceAlertManager(enabled=True)
    alert_mgr = AlertManager(vibration=vib, voice=voice, config=config, cooldown_seconds=0.1)

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

    print("\n" + "=" * 70)
    print("[SUCCESS] Phase 5 Alert System Demonstration Completed Cleanly.\n")


def run_realtime_fusion(config_path: str = "config/config.yaml") -> None:
    """Phase 4 & 5: Real-Time Laptop Webcam + YOLO + Distance Sensors + Sensor Fusion + Risk Analysis + Alert System."""
    config = load_config(config_path)
    cam_cfg = config.get("camera", {})
    det_cfg = config.get("detection", {})
    disp_cfg = det_cfg.get("display", {})

    print("=" * 70)
    print("  AI-Powered Smart Navigation Shoe — Phase 4/5 Sensor Fusion & Alerts")
    print("=" * 70)
    print("Press 'q' or 'ESC' on the camera display window to exit.\n")

    device_id = cam_cfg.get("device_id", 0)
    camera = WebcamCamera(
        device_id=device_id,
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        fps=cam_cfg.get("fps", 30),
        simulation_fallback=False,
    )

    if not camera.connect():
        print(f"[ERROR] Could not connect to webcam device ID {device_id}.")
        print("[HINT] Ensure your laptop webcam is available and not opened in another application.")
        return

    sensor_manager = SensorManager(config=config)
    print("[INFO] Sensor Manager initialized with simulated spatial distance sensors:")
    print(sensor_manager.format_sensor_display(display_unit="cm"))

    model_path = det_cfg.get("model_path", "models/yolov8n.pt")
    detector = YoloDetector(
        model_path=model_path,
        confidence_threshold=det_cfg.get("confidence_threshold", 0.5),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
        device=det_cfg.get("device", "cpu"),
    )

    print(f"\n[INFO] Loading YOLO model weights from '{model_path}'...")
    if not detector.load_model():
        print(f"[ERROR] Failed to load YOLO model weights from '{model_path}'.")
        camera.release()
        return

    fusion_engine = SensorFusionEngine(
        default_range_m=config.get("sensors", {}).get("distance_sensor", {}).get("simulated_default_m", 2.5)
    )
    risk_analyzer = RiskAnalyzer(config=config)

    vibration = SimulatedVibration(enabled=True)
    voice = VoiceAlertManager(enabled=True)
    alert_manager = AlertManager(vibration=vibration, voice=voice, config=config)

    print("[INFO] Webcam, YOLO, Sensor Fusion, Risk Analyzer & Alert Manager ready. Starting stream...\n")
    window_name = disp_cfg.get("window_name", "Smart Navigation Shoe - Phase 5 Real-Time System")

    try:
        import cv2
        last_logged_time = 0.0

        while True:
            success, frame = camera.read_frame()
            if not success or frame is None:
                print("[WARNING] Empty or invalid frame received from webcam. Retrying...")
                time.sleep(0.03)
                continue

            detection_result = detector.detect(frame)
            sensor_readings = sensor_manager.get_readings_list()
            fused_obstacles = fusion_engine.fuse(detection_result.detections, sensor_readings)
            risk_assessment = risk_analyzer.evaluate(fused_obstacles)

            # Trigger Alerts (Vibration simulation + Voice alert debouncing)
            vib_text, alert_status = alert_manager.evaluate_and_trigger(risk_assessment, fused_obstacles)

            current_time = time.time()
            if fused_obstacles and (current_time - last_logged_time > 1.5):
                for obs in fused_obstacles:
                    print(obs.format_display())
                print(f"Overall Risk: {risk_assessment.overall_risk_level.value} | {vib_text} | Status: {alert_status}\n")
                last_logged_time = current_time

            annotated_frame = detector.draw_fused_obstacles(frame, fused_obstacles)
            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[INFO] User requested exit ('q' / ESC pressed). Stopping live video feed...")
                break
    except Exception as e:
        print(f"[ERROR] Error during real-time sensor fusion loop: {e}")
    finally:
        camera.release()
        voice.stop()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[INFO] Camera released and OpenCV display windows closed cleanly.")


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
    left_sensor = sensor_manager.get_sensor(list(sensor_manager.sensors.keys())[0])
    if left_sensor and hasattr(left_sensor, "set_simulated_distance_cm"):
        left_sensor.set_simulated_distance_cm(-50.0)

    print("  Setting RIGHT sensor to 500 cm (Out Of Bounds > 400 cm)...")
    right_sensor = sensor_manager.get_sensor(list(sensor_manager.sensors.keys())[2])
    if right_sensor and hasattr(right_sensor, "set_simulated_distance_cm"):
        right_sensor.set_simulated_distance_cm(500.0)

    print(sensor_manager.format_sensor_display(display_unit="cm"))

    print("\n3. Resetting Sensors to Normal Baseline Values (LEFT=150cm, CENTER=80cm, RIGHT=220cm)...")
    sensor_manager.set_simulated_distances_cm(150.0, 80.0, 220.0)
    print(sensor_manager.format_sensor_display(display_unit="cm"))

    print("\n[SUCCESS] Phase 3 Distance Sensor Demonstration Completed Cleanly.\n")


def run_realtime_detection(config_path: str = "config/config.yaml") -> None:
    """Phase 2: Runs real-time object detection using laptop webcam and Ultralytics YOLO."""
    config = load_config(config_path)
    cam_cfg = config.get("camera", {})
    det_cfg = config.get("detection", {})
    disp_cfg = det_cfg.get("display", {})

    print("=" * 70)
    print("  AI-Powered Smart Navigation Shoe — Phase 2 Real-Time YOLO Detection")
    print("=" * 70)
    print("Press 'q' or 'ESC' on the camera display window to exit.\n")

    device_id = cam_cfg.get("device_id", 0)
    camera = WebcamCamera(
        device_id=device_id,
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        fps=cam_cfg.get("fps", 30),
        simulation_fallback=False,
    )

    if not camera.connect():
        print(f"[ERROR] Could not connect to webcam device ID {device_id}.")
        print("[HINT] Ensure your laptop webcam is available and not opened in another application.")
        return

    model_path = det_cfg.get("model_path", "models/yolov8n.pt")
    detector = YoloDetector(
        model_path=model_path,
        confidence_threshold=det_cfg.get("confidence_threshold", 0.5),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
        device=det_cfg.get("device", "cpu"),
    )

    print(f"[INFO] Loading YOLO model weights from '{model_path}'...")
    if not detector.load_model():
        print(f"[ERROR] Failed to load YOLO model weights from '{model_path}'.")
        camera.release()
        return

    print("[INFO] Webcam & YOLO initialized. Starting live video feed...")
    window_name = disp_cfg.get("window_name", "Smart Navigation Shoe - Phase 2 Real-Time Detection")

    try:
        import cv2
        while True:
            success, frame = camera.read_frame()
            if not success or frame is None:
                print("[WARNING] Empty or invalid frame received from webcam. Retrying...")
                time.sleep(0.03)
                continue

            detection_result = detector.detect(frame)
            annotated_frame = detector.draw_detections(
                frame=frame,
                result=detection_result,
                show_confidence=disp_cfg.get("show_confidence", True),
                show_box=disp_cfg.get("show_box", True),
            )

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] User requested exit ('q' / ESC pressed). Stopping camera stream...")
                break
    except Exception as e:
        print(f"[ERROR] Error during real-time webcam detection loop: {e}")
    finally:
        camera.release()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[INFO] Camera released and OpenCV display windows closed cleanly.")


def run_dashboard_mode(config_path: str = "config/config.yaml") -> None:
    """Phase 6: Real-time Dashboard + Persistent Event & System Logging."""
    config = load_config(config_path)
    cam_cfg = config.get("camera", {})
    det_cfg = config.get("detection", {})
    disp_cfg = det_cfg.get("display", {})

    event_logger = EventLogger(config=config)
    event_logger.log_system_message("Dashboard mode started")

    print("=" * 70)
    print("  AI-Powered Smart Navigation Shoe — Phase 6 Dashboard & Event Logging")
    print("=" * 70)
    print("Press 'q' or 'ESC' on the camera display window to exit.\n")

    device_id = cam_cfg.get("device_id", 0)
    camera = WebcamCamera(
        device_id=device_id,
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        fps=cam_cfg.get("fps", 30),
        simulation_fallback=False,
    )

    if not camera.connect():
        event_logger.log_system_message(f"Could not connect to webcam device ID {device_id}", level="ERROR")
        print(f"[ERROR] Could not connect to webcam device ID {device_id}.")
        print("[HINT] Ensure your laptop webcam is available and not opened in another application.")
        return

    event_logger.log_system_message("Camera connected")

    sensor_manager = SensorManager(config=config)
    event_logger.log_system_message("Sensors initialized (SIMULATED)")

    model_path = det_cfg.get("model_path", "models/yolov8n.pt")
    detector = YoloDetector(
        model_path=model_path,
        confidence_threshold=det_cfg.get("confidence_threshold", 0.5),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
        device=det_cfg.get("device", "cpu"),
    )

    print(f"[INFO] Loading YOLO model weights from '{model_path}'...")
    if not detector.load_model():
        event_logger.log_system_message(f"Failed to load YOLO model weights from '{model_path}'", level="ERROR")
        print(f"[ERROR] Failed to load YOLO model weights from '{model_path}'.")
        camera.release()
        return

    event_logger.log_system_message("YOLO model loaded")

    fusion_engine = SensorFusionEngine(
        default_range_m=config.get("sensors", {}).get("distance_sensor", {}).get("simulated_default_m", 2.5)
    )
    risk_analyzer = RiskAnalyzer(config=config)

    vibration = SimulatedVibration(enabled=True)
    voice = VoiceAlertManager(enabled=True)
    alert_manager = AlertManager(vibration=vibration, voice=voice, config=config)

    dashboard = Dashboard(config=config)
    event_logger.log_system_message("Dashboard started")

    print("[INFO] Dashboard, Webcam, YOLO, Sensor Fusion & Event Logger active.\n")
    window_name = disp_cfg.get("window_name", "Smart Navigation Shoe - Phase 6 Dashboard")

    try:
        import cv2

        last_cli_update_time = 0.0

        while True:
            success, frame = camera.read_frame()
            if not success or frame is None:
                time.sleep(0.03)
                continue

            detection_result = detector.detect(frame)
            sensor_readings = sensor_manager.get_readings_list()
            fused_obstacles = fusion_engine.fuse(detection_result.detections, sensor_readings)
            risk_assessment = risk_analyzer.evaluate(fused_obstacles)

            vib_action, voice_msg = alert_manager.evaluate_and_trigger(risk_assessment, fused_obstacles)
            vib_text = vib_action.replace("VIBRATION: ", "")

            # Log event (debounced)
            event_logger.log_event(
                fused_obstacles=fused_obstacles,
                overall_risk=risk_assessment.overall_risk_level,
                vibration_action=vib_text,
                voice_message=voice_msg,
            )

            # Build Dashboard Snapshot
            snapshot = DashboardSnapshot.from_pipeline_results(
                fused_obstacles=fused_obstacles,
                sensor_readings=sensor_readings,
                overall_risk=risk_assessment.overall_risk_level,
                vibration_action=vib_text,
                voice_message=voice_msg,
                recent_events=event_logger.recent_events,
                camera_status="CONNECTED" if camera.is_connected else "DISCONNECTED",
                yolo_status="ACTIVE" if detector.is_loaded else "INACTIVE",
                sensors_status="ACTIVE (SIMULATED)",
            )

            # Print ASCII terminal dashboard periodically
            now = time.time()
            if now - last_cli_update_time >= dashboard.refresh_interval:
                dashboard.display_cli(snapshot, clear_screen=False)
                last_cli_update_time = now

            # Annotate video frame with bounding boxes and overlay dashboard banner
            annotated_frame = detector.draw_fused_obstacles(frame, fused_obstacles)
            annotated_frame = dashboard.render_video_overlay(annotated_frame, snapshot)

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                event_logger.log_system_message("User requested exit from dashboard mode")
                print("\n[INFO] User requested exit ('q' / ESC pressed). Stopping dashboard stream...")
                break

    except Exception as e:
        event_logger.log_system_message(f"Error during real-time dashboard loop: {e}", level="ERROR")
        print(f"[ERROR] Error during real-time dashboard loop: {e}")
    finally:
        camera.release()
        event_logger.log_system_message("Camera released")
        voice.stop()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        event_logger.log_system_message("Dashboard closed cleanly")
        print("[INFO] Camera released and dashboard display window closed cleanly.")


def main() -> None:
    """Main execution function supporting CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Smart Navigation Shoe - AI Assistive Navigation System"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fusion",
        choices=["fusion", "alerts", "detection", "sensors", "simulation", "architecture", "dashboard"],
        help="Execution mode: 'fusion' for sensor fusion, 'alerts' for alert demo, 'sensors' for sensor demo, 'detection' for live YOLO, 'simulation' for test loop, 'dashboard' for Phase 6 dashboard & logging",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file",
    )

    args, _ = parser.parse_known_args()

    if args.mode == "dashboard":
        run_dashboard_mode(config_path=args.config)
    elif args.mode == "fusion":
        run_realtime_fusion(config_path=args.config)
    elif args.mode == "alerts":
        run_alert_demo(config_path=args.config)
    elif args.mode == "detection":
        run_realtime_detection(config_path=args.config)
    elif args.mode == "sensors":
        run_sensor_demo(config_path=args.config)
    else:
        print("=" * 70)
        print("  AI-Powered Smart Navigation Shoe — Simulation Architecture Run")
        print("=" * 70)
        app = SmartNavigationShoeApp(config_path=args.config)
        if app.initialize():
            print("\n[INFO] Running 3 demonstration architecture steps...\n")
            for i in range(3):
                print(f"--- Iteration {i+1} ---")
                app.run_step()
                time.sleep(0.5)
            app.shutdown()
            print("\n[SUCCESS] Phase 1 architecture loop executed successfully.")
        else:
            print("[ERROR] Application initialization failed.")


if __name__ == "__main__":
    main()

