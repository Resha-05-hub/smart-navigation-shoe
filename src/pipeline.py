"""Shared navigation pipeline: detections -> simulated sensors -> fusion -> risk -> direction -> alerts.

Every execution mode (fusion, dashboard, scenario, architecture) runs the same per-frame processing,
so it lives here once instead of being repeated in each mode.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .alerts.alert_manager import AlertManager
from .alerts.raspberry_pi_vibration import RaspberryPiVibration
from .alerts.simulated_vibration import SimulatedVibration
from .alerts.vibration_interface import VibrationInterface
from .alerts.voice_alert import VoiceAlertManager
from .core.models import DetectionItem, DirectionGuidance, FusedObstacle, RiskAssessment, SensorReading
from .decision.direction_analyzer import DirectionAnalyzer
from .decision.risk_analyzer import RiskAnalyzer
from .detection.object_tracker import ObjectTracker
from .detection.yolo_detector import YoloDetector
from .event_logging.event_logger import EventLogger
from .fusion.sensor_fusion import SensorFusionEngine
from .sensors.scenario_engine import SensorScenarioEngine
from .sensors.sensor_manager import SensorManager


def create_detector(config: Dict[str, Any]) -> YoloDetector:
    """YOLO detector configured from the detection section (weights are loaded separately)."""
    det_cfg = config.get("detection", {})
    return YoloDetector(
        model_path=det_cfg.get("model_path", "models/yolov8n.pt"),
        confidence_threshold=det_cfg.get("confidence_threshold", 0.5),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
        device=det_cfg.get("device", "cpu"),
        target_classes=det_cfg.get("target_classes"),
    )


def create_alert_system(config: Dict[str, Any]) -> Tuple[VibrationInterface, VoiceAlertManager, AlertManager]:
    """Vibration (simulated or GPIO), voice, and the alert manager, honoring the alerts config."""
    alert_cfg = config.get("alerts", {})
    vib_cfg = alert_cfg.get("vibration", {})
    voice_cfg = alert_cfg.get("voice", {})

    vibration: VibrationInterface
    if vib_cfg.get("type", "simulated") == "gpio":
        pins = vib_cfg.get("gpio_pins", {})
        vibration = RaspberryPiVibration(
            left_pin=pins.get("left", 5),
            center_pin=pins.get("center", 6),
            right_pin=pins.get("right", 13),
        )
    else:
        vibration = SimulatedVibration(enabled=vib_cfg.get("enabled", True))

    voice = VoiceAlertManager(
        enabled=voice_cfg.get("enabled", True),
        speech_rate=voice_cfg.get("speech_rate", 160),
        volume=voice_cfg.get("volume", 0.9),
    )
    return vibration, voice, AlertManager(vibration=vibration, voice=voice, config=config)


@dataclass
class PipelineResult:
    """Everything produced for one frame/tick."""
    detections: List[DetectionItem]
    sensor_readings: List[SensorReading]
    fused_obstacles: List[FusedObstacle]
    risk_assessment: RiskAssessment
    direction: DirectionGuidance
    vibration_text: str   # e.g. "BOTH - MEDIUM PULSE" or "OFF"
    alert_status: str     # "[ALERT TRIGGERED] ..." / "[ALERT SUPPRESSED] ..."
    alert_triggered: bool
    last_spoken: str      # Most recent voice message the user heard
    extra: Dict[str, Any] = field(default_factory=dict)


class NavigationPipeline:
    """Per-frame decision pipeline shared by all modes.

    Owns the simulated sensors (and their scenario engine), the object tracker, fusion,
    risk and direction analysis, and the alert system.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        scenario: Optional[str] = None,
        use_tracker: bool = True,
    ) -> None:
        self.config = config
        self.sensor_manager = SensorManager(config=config)
        self.scenario_engine = SensorScenarioEngine(self.sensor_manager, config=config)
        if scenario:
            self.scenario_engine.start_scenario(scenario)

        tracking_enabled = use_tracker and config.get("tracking", {}).get("enabled", True)
        self.tracker: Optional[ObjectTracker] = ObjectTracker(config=config) if tracking_enabled else None
        self.fusion_engine = SensorFusionEngine(config=config)
        self.risk_analyzer = RiskAnalyzer(config=config)
        self.direction_analyzer = DirectionAnalyzer(config=config)
        self.vibration, self.voice, self.alert_manager = create_alert_system(config)

    def process(
        self,
        detections: Optional[List[DetectionItem]] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> PipelineResult:
        """Runs one step. Pass YOLO detections and (width, height) for camera frames; omit both for
        sensor-only operation (scenario mode)."""
        detections = list(detections or [])
        if self.tracker is not None and frame_size is not None:
            detections = self.tracker.update(detections, frame_size[0])

        self.scenario_engine.update(detections, frame_size)
        readings = self.sensor_manager.get_readings_list()
        fused = self.fusion_engine.fuse(detections, readings)
        risk = self.risk_analyzer.evaluate(fused, readings)
        direction = self.direction_analyzer.analyze_path(fused, risk, readings)
        vib_action, alert_status = self.alert_manager.evaluate_and_trigger(risk, fused, direction)

        return PipelineResult(
            detections=detections,
            sensor_readings=readings,
            fused_obstacles=fused,
            risk_assessment=risk,
            direction=direction,
            vibration_text=vib_action.replace("VIBRATION: ", ""),
            alert_status=alert_status,
            alert_triggered=self.alert_manager.last_alert_triggered,
            last_spoken=self.alert_manager.last_triggered_text,
        )

    @staticmethod
    def log_event(event_logger: EventLogger, result: PipelineResult) -> bool:
        """Writes the step to the event CSV (debounced); voice text only when it was actually spoken."""
        return event_logger.log_event(
            fused_obstacles=result.fused_obstacles,
            overall_risk=result.risk_assessment.overall_risk_level,
            vibration_action=result.vibration_text,
            voice_message=result.last_spoken if result.alert_triggered else "",
            alert_triggered=result.alert_triggered,
            direction=result.direction.recommended_direction.value,
        )

    def shutdown(self) -> None:
        """Stops actuators and the voice worker."""
        self.vibration.stop_all()
        self.voice.stop()
