"""Unit tests for the shared navigation pipeline, logging setup, and CLI mode selection."""

import csv
import logging
import sys
from unittest.mock import patch

import pytest

from src import main as app_main
from src.alerts.simulated_vibration import SimulatedVibration
from src.core.enums import DirectionCommand, ObstacleZone, RiskLevel
from src.core.models import BoundingBox, DetectionItem
from src.event_logging.event_logger import EventLogger
from src.pipeline import NavigationPipeline, create_alert_system, create_detector
from src.utils.helpers import load_config
from src.utils.logger import setup_logging, get_logger


@pytest.fixture
def config():
    cfg = load_config()
    cfg["alerts"]["voice"]["enabled"] = False  # No TTS thread in tests
    cfg["sensors"]["simulation"]["noise_cm"] = 0.0
    return cfg


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    yield
    for handler in list(root.handlers):
        if getattr(handler, "_smart_shoe", False):
            root.removeHandler(handler)
            handler.close()
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])


def test_alert_system_honors_config(config):
    config["alerts"]["vibration"]["enabled"] = False
    vibration, voice, alert_manager = create_alert_system(config)
    assert isinstance(vibration, SimulatedVibration) and not vibration.enabled
    assert not voice.enabled
    assert alert_manager.cooldown_seconds == config["alerts"]["voice"]["cooldown_seconds"]


def test_detector_from_config(config):
    detector = create_detector(config)
    assert detector.model_path == config["detection"]["model_path"]
    assert "person" in detector.target_classes


def test_sensor_only_step(config):
    pipeline = NavigationPipeline(config)
    pipeline.scenario_engine.set_distances_cm(300, 40, 300)
    result = pipeline.process()

    assert result.risk_assessment.overall_risk_level == RiskLevel.DANGER
    assert result.direction.recommended_direction == DirectionCommand.STOP
    assert result.alert_triggered
    assert result.last_spoken == "Danger. Obstacle very close. Stop."
    assert result.vibration_text == "BOTH - RAPID PULSES"
    pipeline.shutdown()


def test_camera_step_uses_tracker_and_camera_linked_sensors(config):
    pipeline = NavigationPipeline(config)
    assert pipeline.scenario_engine.mode == "camera"
    person = DetectionItem("person", 0.9, BoundingBox(145, 80, 495, 480), ObstacleZone.CENTER)

    first = pipeline.process([person], (640, 480))
    assert first.detections == []  # Tracker needs 2 sightings

    second = pipeline.process([person], (640, 480))
    assert [d.track_id for d in second.detections] == [1]
    assert second.fused_obstacles[0].object_id == "track_1"
    assert second.risk_assessment.overall_risk_level == RiskLevel.WARNING  # ~71 cm estimated
    pipeline.shutdown()


def test_log_event_records_spoken_text(config, tmp_path):
    pipeline = NavigationPipeline(config)
    pipeline.scenario_engine.set_distances_cm(300, 80, 300)
    logger = EventLogger(event_log_path=str(tmp_path / "e.csv"), system_log_path=str(tmp_path / "s.log"))

    assert NavigationPipeline.log_event(logger, pipeline.process())
    with open(tmp_path / "e.csv", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[0]
    assert row["voice_message"] == "Warning. Obstacle ahead. Slow down. Move slightly right."
    assert row["direction"] == "SLIGHT_RIGHT"
    pipeline.shutdown()


def test_setup_logging_writes_module_logs_once(tmp_path, restore_logging, capsys):
    log_path = tmp_path / "system.log"
    config = {"logging": {"system_log": str(log_path), "console_level": "WARNING"}}
    setup_logging(config)
    setup_logging(config)  # Reconfiguring must not duplicate handlers

    get_logger("alert_manager").info("module info line")
    events = EventLogger(event_log_path=str(tmp_path / "e.csv"), system_log_path=str(log_path))
    events.log_system_message("event line")
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = log_path.read_text(encoding="utf-8")
    assert content.count("module info line") == 1
    assert content.count("event line") == 1
    assert "[smart_shoe.events]" in content
    assert "module info line" not in capsys.readouterr().err  # INFO stays off the console


def test_default_mode_comes_from_config(config):
    config["system"]["mode"] = "sensors"
    with patch.object(app_main, "load_config", return_value=config), \
            patch.object(app_main, "setup_logging"), \
            patch.object(app_main, "run_sensor_demo") as sensors, \
            patch.object(app_main, "run_dashboard_mode") as dashboard, \
            patch.object(sys, "argv", ["main"]):
        app_main.main()
    sensors.assert_called_once()
    dashboard.assert_not_called()


def test_unknown_scenario_rejected(config, capsys):
    with patch.object(app_main, "load_config", return_value=config), \
            patch.object(app_main, "setup_logging"), \
            patch.object(app_main, "run_dashboard_mode") as dashboard, \
            patch.object(sys, "argv", ["main", "--scenario", "nope"]):
        app_main.main()
    dashboard.assert_not_called()
    assert "Unknown scenario 'nope'" in capsys.readouterr().out
