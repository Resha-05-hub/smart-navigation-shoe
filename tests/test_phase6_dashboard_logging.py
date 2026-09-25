"""Unit tests for Phase 6 — Dashboard and Event Logging."""

import os
import csv
from pathlib import Path
import pytest
import numpy as np

from src.core.models import FusedObstacle, BoundingBox, SensorReading
from src.core.enums import RiskLevel, ObstacleZone, SensorStatus
from src.dashboard.dashboard_data import (
    DashboardSnapshot,
    DetectionSummary,
    SensorStatusSummary,
    RecentEvent,
)
from src.dashboard.dashboard import Dashboard
from src.event_logging.event_logger import EventLogger, CSV_HEADERS


@pytest.fixture
def temp_log_dir(tmp_path):
    """Provides temporary log directory for test isolation."""
    event_log = tmp_path / "test_detection_events.csv"
    system_log = tmp_path / "test_system.log"
    return event_log, system_log


def test_1_dashboard_data_creation_from_detection_results():
    """Test 1: Dashboard data can be created from detection results."""
    obstacle = FusedObstacle(
        object_id="1",
        label="person",
        confidence=0.87,
        bbox=BoundingBox(10, 10, 50, 50),
        distance_m=0.8,
        zone=ObstacleZone.CENTER,
        risk_level=RiskLevel.WARNING,
    )
    sensors = [
        SensorReading(distance_m=1.5, position=ObstacleZone.LEFT),
        SensorReading(distance_m=0.8, position=ObstacleZone.CENTER),
        SensorReading(distance_m=2.2, position=ObstacleZone.RIGHT),
    ]

    snapshot = DashboardSnapshot.from_pipeline_results(
        fused_obstacles=[obstacle],
        sensor_readings=sensors,
        overall_risk=RiskLevel.WARNING,
        vibration_action="BOTH - MEDIUM PULSE",
        voice_message="Warning. Obstacle ahead. Slow down.",
    )

    assert isinstance(snapshot, DashboardSnapshot)
    assert len(snapshot.detections) == 1
    assert snapshot.overall_risk == "WARNING"
    assert snapshot.alert_status.vibration_action == "BOTH - MEDIUM PULSE"
    assert snapshot.alert_status.voice_message == "Warning. Obstacle ahead. Slow down."


def test_2_object_name_display():
    """Test 2: Object name is correctly displayed."""
    obstacle = FusedObstacle(
        object_id="1",
        label="person",
        confidence=0.87,
        bbox=None,
        distance_m=0.8,
        zone=ObstacleZone.CENTER,
        risk_level=RiskLevel.WARNING,
    )
    snapshot = DashboardSnapshot.from_pipeline_results(
        fused_obstacles=[obstacle],
        sensor_readings=[],
        overall_risk="WARNING",
    )
    assert snapshot.detections[0].object_name == "person"

    dashboard = Dashboard()
    cli_output = dashboard.render_cli(snapshot)
    assert "person" in cli_output


def test_3_confidence_display():
    """Test 3: Confidence is correctly displayed."""
    obstacle = FusedObstacle(
        object_id="1",
        label="person",
        confidence=0.87,
        bbox=None,
        distance_m=0.8,
        zone=ObstacleZone.CENTER,
        risk_level=RiskLevel.WARNING,
    )
    snapshot = DashboardSnapshot.from_pipeline_results(
        fused_obstacles=[obstacle],
        sensor_readings=[],
        overall_risk="WARNING",
    )
    assert snapshot.detections[0].confidence == 0.87
    assert snapshot.detections[0].confidence_pct_str == "87%"

    dashboard = Dashboard()
    cli_output = dashboard.render_cli(snapshot)
    assert "87%" in cli_output


def test_4_zone_display():
    """Test 4: Zone is correctly displayed."""
    obstacle = FusedObstacle(
        object_id="1",
        label="chair",
        confidence=0.75,
        bbox=None,
        distance_m=1.5,
        zone=ObstacleZone.LEFT,
        risk_level=RiskLevel.CAUTION,
    )
    snapshot = DashboardSnapshot.from_pipeline_results(
        fused_obstacles=[obstacle],
        sensor_readings=[],
        overall_risk="CAUTION",
    )
    assert snapshot.detections[0].zone == "LEFT"

    dashboard = Dashboard()
    cli_output = dashboard.render_cli(snapshot)
    assert "LEFT" in cli_output


def test_5_distance_display():
    """Test 5: Distance is correctly displayed in cm."""
    obstacle = FusedObstacle(
        object_id="1",
        label="person",
        confidence=0.87,
        bbox=None,
        distance_m=0.8,  # 80 cm
        zone=ObstacleZone.CENTER,
        risk_level=RiskLevel.WARNING,
    )
    snapshot = DashboardSnapshot.from_pipeline_results(
        fused_obstacles=[obstacle],
        sensor_readings=[],
        overall_risk="WARNING",
    )
    assert snapshot.detections[0].distance_cm == 80.0

    dashboard = Dashboard()
    cli_output = dashboard.render_cli(snapshot)
    assert "80 cm" in cli_output


def test_6_risk_level_display():
    """Test 6: Risk level is correctly displayed."""
    obstacle = FusedObstacle(
        object_id="1",
        label="suitcase",
        confidence=0.95,
        bbox=None,
        distance_m=0.3,
        zone=ObstacleZone.RIGHT,
        risk_level=RiskLevel.DANGER,
    )
    snapshot = DashboardSnapshot.from_pipeline_results(
        fused_obstacles=[obstacle],
        sensor_readings=[],
        overall_risk=RiskLevel.DANGER,
    )
    assert snapshot.detections[0].risk_level == "DANGER"
    assert snapshot.overall_risk == "DANGER"

    dashboard = Dashboard()
    cli_output = dashboard.render_cli(snapshot)
    assert "DANGER" in cli_output


def test_7_multiple_objects_supported():
    """Test 7: Multiple detected objects are supported in dashboard."""
    obs1 = FusedObstacle("1", "person", 0.87, None, 0.8, ObstacleZone.CENTER, RiskLevel.WARNING)
    obs2 = FusedObstacle("2", "suitcase", 0.55, None, 2.2, ObstacleZone.RIGHT, RiskLevel.SAFE)
    obs3 = FusedObstacle("3", "chair", 0.65, None, 1.5, ObstacleZone.LEFT, RiskLevel.CAUTION)

    snapshot = DashboardSnapshot.from_pipeline_results(
        fused_obstacles=[obs1, obs2, obs3],
        sensor_readings=[],
        overall_risk="WARNING",
    )

    assert len(snapshot.detections) == 3
    names = [d.object_name for d in snapshot.detections]
    assert "person" in names
    assert "suitcase" in names
    assert "chair" in names

    dashboard = Dashboard()
    cli_output = dashboard.render_cli(snapshot)
    assert "person" in cli_output
    assert "suitcase" in cli_output
    assert "chair" in cli_output


def test_8_sensor_values_passed_to_dashboard():
    """Test 8: Sensor values are correctly passed to dashboard data."""
    sensors = [
        SensorReading(distance_m=1.5, position=ObstacleZone.LEFT, is_valid=True),
        SensorReading(distance_m=0.8, position=ObstacleZone.CENTER, is_valid=True),
        SensorReading(distance_m=2.2, position=ObstacleZone.RIGHT, is_valid=True),
    ]
    snapshot = DashboardSnapshot.from_pipeline_results(
        fused_obstacles=[],
        sensor_readings=sensors,
        overall_risk="SAFE",
    )
    assert len(snapshot.sensors) == 3
    s_map = {s.name: s.distance_cm for s in snapshot.sensors}
    assert s_map["LEFT"] == 150.0
    assert s_map["CENTER"] == 80.0
    assert s_map["RIGHT"] == 220.0

    dashboard = Dashboard()
    cli_output = dashboard.render_cli(snapshot)
    assert "LEFT" in cli_output
    assert "150 cm" in cli_output
    assert "CENTER" in cli_output
    assert "80 cm" in cli_output
    assert "RIGHT" in cli_output
    assert "220 cm" in cli_output


def test_9_event_logger_creates_csv_file(temp_log_dir):
    """Test 9: Event logger creates the CSV file."""
    event_path, sys_path = temp_log_dir
    logger = EventLogger(event_log_path=str(event_path), system_log_path=str(sys_path))
    assert event_path.exists()
    assert event_path.is_file()


def test_10_csv_headers_are_correct(temp_log_dir):
    """Test 10: CSV headers are correct."""
    event_path, sys_path = temp_log_dir
    logger = EventLogger(event_log_path=str(event_path), system_log_path=str(sys_path))

    with open(event_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == CSV_HEADERS


def test_11_detection_event_written_correctly(temp_log_dir):
    """Test 11: Detection event is written correctly to CSV."""
    event_path, sys_path = temp_log_dir
    logger = EventLogger(event_log_path=str(event_path), system_log_path=str(sys_path))

    obs = FusedObstacle("1", "person", 0.87, None, 0.8, ObstacleZone.CENTER, RiskLevel.WARNING)
    logged = logger.log_event(
        fused_obstacles=[obs],
        overall_risk=RiskLevel.WARNING,
        vibration_action="BOTH_MEDIUM_PULSE",
        voice_message="Warning. Obstacle ahead. Slow down.",
        force=True,
    )
    assert logged is True

    with open(event_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 2  # Header + 1 data row
    data_row = rows[1]
    # timestamp, object_name, confidence, zone, distance_cm, risk_level, vibration_action, voice_message
    assert data_row[1] == "person"
    assert data_row[2] == "0.87"
    assert data_row[3] == "CENTER"
    assert data_row[4] == "80"
    assert data_row[5] == "WARNING"
    assert data_row[6] == "BOTH_MEDIUM_PULSE"
    assert data_row[7] == "Warning. Obstacle ahead. Slow down."


def test_12_duplicate_continuous_events_not_logged(temp_log_dir):
    """Test 12: Duplicate continuous events are not logged unnecessarily."""
    event_path, sys_path = temp_log_dir
    logger = EventLogger(event_log_path=str(event_path), system_log_path=str(sys_path))

    obs = FusedObstacle("1", "person", 0.87, None, 0.8, ObstacleZone.CENTER, RiskLevel.WARNING)

    # First log
    first = logger.log_event([obs], RiskLevel.WARNING, "BOTH_MEDIUM_PULSE", "Warning. Obstacle ahead.")
    assert first is True

    # Immediate second log with identical parameters (should be debounced)
    second = logger.log_event([obs], RiskLevel.WARNING, "BOTH_MEDIUM_PULSE", "Warning. Obstacle ahead.")
    assert second is False

    with open(event_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2  # Header + 1 data row only

    # Change condition (risk level changes to DANGER)
    obs_danger = FusedObstacle("1", "person", 0.87, None, 0.4, ObstacleZone.CENTER, RiskLevel.DANGER)
    third = logger.log_event([obs_danger], RiskLevel.DANGER, "BOTH_FAST_PULSE", "Danger! Stop!")
    assert third is True

    with open(event_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 3  # Header + 2 data rows


def test_13_system_logging_works(temp_log_dir):
    """Test 13: System logging writes messages to system.log file."""
    event_path, sys_path = temp_log_dir
    logger = EventLogger(event_log_path=str(event_path), system_log_path=str(sys_path))

    logger.log_system_message("Dashboard started", level="INFO")
    logger.log_system_message("Camera connected", level="INFO")

    assert sys_path.exists()
    content = sys_path.read_text(encoding="utf-8")
    assert "Dashboard started" in content
    assert "Camera connected" in content


def test_14_video_overlay_rendering():
    """Test video overlay generation without errors."""
    dashboard = Dashboard()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    snapshot = DashboardSnapshot()
    annotated = dashboard.render_video_overlay(dummy_frame, snapshot)
    assert annotated.shape == dummy_frame.shape


def test_15_camera_is_connected_integration():
    """Test 15: Camera is_connected property integrates cleanly with dashboard snapshot."""
    from src.camera.webcam_camera import WebcamCamera
    camera = WebcamCamera(device_id=999, simulation_fallback=True)
    camera.connect()
    assert camera.is_connected is True

    snapshot = DashboardSnapshot.from_pipeline_results(
        fused_obstacles=[],
        sensor_readings=[],
        overall_risk="SAFE",
        camera_status="CONNECTED" if camera.is_connected else "DISCONNECTED",
    )
    assert snapshot.system_status.camera_status == "CONNECTED"

    camera.release()
    assert camera.is_connected is False

    snapshot_off = DashboardSnapshot.from_pipeline_results(
        fused_obstacles=[],
        sensor_readings=[],
        overall_risk="SAFE",
        camera_status="CONNECTED" if camera.is_connected else "DISCONNECTED",
    )
    assert snapshot_off.system_status.camera_status == "DISCONNECTED"


def test_16_recent_event_deduplication_and_multi_object(temp_log_dir):
    """Test 16: Verify recent event history deduplication and multi-object representation."""
    event_path, sys_path = temp_log_dir
    logger = EventLogger(event_log_path=str(event_path), system_log_path=str(sys_path))

    obs1 = FusedObstacle("1", "person", 0.87, None, 0.8, ObstacleZone.CENTER, RiskLevel.WARNING)
    obs2 = FusedObstacle("2", "chair", 0.65, None, 2.2, ObstacleZone.RIGHT, RiskLevel.SAFE)

    # Frame 1: person and chair detected
    logger.log_event([obs1, obs2], RiskLevel.WARNING, "BOTH_MEDIUM_PULSE", "Warning. Obstacle ahead.")
    assert len(logger.recent_events) == 2
    assert logger.recent_events[0].object_name == "chair"
    assert logger.recent_events[1].object_name == "person"

    # Frame 2: identical person and chair detected -> should be debounced
    logger.log_event([obs1, obs2], RiskLevel.WARNING, "BOTH_MEDIUM_PULSE", "Warning. Obstacle ahead.")
    assert len(logger.recent_events) == 2  # Still 2, no continuous duplicate entries!

    # Frame 3: person moves closer (distance 80cm -> 40cm, risk WARNING -> DANGER)
    obs1_close = FusedObstacle("1", "person", 0.90, None, 0.4, ObstacleZone.CENTER, RiskLevel.DANGER)
    logger.log_event([obs1_close, obs2], RiskLevel.DANGER, "BOTH_FAST_PULSE", "Danger! Stop!")
    assert len(logger.recent_events) == 3
    assert logger.recent_events[0].object_name == "person"
    assert logger.recent_events[0].risk_level == "DANGER"
    assert logger.recent_events[0].distance_cm == 40.0


