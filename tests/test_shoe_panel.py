"""Unit tests for the simulated shoe panel, dashboard frame layout, and risk-colored detections."""

import numpy as np

from src.core.models import BoundingBox, DirectionGuidance, FusedObstacle, SensorReading
from src.core.enums import DirectionCommand, ObstacleZone, RiskLevel, SensorStatus
from src.dashboard.dashboard import Dashboard
from src.dashboard.dashboard_data import DashboardSnapshot
from src.dashboard.shoe_panel import (
    DESIGN_HEIGHT,
    MOTOR_OFF,
    PANEL_WIDTH,
    parse_vibration,
    pulse_is_on,
    render_shoe_panel,
    zone_risks,
)
from src.detection.yolo_detector import YoloDetector
from src.utils.visual import FAULT_COLOR_BGR, FrameRateMeter, risk_color

# (row, col) inside each motor circle, left of the "L"/"R" letter, in the design-size panel
LEFT_MOTOR = (240, 106)
RIGHT_MOTOR = (240, 138)


def snapshot(vibration="OFF", risk=RiskLevel.SAFE, direction=None, obstacles=(), readings=None):
    readings = readings if readings is not None else [
        SensorReading(3.0, position=ObstacleZone.LEFT),
        SensorReading(3.0, position=ObstacleZone.CENTER),
        SensorReading(3.0, position=ObstacleZone.RIGHT),
    ]
    return DashboardSnapshot.from_pipeline_results(
        fused_obstacles=list(obstacles),
        sensor_readings=readings,
        overall_risk=risk,
        vibration_action=vibration,
        direction=direction,
        fps=9.5,
        inference_ms=80,
    )


def pixel(img, row_col):
    return tuple(int(v) for v in img[row_col[0], row_col[1]])


def test_parse_vibration():
    assert parse_vibration("LEFT - SHORT PULSE") == (["LEFT"], "SHORT PULSE")
    assert parse_vibration("VIBRATION: BOTH - RAPID PULSES") == (["LEFT", "RIGHT"], "RAPID PULSES")
    assert parse_vibration("OFF") == ([], "")


def test_pulse_timing_matches_pattern():
    assert pulse_is_on("SHORT PULSE", 0.1) and not pulse_is_on("SHORT PULSE", 0.5)
    assert pulse_is_on("MEDIUM PULSE", 0.6) and not pulse_is_on("MEDIUM PULSE", 0.4)
    assert pulse_is_on("RAPID PULSES", 0.17) and not pulse_is_on("RAPID PULSES", 0.1)
    assert not pulse_is_on("", 0.0)


def test_zone_risks_uses_worst_per_zone():
    snap = snapshot(obstacles=[
        FusedObstacle("1", "chair", 0.9, None, 1.5, ObstacleZone.CENTER, RiskLevel.CAUTION),
        FusedObstacle("2", "person", 0.9, None, 0.4, ObstacleZone.CENTER, RiskLevel.DANGER),
        FusedObstacle("3", "dog", 0.9, None, 2.5, ObstacleZone.LEFT, RiskLevel.SAFE),
    ])
    assert zone_risks(snap) == {"CENTER": "DANGER", "LEFT": "SAFE"}


def test_panel_size_and_scaling():
    assert render_shoe_panel(snapshot(), now=0.0).shape == (DESIGN_HEIGHT, PANEL_WIDTH, 3)
    assert render_shoe_panel(snapshot(), height=360, now=0.0).shape == (360, 195, 3)


def test_active_motor_lights_in_pulse_and_idle_motor_stays_off():
    snap = snapshot(vibration="LEFT - SHORT PULSE", risk=RiskLevel.CAUTION)

    during_pulse = render_shoe_panel(snap, now=0.05)
    assert pixel(during_pulse, LEFT_MOTOR) == risk_color("CAUTION")
    assert pixel(during_pulse, RIGHT_MOTOR) == MOTOR_OFF

    between_pulses = render_shoe_panel(snap, now=0.5)
    assert pixel(between_pulses, LEFT_MOTOR) == MOTOR_OFF


def test_both_motors_light_for_center_hazard():
    panel = render_shoe_panel(snapshot(vibration="BOTH - RAPID PULSES", risk=RiskLevel.DANGER), now=0.01)
    assert pixel(panel, LEFT_MOTOR) == risk_color("DANGER")
    assert pixel(panel, RIGHT_MOTOR) == risk_color("DANGER")


def test_stop_sign_and_arrow():
    stop = render_shoe_panel(
        snapshot(risk=RiskLevel.DANGER, direction=DirectionGuidance(DirectionCommand.STOP)), now=0.0
    )
    assert pixel(stop, (385, 130)) == (0, 0, 220)  # Red octagon above the "STOP" text

    arrow = render_shoe_panel(snapshot(direction=DirectionGuidance(DirectionCommand.TURN_RIGHT)), now=0.0)
    assert pixel(arrow, (400, 140)) == (255, 255, 0)  # Horizontal arrow shaft right of center


def test_faulty_sensor_cone_drawn_gray():
    readings = [
        SensorReading(-1.0, position=ObstacleZone.LEFT, status=SensorStatus.INVALID, is_valid=False),
        SensorReading(3.0, position=ObstacleZone.CENTER),
        SensorReading(3.0, position=ObstacleZone.RIGHT),
    ]
    panel = render_shoe_panel(snapshot(readings=readings), now=0.0)
    left_sensor_dot = (196, 110)
    assert pixel(panel, left_sensor_dot) == FAULT_COLOR_BGR


def test_render_frame_layout():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dashboard = Dashboard()
    combined = dashboard.render_frame(frame, snapshot(), now=0.0)
    assert combined.shape == (480, 640 + PANEL_WIDTH, 3)
    assert combined[:200, int(640 * 0.33)].any()  # Dashed zone divider on the video

    no_panel = Dashboard(config={"dashboard": {"show_side_panel": False}}).render_frame(frame, snapshot())
    assert no_panel.shape == frame.shape


def test_detection_boxes_colored_by_risk():
    detector = YoloDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    danger = FusedObstacle("1", "person", 0.9, BoundingBox(100, 150, 300, 400), 0.4, ObstacleZone.CENTER, RiskLevel.DANGER)
    safe = FusedObstacle("2", "chair", 0.9, BoundingBox(400, 150, 600, 400), 3.0, ObstacleZone.RIGHT, RiskLevel.SAFE)

    out = detector.draw_fused_obstacles(frame, [danger, safe])
    assert pixel(out, (300, 100)) == risk_color("DANGER")  # Left edge of the danger box
    assert pixel(out, (300, 400)) == risk_color("SAFE")

    fixed = detector.draw_fused_obstacles(frame, [danger], box_color=(255, 0, 0))
    assert pixel(fixed, (300, 100)) == (255, 0, 0)


def test_frame_rate_meter():
    clock = iter([0.0, 0.1, 0.2, 0.3])
    meter = FrameRateMeter(smoothing=1.0, clock=lambda: next(clock))
    assert meter.tick() == 0.0
    assert abs(meter.tick() - 10.0) < 1e-6
    assert abs(meter.tick() - 10.0) < 1e-6
