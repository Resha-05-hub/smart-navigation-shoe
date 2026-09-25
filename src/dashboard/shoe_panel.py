"""Side panel visualizing the SIMULATED shoe: sensor cones, vibration motors, direction, and speed.

Everything drawn here represents simulated hardware for the software-only demonstration.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.dashboard.dashboard_data import DashboardSnapshot
from src.decision.risk_analyzer import RISK_PRIORITY
from src.core.enums import RiskLevel
from src.utils.visual import BGR, FAULT_COLOR_BGR, risk_color

PANEL_WIDTH = 260
DESIGN_HEIGHT = 480  # The panel is drawn at this height, then scaled to the video height
MAX_RANGE_CM = 400.0

BACKGROUND: BGR = (28, 28, 28)
TEXT: BGR = (230, 230, 230)
DIM: BGR = (120, 120, 120)
MOTOR_OFF: BGR = (70, 70, 70)
ARROW: BGR = (255, 255, 0)

# Sensor mount points on the toe and the direction each one faces (degrees; -90 = straight ahead)
SENSOR_LAYOUT: Dict[str, Tuple[Tuple[int, int], float]] = {
    "LEFT": ((110, 196), -118.0),
    "CENTER": ((130, 186), -90.0),
    "RIGHT": ((150, 196), -62.0),
}
CONE_HALF_ANGLE = 13.0
CONE_MIN_PX, CONE_MAX_PX = 18.0, 140.0

# Vibration pulse timing (on_seconds, period_seconds), matching SimulatedVibration's patterns
PULSE_TIMING: Dict[str, Tuple[float, float]] = {
    "SHORT PULSE": (0.15, 1.0),
    "MEDIUM PULSE": (0.25, 0.5),
    "RAPID PULSES": (0.08, 0.16),
}

ARROW_ANGLES: Dict[str, float] = {
    "MOVE FORWARD": -90.0,
    "SLIGHT LEFT": -125.0,
    "SLIGHT RIGHT": -55.0,
    "TURN LEFT": 180.0,
    "TURN RIGHT": 0.0,
}


def parse_vibration(action: str) -> Tuple[List[str], str]:
    """'LEFT - SHORT PULSE' -> (['LEFT'], 'SHORT PULSE'); 'BOTH - ...' -> both motors; 'OFF' -> ([], '')."""
    action = action.replace("VIBRATION: ", "").strip()
    if " - " not in action:
        return [], ""
    motors, pattern = (part.strip() for part in action.split(" - ", 1))
    active = ["LEFT", "RIGHT"] if motors == "BOTH" else [motors] if motors in ("LEFT", "RIGHT") else []
    return active, pattern


def pulse_is_on(pattern: str, now: float) -> bool:
    """Whether a vibration pattern is in its 'on' phase at time now."""
    if pattern not in PULSE_TIMING:
        return False
    on_s, period_s = PULSE_TIMING[pattern]
    return (now % period_s) < on_s


def zone_risks(snapshot: DashboardSnapshot) -> Dict[str, str]:
    """Worst risk level per zone among current detections."""
    worst: Dict[str, str] = {}
    for det in snapshot.detections:
        current = worst.get(det.zone)
        if current is None or RISK_PRIORITY.get(RiskLevel(det.risk_level), 0) > RISK_PRIORITY.get(RiskLevel(current), 0):
            worst[det.zone] = det.risk_level
    return worst


def render_shoe_panel(snapshot: DashboardSnapshot, height: int = DESIGN_HEIGHT, now: Optional[float] = None) -> np.ndarray:
    """Draws the simulated shoe panel and scales it to the given video height."""
    import cv2

    now = time.monotonic() if now is None else now
    panel = np.full((DESIGN_HEIGHT, PANEL_WIDTH, 3), BACKGROUND, dtype=np.uint8)

    _text(panel, "SIMULATED SHOE", (12, 22), 0.55, TEXT, 2)
    _text(panel, "(software demo - no hardware)", (12, 40), 0.38, DIM)

    _draw_shoe_and_motors(panel, snapshot, now)
    _draw_sensor_cones(panel, snapshot)  # After the shoe, so the sensors on the toe stay visible
    _draw_direction(panel, snapshot)

    stats = f"FPS {snapshot.fps:4.1f}   YOLO {snapshot.inference_ms:4.0f} ms"
    _text(panel, stats, (12, DESIGN_HEIGHT - 12), 0.42, DIM)

    if height != DESIGN_HEIGHT:
        width = max(1, int(round(PANEL_WIDTH * height / DESIGN_HEIGHT)))
        panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
    return panel


def _text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float, color: BGR, thickness: int = 1) -> None:
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_sensor_cones(panel: np.ndarray, snapshot: DashboardSnapshot) -> None:
    """One ultrasonic cone per sensor: length = distance, color = worst risk in that zone."""
    import cv2

    risks = zone_risks(snapshot)
    sensors = {s.name: s for s in snapshot.sensors}

    for name, ((ox, oy), angle) in SENSOR_LAYOUT.items():
        sensor = sensors.get(name)
        healthy = sensor is not None and sensor.health_status == "HEALTHY"
        distance_cm = sensor.distance_cm if healthy else MAX_RANGE_CM
        length = CONE_MIN_PX + (CONE_MAX_PX - CONE_MIN_PX) * min(distance_cm, MAX_RANGE_CM) / MAX_RANGE_CM

        points = [(ox, oy)]
        for step in range(-4, 5):
            a = math.radians(angle + CONE_HALF_ANGLE * step / 4)
            points.append((int(round(ox + length * math.cos(a))), int(round(oy + length * math.sin(a)))))
        polygon = np.array(points, dtype=np.int32)

        if healthy:
            color = risk_color(risks.get(name, "SAFE"))
            overlay = panel.copy()
            cv2.fillPoly(overlay, [polygon], color)
            cv2.addWeighted(overlay, 0.45, panel, 0.55, 0, panel)
            cv2.polylines(panel, [polygon], True, color, 1, cv2.LINE_AA)
            label = f"{int(round(distance_cm))}cm"
        else:
            color = FAULT_COLOR_BGR
            cv2.polylines(panel, [polygon], True, color, 1, cv2.LINE_AA)
            label = "FAULT" if sensor is not None else "--"

        tip_a = math.radians(angle)
        tx = int(ox + (length + 14) * math.cos(tip_a))
        ty = int(oy + (length + 14) * math.sin(tip_a))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        tx = min(max(tx - tw // 2, 2), PANEL_WIDTH - tw - 2)
        ty = min(max(ty + th // 2, 56), 190)
        _text(panel, label, (tx, ty), 0.4, color if not healthy else TEXT)
        cv2.circle(panel, (ox, oy), 4, color, -1, cv2.LINE_AA)


def _draw_shoe_and_motors(panel: np.ndarray, snapshot: DashboardSnapshot, now: float) -> None:
    """Top view of the shoe with LEFT/RIGHT motors that blink in the active vibration pattern."""
    import cv2

    cv2.ellipse(panel, (130, 258), (38, 72), 0, 0, 360, (90, 90, 90), -1, cv2.LINE_AA)
    cv2.ellipse(panel, (130, 258), (38, 72), 0, 0, 360, (160, 160, 160), 2, cv2.LINE_AA)
    cv2.ellipse(panel, (130, 300), (22, 20), 0, 0, 360, (70, 70, 70), -1, cv2.LINE_AA)  # Heel pad

    active, pattern = parse_vibration(snapshot.alert_status.vibration_action)
    on = pulse_is_on(pattern, now)
    lit = risk_color(snapshot.overall_risk)

    for name, center in (("LEFT", (114, 240)), ("RIGHT", (146, 240))):
        is_active = name in active
        color = lit if is_active and on else MOTOR_OFF
        cv2.circle(panel, center, 11, color, -1, cv2.LINE_AA)
        cv2.circle(panel, center, 11, (200, 200, 200) if is_active else (110, 110, 110), 1, cv2.LINE_AA)
        if is_active and on:
            for radius in (16, 21):  # Vibration ripples
                cv2.circle(panel, center, radius, lit, 1, cv2.LINE_AA)
        _text(panel, name[0], (center[0] - 4, center[1] + 4), 0.38, (0, 0, 0) if is_active and on else TEXT)

    motor_text = snapshot.alert_status.vibration_action.replace("VIBRATION: ", "") if active else "OFF"
    _text(panel, f"Motors: {motor_text}", (12, 350), 0.4, TEXT)


def _draw_direction(panel: np.ndarray, snapshot: DashboardSnapshot) -> None:
    """Direction arrow (or STOP sign) with the guidance text."""
    import cv2

    direction = snapshot.alert_status.direction
    center = (130, 400)
    _text(panel, "GUIDANCE", (12, 372), 0.4, DIM)

    if direction == "STOP":
        octagon = [
            (int(center[0] + 26 * math.cos(math.radians(22.5 + 45 * k))),
             int(center[1] + 26 * math.sin(math.radians(22.5 + 45 * k))))
            for k in range(8)
        ]
        cv2.fillPoly(panel, [np.array(octagon, dtype=np.int32)], (0, 0, 220), cv2.LINE_AA)
        _text(panel, "STOP", (center[0] - 20, center[1] + 5), 0.5, (255, 255, 255), 2)
    elif direction in ARROW_ANGLES:
        a = math.radians(ARROW_ANGLES[direction])
        start = (int(center[0] - 24 * math.cos(a)), int(center[1] - 24 * math.sin(a)))
        end = (int(center[0] + 24 * math.cos(a)), int(center[1] + 24 * math.sin(a)))
        cv2.arrowedLine(panel, start, end, ARROW, 5, cv2.LINE_AA, tipLength=0.45)
    else:
        _text(panel, "--", (center[0] - 8, center[1] + 5), 0.5, DIM)

    label = direction or "NO GUIDANCE"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    _text(panel, label, (center[0] - tw // 2, 448), 0.5, ARROW if direction else DIM, 1)
