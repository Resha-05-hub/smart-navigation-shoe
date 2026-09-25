"""Directional analysis logic for steering the user toward clear paths."""

from typing import Any, Dict, List, Optional

from ..core.models import FusedObstacle, DirectionGuidance, RiskAssessment, SensorReading
from ..core.enums import DirectionCommand, ObstacleZone, RiskLevel, SensorStatus
from .risk_analyzer import RISK_PRIORITY
from ..utils.logger import get_logger

logger = get_logger("direction_analyzer")

# Spoken instruction appended to hazard voice alerts
DIRECTION_VOICE: Dict[DirectionCommand, str] = {
    DirectionCommand.MOVE_FORWARD: "Keep straight.",
    DirectionCommand.SLIGHT_LEFT: "Move slightly left.",
    DirectionCommand.SLIGHT_RIGHT: "Move slightly right.",
    DirectionCommand.TURN_LEFT: "Turn left.",
    DirectionCommand.TURN_RIGHT: "Turn right.",
    DirectionCommand.STOP: "Stop.",
}

NO_OBSTACLE_M = 10.0  # Distance assumed for a zone with no obstacle


class DirectionAnalyzer:
    """Analyzes spatial zone occupancy to recommend clear navigation paths.

    Rules:
    - STOP when the obstacle straight ahead is at DANGER level, or every escape side is too.
    - Center clear: keep straight, or move slightly away from a side obstacle at WARNING or worse.
    - Center blocked: steer toward the clearer side (slight turn if it is fully clear, else turn).
    - A zone whose sensor is faulty is unknown: not clear ahead, and never chosen as the escape route.
    - The previously chosen side is kept unless the other side is clearer by side_hysteresis_m,
      so small distance fluctuations do not flip the instruction back and forth.
    """

    def __init__(
        self,
        clear_path_threshold_m: float = 2.0,
        side_hysteresis_m: float = 0.3,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.clear_path_threshold_m = clear_path_threshold_m
        self.side_hysteresis_m = side_hysteresis_m
        if config:
            dir_cfg = config.get("direction", {})
            self.clear_path_threshold_m = dir_cfg.get("clear_path_threshold", clear_path_threshold_m)
            self.side_hysteresis_m = dir_cfg.get("side_hysteresis_m", side_hysteresis_m)
        self._last_side: Optional[ObstacleZone] = None

    def analyze_path(
        self,
        obstacles: List[FusedObstacle],
        risk_assessment: RiskAssessment,
        sensor_readings: Optional[List[SensorReading]] = None,
    ) -> DirectionGuidance:
        """Determines best navigation command based on spatial distribution of obstacles.

        Args:
            obstacles: List of fused obstacles (risk levels assigned by RiskAnalyzer).
            risk_assessment: Current risk assessment.
            sensor_readings: Optional distance readings; faulty side sensors rule that side out.

        Returns:
            DirectionGuidance object.
        """
        zones = (ObstacleZone.LEFT, ObstacleZone.CENTER, ObstacleZone.RIGHT)
        zone_dist = {zone: NO_OBSTACLE_M for zone in zones}
        zone_risk = {zone: 0 for zone in zones}

        for obs in obstacles:
            if obs.zone in zone_dist:
                zone_dist[obs.zone] = min(zone_dist[obs.zone], obs.distance_m)
                zone_risk[obs.zone] = max(zone_risk[obs.zone], RISK_PRIORITY.get(obs.risk_level, 0))

        # A failed sensor's zone is unknown space: never "clear ahead" and never an escape route.
        # OUT_OF_RANGE (no echo) is not a fault; it means nothing is in range.
        center_faulty = False
        for reading in sensor_readings or []:
            if reading.status == SensorStatus.INVALID and reading.position in zone_dist:
                zone_dist[reading.position] = 0.0
                center_faulty = center_faulty or reading.position == ObstacleZone.CENTER

        danger = RISK_PRIORITY[RiskLevel.DANGER]
        warning = RISK_PRIORITY[RiskLevel.WARNING]
        center_dist = zone_dist[ObstacleZone.CENTER]
        left_dist, right_dist = zone_dist[ObstacleZone.LEFT], zone_dist[ObstacleZone.RIGHT]

        # Case 1: Obstacle dangerously close straight ahead
        if zone_risk[ObstacleZone.CENTER] >= danger:
            return DirectionGuidance(
                recommended_direction=DirectionCommand.STOP,
                clear_path_score=0.0,
                safety_notes="STOP: Critical obstacle immediately ahead.",
            )

        # Case 2: Center is open; step away from close side obstacles
        if center_dist >= self.clear_path_threshold_m:
            left_close = zone_risk[ObstacleZone.LEFT] >= warning
            right_close = zone_risk[ObstacleZone.RIGHT] >= warning
            score = min(1.0, center_dist / 4.0)
            if left_close and not right_close:
                self._last_side = ObstacleZone.RIGHT
                return DirectionGuidance(DirectionCommand.SLIGHT_RIGHT, score, "Obstacle close on the left. Move slightly right.")
            if right_close and not left_close:
                self._last_side = ObstacleZone.LEFT
                return DirectionGuidance(DirectionCommand.SLIGHT_LEFT, score, "Obstacle close on the right. Move slightly left.")

            self._last_side = None
            note = "Obstacles on both sides. Keep straight." if left_close else "Center path clear. Move forward."
            return DirectionGuidance(DirectionCommand.MOVE_FORWARD, score, note)

        # Case 3: Center blocked; steer toward the clearer side
        side = self._choose_side(left_dist, right_dist)
        side_dist = zone_dist[side]
        if zone_risk[side] >= danger or side_dist <= 0.0:
            return DirectionGuidance(
                recommended_direction=DirectionCommand.STOP,
                clear_path_score=0.0,
                safety_notes="STOP: No safe direction available.",
            )

        is_left = side == ObstacleZone.LEFT
        side_word = "left" if is_left else "right"
        score = min(1.0, side_dist / 4.0)
        if side_dist >= self.clear_path_threshold_m:
            command = DirectionCommand.SLIGHT_LEFT if is_left else DirectionCommand.SLIGHT_RIGHT
            reason = "Center sensor fault" if center_faulty else "Center blocked"
            note = f"{reason}. Veer slight {side_word}."
        else:
            command = DirectionCommand.TURN_LEFT if is_left else DirectionCommand.TURN_RIGHT
            reason = "Center sensor fault" if center_faulty else "Front path obstructed"
            note = f"{reason}. Turn {side_word}."
        return DirectionGuidance(recommended_direction=command, clear_path_score=score, safety_notes=note)

    def _choose_side(self, left_dist: float, right_dist: float) -> ObstacleZone:
        """Picks the clearer side, keeping the previous choice unless the other is clearly better."""
        if self._last_side == ObstacleZone.LEFT and left_dist >= right_dist - self.side_hysteresis_m:
            side = ObstacleZone.LEFT
        elif self._last_side == ObstacleZone.RIGHT and right_dist >= left_dist - self.side_hysteresis_m:
            side = ObstacleZone.RIGHT
        else:
            side = ObstacleZone.LEFT if left_dist > right_dist else ObstacleZone.RIGHT
        self._last_side = side
        return side
