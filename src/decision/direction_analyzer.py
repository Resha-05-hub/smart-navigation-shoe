"""Directional analysis logic for steering the user toward clear paths."""

from typing import List

from ..core.models import FusedObstacle, DirectionGuidance, RiskAssessment
from ..core.enums import DirectionCommand, ObstacleZone, RiskLevel
from ..utils.logger import get_logger

logger = get_logger("direction_analyzer")


class DirectionAnalyzer:
    """Analyzes spatial zone occupancy to recommend clear navigation paths."""

    def __init__(self, clear_path_threshold_m: float = 2.0) -> None:
        self.clear_path_threshold_m = clear_path_threshold_m

    def analyze_path(
        self,
        obstacles: List[FusedObstacle],
        risk_assessment: RiskAssessment,
    ) -> DirectionGuidance:
        """Determines best navigation command based on spatial distribution of obstacles.

        Args:
            obstacles: List of fused obstacles.
            risk_assessment: Current risk assessment.

        Returns:
            DirectionGuidance object.
        """
        # If danger or critical risk, stop immediately
        if risk_assessment.overall_risk_level in (RiskLevel.DANGER, RiskLevel.CRITICAL):
            return DirectionGuidance(
                recommended_direction=DirectionCommand.STOP,
                clear_path_score=0.0,
                safety_notes="STOP: Critical obstacle immediately ahead.",
            )

        # Evaluate minimum obstacle distance per zone
        zone_min_distances = {
            ObstacleZone.LEFT: 10.0,
            ObstacleZone.CENTER: 10.0,
            ObstacleZone.RIGHT: 10.0,
        }

        for obs in obstacles:
            if obs.zone in zone_min_distances:
                if obs.distance_m < zone_min_distances[obs.zone]:
                    zone_min_distances[obs.zone] = obs.distance_m

        center_dist = zone_min_distances[ObstacleZone.CENTER]
        left_dist = zone_min_distances[ObstacleZone.LEFT]
        right_dist = zone_min_distances[ObstacleZone.RIGHT]

        # Case 1: Center is open
        if center_dist >= self.clear_path_threshold_m:
            return DirectionGuidance(
                recommended_direction=DirectionCommand.MOVE_FORWARD,
                clear_path_score=min(1.0, center_dist / 4.0),
                safety_notes="Center path clear. Move forward.",
            )

        # Case 2: Center blocked, evaluate sides
        if left_dist > right_dist and left_dist >= self.clear_path_threshold_m:
            return DirectionGuidance(
                recommended_direction=DirectionCommand.SLIGHT_LEFT,
                clear_path_score=min(1.0, left_dist / 4.0),
                safety_notes="Center blocked. Veer slight left.",
            )
        elif right_dist >= self.clear_path_threshold_m:
            return DirectionGuidance(
                recommended_direction=DirectionCommand.SLIGHT_RIGHT,
                clear_path_score=min(1.0, right_dist / 4.0),
                safety_notes="Center blocked. Veer slight right.",
            )
        elif left_dist > right_dist:
            return DirectionGuidance(
                recommended_direction=DirectionCommand.TURN_LEFT,
                clear_path_score=min(1.0, left_dist / 4.0),
                safety_notes="Front path obstructed. Turn left.",
            )
        else:
            return DirectionGuidance(
                recommended_direction=DirectionCommand.TURN_RIGHT,
                clear_path_score=min(1.0, right_dist / 4.0),
                safety_notes="Front path obstructed. Turn right.",
            )
