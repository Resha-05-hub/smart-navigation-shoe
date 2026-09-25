"""Risk analysis logic for evaluating collision threats from fused obstacles."""

from typing import List, Optional, Dict, Any

from ..core.models import FusedObstacle, RiskAssessment, SensorReading
from ..core.enums import RiskLevel, ObstacleZone, SensorStatus
from ..utils.logger import get_logger

logger = get_logger("risk_analyzer")

# Severity ranking shared by the risk analyzer and alert manager (legacy levels map onto the Phase 4 scale)
RISK_PRIORITY: Dict[RiskLevel, int] = {
    RiskLevel.SAFE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.CAUTION: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.WARNING: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.DANGER: 4,
    RiskLevel.CRITICAL: 4,
}


class RiskAnalyzer:
    """Evaluates individual obstacle risks and overall environment safety level."""

    def __init__(
        self,
        danger_distance_m: float = 0.5,       # <= 50 cm = DANGER
        warning_distance_m: float = 1.0,      # 50 - 100 cm = WARNING
        caution_distance_m: float = 2.0,      # 100 - 200 cm = CAUTION
        critical_distance_m: Optional[float] = None,
        safe_distance_m: Optional[float] = None,
        hysteresis_m: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.hysteresis_m = hysteresis_m
        self._zone_levels: Dict[ObstacleZone, RiskLevel] = {}  # Last evaluated risk per zone
        self.is_legacy_mode = (critical_distance_m is not None)
        self.critical_distance_m = critical_distance_m or 0.8
        self.danger_distance_m = danger_distance_m
        self.warning_distance_m = warning_distance_m
        self.caution_distance_m = caution_distance_m
        self.safe_distance_m = safe_distance_m or 3.0

        if self.is_legacy_mode:
            self.danger_distance_m = self.critical_distance_m
            self.caution_distance_m = self.safe_distance_m

        if config:
            risk_cfg = config.get("risk_analysis", {})
            self.hysteresis_m = risk_cfg.get("hysteresis_cm", hysteresis_m * 100.0) / 100.0
            thresh_cm = risk_cfg.get("thresholds_cm", {})
            if "danger_cm" in thresh_cm:
                self.danger_distance_m = thresh_cm["danger_cm"] / 100.0
                self.is_legacy_mode = False
            elif "critical_distance_m" in risk_cfg:
                self.danger_distance_m = risk_cfg["critical_distance_m"]

            if "warning_cm" in thresh_cm:
                self.warning_distance_m = thresh_cm["warning_cm"] / 100.0
            elif "warning_distance_m" in risk_cfg:
                self.warning_distance_m = risk_cfg["warning_distance_m"]

            if "caution_cm" in thresh_cm:
                self.caution_distance_m = thresh_cm["caution_cm"] / 100.0
            elif "safe_distance_m" in risk_cfg:
                self.caution_distance_m = risk_cfg["safe_distance_m"]

    def classify_obstacle_risk(self, obstacle: FusedObstacle) -> RiskLevel:
        """Determines risk level for a single obstacle based on distance thresholds (stateless)."""
        return self._classify_distance(obstacle.distance_m, obstacle.zone)

    def _classify_with_hysteresis(self, obstacle: FusedObstacle) -> RiskLevel:
        """Escalates immediately, but only de-escalates once the obstacle is hysteresis_m past the threshold.

        Without this, a stationary obstacle near a boundary (e.g. 200 cm with +/-1 cm sensor noise)
        flips between two levels every frame.
        """
        raw = self._classify_distance(obstacle.distance_m, obstacle.zone)
        previous = self._zone_levels.get(obstacle.zone)
        if self.hysteresis_m <= 0 or previous is None or RISK_PRIORITY[raw] >= RISK_PRIORITY[previous]:
            return raw

        candidate = self._classify_distance(obstacle.distance_m - self.hysteresis_m, obstacle.zone)
        return candidate if RISK_PRIORITY[candidate] < RISK_PRIORITY[previous] else previous

    def _classify_distance(self, dist: float, zone: ObstacleZone) -> RiskLevel:
        if self.is_legacy_mode:
            if dist <= self.critical_distance_m:
                return RiskLevel.CRITICAL
            elif dist <= self.warning_distance_m:
                if zone == ObstacleZone.CENTER:
                    return RiskLevel.HIGH
                return RiskLevel.MEDIUM
            elif dist <= self.safe_distance_m:
                return RiskLevel.LOW
            else:
                return RiskLevel.SAFE

        # Phase 4 Prototype Thresholds:
        # <= 50 cm (0.5 m)          -> DANGER
        # 50 cm - 100 cm (1.0 m)   -> WARNING
        # 100 cm - 200 cm (2.0 m)  -> CAUTION
        # > 200 cm (> 2.0 m)       -> SAFE
        if dist <= self.danger_distance_m:
            return RiskLevel.DANGER
        elif dist <= self.warning_distance_m:
            return RiskLevel.WARNING
        elif dist <= self.caution_distance_m:
            return RiskLevel.CAUTION
        else:
            return RiskLevel.SAFE

    def evaluate(
        self,
        obstacles: List[FusedObstacle],
        sensor_readings: Optional[List[SensorReading]] = None,
    ) -> RiskAssessment:
        """Evaluates all fused obstacles to generate an overall system RiskAssessment.

        Args:
            obstacles: List of FusedObstacle instances.
            sensor_readings: Optional readings; a failed sensor (INVALID) means that zone is unknown,
                so overall risk is raised to at least CAUTION. OUT_OF_RANGE (no echo) counts as clear.

        Returns:
            RiskAssessment object containing overall risk rating and critical threat items.
        """
        faulty_zones = [
            r.position for r in (sensor_readings or [])
            if r.status == SensorStatus.INVALID and r.position != ObstacleZone.UNKNOWN
        ]

        if not obstacles and not faulty_zones:
            self._zone_levels = {}
            return RiskAssessment(
                overall_risk_level=RiskLevel.SAFE,
                critical_obstacles=[],
                recommended_action="Path clear. Proceed safely.",
            )

        highest_risk = RiskLevel.SAFE
        critical_items: List[FusedObstacle] = []
        zone_levels: Dict[ObstacleZone, RiskLevel] = {}

        for obs in obstacles:
            risk = self._classify_with_hysteresis(obs)
            obs.risk_level = risk
            if RISK_PRIORITY[risk] > RISK_PRIORITY.get(zone_levels.get(obs.zone, RiskLevel.SAFE), 0):
                zone_levels[obs.zone] = risk
            else:
                zone_levels.setdefault(obs.zone, risk)

            if RISK_PRIORITY.get(risk, 0) > RISK_PRIORITY.get(highest_risk, 0):
                highest_risk = risk

            if risk in (RiskLevel.DANGER, RiskLevel.WARNING, RiskLevel.HIGH, RiskLevel.CRITICAL):
                critical_items.append(obs)

        self._zone_levels = zone_levels  # Zones without obstacles start fresh next frame

        if faulty_zones and RISK_PRIORITY[highest_risk] < RISK_PRIORITY[RiskLevel.CAUTION]:
            highest_risk = RiskLevel.CAUTION

        action_messages = {
            RiskLevel.DANGER: "DANGER! Immediate obstacle ahead. Stop movement.",
            RiskLevel.CRITICAL: "CRITICAL! Immediate danger detected.",
            RiskLevel.WARNING: "WARNING: Obstacle nearby. Slow down.",
            RiskLevel.HIGH: "HIGH RISK: Obstacle detected.",
            RiskLevel.CAUTION: "CAUTION: Approach with care.",
            RiskLevel.MEDIUM: "CAUTION: Object detected.",
            RiskLevel.LOW: "LOW RISK: Object nearby.",
            RiskLevel.SAFE: "Path clear.",
        }

        recommended_action = action_messages.get(highest_risk, "Proceed with caution.")
        if faulty_zones:
            recommended_action += " Sensor fault: " + ", ".join(z.value for z in faulty_zones) + "."

        return RiskAssessment(
            overall_risk_level=highest_risk,
            critical_obstacles=critical_items,
            recommended_action=recommended_action,
            faulty_zones=faulty_zones,
        )
