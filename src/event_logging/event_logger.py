"""Persistent event and system logging module for Smart Navigation Shoe."""

import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Union
import time

from src.core.models import FusedObstacle, RiskAssessment
from src.core.enums import RiskLevel, ObstacleZone
from src.dashboard.dashboard_data import RecentEvent
from src.utils.logger import setup_logger, get_logger


CSV_HEADERS = [
    "timestamp",
    "object_name",
    "confidence",
    "zone",
    "distance_cm",
    "risk_level",
    "vibration_action",
    "voice_message",
    "alert_status",
    "direction",
]


class EventLogger:
    """Manages persistent CSV detection event logging and system logging."""

    def __init__(
        self,
        event_log_path: str = "logs/detection_events.csv",
        system_log_path: str = "logs/system.log",
        enabled: bool = True,
        max_recent_events: int = 10,
        config: Optional[dict] = None,
    ) -> None:
        if config:
            log_cfg = config.get("logging", {})
            event_log_path = log_cfg.get("event_log", event_log_path)
            system_log_path = log_cfg.get("system_log", system_log_path)
            enabled = log_cfg.get("enabled", enabled)

        self.event_log_path = Path(event_log_path)
        self.system_log_path = Path(system_log_path)
        self.enabled = enabled
        self.max_recent_events = max_recent_events

        self.recent_events: List[RecentEvent] = []
        self._last_event_key: Optional[Tuple] = None
        self._last_log_timestamp: float = 0.0

        if self.enabled:
            self._ensure_log_files()

        # System logger instance
        logger_name = f"smart_shoe_system_{abs(hash(str(self.system_log_path)))}"
        self.system_logger = logging.getLogger(logger_name)
        self.system_logger.setLevel(logging.INFO)

        if not self.system_logger.handlers:
            fmt = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
            formatter = logging.Formatter(fmt)

            # File Handler
            if self.system_log_path:
                fh = logging.FileHandler(str(self.system_log_path), encoding="utf-8")
                fh.setFormatter(formatter)
                self.system_logger.addHandler(fh)

            # Console Handler
            import sys
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            self.system_logger.addHandler(ch)



    def _ensure_log_files(self) -> None:
        """Creates log directory and initializes CSV header if needed."""
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.system_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Keep a CSV written with an older column layout instead of appending mismatched rows
        if self.event_log_path.exists() and self.event_log_path.stat().st_size > 0:
            with open(self.event_log_path, mode="r", newline="", encoding="utf-8") as f:
                existing_header = next(csv.reader(f), [])
            if existing_header != CSV_HEADERS:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                legacy_path = self.event_log_path.with_name(
                    f"{self.event_log_path.stem}.legacy_{stamp}{self.event_log_path.suffix}"
                )
                self.event_log_path.rename(legacy_path)

        if not self.event_log_path.exists() or self.event_log_path.stat().st_size == 0:
            with open(self.event_log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)

        if not self.system_log_path.exists():
            with open(self.system_log_path, mode="a", encoding="utf-8") as f:
                pass

    def _is_duplicate_recent_event(
        self,
        obj_name: str,
        zone_str: str,
        dist_cm: float,
        risk_str: str,
        is_alert_triggered: bool = False,
    ) -> bool:
        """Checks if an event state matches a recent event for the same object and zone."""
        for rec in self.recent_events[:5]:
            if rec.object_name == obj_name and rec.zone == zone_str:
                same_risk = (rec.risk_level == risk_str)
                same_dist = (abs(rec.distance_cm - dist_cm) < 15.0)  # 15 cm threshold
                if same_risk and same_dist:
                    return True
                else:
                    return False
        return False

    def log_event(
        self,
        fused_obstacles: List[FusedObstacle],
        overall_risk: Union[str, RiskLevel],
        vibration_action: str,
        voice_message: str,
        force: bool = False,
        alert_triggered: bool = False,
        direction: str = "",
    ) -> bool:
        """Logs detection events to CSV and recent_events buffer if not duplicate continuous frame state.

        Args:
            voice_message: Message actually spoken this frame (empty when the voice alert was suppressed).
            alert_triggered: True if the alert manager spoke a voice alert this frame.
            direction: Recommended navigation command, e.g. "SLIGHT_LEFT".

        Returns True if at least one new event row was written, False if debounced/skipped.
        """
        if not self.enabled:
            return False

        risk_str = overall_risk.value if isinstance(overall_risk, RiskLevel) else str(overall_risk)
        is_alert_triggered = alert_triggered
        alert_status_str = "TRIGGERED" if alert_triggered else "SUPPRESSED"
        now_ts = time.time()
        logged_any = False

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_only_str = datetime.now().strftime("%H:%M:%S")

        if not fused_obstacles:
            obj_name = "none"
            zone_str = "NONE"
            dist_cm = 0.0

            if force or not self._is_duplicate_recent_event(obj_name, zone_str, dist_cm, risk_str, is_alert_triggered):
                with open(self.event_log_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        now_str,
                        obj_name,
                        "0.00",
                        zone_str,
                        0,
                        risk_str,
                        vibration_action,
                        voice_message,
                        alert_status_str,
                        direction,
                    ])

                rec_ev = RecentEvent(
                    timestamp_str=time_only_str,
                    object_name=obj_name,
                    zone=zone_str,
                    distance_cm=0.0,
                    risk_level=risk_str,
                )
                self.recent_events.insert(0, rec_ev)
                if len(self.recent_events) > self.max_recent_events:
                    self.recent_events = self.recent_events[: self.max_recent_events]

                self._last_event_key = (obj_name, zone_str, 0, risk_str, vibration_action, voice_message)
                self._last_log_timestamp = now_ts
                logged_any = True
        else:
            for obs in fused_obstacles:
                obj_name = obs.label
                conf = round(obs.confidence, 2)
                zone_str = (
                    obs.zone.value
                    if isinstance(obs.zone, ObstacleZone)
                    else str(obs.zone)
                )
                dist_cm = round(obs.distance_m * 100.0, 1)
                obs_risk = (
                    obs.risk_level.value
                    if isinstance(obs.risk_level, RiskLevel)
                    else str(obs.risk_level)
                )

                if force or not self._is_duplicate_recent_event(obj_name, zone_str, dist_cm, obs_risk, is_alert_triggered):
                    with open(self.event_log_path, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            now_str,
                            obj_name,
                            f"{conf:.2f}",
                            zone_str,
                            int(round(dist_cm)),
                            obs_risk,
                            vibration_action,
                            voice_message,
                            alert_status_str,
                            direction,
                        ])

                    rec_ev = RecentEvent(
                        timestamp_str=time_only_str,
                        object_name=obj_name,
                        zone=zone_str,
                        distance_cm=dist_cm,
                        risk_level=obs_risk,
                    )
                    self.recent_events.insert(0, rec_ev)
                    if len(self.recent_events) > self.max_recent_events:
                        self.recent_events = self.recent_events[: self.max_recent_events]

                    if obs_risk in ("WARNING", "DANGER") and is_alert_triggered:
                        self.system_logger.info(
                            f"{obs_risk} alert triggered | Object: {obj_name} ({zone_str}, {int(round(dist_cm))}cm) | Vib: {vibration_action} | Voice: '{voice_message}'"
                        )
                    self._last_event_key = (obj_name, zone_str, int(round(dist_cm)), obs_risk, vibration_action, voice_message)
                    self._last_log_timestamp = now_ts
                    logged_any = True

        return logged_any


    def log_system_message(self, message: str, level: str = "INFO") -> None:
        """Logs a system debugging message to system log file."""
        log_func = getattr(self.system_logger, level.lower(), self.system_logger.info)
        log_func(message)
        for h in self.system_logger.handlers:
            h.flush()

