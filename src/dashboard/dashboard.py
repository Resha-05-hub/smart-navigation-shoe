"""Real-time dashboard interface for Smart Navigation Shoe."""

import os
from typing import Optional, List
import numpy as np

from src.dashboard.dashboard_data import DashboardSnapshot, DetectionSummary, SensorStatusSummary


class Dashboard:
    """Lightweight Python terminal and video overlay dashboard renderer."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        dash_cfg = self.config.get("dashboard", {})
        self.enabled = dash_cfg.get("enabled", True)
        self.refresh_interval = dash_cfg.get("refresh_interval", 0.2)

        # Enable ANSI escape handling in the Windows console so the CLI can redraw in place
        if os.name == "nt":
            os.system("")

    def render_cli(self, snapshot: DashboardSnapshot) -> str:
        """Renders formatted ASCII status dashboard text string matching Phase 6 layout."""
        lines = []
        lines.append("--------------------------------------------------")
        lines.append("    SMART NAVIGATION SHOE")
        lines.append("--------------------------------------------------")
        lines.append("SYSTEM STATUS")
        lines.append("")
        lines.append(f"Camera          : {snapshot.system_status.camera_status}")
        lines.append(f"YOLO            : {snapshot.system_status.yolo_status}")
        lines.append(f"Sensors         : {snapshot.system_status.sensors_status}")
        lines.append(f"Simulation      : {snapshot.sensor_mode}")
        lines.append("")
        lines.append("CURRENT DETECTION")
        lines.append("")

        if snapshot.detections:
            lines.append(f"{'Object':<12} {'Confidence':<13} {'Zone':<10} {'Distance':<11} {'Risk':<10}")
            for det in snapshot.detections:
                dist_str = f"{int(round(det.distance_cm))} cm"
                lines.append(
                    f"{det.object_name:<12} {det.confidence_pct_str:<13} {det.zone:<10} {dist_str:<11} {det.risk_level:<10}"
                )
        else:
            lines.append("No obstacles detected.")

        lines.append("")
        lines.append("ALERT STATUS")
        lines.append("")
        lines.append(f"Vibration       : {snapshot.alert_status.vibration_action}")
        lines.append(f"Voice           : {snapshot.alert_status.voice_message}")
        lines.append("")
        lines.append("SENSOR STATUS")
        lines.append("")
        for sensor in snapshot.sensors:
            dist_str = f"{int(round(sensor.distance_cm))} cm" if sensor.health_status == "HEALTHY" else "N/A"
            lines.append(f"{sensor.name:<15} : {dist_str:<12} {sensor.health_status}")

        lines.append("")
        lines.append("RECENT EVENTS")
        lines.append("")
        if snapshot.recent_events:
            lines.append(f"{'Time':<10} {'Object':<11} {'Zone':<9} {'Distance':<11} {'Risk':<10}")
            for ev in snapshot.recent_events:
                dist_str = f"{int(round(ev.distance_cm))} cm"
                lines.append(
                    f"{ev.timestamp_str:<10} {ev.object_name:<11} {ev.zone:<9} {dist_str:<11} {ev.risk_level:<10}"
                )
        else:
            lines.append("No events logged yet.")

        lines.append("--------------------------------------------------")
        if snapshot.controls_hint:
            lines.append(snapshot.controls_hint)
        return "\n".join(lines)

    def display_cli(self, snapshot: DashboardSnapshot, clear_screen: bool = False) -> None:
        """Displays dashboard in console terminal, redrawing in place when clear_screen is set."""
        if clear_screen:
            # Cursor home + clear to end of screen: no subprocess and no flicker, unlike cls/clear
            print("\033[H\033[J", end="")
        print(self.render_cli(snapshot), flush=True)

    def render_video_overlay(self, frame: np.ndarray, snapshot: DashboardSnapshot) -> np.ndarray:
        """Renders dashboard status panel overlay onto an OpenCV camera frame."""
        import cv2

        annotated = frame.copy()
        h, w, _ = annotated.shape

        # Create dark overlay bar at the top of the video frame
        overlay_height = 95
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, overlay_height), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Title
        cv2.putText(
            annotated,
            "SMART NAVIGATION SHOE - DASHBOARD",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Risk level indicator
        risk_colors = {
            "SAFE": (0, 255, 0),
            "CAUTION": (0, 255, 255),
            "WARNING": (0, 165, 255),
            "DANGER": (0, 0, 255),
        }
        color = risk_colors.get(snapshot.overall_risk, (255, 255, 255))
        cv2.putText(
            annotated,
            f"RISK: {snapshot.overall_risk}",
            (w - 170, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        # Sensors Bar
        sensor_str = f"Sensors ({snapshot.sensor_mode}): " + " | ".join(
            [
                f"{s.name}: {int(round(s.distance_cm))}cm" if s.health_status == "HEALTHY" else f"{s.name}: N/A"
                for s in snapshot.sensors
            ]
        )
        cv2.putText(
            annotated,
            sensor_str,
            (15, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Alert Summary
        alert_str = f"VIB: {snapshot.alert_status.vibration_action} | VOICE: {snapshot.alert_status.voice_message}"
        if len(alert_str) > 75:
            alert_str = alert_str[:72] + "..."
        cv2.putText(
            annotated,
            alert_str,
            (15, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return annotated
