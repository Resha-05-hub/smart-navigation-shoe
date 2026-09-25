"""Real-time dashboard interface for Smart Navigation Shoe."""

import os
from typing import Optional, List
import numpy as np

from src.dashboard.dashboard_data import DashboardSnapshot, DetectionSummary, SensorStatusSummary
from src.dashboard.shoe_panel import render_shoe_panel
from src.utils.visual import risk_color


class Dashboard:
    """Lightweight Python terminal and video overlay dashboard renderer."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        dash_cfg = self.config.get("dashboard", {})
        self.enabled = dash_cfg.get("enabled", True)
        self.refresh_interval = dash_cfg.get("refresh_interval", 0.2)
        self.show_side_panel = dash_cfg.get("show_side_panel", True)
        self.show_zone_lines = dash_cfg.get("show_zone_lines", True)
        zones_cfg = self.config.get("risk_analysis", {}).get("zones", {})
        self.zone_boundaries = (zones_cfg.get("left_boundary", 0.33), zones_cfg.get("right_boundary", 0.66))

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
        if snapshot.alert_status.direction:
            lines.append(
                f"Direction       : {snapshot.alert_status.direction} ({snapshot.alert_status.direction_note})"
            )
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

    def render_frame(
        self,
        frame: np.ndarray,
        snapshot: DashboardSnapshot,
        now: Optional[float] = None,
    ) -> np.ndarray:
        """Full dashboard view: zone lines + status on the video, simulated shoe panel alongside.

        With the side panel (which shows sensors, motors, and guidance), the video keeps only a slim
        bottom status strip so detection labels at the top of the frame stay visible.
        """
        if not self.show_side_panel:
            annotated = self.draw_zone_lines(frame) if self.show_zone_lines else frame
            return self.render_video_overlay(annotated, snapshot)

        annotated = self.draw_zone_lines(frame, label_bottom_margin=56) if self.show_zone_lines else frame
        annotated = self.render_status_strip(annotated, snapshot)
        panel = render_shoe_panel(snapshot, height=annotated.shape[0], now=now)
        return np.hstack([annotated, panel])

    def render_status_strip(self, frame: np.ndarray, snapshot: DashboardSnapshot) -> np.ndarray:
        """Two-line status bar along the bottom of the video: risk and source, then the spoken message."""
        import cv2

        annotated = frame.copy()
        h, w = annotated.shape[:2]
        top = h - 46
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, top), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        cv2.putText(annotated, f"RISK: {snapshot.overall_risk}", (10, top + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, risk_color(snapshot.overall_risk), 2, cv2.LINE_AA)
        source = f"Camera: {snapshot.system_status.camera_status}"
        (sw, _), _ = cv2.getTextSize(source, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(annotated, source, (max(w - sw - 10, 180), top + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (190, 190, 190), 1, cv2.LINE_AA)

        voice = f"VOICE: {snapshot.alert_status.voice_message}"
        max_chars = max(20, w // 8)
        if len(voice) > max_chars:
            voice = voice[: max_chars - 3] + "..."
        cv2.putText(annotated, voice, (10, top + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return annotated

    def draw_zone_lines(self, frame: np.ndarray, label_bottom_margin: int = 10) -> np.ndarray:
        """Dashed LEFT | CENTER | RIGHT zone dividers, matching the zone boundaries used for fusion."""
        import cv2

        annotated = frame.copy()
        h, w = annotated.shape[:2]
        for boundary in self.zone_boundaries:
            x = int(w * boundary)
            for y in range(0, h, 16):
                cv2.line(annotated, (x, y), (x, min(y + 8, h)), (200, 200, 200), 1, cv2.LINE_AA)

        left_x, right_x = (int(w * b) for b in self.zone_boundaries)
        label_y = h - label_bottom_margin
        for label, x0, x1 in (("LEFT", 0, left_x), ("CENTER", left_x, right_x), ("RIGHT", right_x, w)):
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(annotated, label, ((x0 + x1 - tw) // 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        return annotated

    def render_video_overlay(self, frame: np.ndarray, snapshot: DashboardSnapshot) -> np.ndarray:
        """Renders dashboard status panel overlay onto an OpenCV camera frame."""
        import cv2

        annotated = frame.copy()
        h, w, _ = annotated.shape

        # Create dark overlay bar at the top of the video frame
        overlay_height = 118 if snapshot.alert_status.direction else 95
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
        color = risk_color(snapshot.overall_risk)
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

        # Direction guidance
        if snapshot.alert_status.direction:
            cv2.putText(
                annotated,
                f"GO: {snapshot.alert_status.direction}",
                (15, 106),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated
