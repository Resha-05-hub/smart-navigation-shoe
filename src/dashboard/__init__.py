"""Dashboard module for real-time visualization and system metrics."""

from .dashboard_data import (
    DashboardSnapshot,
    DetectionSummary,
    SensorStatusSummary,
    RecentEvent,
    SystemStatus,
    AlertStatusSummary,
)
from .dashboard import Dashboard

__all__ = [
    "DashboardSnapshot",
    "DetectionSummary",
    "SensorStatusSummary",
    "RecentEvent",
    "SystemStatus",
    "AlertStatusSummary",
    "Dashboard",
]
