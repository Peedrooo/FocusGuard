"""OS telemetry and ephemeral screen capture subsystem for FocusGuard."""

from __future__ import annotations

from focus_guard.sensors.os_monitor.os_sensor import UsageBehaviorTracker, WindowsOSSensor
from focus_guard.sensors.os_monitor.screen_capture import MSSScreenCapture, compute_dhash

__all__ = [
    "MSSScreenCapture",
    "UsageBehaviorTracker",
    "WindowsOSSensor",
    "compute_dhash",
]
