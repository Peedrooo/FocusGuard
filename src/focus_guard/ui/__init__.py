"""FocusGuard UI subsystem: FastAPI/WebSocket dashboard, Windows tray, and interventions."""

from focus_guard.ui.api_server import WebSocketManager, create_api_app
from focus_guard.ui.interventions import WindowsInterventionDispatcher
from focus_guard.ui.tray import (
    SystemTrayController,
    create_tray_icon_image,
    get_state_color,
)

__all__ = [
    "SystemTrayController",
    "WebSocketManager",
    "WindowsInterventionDispatcher",
    "create_api_app",
    "create_tray_icon_image",
    "get_state_color",
]
