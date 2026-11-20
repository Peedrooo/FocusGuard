"""FocusGuard application composition and CLI entrypoints."""

from focus_guard.app.container import AppContainer
from focus_guard.app.main import build_arg_parser, cli, run_app

__all__ = [
    "AppContainer",
    "build_arg_parser",
    "cli",
    "run_app",
]
