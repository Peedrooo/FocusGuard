"""Windows System Tray icon (pystray + Pillow) and global hotkeys (Win32 RegisterHotKey + pynput fallback)."""

from __future__ import annotations

import asyncio
import ctypes
import inspect
import logging
import sys
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from PIL import Image, ImageDraw

from focus_guard.core.events import FocusState

logger = logging.getLogger(__name__)

_MOD_ALT = 0x0001
_MOD_CONTROL = 0x0002
_MOD_SHIFT = 0x0004
_MOD_WIN = 0x0008
_MOD_NOREPEAT = 0x4000
_WM_HOTKEY = 0x0312
_WM_QUIT = 0x0012
_PM_NOREMOVE = 0x0000

_VK_NAMES: dict[str, int] = {
    "space": 0x20,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "esc": 0x1B,
    "escape": 0x1B,
    "backspace": 0x08,
    "pause": 0x13,
    "insert": 0x2D,
    "delete": 0x2E,
    "home": 0x24,
    "end": 0x23,
    "page_up": 0x21,
    "page_down": 0x22,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
}


def _parse_win32_hotkey(hotkey: str) -> tuple[int, int] | None:
    """Parse pynput-style hotkey string (e.g. '<ctrl>+<shift>+p') to Win32 (mods, vk)."""
    raw = (hotkey or "").strip().lower()
    if not raw:
        return None
    tokens = [part.strip() for part in raw.split("+") if part.strip()]
    if not tokens:
        return None

    mods = 0
    vk: int | None = None
    for tok in tokens:
        clean = tok[1:-1] if tok.startswith("<") and tok.endswith(">") else tok
        if clean in ("ctrl", "control"):
            mods |= _MOD_CONTROL
        elif clean == "shift":
            mods |= _MOD_SHIFT
        elif clean == "alt":
            mods |= _MOD_ALT
        elif clean in ("cmd", "win", "super"):
            mods |= _MOD_WIN
        elif vk is not None:
            return None
        elif clean in _VK_NAMES:
            vk = _VK_NAMES[clean]
        elif len(clean) == 1 and "a" <= clean <= "z":
            vk = ord(clean.upper())
        elif len(clean) == 1 and "0" <= clean <= "9":
            vk = ord(clean)
        elif clean.startswith("f") and clean[1:].isdigit():
            fn_num = int(clean[1:])
            if 1 <= fn_num <= 12:
                vk = 0x70 + (fn_num - 1)
            else:
                return None
        else:
            return None

    if vk is None:
        return None
    return (mods, vk)

STATE_COLORS: dict[FocusState, str] = {
    FocusState.FOCUSED: "#10b981",
    FocusState.GRACE_WINDOW: "#f59e0b",
    FocusState.BREAK_MODE: "#f59e0b",
    FocusState.CONVERSATION_MODE: "#f59e0b",
    FocusState.INTERVENING: "#ef4444",
    FocusState.DRIFT_SUSPECTED: "#ef4444",
    FocusState.PAUSED: "#64748b",
    FocusState.PRIVACY_MODE: "#64748b",
}


def get_state_color(state: FocusState) -> str:
    """Return hex color associated with a FocusState."""
    return STATE_COLORS.get(state, "#64748b")


def create_tray_icon_image(state: FocusState, size: int = 64) -> Image.Image:
    """Render a crisp circular status icon purely in RAM (never written to disk)."""
    color_hex = get_state_color(state)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = max(2, size // 16)
    # Dark outer badge circle
    draw.ellipse(
        (pad, pad, size - pad, size - pad),
        fill="#0f172a",
        outline="#1e293b",
        width=max(2, size // 20),
    )

    inner_pad = max(6, size // 5)
    draw.ellipse(
        (inner_pad, inner_pad, size - inner_pad, size - inner_pad),
        fill=color_hex,
    )

    # Subtle inner highlight for visibility on dark & light taskbars
    highlight_pad = max(10, size // 3)
    draw.ellipse(
        (highlight_pad, highlight_pad, size - highlight_pad, size - highlight_pad),
        outline="#ffffff",
        width=max(1, size // 32),
    )
    return img


class SystemTrayController:
    """Controls the Windows system tray icon, menu actions, and global hotkeys."""

    def __init__(
        self,
        on_toggle_session: Callable[[], Any] | None = None,
        on_toggle_privacy: Callable[[], Any] | None = None,
        on_talk_coach: Callable[[], Any] | None = None,
        on_open_dashboard: Callable[[], Any] | None = None,
        on_quit: Callable[[], Any] | None = None,
        hotkey_privacy: str = "<ctrl>+<shift>+p",
        hotkey_talk: str = "<ctrl>+<shift>+<space>",
        enable_tray: bool = True,
        enable_hotkeys: bool = True,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._on_toggle_session = on_toggle_session
        self._on_toggle_privacy = on_toggle_privacy
        self._on_talk_coach = on_talk_coach
        self._on_open_dashboard = on_open_dashboard
        self._on_quit = on_quit
        self._hotkey_privacy = hotkey_privacy
        self._hotkey_talk = hotkey_talk
        self._enable_tray = enable_tray
        self._enable_hotkeys = enable_hotkeys
        self._loop = loop

        self._current_state = FocusState.PAUSED
        self._icon: Any = None
        self._tray_thread: threading.Thread | None = None
        self._hotkey_listener: Any = None
        self._win32_thread: threading.Thread | None = None
        self._win32_thread_id: int | None = None
        self._win32_stop_event = threading.Event()
        self._started = False

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def create_icon_image(self, state: FocusState | None = None, size: int = 64) -> Image.Image:
        return create_tray_icon_image(state or self._current_state, size=size)

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start tray icon thread and global hotkey listener safely."""
        if loop is not None:
            self._loop = loop
        elif self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

        if self._started:
            return
        self._started = True

        if self._enable_hotkeys:
            self._start_hotkeys()
        if self._enable_tray:
            self._start_tray()

    def stop(self) -> None:
        """Stop tray icon and global hotkey listener cleanly."""
        self._started = False
        self._win32_stop_event.set()
        tid = self._win32_thread_id
        if tid is not None:
            apis = self._get_win32_apis()
            if apis is not None:
                user32, _kernel32 = apis
                try:
                    user32.PostThreadMessageW(tid, _WM_QUIT, 0, 0)
                except Exception:
                    logger.debug("Error posting WM_QUIT to Win32 hotkey thread.", exc_info=True)
        if self._win32_thread is not None:
            self._win32_thread.join(timeout=1.5)
            self._win32_thread = None

        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
            except Exception:
                logger.debug("Error stopping pynput GlobalHotKeys.", exc_info=True)
            finally:
                self._hotkey_listener = None

        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                logger.debug("Error stopping pystray icon.", exc_info=True)
            finally:
                self._icon = None

    def update_state(self, state: FocusState, tooltip: str | None = None) -> None:
        """Update tray icon color and tooltip for the new FocusState."""
        self._current_state = state
        if self._icon is None:
            return
        try:
            self._icon.icon = self.create_icon_image(state)
            self._icon.title = tooltip or f"FocusGuard — {state.value}"
            self._icon.update_menu()
        except Exception:
            logger.debug("Failed to update tray icon state.", exc_info=True)

    def _invoke(self, callback: Callable[[], Any] | None) -> None:
        if callback is None:
            return
        try:
            result = callback()
            if inspect.isawaitable(result):
                async def _runner(aw: Awaitable[Any]) -> None:
                    await aw

                coro = _runner(result)
                if self._loop is not None and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(coro, self._loop)
                else:
                    threading.Thread(
                        target=lambda: asyncio.run(coro),
                        daemon=True,
                    ).start()
        except Exception:
            logger.exception("System tray callback raised an error.")

    def _handle_privacy_hotkey(self) -> None:
        self._invoke(self._on_toggle_privacy)

    def _handle_talk_hotkey(self) -> None:
        self._invoke(self._on_talk_coach)

    @staticmethod
    def _get_win32_apis() -> tuple[Any, Any] | None:
        if sys.platform != "win32":
            return None
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return None
        try:
            return windll.user32, windll.kernel32
        except Exception:
            return None

    def _start_win32_hotkeys(self) -> bool:
        apis = self._get_win32_apis()
        if apis is None:
            return False

        specs: list[tuple[int, int, int, Callable[[], None]]] = []
        if self._hotkey_privacy:
            parsed = _parse_win32_hotkey(self._hotkey_privacy)
            if parsed is None:
                return False
            specs.append((1, parsed[0], parsed[1], self._handle_privacy_hotkey))
        if self._hotkey_talk:
            parsed = _parse_win32_hotkey(self._hotkey_talk)
            if parsed is None:
                return False
            specs.append((2, parsed[0], parsed[1], self._handle_talk_hotkey))

        if not specs:
            return True

        user32, kernel32 = apis
        self._win32_stop_event.clear()
        self._win32_thread_id = None
        ready_event = threading.Event()
        result_box: dict[str, bool] = {"ok": False}

        def _pump() -> None:
            from ctypes import wintypes

            msg = wintypes.MSG()
            self._win32_thread_id = int(kernel32.GetCurrentThreadId())
            user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, _PM_NOREMOVE)

            registered_ids: list[int] = []
            id_to_cb: dict[int, Callable[[], None]] = {}
            try:
                for hk_id, mods, vk, cb in specs:
                    if not user32.RegisterHotKey(None, hk_id, mods | _MOD_NOREPEAT, vk):
                        logger.warning(
                            "Win32 RegisterHotKey failed for id=%s (vk=0x%X); falling back to pynput.",
                            hk_id,
                            vk,
                        )
                        return
                    registered_ids.append(hk_id)
                    id_to_cb[hk_id] = cb
                result_box["ok"] = True
            except Exception:
                logger.warning(
                    "Win32 RegisterHotKey raised error; falling back to pynput.", exc_info=True
                )
                return
            finally:
                if not result_box["ok"]:
                    for rid in registered_ids:
                        try:
                            user32.UnregisterHotKey(None, rid)
                        except Exception:
                            pass
                ready_event.set()

            try:
                while not self._win32_stop_event.is_set():
                    res = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                    if res <= 0:
                        break
                    if msg.message == _WM_HOTKEY:
                        handler = id_to_cb.get(int(msg.wParam))
                        if handler is not None:
                            handler()
            except Exception:
                logger.debug("Win32 hotkey message pump terminated with error.", exc_info=True)
            finally:
                for rid in registered_ids:
                    try:
                        user32.UnregisterHotKey(None, rid)
                    except Exception:
                        pass
                self._win32_thread_id = None

        thread = threading.Thread(
            target=_pump,
            name="FocusGuard-Win32Hotkeys",
            daemon=True,
        )
        thread.start()
        ready_event.wait(timeout=2.0)

        if result_box["ok"]:
            self._win32_thread = thread
            return True

        self._win32_stop_event.set()
        thread.join(timeout=0.5)
        return False

    def _start_pynput_hotkeys(self) -> None:
        try:
            from pynput import keyboard

            hotkeys: dict[str, Callable[[], None]] = {}
            if self._hotkey_privacy:
                hotkeys[self._hotkey_privacy] = self._handle_privacy_hotkey
            if self._hotkey_talk:
                hotkeys[self._hotkey_talk] = self._handle_talk_hotkey

            if not hotkeys:
                return

            listener = keyboard.GlobalHotKeys(hotkeys)
            listener.daemon = True
            listener.start()
            self._hotkey_listener = listener
        except Exception:
            logger.warning("Global hotkeys disabled or unavailable on this system.", exc_info=True)

    def _start_hotkeys(self) -> None:
        if self._start_win32_hotkeys():
            return
        self._start_pynput_hotkeys()

    def _start_tray(self) -> None:
        try:
            import pystray

            menu = pystray.Menu(
                pystray.MenuItem(
                    "Iniciar/Pausar Sessão",
                    lambda icon, item: self._invoke(self._on_toggle_session),
                ),
                pystray.MenuItem(
                    "Modo Privacidade",
                    lambda icon, item: self._invoke(self._on_toggle_privacy),
                    checked=lambda item: self._current_state == FocusState.PRIVACY_MODE,
                ),
                pystray.MenuItem(
                    "Falar com Coach (Ctrl+Shift+Space)",
                    lambda icon, item: self._invoke(self._on_talk_coach),
                ),
                pystray.MenuItem(
                    "Abrir Dashboard",
                    lambda icon, item: self._invoke(self._on_open_dashboard),
                    default=True,
                ),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(
                    "Sair",
                    lambda icon, item: self._handle_quit(),
                ),
            )

            icon = pystray.Icon(
                name="FocusGuard",
                icon=self.create_icon_image(self._current_state),
                title=f"FocusGuard — {self._current_state.value}",
                menu=menu,
            )
            self._icon = icon
            self._tray_thread = threading.Thread(
                target=self._run_icon_loop,
                args=(icon,),
                name="FocusGuard-Tray",
                daemon=True,
            )
            self._tray_thread.start()
        except Exception:
            logger.warning("System tray unavailable on this system.", exc_info=True)
            self._icon = None

    def _run_icon_loop(self, icon: Any) -> None:
        try:
            icon.run()
        except Exception:
            logger.warning("pystray icon loop terminated with error.", exc_info=True)

    def _handle_quit(self) -> None:
        self._invoke(self._on_quit)
        self.stop()
