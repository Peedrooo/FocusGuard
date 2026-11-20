"""Windows non-destructive intervention dispatcher (Toast, TTS Voice, Topmost Overlay)."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from focus_guard.core.events import InterventionCommand, InterventionLevel
from focus_guard.core.ports import TTSPort

logger = logging.getLogger(__name__)


class WindowsInterventionDispatcher:
    """Dispatches Level 1-3 focus interventions gracefully on Windows.

    Implements `InterventionDispatcherPort`. When `enable_ui=False` (e.g. in
    headless unit tests), visual UI elements (winotify toast, winsound beep,
    tkinter overlay) are skipped while TTS calls still execute normally.
    """

    def __init__(
        self,
        tts: TTSPort,
        on_ack_focus: Callable[[], Any] | None = None,
        on_break_5m: Callable[[], Any] | None = None,
        on_open_coach: Callable[[], Any] | None = None,
        enable_ui: bool = True,
        show_windows_toast: bool = False,
        show_overlay: bool = False,
    ) -> None:
        self._tts = tts
        self._on_ack_focus = on_ack_focus
        self._on_break_5m = on_break_5m
        self._on_open_coach = on_open_coach
        self._enable_ui = enable_ui
        self._show_windows_toast = show_windows_toast
        self._show_overlay = show_overlay

        self._loop: asyncio.AbstractEventLoop | None = None
        self._overlay_lock = threading.Lock()
        self._overlay_root: Any = None
        self._overlay_title_var: Any = None
        self._overlay_msg_var: Any = None

    async def dispatch(self, command: InterventionCommand) -> None:
        """Dispatch an intervention according to its escalation level."""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        level = InterventionLevel(command.level)
        if level == InterventionLevel.NONE:
            return

        if level == InterventionLevel.SOFT:
            if self._enable_ui and self._show_windows_toast:
                self._show_toast_async(command.title, command.message)
                self._play_soft_beep_async()
            if command.speak_aloud:
                await self._safe_speak(command.message)
            return

        if level == InterventionLevel.VOICE:
            if self._enable_ui and self._show_windows_toast:
                self._show_toast_async(command.title, command.message)
            await self._safe_speak(command.message)
            return

        # Level 3 (FIRM_OVERLAY)
        if self._enable_ui and self._show_overlay:
            self._show_or_update_overlay(command.title, command.message)
        await self._safe_speak(command.message)

    def dismiss_overlay(self) -> None:
        """Close the topmost tkinter banner if currently open."""
        with self._overlay_lock:
            root = self._overlay_root
            self._overlay_root = None
            self._overlay_title_var = None
            self._overlay_msg_var = None

        if root is not None:
            try:
                root.after(0, root.destroy)
            except Exception:
                logger.debug("Overlay already closed or unavailable during dismiss.", exc_info=True)

    def stop_speech(self) -> None:
        """Immediately silence any in-progress intervention TTS playback."""
        try:
            self._tts.stop()
        except Exception:
            logger.debug("Ignoring TTS stop error during stop_speech.", exc_info=True)

    async def _safe_speak(self, message: str) -> None:
        if not message.strip():
            return
        try:
            await self._tts.speak(message)
        except Exception:
            logger.exception("TTS speak failed during intervention dispatch.")

    def _invoke_callback(self, callback: Callable[[], Any] | None) -> None:
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
            logger.exception("Intervention action callback failed.")

    def _show_toast_async(self, title: str, message: str) -> None:
        def _worker() -> None:
            try:
                from winotify import Notification

                toast = Notification(
                    app_id="FocusGuard",
                    title=title[:120] or "FocusGuard",
                    msg=message[:300] or "Hora de voltar ao foco.",
                    duration="short",
                )
                toast.show()
            except Exception:
                logger.warning("Failed to display Windows toast notification.", exc_info=True)

        threading.Thread(target=_worker, name="FocusGuard-Toast", daemon=True).start()

    def _play_soft_beep_async(self) -> None:
        def _worker() -> None:
            try:
                import winsound

                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            except Exception:
                logger.debug("Soft winsound beep unavailable.", exc_info=True)

        threading.Thread(target=_worker, name="FocusGuard-Beep", daemon=True).start()

    def _show_or_update_overlay(self, title: str, message: str) -> None:
        with self._overlay_lock:
            root = self._overlay_root
            title_var = self._overlay_title_var
            msg_var = self._overlay_msg_var

        if root is not None and title_var is not None and msg_var is not None:
            try:
                def _update() -> None:
                    title_var.set(title[:90])
                    msg_var.set(message[:260])

                root.after(0, _update)
                return
            except Exception:
                self.dismiss_overlay()

        threading.Thread(
            target=self._run_overlay_window,
            args=(title, message),
            name="FocusGuard-Overlay",
            daemon=True,
        ).start()

    def _run_overlay_window(self, title: str, message: str) -> None:
        try:
            import tkinter as tk
        except Exception:
            logger.warning("Tkinter unavailable; cannot show Level 3 overlay.", exc_info=True)
            return

        root = None
        try:
            root = tk.Tk()
            root.title("FocusGuard — Intervenção")
            root.overrideredirect(True)
            root.attributes("-topmost", True)

            screen_w = root.winfo_screenwidth()
            banner_w = min(940, max(580, screen_w - 80))
            banner_h = 112
            pos_x = max(20, (screen_w - banner_w) // 2)
            pos_y = 16
            root.geometry(f"{banner_w}x{banner_h}+{pos_x}+{pos_y}")

            bg_outer = "#ef4444"
            bg_card = "#160b0f"
            accent_amber = "#f59e0b"
            text_primary = "#f8fafc"

            root.configure(bg=bg_outer)

            card = tk.Frame(root, bg=bg_card, bd=0)
            card.place(x=2, y=2, width=banner_w - 4, height=banner_h - 4)

            top_strip = tk.Frame(card, bg=accent_amber, height=4)
            top_strip.pack(fill="x", side="top")

            body = tk.Frame(card, bg=bg_card)
            body.pack(fill="both", expand=True, padx=16, pady=10)

            title_var = tk.StringVar(value=title[:90])
            msg_var = tk.StringVar(value=message[:260])

            left = tk.Frame(body, bg=bg_card)
            left.pack(side="left", fill="both", expand=True, padx=(0, 14))

            lbl_title = tk.Label(
                left,
                textvariable=title_var,
                bg=bg_card,
                fg="#fbbf24",
                font=("Segoe UI", 11, "bold"),
                anchor="w",
            )
            lbl_title.pack(fill="x", anchor="w")

            lbl_msg = tk.Label(
                left,
                textvariable=msg_var,
                bg=bg_card,
                fg=text_primary,
                font=("Segoe UI", 10),
                anchor="nw",
                justify="left",
                wraplength=max(260, banner_w - 390),
            )
            lbl_msg.pack(fill="both", expand=True, pady=(3, 0))

            right = tk.Frame(body, bg=bg_card)
            right.pack(side="right", fill="y")

            def _handle_ack() -> None:
                self.dismiss_overlay()
                self._invoke_callback(self._on_ack_focus)

            def _handle_break() -> None:
                self.dismiss_overlay()
                self._invoke_callback(self._on_break_5m)

            def _handle_coach() -> None:
                self.dismiss_overlay()
                self._invoke_callback(self._on_open_coach)

            def _make_btn(
                label: str,
                bg_color: str,
                fg_color: str,
                active_bg: str,
                cmd: Callable[[], None],
            ) -> Any:
                btn = tk.Button(
                    right,
                    text=label,
                    bg=bg_color,
                    fg=fg_color,
                    activebackground=active_bg,
                    command=cmd,
                    font=("Segoe UI", 9, "bold"),
                    bd=0,
                    relief="flat",
                    cursor="hand2",
                    padx=10,
                    pady=6,
                )
                btn.pack(side="left", padx=4, pady=18)
                return btn

            _make_btn("Voltei pro Foco", "#10b981", "#052e16", "#34d399", _handle_ack)
            _make_btn("Pausa 5 min", "#f59e0b", "#1c1917", "#fbbf24", _handle_break)
            _make_btn("Abrir Coach", "#38bdf8", "#082f49", "#7dd3fc", _handle_coach)

            with self._overlay_lock:
                self._overlay_root = root
                self._overlay_title_var = title_var
                self._overlay_msg_var = msg_var

            root.mainloop()
        except Exception:
            logger.warning("Level 3 tkinter overlay failed cleanly.", exc_info=True)
        finally:
            with self._overlay_lock:
                if self._overlay_root is root:
                    self._overlay_root = None
                    self._overlay_title_var = None
                    self._overlay_msg_var = None
            if root is not None:
                try:
                    root.destroy()
                except Exception:
                    pass
