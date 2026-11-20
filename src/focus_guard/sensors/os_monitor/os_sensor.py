"""Windows OS telemetry sensor (windows, monitors, processes, behavior, URL, audio)."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import ctypes.wintypes as wintypes
import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Sequence
from typing import Any

try:
    import comtypes
    import psutil
    import pycaw.pycaw as pycaw
    import uiautomation as auto
    import win32api
    import win32gui
    import win32process
except ImportError:  # pragma: no cover - non-Windows fallback
    comtypes = None  # type: ignore[assignment]
    psutil = None  # type: ignore[assignment]
    pycaw = None  # type: ignore[assignment]
    auto = None  # type: ignore[assignment]
    win32api = None  # type: ignore[assignment]
    win32gui = None  # type: ignore[assignment]
    win32process = None  # type: ignore[assignment]

from focus_guard.core.config import AppConfig
from focus_guard.core.event_bus import EventBus
from focus_guard.core.events import (
    OSContextEvent,
    ProcessActivityInfo,
    UsagePatternMetrics,
    VisibleWindowInfo,
)
from focus_guard.core.privacy import PrivacyShield

logger = logging.getLogger(__name__)

BROWSER_PROCESSES: frozenset[str] = frozenset(
    {"chrome.exe", "msedge.exe", "brave.exe", "firefox.exe"}
)
_ADDRESS_BAR_AUTOMATION_IDS: frozenset[str] = frozenset({"addresseditbox", "urlbar-input"})
_ADDRESS_BAR_NAME_TOKENS: tuple[str, ...] = (
    "address",
    "endereço",
    "endereco",
    "url",
    "barra de",
)
_INTERNAL_URL_PREFIXES: tuple[str, ...] = (
    "about:",
    "chrome:",
    "edge:",
    "brave:",
    "moz-extension:",
    "view-source:",
)

_EXCLUDED_WINDOW_CLASSES: frozenset[str] = frozenset(
    {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}
)
_MIN_VISIBLE_WINDOW_WIDTH = 150
_MIN_VISIBLE_WINDOW_HEIGHT = 100
_MAX_VISIBLE_WINDOWS = 10
_MAX_NOTABLE_PROCESSES = 12
_NOTABLE_MEMORY_THRESHOLD_MB = 120.0

_IGNORED_BG_PROCESSES: frozenset[str] = frozenset(
    {
        "",
        "svchost.exe",
        "conhost.exe",
        "dwm.exe",
        "csrss.exe",
        "system",
        "system idle process",
        "registry",
        "idle",
        "smss.exe",
        "wininit.exe",
        "services.exe",
        "lsass.exe",
        "lsm.exe",
        "winlogon.exe",
        "fontdrvhost.exe",
        "sihost.exe",
        "taskhostw.exe",
        "runtimebroker.exe",
        "ctfmon.exe",
        "dllhost.exe",
        "wmiprvse.exe",
        "audiodg.exe",
        "spoolsv.exe",
        "dashost.exe",
        "memory compression",
        "memcompression",
        "vmmem",
        "vmmemwsl",
    }
)

_PROCESS_CATEGORIES: dict[str, str] = {
    # Study / Productivity
    "anki.exe": "study",
    "obsidian.exe": "study",
    "notion.exe": "study",
    "acrobat.exe": "study",
    "acrord32.exe": "study",
    "foxitpdfreader.exe": "study",
    "sumatrapdf.exe": "study",
    "onenote.exe": "study",
    "zotero.exe": "study",
    "xodo.exe": "study",
    "remnote.exe": "study",
    "logseq.exe": "study",
    "calibre.exe": "study",
    "winword.exe": "study",
    "excel.exe": "study",
    "powerpnt.exe": "study",
    "soffice.bin": "study",
    # Browsers
    "chrome.exe": "browser",
    "msedge.exe": "browser",
    "brave.exe": "browser",
    "firefox.exe": "browser",
    "opera.exe": "browser",
    "opera_gx.exe": "browser",
    "vivaldi.exe": "browser",
    "arc.exe": "browser",
    "zen.exe": "browser",
    "chromium.exe": "browser",
    # Media
    "spotify.exe": "media",
    "vlc.exe": "media",
    "mpc-hc64.exe": "media",
    "mpc-hc.exe": "media",
    "mpv.exe": "media",
    "potplayermini64.exe": "media",
    "itunes.exe": "media",
    "applemusic.exe": "media",
    "foobar2000.exe": "media",
    "musicbee.exe": "media",
    "netflix.exe": "media",
    "tidal.exe": "media",
    "deezer.exe": "media",
    "obs64.exe": "media",
    "obs32.exe": "media",
    # Communication
    "discord.exe": "comm",
    "whatsapp.exe": "comm",
    "whatsapp.root.exe": "comm",
    "telegram.exe": "comm",
    "slack.exe": "comm",
    "teams.exe": "comm",
    "ms-teams.exe": "comm",
    "zoom.exe": "comm",
    "skype.exe": "comm",
    "signal.exe": "comm",
    # Dev
    "code.exe": "dev",
    "cursor.exe": "dev",
    "antigravity.exe": "dev",
    "pycharm64.exe": "dev",
    "idea64.exe": "dev",
    "webstorm64.exe": "dev",
    "devenv.exe": "dev",
    "windowsterminal.exe": "dev",
    "powershell.exe": "dev",
    "pwsh.exe": "dev",
    "cmd.exe": "dev",
    "wt.exe": "dev",
    "git-bash.exe": "dev",
    "python.exe": "dev",
    "pythonw.exe": "dev",
    "node.exe": "dev",
    "docker desktop.exe": "dev",
    "sublime_text.exe": "dev",
    "zed.exe": "dev",
    "windsurf.exe": "dev",
    # Games / Launchers
    "steam.exe": "game_or_other",
    "steamwebhelper.exe": "game_or_other",
    "epicgameslauncher.exe": "game_or_other",
    "riotclientservices.exe": "game_or_other",
    "valorant.exe": "game_or_other",
    "leagueclient.exe": "game_or_other",
    "battle.net.exe": "game_or_other",
    "gogalaxy.exe": "game_or_other",
    "xboxapp.exe": "game_or_other",
    "retroarch.exe": "game_or_other",
    "minecraft.exe": "game_or_other",
    "robloxplayerbeta.exe": "game_or_other",
}


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


def _categorize_process(process_name: str) -> str:
    return _PROCESS_CATEGORIES.get(process_name.strip().lower(), "app")


def _normalize_browser_url(raw: str) -> str | None:
    """Normalize raw omnibox text into a URL, filtering empty strings and search queries."""
    text = raw.strip()
    if not text or any(ch.isspace() for ch in text):
        return None
    if "://" in text or text.startswith(_INTERNAL_URL_PREFIXES):
        return text

    host = text.split("/", 1)[0].split("?", 1)[0]
    if host.startswith("localhost") or (
        "." in host and len(host) >= 3 and not host.startswith(".") and not host.endswith(".")
    ):
        return f"https://{text}"
    return None


def _looks_like_explicit_url(raw: str) -> bool:
    """Return True when raw text has an explicit scheme or www prefix."""
    text = raw.strip()
    normalized = _normalize_browser_url(text)
    return normalized is not None and ("://" in text or text.startswith("www."))


class UsageBehaviorTracker:
    """In-memory 60-second sliding-window tracker of user interaction behavior (no keylogging)."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self._window_seconds = max(1.0, float(window_seconds))
        self._lock = threading.Lock()
        self._samples: deque[tuple[float, bool, str]] = deque()
        self._last_sig: str | None = None
        self._same_screen_since: float | None = None
        self._prev_idle_seconds: float = 0.0
        self._latest_metrics = UsagePatternMetrics()

    @property
    def latest_metrics(self) -> UsagePatternMetrics:
        with self._lock:
            return self._latest_metrics

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._last_sig = None
            self._same_screen_since = None
            self._prev_idle_seconds = 0.0
            self._latest_metrics = UsagePatternMetrics()

    def record(
        self,
        context_signature: str,
        idle_seconds: float,
        background_audio_apps: Sequence[str] = (),
        now: float | None = None,
    ) -> UsagePatternMetrics:
        """Record a non-intrusive telemetry sample and compute 60s interaction metrics."""
        ts = time.monotonic() if now is None else float(now)
        idle_val = max(0.0, float(idle_seconds))
        is_active_input = idle_val < 2.0
        sig = context_signature

        with self._lock:
            self._samples.append((ts, is_active_input, sig))
            cutoff = ts - self._window_seconds
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

            total = len(self._samples)
            active_count = sum(1 for _, active, _ in self._samples if active)
            active_ratio = round(min(1.0, max(0.0, active_count / total)), 3) if total > 0 else 0.0

            switches = 0
            for idx in range(1, total):
                if self._samples[idx][2] != self._samples[idx - 1][2]:
                    switches += 1

            if (
                self._last_sig is None
                or self._same_screen_since is None
                or sig != self._last_sig
                or idle_val >= 60.0
                or (self._prev_idle_seconds >= 45.0 and is_active_input)
            ):
                self._last_sig = sig
                self._same_screen_since = ts

            self._prev_idle_seconds = idle_val
            continuous_sec = round(max(0.0, ts - self._same_screen_since), 1)

            if switches >= 8:
                mode: Any = "rapid_switching"
            elif active_ratio >= 0.35:
                mode = "active_work"
            elif idle_val >= 15.0 and len(background_audio_apps) > 0:
                mode = "idle_with_media"
            elif idle_val < 45.0:
                mode = "reading_or_watching"
            else:
                mode = "idle"

            metrics = UsagePatternMetrics(
                active_input_ratio_60s=active_ratio,
                window_switches_60s=switches,
                continuous_same_screen_sec=continuous_sec,
                interaction_mode=mode,
            )
            self._latest_metrics = metrics
            return metrics


class WindowsOSSensor:
    """OSSensorPort implementation collecting Windows context, windows, processes, and behavior."""

    def __init__(
        self,
        config: AppConfig,
        event_bus: EventBus,
        privacy_shield: PrivacyShield | None = None,
        poll_interval_sec: float = 1.0,
        uia_timeout_sec: float = 0.15,
        url_cache_maxsize: int = 128,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._privacy_shield = privacy_shield or PrivacyShield(config.privacy)
        self._poll_interval_sec = max(0.01, float(poll_interval_sec))
        self._uia_timeout_sec = max(0.02, float(uia_timeout_sec))
        self._url_cache_maxsize = max(1, int(url_cache_maxsize))

        self._lock = threading.Lock()
        self._url_cache: OrderedDict[tuple[int, str], str] = OrderedDict()
        self._last_browser_key: tuple[int, str] | None = None

        self._monitors_cache_ttl_sec = 5.0
        self._monitors_cache_ts: float = 0.0
        self._monitors_cache: list[tuple[tuple[int, int, int, int], str, bool]] | None = None

        self._proc_cache_ttl_sec = 3.0
        self._proc_cache_ts: float = 0.0
        self._proc_cache: dict[str, float] | None = None

        self._behavior_tracker = UsageBehaviorTracker(window_seconds=60.0)

        self._uia_inflight_lock = threading.Lock()
        self._uia_inflight = 0

        self._stop_event: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start background polling loop publishing OSContextEvent to EventBus."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._poll_loop(), name="windows-os-sensor")

    async def stop(self) -> None:
        """Stop background polling loop cleanly."""
        if self._stop_event is not None:
            self._stop_event.set()
        task = self._task
        self._task = None
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
            except TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            except asyncio.CancelledError:
                pass

    def sample_context(self) -> OSContextEvent:
        """Collect current OS context synchronously with graceful degradation."""
        hwnd, process_name, window_title = self._get_foreground_window_info()
        idle_seconds = self._get_idle_seconds()
        browser_url = self._resolve_browser_url(hwnd, process_name, window_title)
        background_audio_apps = self._get_active_audio_apps()
        is_sensitive, sensitive_reason = self._privacy_shield.is_sensitive(
            process_name, window_title, browser_url
        )

        monitors = self._get_monitors()
        visible_windows, active_monitor_label = self._collect_visible_windows(
            hwnd, monitors, browser_url
        )
        notable_running_processes = self._collect_notable_processes(
            visible_windows, background_audio_apps
        )

        event = OSContextEvent(
            process_name=process_name,
            window_title=window_title,
            browser_url=browser_url,
            background_audio_apps=background_audio_apps,
            idle_seconds=idle_seconds,
            is_sensitive_window=is_sensitive,
            sensitive_reason=sensitive_reason,
            active_monitor_label=active_monitor_label,
            visible_windows=visible_windows,
            notable_running_processes=notable_running_processes,
        )
        usage_pattern = self._behavior_tracker.record(
            event.context_signature(),
            idle_seconds,
            background_audio_apps,
        )
        return event.model_copy(update={"usage_pattern": usage_pattern})

    async def _poll_loop(self) -> None:
        stop_event = self._stop_event
        if stop_event is None:
            return
        while not stop_event.is_set():
            try:
                event = await asyncio.to_thread(self.sample_context)
                if stop_event.is_set():
                    break
                await self._event_bus.publish(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("WindowsOSSensor poll iteration failed")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._poll_interval_sec)
            except TimeoutError:
                continue

    def _get_foreground_window_info(self) -> tuple[int, str, str]:
        if win32gui is None or win32process is None or psutil is None:
            return 0, "", ""
        try:
            hwnd = int(win32gui.GetForegroundWindow() or 0)
        except Exception as exc:
            logger.debug("GetForegroundWindow failed: %s", exc)
            return 0, "", ""

        if hwnd <= 0:
            return 0, "", ""

        window_title = ""
        try:
            window_title = str(win32gui.GetWindowText(hwnd) or "")
        except Exception as exc:
            logger.debug("GetWindowText failed for hwnd=%s: %s", hwnd, exc)

        process_name = self._lookup_process_name_for_hwnd(hwnd)
        return hwnd, process_name, window_title

    def _lookup_process_name_for_hwnd(self, hwnd: int) -> str:
        if win32process is None or psutil is None or hwnd <= 0:
            return ""
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid and pid > 0:
                return str(psutil.Process(pid).name() or "")
        except Exception as exc:
            logger.debug("Process lookup failed for hwnd=%s: %s", hwnd, exc)
        return ""

    def _get_idle_seconds(self) -> float:
        if win32api is None:
            return 0.0
        try:
            raw_last_input = win32api.GetLastInputInfo()
            last_input = int(
                raw_last_input["dwTime"] if isinstance(raw_last_input, dict) else raw_last_input
            )
            tick_count = int(win32api.GetTickCount())
            elapsed_ms = (tick_count - last_input) & 0xFFFFFFFF
            return round(elapsed_ms / 1000.0, 3)
        except Exception as exc:
            logger.debug("Idle time lookup failed: %s", exc)
            return 0.0

    def _get_monitors(self) -> list[tuple[tuple[int, int, int, int], str, bool]]:
        now = time.monotonic()
        with self._lock:
            if (
                self._monitors_cache is not None
                and (now - self._monitors_cache_ts) < self._monitors_cache_ttl_sec
            ):
                return list(self._monitors_cache)

        raw_monitors: list[tuple[tuple[int, int, int, int], bool]] = []
        if win32api is not None and hasattr(win32api, "EnumDisplayMonitors"):
            try:
                enumerated = win32api.EnumDisplayMonitors() or []
                for entry in enumerated:
                    try:
                        hmon, _, rect_tuple = entry
                        rect = (
                            int(rect_tuple[0]),
                            int(rect_tuple[1]),
                            int(rect_tuple[2]),
                            int(rect_tuple[3]),
                        )
                        is_primary = rect[0] == 0 and rect[1] == 0
                        if hasattr(win32api, "GetMonitorInfo"):
                            try:
                                info = win32api.GetMonitorInfo(hmon)
                                if isinstance(info, dict) and "Monitor" in info:
                                    m_rect = info["Monitor"]
                                    rect = (
                                        int(m_rect[0]),
                                        int(m_rect[1]),
                                        int(m_rect[2]),
                                        int(m_rect[3]),
                                    )
                                    is_primary = bool(int(info.get("Flags", 0)) & 1)
                            except Exception:
                                pass
                        if rect[2] > rect[0] and rect[3] > rect[1]:
                            raw_monitors.append((rect, is_primary))
                    except Exception:
                        continue
            except Exception as exc:
                logger.debug("EnumDisplayMonitors failed: %s", exc)

        if not raw_monitors:
            raw_monitors = self._enumerate_monitors_ctypes()

        labeled = self._label_monitors(raw_monitors)
        with self._lock:
            self._monitors_cache = list(labeled)
            self._monitors_cache_ts = time.monotonic()
        return labeled

    @staticmethod
    def _enumerate_monitors_ctypes() -> list[tuple[tuple[int, int, int, int], bool]]:
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return []
        try:
            user32 = windll.user32
            results: list[tuple[tuple[int, int, int, int], bool]] = []
            proc_type = ctypes.WINFUNCTYPE(
                wintypes.BOOL,
                wintypes.HMONITOR,
                wintypes.HDC,
                ctypes.POINTER(wintypes.RECT),
                wintypes.LPARAM,
            )

            def _cb(
                hmon: Any,
                _hdc: Any,
                lprect: Any,
                _lparam: Any,
            ) -> bool:
                try:
                    mi = _MONITORINFO()
                    mi.cbSize = ctypes.sizeof(_MONITORINFO)
                    if user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                        rc = mi.rcMonitor
                        rect = (int(rc.left), int(rc.top), int(rc.right), int(rc.bottom))
                        is_primary = bool(mi.dwFlags & 1)
                    else:
                        rc = lprect.contents
                        rect = (int(rc.left), int(rc.top), int(rc.right), int(rc.bottom))
                        is_primary = rect[0] == 0 and rect[1] == 0
                    if rect[2] > rect[0] and rect[3] > rect[1]:
                        results.append((rect, is_primary))
                except Exception:
                    pass
                return True

            user32.EnumDisplayMonitors(None, None, proc_type(_cb), 0)
            return results
        except Exception as exc:
            logger.debug("ctypes EnumDisplayMonitors fallback failed: %s", exc)
            return []

    @staticmethod
    def _label_monitors(
        raw_monitors: list[tuple[tuple[int, int, int, int], bool]],
    ) -> list[tuple[tuple[int, int, int, int], str, bool]]:
        if not raw_monitors:
            return [((0, 0, 1920, 1080), "M1-Principal", True)]

        primary_idx = next((i for i, (_, prim) in enumerate(raw_monitors) if prim), None)
        if primary_idx is None:
            primary_idx = next(
                (i for i, (rect, _) in enumerate(raw_monitors) if rect[0] == 0 and rect[1] == 0),
                0,
            )

        primary_rect = raw_monitors[primary_idx][0]
        labeled: list[tuple[tuple[int, int, int, int], str, bool]] = [
            (primary_rect, "M1-Principal", True)
        ]

        secondaries = [
            rect for idx, (rect, _) in enumerate(raw_monitors) if idx != primary_idx
        ]
        secondaries.sort(key=lambda r: (r[0], r[1]))
        for idx, rect in enumerate(secondaries, start=2):
            label = "M2-Secundario" if idx == 2 else f"M{idx}"
            labeled.append((rect, label, False))
        return labeled

    @staticmethod
    def _resolve_monitor_for_rect(
        rect: tuple[int, int, int, int],
        monitors: list[tuple[tuple[int, int, int, int], str, bool]],
    ) -> tuple[str, float]:
        if not monitors:
            return "M1-Principal", 0.0

        left, top, right, bottom = rect
        win_area = max(0, right - left) * max(0, bottom - top)
        cx = (left + right) // 2
        cy = (top + bottom) // 2

        matched = None
        for m_rect, label, is_primary in monitors:
            ml, mt, mr, mb = m_rect
            if ml <= cx < mr and mt <= cy < mb:
                matched = (m_rect, label, is_primary)
                break

        if matched is None:
            best_overlap = 0
            for m_rect, label, is_primary in monitors:
                ml, mt, mr, mb = m_rect
                overlap_w = max(0, min(right, mr) - max(left, ml))
                overlap_h = max(0, min(bottom, mb) - max(top, mt))
                overlap = overlap_w * overlap_h
                if overlap > best_overlap:
                    best_overlap = overlap
                    matched = (m_rect, label, is_primary)

        if matched is None:
            matched = monitors[0]

        m_rect, label, _ = matched
        mon_area = max(1, (m_rect[2] - m_rect[0]) * (m_rect[3] - m_rect[1]))
        pct = max(0.0, min(100.0, round((win_area / mon_area) * 100.0, 1)))
        return label, pct

    def _collect_visible_windows(
        self,
        fg_hwnd: int,
        monitors: list[tuple[tuple[int, int, int, int], str, bool]],
        fg_browser_url: str | None,
    ) -> tuple[list[VisibleWindowInfo], str]:
        default_label = monitors[0][1] if monitors else "M1-Principal"
        if win32gui is None or not hasattr(win32gui, "EnumWindows"):
            return [], default_label

        hwnds: list[int] = []

        def _enum_cb(hwnd: int, _extra: Any) -> bool:
            try:
                hwnd_int = int(hwnd)
                if hwnd_int > 0:
                    hwnds.append(hwnd_int)
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(_enum_cb, None)
        except Exception as exc:
            logger.debug("EnumWindows failed: %s", exc)

        windows: list[VisibleWindowInfo] = []
        pid_names: dict[int, str] = {}

        for hwnd in hwnds:
            try:
                if hasattr(win32gui, "IsWindowVisible") and not win32gui.IsWindowVisible(hwnd):
                    continue
                if hasattr(win32gui, "IsIconic") and win32gui.IsIconic(hwnd):
                    continue

                raw_title = str(win32gui.GetWindowText(hwnd) or "") if hasattr(win32gui, "GetWindowText") else ""
                title = raw_title.strip()
                if not title:
                    continue

                if hasattr(win32gui, "GetClassName"):
                    try:
                        cls_name = str(win32gui.GetClassName(hwnd) or "")
                        if cls_name in _EXCLUDED_WINDOW_CLASSES:
                            continue
                    except Exception:
                        pass

                if not hasattr(win32gui, "GetWindowRect"):
                    continue
                raw_rect = win32gui.GetWindowRect(hwnd)
                rect = (
                    int(raw_rect[0]),
                    int(raw_rect[1]),
                    int(raw_rect[2]),
                    int(raw_rect[3]),
                )
                if (
                    (rect[2] - rect[0]) < _MIN_VISIBLE_WINDOW_WIDTH
                    or (rect[3] - rect[1]) < _MIN_VISIBLE_WINDOW_HEIGHT
                ):
                    continue

                proc_name = self._lookup_process_name_cached(hwnd, pid_names)
                is_fg = fg_hwnd > 0 and hwnd == fg_hwnd
                url_hint = fg_browser_url if is_fg else None
                is_sensitive, _ = self._privacy_shield.is_sensitive(
                    proc_name, title, url_hint
                )
                masked_title = "[JANELA PROTEGIDA]" if is_sensitive else title

                monitor_label, area_pct = self._resolve_monitor_for_rect(rect, monitors)
                windows.append(
                    VisibleWindowInfo(
                        hwnd=hwnd,
                        process_name=proc_name,
                        window_title=masked_title,
                        rect=rect,
                        monitor_label=monitor_label,
                        screen_area_pct=area_pct,
                        is_foreground=is_fg,
                        is_sensitive=is_sensitive,
                    )
                )
            except Exception as exc:
                logger.debug("Skipping window hwnd=%s during enumeration: %s", hwnd, exc)
                continue

        windows.sort(key=lambda w: 0 if w.is_foreground else 1)
        limited = windows[:_MAX_VISIBLE_WINDOWS]

        active_monitor_label = default_label
        fg_win = next((w for w in limited if w.is_foreground), None)
        if fg_win is not None:
            active_monitor_label = fg_win.monitor_label
        elif fg_hwnd > 0 and hasattr(win32gui, "GetWindowRect"):
            try:
                raw_fg_rect = win32gui.GetWindowRect(fg_hwnd)
                fg_rect = (
                    int(raw_fg_rect[0]),
                    int(raw_fg_rect[1]),
                    int(raw_fg_rect[2]),
                    int(raw_fg_rect[3]),
                )
                if fg_rect[2] > fg_rect[0] and fg_rect[3] > fg_rect[1]:
                    active_monitor_label, _ = self._resolve_monitor_for_rect(fg_rect, monitors)
            except Exception:
                pass

        return limited, active_monitor_label

    def _lookup_process_name_cached(self, hwnd: int, pid_names: dict[int, str]) -> str:
        if win32process is None or psutil is None or hwnd <= 0:
            return ""
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if not pid or pid <= 0:
                return ""
            if pid in pid_names:
                return pid_names[pid]
            name = str(psutil.Process(pid).name() or "")
            pid_names[pid] = name
            return name
        except Exception:
            return ""

    def _get_process_memory_snapshot(self) -> dict[str, float]:
        now = time.monotonic()
        with self._lock:
            if (
                self._proc_cache is not None
                and (now - self._proc_cache_ts) < self._proc_cache_ttl_sec
            ):
                return dict(self._proc_cache)

        aggregated: dict[str, float] = {}
        if psutil is not None and hasattr(psutil, "process_iter"):
            try:
                for proc in psutil.process_iter(["pid", "name", "memory_info"]):
                    try:
                        info = getattr(proc, "info", None)
                        if isinstance(info, dict):
                            raw_name = info.get("name") or ""
                            mem_info = info.get("memory_info")
                        else:
                            raw_name = proc.name() if callable(getattr(proc, "name", None)) else ""
                            mem_info = (
                                proc.memory_info()
                                if callable(getattr(proc, "memory_info", None))
                                else None
                            )

                        proc_name = str(raw_name).strip().lower()
                        if proc_name in _IGNORED_BG_PROCESSES:
                            continue

                        rss_bytes = getattr(mem_info, "rss", 0) if mem_info is not None else 0
                        mem_mb = max(0.0, float(rss_bytes) / (1024.0 * 1024.0))
                        aggregated[proc_name] = aggregated.get(proc_name, 0.0) + mem_mb
                    except Exception:
                        continue
            except Exception as exc:
                logger.debug("psutil.process_iter failed: %s", exc)

        with self._lock:
            self._proc_cache = dict(aggregated)
            self._proc_cache_ts = time.monotonic()
        return aggregated

    def _collect_notable_processes(
        self,
        visible_windows: list[VisibleWindowInfo],
        background_audio_apps: list[str],
    ) -> list[ProcessActivityInfo]:
        mem_by_proc = self._get_process_memory_snapshot()

        window_counts: dict[str, int] = {}
        for win in visible_windows:
            name = win.process_name.strip().lower()
            if name and name not in _IGNORED_BG_PROCESSES:
                window_counts[name] = window_counts.get(name, 0) + 1

        active_audio_set = {
            app.strip().lower()
            for app in background_audio_apps
            if app.strip() and app.strip().lower() not in _IGNORED_BG_PROCESSES
        }

        all_names = set(mem_by_proc) | set(window_counts) | active_audio_set
        notable: list[ProcessActivityInfo] = []

        for proc_name in all_names:
            if proc_name in _IGNORED_BG_PROCESSES:
                continue
            mem_mb = round(mem_by_proc.get(proc_name, 0.0), 1)
            win_count = window_counts.get(proc_name, 0)
            has_audio = proc_name in active_audio_set
            category = _categorize_process(proc_name)

            if (
                has_audio
                or win_count > 0
                or mem_mb >= _NOTABLE_MEMORY_THRESHOLD_MB
                or category != "app"
            ):
                notable.append(
                    ProcessActivityInfo(
                        process_name=proc_name,
                        cpu_percent=0.0,
                        memory_mb=mem_mb,
                        has_active_audio=has_audio,
                        window_count=win_count,
                        category=category,
                    )
                )

        notable.sort(
            key=lambda p: (int(p.has_active_audio), p.window_count, p.memory_mb),
            reverse=True,
        )
        return notable[:_MAX_NOTABLE_PROCESSES]

    def _resolve_browser_url(
        self, hwnd: int, process_name: str, window_title: str
    ) -> str | None:
        if hwnd <= 0 or process_name.lower() not in BROWSER_PROCESSES:
            with self._lock:
                self._last_browser_key = None
            return None

        cache_key = (hwnd, window_title)
        with self._lock:
            if cache_key in self._url_cache:
                self._url_cache.move_to_end(cache_key)
                self._last_browser_key = cache_key
                return self._url_cache[cache_key]
            if cache_key == self._last_browser_key:
                return None
            self._last_browser_key = cache_key

        url = self._lookup_browser_url_with_timeout(hwnd, process_name)
        if url:
            with self._lock:
                self._url_cache[cache_key] = url
                self._url_cache.move_to_end(cache_key)
                while len(self._url_cache) > self._url_cache_maxsize:
                    self._url_cache.popitem(last=False)
        return url

    def _lookup_browser_url_with_timeout(self, hwnd: int, process_name: str) -> str | None:
        if auto is None:
            return None

        with self._uia_inflight_lock:
            if self._uia_inflight >= 3:
                logger.debug(
                    "Skipping UIA lookup for hwnd=%s; %d worker(s) already in flight",
                    hwnd,
                    self._uia_inflight,
                )
                return None
            self._uia_inflight += 1

        result_box: dict[str, str | None] = {"url": None}
        done_event = threading.Event()

        def _worker() -> None:
            try:
                result_box["url"] = self._query_uia_browser_url(hwnd, process_name)
            except Exception as exc:
                logger.debug("UIA browser URL lookup failed for hwnd=%s: %s", hwnd, exc)
            finally:
                with self._uia_inflight_lock:
                    self._uia_inflight = max(0, self._uia_inflight - 1)
                done_event.set()

        try:
            worker = threading.Thread(target=_worker, name="focusguard-uia", daemon=True)
            worker.start()
        except Exception as exc:
            with self._uia_inflight_lock:
                self._uia_inflight = max(0, self._uia_inflight - 1)
            logger.debug("Failed to start UIA lookup thread: %s", exc)
            return None

        if not done_event.wait(timeout=self._uia_timeout_sec):
            logger.debug(
                "UIA browser URL lookup timed out after %.3fs for hwnd=%s (%s)",
                self._uia_timeout_sec,
                hwnd,
                process_name,
            )
            return None
        return result_box["url"]

    def _query_uia_browser_url(self, hwnd: int, process_name: str) -> str | None:
        if auto is None:
            return None
        with auto.UIAutomationInitializerInThread():
            ctrl = auto.ControlFromHandle(hwnd)
            if ctrl is None:
                return None
            return self._extract_url_from_control(ctrl, process_name)

    def _extract_url_from_control(self, ctrl: Any, process_name: str) -> str | None:
        if auto is not None and isinstance(ctrl, auto.Control):
            return self._walk_browser_controls_for_url(ctrl)

        if hasattr(ctrl, "EditControl"):
            edit = ctrl.EditControl(searchDepth=12)
            if edit is not None:
                if hasattr(edit, "Exists") and not edit.Exists(0.08, 0.02):
                    return None
                vp = edit.GetValuePattern() if hasattr(edit, "GetValuePattern") else None
                raw_val = str(getattr(vp, "Value", "") or "") if vp is not None else ""
                return _normalize_browser_url(raw_val)
        return None

    def _walk_browser_controls_for_url(self, root: Any) -> str | None:
        if auto is None:
            return None

        def _get_first_child(node: Any) -> Any:
            try:
                if node.ControlType == auto.ControlType.DocumentControl:
                    return None
                return node.GetFirstChildControl()
            except Exception:
                return None

        def _get_next_sibling(node: Any) -> Any:
            try:
                return node.GetNextSiblingControl()
            except Exception:
                return None

        doc_fallback: str | None = None
        visited = 0
        try:
            for node, _depth in auto.WalkTree(
                root,
                getFirstChild=_get_first_child,
                getNextSibling=_get_next_sibling,
                maxDepth=16,
            ):
                visited += 1
                if visited > 300:
                    break
                try:
                    ctype = node.ControlType
                    if ctype == auto.ControlType.EditControl:
                        aid = str(node.AutomationId or "").lower()
                        name = str(node.Name or "").lower()
                        cls = str(node.ClassName or "").lower()
                        access_key = str(getattr(node, "AccessKey", "") or "").lower()
                        is_address_bar = (
                            aid in _ADDRESS_BAR_AUTOMATION_IDS
                            or "omnibox" in cls
                            or "ctrl+l" in access_key
                            or any(tok in name for tok in _ADDRESS_BAR_NAME_TOKENS)
                        )
                        vp = node.GetPattern(auto.PatternId.ValuePattern)
                        raw_val = str(getattr(vp, "Value", "") or "") if vp is not None else ""
                        normalized = _normalize_browser_url(raw_val)
                        if normalized and (is_address_bar or _looks_like_explicit_url(raw_val)):
                            return normalized
                    elif ctype == auto.ControlType.DocumentControl and doc_fallback is None:
                        vp = node.GetPattern(auto.PatternId.ValuePattern)
                        raw_val = str(getattr(vp, "Value", "") or "") if vp is not None else ""
                        normalized = _normalize_browser_url(raw_val)
                        if normalized and "://" in normalized:
                            doc_fallback = normalized
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("UIA tree walk failed: %s", exc)

        return doc_fallback

    def _get_active_audio_apps(self) -> list[str]:
        if pycaw is None or comtypes is None:
            return []

        co_initialized = False
        try:
            comtypes.CoInitialize()
            co_initialized = True
        except Exception:
            pass

        active_apps: set[str] = set()
        try:
            sessions = pycaw.AudioUtilities.GetAllSessions() or []
            for session in sessions:
                try:
                    proc = getattr(session, "Process", None)
                    if proc is None:
                        continue

                    is_playing = getattr(session, "State", 0) == 1
                    if not is_playing:
                        ctl = getattr(session, "_ctl", None)
                        if ctl is not None:
                            meter = ctl.QueryInterface(pycaw.IAudioMeterInformation)
                            if meter is not None:
                                peak = meter.GetPeakValue()
                                if isinstance(peak, (int, float)) and peak > 0.001:
                                    is_playing = True

                    if is_playing:
                        name = str(proc.name() or "").strip().lower()
                        if name:
                            active_apps.add(name)
                except Exception as exc:
                    logger.debug("Skipping audio session due to error: %s", exc)
        except Exception as exc:
            logger.debug("Failed to enumerate audio sessions: %s", exc)
        finally:
            if co_initialized:
                try:
                    comtypes.CoUninitialize()
                except Exception:
                    pass

        return sorted(active_apps)
