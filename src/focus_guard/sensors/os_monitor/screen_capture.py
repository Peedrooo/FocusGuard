"""Ephemeral in-memory screen capture, monitor selection, and RAM privacy masking."""

from __future__ import annotations

import io
import logging
import warnings
from typing import Any

import mss
from PIL import Image, ImageDraw

from focus_guard.core.config import AppConfig
from focus_guard.core.events import OSContextEvent, ScreenSnapshot
from focus_guard.core.privacy import PrivacyShield

logger = logging.getLogger(__name__)


def compute_dhash(image: Image.Image) -> str:
    """Compute a 64-bit perceptual difference hash as a 16-character hex string."""
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = gray.tobytes()
    hash_bits = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            hash_bits = (hash_bits << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return f"{hash_bits:016x}"


class MSSScreenCapture:
    """ScreenCapturePort implementation keeping all screenshots strictly in RAM."""

    def __init__(
        self,
        config: AppConfig,
        privacy_shield: PrivacyShield | None = None,
    ) -> None:
        self._config = config
        self._privacy_shield = privacy_shield or PrivacyShield(config.privacy)

    def capture(self, os_context: OSContextEvent | None = None) -> ScreenSnapshot | None:
        """Capture selected monitor(s) in RAM unless blocked or masked by PrivacyShield."""
        if os_context is not None:
            if os_context.is_sensitive_window:
                logger.debug(
                    "Skipping screen capture for sensitive window (%s)",
                    os_context.sensitive_reason or os_context.process_name,
                )
                return None
            is_sensitive, reason = self._privacy_shield.is_sensitive(
                os_context.process_name,
                os_context.window_title,
                os_context.browser_url,
            )
            if is_sensitive:
                logger.debug("Skipping screen capture matched by PrivacyShield (%s)", reason)
                return None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                with mss.mss() as sct:
                    monitors = getattr(sct, "monitors", None) or []
                    if not monitors:
                        logger.warning("No monitors returned by mss; skipping capture")
                        return None
                    monitor_index, monitor = self._select_monitor(monitors, os_context)
                    sct_img = sct.grab(monitor)

            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            if img.width <= 0 or img.height <= 0:
                logger.warning("Captured empty frame size=%s; skipping", img.size)
                return None

            self._apply_privacy_masks(img, monitor, os_context)

            max_width = int(self._config.ai.screenshot_max_width)
            if max_width > 0 and img.width > max_width:
                ratio = max_width / img.width
                target_height = max(1, round(img.height * ratio))
                resized = img.resize((max_width, target_height), Image.Resampling.LANCZOS)
            else:
                resized = img

            buf = io.BytesIO()
            quality = max(1, min(95, int(self._config.ai.screenshot_jpeg_quality)))
            resized.save(buf, format="JPEG", quality=quality)
            jpeg_bytes = buf.getvalue()

            perceptual_hash = compute_dhash(resized)
            return ScreenSnapshot(
                image_bytes_jpeg=jpeg_bytes,
                perceptual_hash=perceptual_hash,
                width=resized.width,
                height=resized.height,
                monitor_index=monitor_index,
            )
        except Exception as exc:
            logger.warning("Screen capture failed (%s): %s", type(exc).__name__, exc)
            return None

    def _select_monitor(
        self,
        monitors: list[dict[str, Any]],
        os_context: OSContextEvent | None,
    ) -> tuple[int, dict[str, Any]]:
        mode = getattr(self._config.ai, "capture_mode", "all_monitors")
        if mode == "primary_monitor":
            return self._select_primary_monitor(monitors)
        if mode == "active_monitor":
            return self._select_active_monitor(monitors, os_context)
        return 0, monitors[0]

    @staticmethod
    def _select_primary_monitor(monitors: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        for idx, mon in enumerate(monitors[1:], start=1):
            if mon.get("is_primary") is True:
                return idx, mon
        for idx, mon in enumerate(monitors[1:], start=1):
            if mon.get("left") == 0 and mon.get("top") == 0:
                return idx, mon
        if len(monitors) > 1:
            return 1, monitors[1]
        return 0, monitors[0]

    def _select_active_monitor(
        self,
        monitors: list[dict[str, Any]],
        os_context: OSContextEvent | None,
    ) -> tuple[int, dict[str, Any]]:
        if os_context is not None and os_context.visible_windows:
            fg_win = next((w for w in os_context.visible_windows if w.is_foreground), None)
            if fg_win is not None:
                left, top, right, bottom = fg_win.rect
                if right > left and bottom > top:
                    cx = (left + right) // 2
                    cy = (top + bottom) // 2
                    for idx, mon in enumerate(monitors[1:], start=1):
                        m_left = int(mon.get("left", 0))
                        m_top = int(mon.get("top", 0))
                        m_width = int(mon.get("width", 0))
                        m_height = int(mon.get("height", 0))
                        if (
                            m_left <= cx < (m_left + m_width)
                            and m_top <= cy < (m_top + m_height)
                        ):
                            return idx, mon
        return self._select_primary_monitor(monitors)

    def _apply_privacy_masks(
        self,
        img: Image.Image,
        monitor: dict[str, Any],
        os_context: OSContextEvent | None,
    ) -> None:
        if os_context is None or not os_context.visible_windows:
            return

        mon_left = int(monitor.get("left", 0))
        mon_top = int(monitor.get("top", 0))
        draw: ImageDraw.ImageDraw | None = None

        for win in os_context.visible_windows:
            is_win_sensitive = win.is_sensitive or self._privacy_shield.is_sensitive(
                win.process_name,
                win.window_title,
            )[0]
            if not is_win_sensitive:
                continue

            left, top, right, bottom = win.rect
            if right <= left or bottom <= top:
                continue

            x0 = max(0, min(img.width, int(left) - mon_left))
            y0 = max(0, min(img.height, int(top) - mon_top))
            x1 = max(0, min(img.width, int(right) - mon_left))
            y1 = max(0, min(img.height, int(bottom) - mon_top))
            if x1 > x0 and y1 > y0:
                if draw is None:
                    draw = ImageDraw.Draw(img)
                draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
