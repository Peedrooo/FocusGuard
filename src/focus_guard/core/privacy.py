"""Pre-capture PrivacyShield blocking sensitive windows (banks, password managers, incognito)."""

from __future__ import annotations

import unicodedata

from focus_guard.core.config import PrivacyConfig


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return ascii_only.lower().strip()


class PrivacyShield:
    """Evaluates whether an active window must be excluded from screenshots and cloud LLM calls."""

    def __init__(self, config: PrivacyConfig) -> None:
        self._blocked_processes = {_normalize(p) for p in config.blocked_process_names if p.strip()}
        self._blocked_keywords = [_normalize(k) for k in config.blocked_title_keywords if k.strip()]

    def is_sensitive(
        self,
        process_name: str,
        window_title: str,
        browser_url: str | None = None,
    ) -> tuple[bool, str | None]:
        proc = _normalize(process_name)
        if proc in self._blocked_processes:
            return True, f"process:{proc}"

        haystack = f"{_normalize(window_title)} {_normalize(browser_url or '')}"
        for kw in self._blocked_keywords:
            if kw and kw in haystack:
                return True, f"keyword:{kw}"

        return False, None
