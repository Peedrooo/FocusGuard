"""In-memory Voice Pipeline: Edge-TTS / pyttsx3 offline TTS & SoundDevice mic recorder."""

from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import edge_tts
import numpy as np
import pygame
import pyttsx3
import sounddevice as sd
import soundfile as sf

try:
    import pygame._sdl2.audio as _pygame_sdl2_audio  # noqa: F401
except Exception:
    _pygame_sdl2_audio = None  # type: ignore[assignment]

try:
    import winsound
except ImportError:
    winsound = None  # type: ignore[assignment]

from focus_guard.core.config import AppConfig, VoiceConfig
from focus_guard.core.event_bus import EventBus
from focus_guard.core.events import MicStateEvent

logger = logging.getLogger(__name__)

_MAPPER_NAME_PREFIXES = (
    "mapeador de som",
    "microsoft sound mapper",
    "driver de captura de som",
    "driver de som prim",
    "primary sound",
)


_TTS_MIXER_FREQ = 44100
_TTS_MIXER_SIZE = -16
_TTS_MIXER_CHANNELS = 2
_TTS_MIXER_BUFFER = 2048
_MARKDOWN_TTS_RE = re.compile(r"[`*_~#]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text_for_tts(text: str) -> str:
    """Strip markdown formatting noise so TTS speaks naturally without awkward pauses."""
    cleaned = _MARKDOWN_TTS_RE.sub("", text or "")
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _resolve_voice_config(config: VoiceConfig | AppConfig | None) -> VoiceConfig:
    if isinstance(config, AppConfig):
        return config.voice
    if config is not None:
        return config
    return VoiceConfig()


def _match_sdl2_device_name(target: str | None, sdl2_names: list[str]) -> str | None:
    """Match a PortAudio/user device name against SDL2 audio output names."""
    clean_target = (target or "").strip().lower()
    if not clean_target or not sdl2_names:
        return None

    # 1. Exact case-insensitive match
    for name in sdl2_names:
        if name.strip().lower() == clean_target:
            return name

    # 2. Prefix match (handles Windows PortAudio 31-char truncation)
    non_hf = [n for n in sdl2_names if "hands-free" not in n.lower()]
    ordered_pools = (non_hf, sdl2_names)

    for pool in ordered_pools:
        for name in pool:
            n_low = name.strip().lower()
            if n_low.startswith(clean_target) or clean_target.startswith(n_low):
                return name

    # 3. Substring match
    for pool in ordered_pools:
        for name in pool:
            n_low = name.strip().lower()
            if clean_target in n_low or n_low in clean_target:
                return name

    return None


def _is_mapper_or_empty(name: str) -> bool:
    low = name.strip().lower()
    if not low or low in {"input ()", "output ()", "alto-falantes ()", "fones de ouvido ()"}:
        return True
    return any(low.startswith(prefix) for prefix in _MAPPER_NAME_PREFIXES)


class EdgeAndOfflineTTS:
    """TTSPort implementation using edge-tts in RAM + pygame.mixer, with pyttsx3 fallback."""

    def __init__(
        self,
        config: VoiceConfig | AppConfig | None = None,
        *,
        communicate_factory: Callable[..., Any] | None = None,
        pyttsx3_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._voice_config = _resolve_voice_config(config)
        self._communicate_factory = communicate_factory or edge_tts.Communicate
        self._pyttsx3_factory = pyttsx3_factory or pyttsx3.init
        self._play_lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._speech_gen: int = 0
        self._stop_event = threading.Event()
        self._active_pyttsx3: Any | None = None
        self._active_sdl_device: str | None = None
        self._device_dirty: bool = False

    def _current_gen(self) -> int:
        with self._gen_lock:
            return self._speech_gen

    def _bump_gen(self) -> int:
        with self._gen_lock:
            self._speech_gen += 1
            return self._speech_gen

    def _get_sdl2_output_names(self, *, allow_temp_init: bool = False) -> tuple[list[str], bool]:
        """Return (sdl2_output_device_names, did_initialize_mixer)."""
        sdl2_mod = getattr(pygame, "_sdl2", None)
        audio_mod = getattr(sdl2_mod, "audio", None) if sdl2_mod is not None else None
        getter = getattr(audio_mod, "get_audio_device_names", None) if audio_mod is not None else None
        if getter is None and _pygame_sdl2_audio is not None:
            getter = getattr(_pygame_sdl2_audio, "get_audio_device_names", None)
        if getter is None:
            return [], False

        try:
            names = list(getter(False) or [])
            return names, False
        except Exception:
            if not allow_temp_init:
                return [], False

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(
                    frequency=_TTS_MIXER_FREQ,
                    size=_TTS_MIXER_SIZE,
                    channels=_TTS_MIXER_CHANNELS,
                    buffer=_TTS_MIXER_BUFFER,
                )
                names = list(getter(False) or [])
                return names, True
        except Exception:
            logger.debug("Could not enumerate SDL2 audio output devices.", exc_info=True)
        return [], False

    def _query_default_output_name(self) -> str | None:
        try:
            default_dev = sd.query_devices(kind="output")
            if isinstance(default_dev, dict):
                name = str(default_dev.get("name") or "").strip()
                return name or None
        except Exception:
            logger.debug("Could not query default output device via sounddevice.", exc_info=True)
        return None

    def _resolve_desired_sdl2_device(self, *, allow_temp_init: bool = True) -> tuple[str | None, bool]:
        target = (self._voice_config.output_device_name or "").strip() or None
        if target is None:
            target = self._query_default_output_name()
        sdl2_names, did_temp_init = self._get_sdl2_output_names(allow_temp_init=allow_temp_init)
        resolved = _match_sdl2_device_name(target, sdl2_names)
        return resolved, did_temp_init

    def list_output_devices(self) -> list[dict[str, Any]]:
        """Safely list available audio output devices matching PortAudio and SDL2 names."""
        sd_outputs: list[dict[str, Any]] = []
        try:
            raw = sd.query_devices()
            if isinstance(raw, dict):
                raw_list = [raw]
            elif raw is not None:
                raw_list = [dict(d) for d in raw]
            else:
                raw_list = []
            for dev in raw_list:
                if int(dev.get("max_output_channels", 0) or 0) > 0:
                    sd_outputs.append(dev)
        except Exception:
            logger.debug("sounddevice.query_devices() failed in list_output_devices.", exc_info=True)

        default_name = self._query_default_output_name()
        configured_name = (self._voice_config.output_device_name or "").strip() or None

        sdl2_names: list[str] = []
        if pygame.mixer.get_init():
            sdl2_names, _ = self._get_sdl2_output_names(allow_temp_init=False)
        else:
            with self._play_lock:
                temp_inited = False
                try:
                    sdl2_names, temp_inited = self._get_sdl2_output_names(allow_temp_init=True)
                finally:
                    if temp_inited:
                        try:
                            pygame.mixer.quit()
                        except Exception:
                            pass

        if sdl2_names:
            matched_default = _match_sdl2_device_name(default_name, sdl2_names)
            matched_cfg = _match_sdl2_device_name(configured_name, sdl2_names)
            results: list[dict[str, Any]] = []
            for sdl_name in sdl2_names:
                sdl_low = sdl_name.lower()
                matched_sd = next(
                    (
                        d
                        for d in sd_outputs
                        if str(d.get("name") or "").strip().lower() in sdl_low
                        or sdl_low.startswith(str(d.get("name") or "").strip().lower())
                    ),
                    None,
                )
                is_default = sdl_name == matched_default
                is_selected = (sdl_name == matched_cfg) if matched_cfg else is_default
                results.append(
                    {
                        "name": sdl_name,
                        "sdl_name": sdl_name,
                        "is_default": is_default,
                        "is_selected": is_selected,
                        "max_output_channels": int((matched_sd or {}).get("max_output_channels", 2) or 2),
                        "default_samplerate": int(
                            float((matched_sd or {}).get("default_samplerate", 44100) or 44100)
                        ),
                    }
                )
            return results

        # Fallback to sounddevice list when SDL2 names are unavailable
        dedup: dict[str, dict[str, Any]] = {}
        for dev in sd_outputs:
            name = str(dev.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key not in dedup or int(dev.get("hostapi", 0) or 0) != 3:
                dedup[key] = dev

        results = []
        for dev in dedup.values():
            name = str(dev.get("name") or "")
            is_default = bool(default_name and name.lower() == default_name.lower())
            is_selected = (
                bool(configured_name and configured_name.lower() in name.lower())
                if configured_name
                else is_default
            )
            results.append(
                {
                    "name": name,
                    "sdl_name": name,
                    "is_default": is_default,
                    "is_selected": is_selected,
                    "max_output_channels": int(dev.get("max_output_channels", 2) or 2),
                    "default_samplerate": int(float(dev.get("default_samplerate", 44100) or 44100)),
                }
            )
        return results

    def set_output_device(self, device_name: str | None) -> None:
        """Select preferred audio output device (None resets to system default)."""
        clean = (device_name or "").strip() or None
        self._voice_config.output_device_name = clean
        self._device_dirty = True

    def _ensure_mixer_initialized(self) -> None:
        is_init = bool(pygame.mixer.get_init())
        resolved, did_temp_init = self._resolve_desired_sdl2_device(allow_temp_init=not is_init)
        is_init = bool(pygame.mixer.get_init())
        if is_init and not did_temp_init and not self._device_dirty and resolved == self._active_sdl_device:
            return

        if is_init:
            try:
                pygame.mixer.quit()
            except Exception:
                logger.debug("Ignoring pygame.mixer.quit error.", exc_info=True)

        initialized = False
        if resolved:
            try:
                pygame.mixer.init(
                    frequency=_TTS_MIXER_FREQ,
                    size=_TTS_MIXER_SIZE,
                    channels=_TTS_MIXER_CHANNELS,
                    buffer=_TTS_MIXER_BUFFER,
                    devicename=resolved,
                )
                self._active_sdl_device = resolved
                initialized = True
            except Exception:
                logger.warning(
                    "pygame.mixer.init(devicename=%r) failed; falling back to default device.",
                    resolved,
                    exc_info=True,
                )

        if not initialized:
            pygame.mixer.init(
                frequency=_TTS_MIXER_FREQ,
                size=_TTS_MIXER_SIZE,
                channels=_TTS_MIXER_CHANNELS,
                buffer=_TTS_MIXER_BUFFER,
            )
            self._active_sdl_device = None

        self._device_dirty = False

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        """Synthesize PT-BR speech strictly in RAM using edge-tts."""
        clean = _clean_text_for_tts(text)
        if not clean or not self._voice_config.enabled:
            return b"", "audio/mpeg"

        try:
            communicate = self._communicate_factory(
                clean,
                voice=self._voice_config.tts_voice_ptbr,
                rate=self._voice_config.tts_rate,
            )
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if isinstance(chunk, dict) and chunk.get("type") == "audio":
                    data = chunk.get("data")
                    if isinstance(data, (bytes, bytearray)) and data:
                        buf.write(data)
            audio = buf.getvalue()
            if audio:
                return audio, "audio/mpeg"
            logger.warning("edge-tts returned empty audio stream; falling back safely.")
            return b"", "audio/wav"
        except Exception:
            logger.warning("edge-tts synthesis failed; falling back gracefully.", exc_info=True)
            return b"", "audio/wav"

    async def speak(self, text: str) -> bool:
        """Synthesize and play audio in RAM, falling back to offline pyttsx3 if needed."""
        clean = _clean_text_for_tts(text)
        if not clean or not self._voice_config.enabled:
            return False

        # Claim latest speech generation and interrupt any previous playback immediately.
        my_gen = self._bump_gen()
        self._stop_active_playback()

        audio_bytes, _ = await self.synthesize(clean)
        if my_gen != self._current_gen():
            return False

        if audio_bytes:
            outcome = await asyncio.to_thread(
                self._play_mp3_bytes_blocking, audio_bytes, my_gen
            )
            if outcome == "played":
                return True
            if outcome == "interrupted":
                return False

        if my_gen != self._current_gen():
            return False

        return await asyncio.to_thread(self._speak_pyttsx3_blocking, clean, my_gen)

    def _stop_active_playback(self) -> None:
        self._stop_event.set()
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            logger.debug("Ignoring pygame.mixer.music.stop error.", exc_info=True)

        engine = self._active_pyttsx3
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                logger.debug("Ignoring pyttsx3 stop error.", exc_info=True)

    def stop(self) -> None:
        """Stop active playback cleanly and cancel any in-flight synthesis."""
        self._bump_gen()
        self._stop_active_playback()

    @staticmethod
    def _safe_unload_music() -> None:
        unload_fn = getattr(pygame.mixer.music, "unload", None)
        if callable(unload_fn):
            try:
                unload_fn()
            except Exception:
                logger.debug("Ignoring pygame.mixer.music.unload error.", exc_info=True)

    def _play_mp3_bytes_blocking(
        self, audio_bytes: bytes, expected_gen: int | None = None
    ) -> str:
        with self._play_lock:
            if expected_gen is not None and expected_gen != self._current_gen():
                return "interrupted"
            self._stop_event.clear()
            try:
                self._ensure_mixer_initialized()
                pygame.mixer.music.stop()
                self._safe_unload_music()
                stream = io.BytesIO(audio_bytes)
                try:
                    pygame.mixer.music.load(stream, namehint="mp3")
                except TypeError:
                    pygame.mixer.music.load(stream)
                if self._stop_event.is_set() or (
                    expected_gen is not None and expected_gen != self._current_gen()
                ):
                    return "interrupted"
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    if self._stop_event.is_set() or (
                        expected_gen is not None and expected_gen != self._current_gen()
                    ):
                        pygame.mixer.music.stop()
                        return "interrupted"
                    time.sleep(0.03)
                if self._stop_event.is_set() or (
                    expected_gen is not None and expected_gen != self._current_gen()
                ):
                    return "interrupted"
                return "played"
            except Exception:
                logger.warning("pygame.mixer playback failed; resetting mixer and trying fallback.", exc_info=True)
                self._active_sdl_device = None
                self._device_dirty = True
                try:
                    if pygame.mixer.get_init():
                        pygame.mixer.quit()
                except Exception:
                    pass
                return "failed"
            finally:
                self._safe_unload_music()

    def _speak_pyttsx3_blocking(
        self, text: str, expected_gen: int | None = None
    ) -> bool:
        with self._play_lock:
            if expected_gen is not None and expected_gen != self._current_gen():
                return False
            self._stop_event.clear()
            engine: Any | None = None
            try:
                engine = self._pyttsx3_factory()
                self._active_pyttsx3 = engine
                self._select_ptbr_voice(engine)
                engine.say(text)
                engine.runAndWait()
                return not self._stop_event.is_set() and (
                    expected_gen is None or expected_gen == self._current_gen()
                )
            except Exception:
                logger.warning("Offline pyttsx3 speech failed.", exc_info=True)
                return False
            finally:
                self._active_pyttsx3 = None

    @staticmethod
    def _select_ptbr_voice(engine: Any) -> None:
        try:
            voices = engine.getProperty("voices") or []
            for voice in voices:
                voice_id = str(getattr(voice, "id", "") or "").lower()
                voice_name = str(getattr(voice, "name", "") or "").lower()
                langs = [str(lang).lower() for lang in (getattr(voice, "languages", None) or [])]
                if "pt-br" in voice_id or "pt_br" in voice_id or "brazil" in voice_name or any(
                    "pt" in lang for lang in langs
                ):
                    engine.setProperty("voice", voice.id)
                    break
        except Exception:
            logger.debug("Could not enumerate pyttsx3 voices.", exc_info=True)


class SoundDeviceMicRecorder:
    """MicRecorderPort implementation capturing native audio in RAM and encoding 16kHz mono WAV."""

    def __init__(
        self,
        config: VoiceConfig | AppConfig | None = None,
        event_bus: EventBus | None = None,
        *,
        rms_threshold: float = 0.012,
        chunk_duration_sec: float = 0.1,
        input_stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._voice_config = _resolve_voice_config(config)
        self._event_bus = event_bus
        self._rms_threshold = max(0.0001, float(rms_threshold))
        self._chunk_duration_sec = max(0.02, float(chunk_duration_sec))
        self._input_stream_factory = input_stream_factory
        self._manual_stop_event = threading.Event()

    def is_available(self) -> bool:
        if not self._voice_config.enabled:
            return False
        try:
            devices = sd.query_devices()
        except Exception:
            logger.warning("Failed to query audio input devices.", exc_info=True)
            return False

        if isinstance(devices, dict):
            return int(devices.get("max_input_channels", 0) or 0) > 0
        try:
            return any(int(dev.get("max_input_channels", 0) or 0) > 0 for dev in devices)
        except Exception:
            return False

    def list_input_devices(self) -> list[dict[str, Any]]:
        """Safely list available audio input devices."""
        try:
            raw = sd.query_devices()
        except Exception:
            logger.debug("sounddevice.query_devices() failed in list_input_devices.", exc_info=True)
            return []

        raw_list = [raw] if isinstance(raw, dict) else [dict(d) for d in (raw or [])]
        default_name = ""
        try:
            default_dev = sd.query_devices(kind="input")
            if isinstance(default_dev, dict):
                default_name = str(default_dev.get("name") or "").strip()
        except Exception:
            pass

        input_entries: list[tuple[int, dict[str, Any]]] = []
        for i, dev in enumerate(raw_list):
            if int(dev.get("max_input_channels", 0) or 0) > 0:
                idx = int(dev.get("index", i) if dev.get("index") is not None else i)
                input_entries.append((idx, dev))

        if not input_entries:
            return []

        non_mappers = [
            (idx, d) for idx, d in input_entries if not _is_mapper_or_empty(str(d.get("name") or ""))
        ]
        pool = non_mappers if non_mappers else input_entries

        # Deduplicate across Windows hostAPIs preferring non-WDM-KS and highest channels
        by_name: dict[str, tuple[int, dict[str, Any]]] = {}
        for idx, dev in pool:
            name = str(dev.get("name") or f"Input {idx}").strip()
            key = name.lower()
            existing = by_name.get(key)
            if existing is None:
                by_name[key] = (idx, dev)
                continue
            _, prev = existing
            cur_score = (
                int(dev.get("hostapi", 0) or 0) != 3,
                int(dev.get("max_input_channels", 0) or 0),
                int(float(dev.get("default_samplerate", 0) or 0)),
            )
            prev_score = (
                int(prev.get("hostapi", 0) or 0) != 3,
                int(prev.get("max_input_channels", 0) or 0),
                int(float(prev.get("default_samplerate", 0) or 0)),
            )
            if cur_score > prev_score:
                by_name[key] = (idx, dev)

        configured = (self._voice_config.input_device_name or "").strip().lower()
        results: list[dict[str, Any]] = []
        for idx, dev in by_name.values():
            name = str(dev.get("name") or f"Input {idx}").strip()
            name_low = name.lower()
            is_default = bool(
                default_name
                and (
                    name_low == default_name.lower()
                    or name_low.startswith(default_name.lower())
                    or default_name.lower().startswith(name_low)
                )
            )
            is_selected = bool(configured and configured in name_low) if configured else is_default
            results.append(
                {
                    "index": idx,
                    "name": name,
                    "max_input_channels": int(dev.get("max_input_channels", 1) or 1),
                    "default_samplerate": int(float(dev.get("default_samplerate", 44100) or 44100)),
                    "is_default": is_default,
                    "is_selected": is_selected,
                }
            )
        results.sort(
            key=lambda item: (
                not item["is_selected"],
                not item["is_default"],
                "hands-free" in item["name"].lower(),
                item["name"].lower(),
            )
        )
        return results

    def set_input_device(self, device_name: str | None) -> None:
        """Select preferred microphone device name (None resets to system default)."""
        clean = (device_name or "").strip() or None
        self._voice_config.input_device_name = clean

    def stop_recording(self) -> None:
        """Signal any active capture loop to stop cleanly and return captured speech."""
        self._manual_stop_event.set()

    def _resolve_input_device(self) -> tuple[int | None, str, int, int]:
        """Return (device_index, device_name, default_samplerate, max_input_channels)."""
        preferred = (self._voice_config.input_device_name or "").strip().lower()
        all_devices: list[dict[str, Any]] = []
        try:
            raw = sd.query_devices()
            if isinstance(raw, dict):
                all_devices = [raw]
            elif raw is not None:
                all_devices = [dict(d) for d in raw]
        except Exception:
            logger.debug("Could not enumerate input devices.", exc_info=True)

        input_devs: list[tuple[int | None, dict[str, Any]]] = []
        for i, dev in enumerate(all_devices):
            if int(dev.get("max_input_channels", 0) or 0) > 0:
                idx = dev.get("index", i)
                input_devs.append((int(idx) if idx is not None else i, dev))

        if preferred and input_devs:
            matches = [
                (idx, dev)
                for idx, dev in input_devs
                if preferred in str(dev.get("name") or "").lower()
            ]
            if matches:
                matches.sort(
                    key=lambda item: (
                        "hands-free" not in str(item[1].get("name") or "").lower(),
                        int(item[1].get("hostapi", 0) or 0) != 3,
                        int(item[1].get("max_input_channels", 0) or 0),
                    ),
                    reverse=True,
                )
                idx, dev = matches[0]
                return (
                    idx,
                    str(dev.get("name") or "Mic"),
                    int(float(dev.get("default_samplerate") or 0)),
                    int(dev.get("max_input_channels", 1) or 1),
                )

        try:
            default_dev = sd.query_devices(kind="input")
            if isinstance(default_dev, dict) and int(default_dev.get("max_input_channels", 0) or 0) > 0:
                idx = default_dev.get("index")
                return (
                    int(idx) if idx is not None else None,
                    str(default_dev.get("name") or "Default Mic"),
                    int(float(default_dev.get("default_samplerate") or 0)),
                    int(default_dev.get("max_input_channels", 1) or 1),
                )
        except Exception:
            pass

        if input_devs:
            idx, dev = input_devs[0]
            return (
                idx,
                str(dev.get("name") or "Mic"),
                int(float(dev.get("default_samplerate") or 0)),
                int(dev.get("max_input_channels", 1) or 1),
            )

        return None, "Default Mic", 0, 1

    def _play_chime_sync(self, freq_hz: int, duration_ms: int = 65) -> None:
        if not self._voice_config.play_mic_chime or self._input_stream_factory is not None:
            return
        if winsound is None:
            return
        try:
            winsound.Beep(freq_hz, duration_ms)
        except Exception:
            pass

    def _play_chime_async(self, freq_hz: int, duration_ms: int = 65) -> None:
        if not self._voice_config.play_mic_chime or self._input_stream_factory is not None:
            return
        if winsound is None:
            return

        threading.Thread(
            target=self._play_chime_sync,
            args=(freq_hz, duration_ms),
            daemon=True,
        ).start()

    def _emit_mic_state(
        self,
        loop: asyncio.AbstractEventLoop | None,
        state: str,
        rms_level: float = 0.0,
        device_name: str = "",
        message: str = "",
    ) -> None:
        if self._event_bus is None or loop is None:
            return
        try:
            if not loop.is_running():
                return
            event = MicStateEvent(
                state=state,  # type: ignore[arg-type]
                rms_level=round(float(rms_level), 5),
                device_name=device_name,
                message=message,
            )
            asyncio.run_coroutine_threadsafe(self._event_bus.publish(event), loop)
        except Exception:
            logger.debug("Ignoring MicStateEvent publish error.", exc_info=True)

    def _open_input_stream(
        self,
        device_idx: int | None,
        dev_default_sr: int,
        target_rate: int,
    ) -> tuple[Any, int, int, int] | None:
        if self._input_stream_factory is not None and dev_default_sr <= 0:
            raw_rates = [target_rate, 48000, 44100, 16000]
        else:
            raw_rates = [dev_default_sr, 48000, 44100, 16000, target_rate]

        candidate_rates = list(dict.fromkeys(r for r in raw_rates if r and r > 0))
        if not candidate_rates:
            candidate_rates = [16000]

        for cand_sr in candidate_rates:
            chunk_samples = max(1, int(round(cand_sr * self._chunk_duration_sec)))
            for ch in (1, 2):
                try:
                    if self._input_stream_factory is not None:
                        stream = self._input_stream_factory(
                            samplerate=cand_sr,
                            channels=ch,
                            dtype="float32",
                            blocksize=chunk_samples,
                        )
                    else:
                        kwargs: dict[str, Any] = {
                            "samplerate": cand_sr,
                            "channels": ch,
                            "dtype": "float32",
                            "blocksize": chunk_samples,
                        }
                        if device_idx is not None:
                            kwargs["device"] = device_idx
                        stream = sd.InputStream(**kwargs)
                    return stream, cand_sr, ch, chunk_samples
                except Exception:
                    logger.debug(
                        "InputStream open failed (device=%r, sr=%s, ch=%s); trying next.",
                        device_idx,
                        cand_sr,
                        ch,
                    )
        return None

    async def record_until_silence(
        self,
        max_seconds: float | None = None,
        silence_timeout: float | None = None,
    ) -> bytes | None:
        """Record audio in RAM until silence timeout or manual stop, returning 16kHz mono WAV."""
        if not self.is_available():
            return None

        resolved_max = float(
            max_seconds if max_seconds is not None else self._voice_config.mic_max_seconds
        )
        resolved_silence = float(
            silence_timeout
            if silence_timeout is not None
            else self._voice_config.mic_silence_timeout_sec
        )
        sample_rate = int(self._voice_config.mic_sample_rate)

        self._manual_stop_event.clear()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        _, dev_name, _, _ = self._resolve_input_device()
        try:
            return await asyncio.to_thread(
                self._record_blocking,
                resolved_max,
                resolved_silence,
                sample_rate,
                loop,
            )
        finally:
            if self._event_bus is not None:
                try:
                    await self._event_bus.publish(
                        MicStateEvent(state="idle", rms_level=0.0, device_name=dev_name)
                    )
                except Exception:
                    logger.debug("Failed to publish idle MicStateEvent.", exc_info=True)

    def _record_blocking(
        self,
        max_seconds: float,
        silence_timeout: float,
        sample_rate: int,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> bytes | None:
        if max_seconds <= 0 or sample_rate <= 0:
            return None

        device_idx, device_name, dev_default_sr, _ = self._resolve_input_device()
        opened = self._open_input_stream(
            device_idx=device_idx,
            dev_default_sr=dev_default_sr,
            target_rate=sample_rate,
        )
        if opened is None:
            logger.warning("Could not open any microphone input stream.")
            return None

        stream, capture_rate, _channels, chunk_samples = opened

        pre_roll_maxlen = max(1, int(round(0.6 / self._chunk_duration_sec)))
        pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_maxlen)
        quiet_frames: list[np.ndarray] = []
        frames: list[np.ndarray] = []

        use_adaptive_vad = self._input_stream_factory is None
        calib_chunks_needed = max(1, int(round(0.3 / self._chunk_duration_sec)))
        calib_rms: list[float] = []
        active_thresh = self._rms_threshold

        speech_started = False
        silent_elapsed = 0.0
        audio_elapsed = 0.0
        max_rms = 0.0
        wall_start = time.monotonic()
        last_emit_ts = 0.0

        # Play start chime BEFORE opening the capture stream so the beep never leaks into VAD calibration.
        self._play_chime_sync(660, 65)
        wall_start = time.monotonic()

        try:
            with stream:
                self._emit_mic_state(loop, "listening", rms_level=0.0, device_name=device_name)
                last_emit_ts = time.monotonic()

                while (
                    not self._manual_stop_event.is_set()
                    and audio_elapsed < max_seconds
                    and (time.monotonic() - wall_start) < (max_seconds + 2.0)
                ):
                    raw_chunk, _ = stream.read(chunk_samples)
                    arr = np.asarray(raw_chunk)
                    if arr.size == 0:
                        if (time.monotonic() - wall_start) >= max_seconds:
                            break
                        continue

                    if np.issubdtype(arr.dtype, np.integer):
                        arr_f32 = arr.astype(np.float32) / 32768.0
                    else:
                        arr_f32 = arr.astype(np.float32)

                    if arr_f32.ndim > 1:
                        if arr_f32.shape[1] >= 2:
                            chunk_f32 = (arr_f32[:, 0] + arr_f32[:, 1]) * np.float32(0.5)
                        else:
                            chunk_f32 = arr_f32[:, 0]
                    else:
                        chunk_f32 = arr_f32

                    chunk_sec = float(chunk_f32.size) / float(capture_rate)
                    audio_elapsed += chunk_sec

                    rms = float(np.sqrt(np.mean(np.square(chunk_f32))))
                    if rms > max_rms:
                        max_rms = rms

                    now = time.monotonic()
                    if now - last_emit_ts >= 0.25:
                        last_emit_ts = now
                        self._emit_mic_state(
                            loop, "listening", rms_level=rms, device_name=device_name
                        )

                    if use_adaptive_vad and len(calib_rms) < calib_chunks_needed:
                        calib_rms.append(rms)
                        pre_roll.append(chunk_f32)
                        quiet_frames.append(chunk_f32)
                        if len(calib_rms) >= calib_chunks_needed:
                            noise_floor = float(np.percentile(calib_rms, 25))
                            active_thresh = max(0.0028, min(0.015, noise_floor * 2.2))
                        continue

                    if rms >= active_thresh:
                        if not speech_started:
                            speech_started = True
                            frames.extend(pre_roll)
                            pre_roll.clear()
                            quiet_frames.clear()
                        frames.append(chunk_f32)
                        silent_elapsed = 0.0
                    elif speech_started:
                        frames.append(chunk_f32)
                        silent_elapsed += chunk_sec
                        if silent_elapsed >= silence_timeout:
                            break
                    else:
                        pre_roll.append(chunk_f32)
                        quiet_frames.append(chunk_f32)
        except Exception:
            logger.warning("Microphone capture failed gracefully.", exc_info=True)
            return None
        finally:
            self._play_chime_async(880, 65)

        if not speech_started:
            if max_rms >= 0.0018 and len(quiet_frames) >= 3:
                frames = quiet_frames
            else:
                return None

        if not frames:
            return None

        self._emit_mic_state(loop, "processing", rms_level=max_rms, device_name=device_name)

        pcm = np.concatenate(frames).astype(np.float32)
        if capture_rate != sample_rate and pcm.size > 1:
            out_len = max(1, int(round(pcm.size * (float(sample_rate) / float(capture_rate)))))
            x_old = np.linspace(0.0, 1.0, num=pcm.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
            pcm = np.interp(x_new, x_old, pcm).astype(np.float32)

        peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
        if 0.0005 < peak < 0.25:
            gain = min(8.0, 0.45 / peak)
            pcm = np.clip(pcm * np.float32(gain), -0.95, 0.95)

        buf = io.BytesIO()
        try:
            sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
            return buf.getvalue()
        except Exception:
            logger.warning("Failed to encode WAV audio in memory.", exc_info=True)
            return None
