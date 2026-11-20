"""Typed configuration loading & safe .env persistence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from focus_guard.core.events import CoachPersona

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"
DEFAULT_USER_CONFIG_PATH = PROJECT_ROOT / "data" / "user_config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


class AIConfig(BaseModel):
    primary_model: str = "gemini-3.5-flash-lite"
    fallback_models: list[str] = Field(
        default_factory=lambda: ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]
    )
    max_daily_usd: float = 0.35
    max_monthly_usd: float = 8.00
    brl_exchange_rate: float = 5.80
    capture_mode: Literal["all_monitors", "primary_monitor", "active_monitor"] = "all_monitors"
    screenshot_max_width: int = 1400
    screenshot_jpeg_quality: int = 70
    focused_recheck_interval_sec: int = 35
    ambiguous_recheck_interval_sec: int = 15
    window_switch_debounce_sec: int = 6


class BehaviorConfig(BaseModel):
    youtube_music_grace_sec: int = 30
    neutral_grace_sec: int = 25
    distraction_level1_sec: int = 25
    distraction_level2_sec: int = 90
    distraction_level3_sec: int = 180
    intervention_cooldown_sec: int = 45
    absence_alert_sec: int = 120
    camera_not_seeing_user_sec: int = 20
    passive_freeze_alert_sec: int = 240
    phone_distraction_sec: int = 15
    idle_webcam_check_sec: int = 12
    speak_on_all_interventions: bool = True
    show_windows_toast: bool = False
    show_level3_overlay: bool = False
    allow_looking_down_for_notes: bool = True
    require_ai_phone_confirmation: bool = True
    always_attach_webcam_to_judge: bool = True
    drowsiness_perclos_threshold: float = 0.45
    drowsiness_window_sec: int = 25
    pomodoro_focus_suggestion_min: int = 50


class HintsConfig(BaseModel):
    likely_study_keywords: list[str] = Field(default_factory=list)
    context_dependent_apps: list[str] = Field(default_factory=list)
    high_suspicion_patterns: list[str] = Field(default_factory=list)


class PrivacyConfig(BaseModel):
    blocked_process_names: list[str] = Field(default_factory=list)
    blocked_title_keywords: list[str] = Field(default_factory=list)


class VoiceConfig(BaseModel):
    enabled: bool = True
    tts_voice_ptbr: str = "pt-BR-AntonioNeural"
    tts_rate: str = "+5%"
    output_device_name: str | None = None
    input_device_name: str | None = None
    play_mic_chime: bool = True
    mic_sample_rate: int = 16000
    mic_max_seconds: int = 25
    mic_silence_timeout_sec: float = 1.8
    hotkey_privacy: str = "<ctrl>+<shift>+p"
    hotkey_talk: str = "<ctrl>+<shift>+<space>"


class VisionConfig(BaseModel):
    enabled: bool = True
    camera_index: int = 0
    target_fps: float = 1.5
    enable_person_detector: bool = True
    enable_phone_detector: bool = False
    phone_detector_interval_sec: float = 1.2
    phone_score_threshold: float = 0.60
    camera_position: Literal["center", "left", "right"] = "right"
    calibrated: bool = False
    neutral_yaw_deg: float = 0.0
    screen_yaw_min_deg: float = -55.0
    screen_yaw_max_deg: float = 25.0
    neutral_pitch_deg: float = 0.0
    neutral_eye_look_down: float = 0.22
    phone_pitch_delta_deg: float | None = None
    yaw_tolerance_deg: float = 22.0
    pitch_tolerance_deg: float = 18.0


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    open_browser_on_start: bool = True


class AppConfig(BaseModel):
    user_name: str = "Pedro"
    study_goal: str = "Estudo para Concurso Publico"
    current_subject: str = "Geral"
    coach_persona: CoachPersona = CoachPersona.FIRME
    gemini_api_key: str = Field(default="", repr=False)
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    ai: AIConfig = Field(default_factory=AIConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    hints: HintsConfig = Field(default_factory=HintsConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "focus_guard.db"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(
    config_path: Path | None = None,
    user_config_path: Path | None = None,
    env_path: Path | None = None,
) -> AppConfig:
    """Load YAML defaults, user overrides, and .env secrets."""
    resolved_env = env_path or DEFAULT_ENV_PATH
    if resolved_env.exists():
        load_dotenv(resolved_env, override=False)

    data: dict[str, Any] = {}
    base_path = config_path or DEFAULT_CONFIG_PATH
    if base_path.exists():
        with base_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                data = _deep_merge(data, loaded)

    override_path = user_config_path or DEFAULT_USER_CONFIG_PATH
    if override_path.exists():
        with override_path.open("r", encoding="utf-8") as f:
            loaded_override = yaml.safe_load(f) or {}
            if isinstance(loaded_override, dict):
                data = _deep_merge(data, loaded_override)

    cfg = AppConfig.model_validate(data)

    # Apply environment variable overrides
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        cfg.gemini_api_key = env_key

    env_model = os.getenv("FOCUS_GUARD_AI_MODEL", "").strip()
    if env_model:
        cfg.ai.primary_model = env_model

    env_daily = os.getenv("FOCUS_GUARD_MAX_DAILY_USD", "").strip()
    if env_daily:
        try:
            cfg.ai.max_daily_usd = float(env_daily)
        except ValueError:
            pass

    env_monthly = os.getenv("FOCUS_GUARD_MAX_MONTHLY_USD", "").strip()
    if env_monthly:
        try:
            cfg.ai.max_monthly_usd = float(env_monthly)
        except ValueError:
            pass

    env_port = os.getenv("FOCUS_GUARD_PORT", "").strip()
    if env_port:
        try:
            cfg.server.port = int(env_port)
        except ValueError:
            pass

    return cfg


def save_user_config(config: AppConfig, path: Path | None = None) -> Path:
    """Persist non-secret user configuration to data/user_config.yaml."""
    target = path or DEFAULT_USER_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json", exclude={"gemini_api_key", "data_dir"})
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return target


def save_api_key_to_env(api_key: str, env_path: Path | None = None) -> Path:
    """Safely store GEMINI_API_KEY into local .env file (git-ignored)."""
    target = env_path or DEFAULT_ENV_PATH
    clean_key = api_key.strip()
    lines: list[str] = []
    found = False
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                lines.append(f"GEMINI_API_KEY={clean_key}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"GEMINI_API_KEY={clean_key}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ["GEMINI_API_KEY"] = clean_key
    return target
