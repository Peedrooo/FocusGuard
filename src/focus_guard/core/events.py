"""Canonical immutable-friendly data models and event contracts for FocusGuard."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid4().hex[:12]


class FocusState(StrEnum):
    PAUSED = "PAUSED"
    PRIVACY_MODE = "PRIVACY_MODE"
    FOCUSED = "FOCUSED"
    GRACE_WINDOW = "GRACE_WINDOW"
    DRIFT_SUSPECTED = "DRIFT_SUSPECTED"
    INTERVENING = "INTERVENING"
    BREAK_MODE = "BREAK_MODE"
    CONVERSATION_MODE = "CONVERSATION_MODE"


class ActivityVerdict(StrEnum):
    STUDYING = "STUDYING"
    MUSIC_ALLOWED = "MUSIC_ALLOWED"
    DISTRACTED = "DISTRACTED"
    UNCERTAIN = "UNCERTAIN"
    AWAY = "AWAY"


class InterventionLevel(IntEnum):
    NONE = 0
    SOFT = 1
    VOICE = 2
    FIRM_OVERLAY = 3


class CoachPersona(StrEnum):
    FIRME = "firme"
    EQUILIBRADO = "equilibrado"
    SARGENTO = "sargento"


class VisibleWindowInfo(BaseModel):
    hwnd: int = 0
    process_name: str
    window_title: str
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    monitor_label: str = "M1"
    screen_area_pct: float = 0.0
    is_foreground: bool = False
    is_sensitive: bool = False


class ProcessActivityInfo(BaseModel):
    process_name: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    has_active_audio: bool = False
    window_count: int = 0
    category: str = "app"


class UsagePatternMetrics(BaseModel):
    active_input_ratio_60s: float = Field(default=0.0, ge=0.0, le=1.0)
    window_switches_60s: int = 0
    continuous_same_screen_sec: float = 0.0
    interaction_mode: Literal[
        "active_work", "reading_or_watching", "rapid_switching", "idle_with_media", "idle"
    ] = "idle"


class OSContextEvent(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    process_name: str
    window_title: str
    browser_url: str | None = None
    background_audio_apps: list[str] = Field(default_factory=list)
    idle_seconds: float = 0.0
    is_sensitive_window: bool = False
    sensitive_reason: str | None = None
    active_monitor_label: str = "M1"
    visible_windows: list[VisibleWindowInfo] = Field(default_factory=list)
    notable_running_processes: list[ProcessActivityInfo] = Field(default_factory=list)
    usage_pattern: UsagePatternMetrics = Field(default_factory=UsagePatternMetrics)

    def context_signature(self) -> str:
        """Stable short key representing the active app + window + url."""
        url_part = (self.browser_url or "").split("#")[0][:120]
        return f"{self.process_name.lower()}|{self.window_title.strip()[:120]}|{url_part}"


class ScreenSnapshot(BaseModel):
    """Ephemeral in-memory screenshot. MUST NEVER be persisted to disk."""

    timestamp: datetime = Field(default_factory=utc_now)
    image_bytes_jpeg: bytes = Field(repr=False)
    perceptual_hash: str
    width: int
    height: int
    monitor_index: int = 1

    def to_base64(self) -> str:
        return base64.b64encode(self.image_bytes_jpeg).decode("ascii")


class PhoneBox(BaseModel):
    xmin: float = 0.0
    ymin: float = 0.0
    xmax: float = 0.0
    ymax: float = 0.0
    score: float = 0.0
    label: str = "cell phone"


class VisionSignalEvent(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    camera_available: bool = True
    face_present: bool = True
    gaze_on_screen_score: float = Field(default=1.0, ge=0.0, le=1.0)
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    head_roll_deg: float = 0.0
    drowsiness_perclos: float = Field(default=0.0, ge=0.0, le=1.0)
    yawn_score: float = Field(default=0.0, ge=0.0, le=1.0)
    looking_down_score: float = Field(default=0.0, ge=0.0, le=1.0)
    phone_detected: bool = False
    phone_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    camera_position: str = "right"
    calibrated: bool = False
    looking_at_screen: bool = True
    calibrated_yaw_delta_deg: float = 0.0
    calibrated_pitch_delta_deg: float = 0.0
    camera_obstructed: bool = False
    phone_boxes: list[PhoneBox] = Field(default_factory=list)
    error: str | None = None


class SemanticVerdict(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    verdict: ActivityVerdict
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    category: str = "unknown"
    app_context: str = ""
    reasoning: str = ""
    suggested_message: str = ""
    used_vision: bool = False
    used_webcam: bool = False
    phone_in_use: bool = False
    physical_state: str = "focused_on_screen"
    model_used: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: float = 0.0
    cached: bool = False


class MicStateEvent(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    state: Literal["idle", "listening", "processing", "error"] = "idle"
    rms_level: float = 0.0
    device_name: str = ""
    message: str = ""


class FocusStateChangedEvent(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    previous_state: FocusState
    new_state: FocusState
    reason: str
    distraction_seconds: float = 0.0
    escalation_level: InterventionLevel = InterventionLevel.NONE
    focus_score: float = 100.0


class InterventionCommand(BaseModel):
    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)
    level: InterventionLevel
    title: str
    message: str
    speak_aloud: bool = False
    reason: str = ""
    actions: list[str] = Field(
        default_factory=lambda: ["back_to_focus", "break_5m", "talk_to_coach"]
    )


class ProactiveSuggestionEvent(BaseModel):
    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)
    kind: Literal["fatigue", "pomodoro", "stuck", "info"] = "info"
    message: str
    speak_aloud: bool = False


class ChatMessage(BaseModel):
    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)
    role: Literal["user", "coach", "system"]
    text: str
    audio_base64: str | None = Field(default=None, repr=False)
    audio_mime: str | None = None
    tool_calls: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0


class TemporaryAllowance(BaseModel):
    id: str = Field(default_factory=new_id)
    pattern: str
    reason: str
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime

    def is_active(self, now: datetime | None = None) -> bool:
        ref = now or utc_now()
        return ref < self.expires_at


class StudyNote(BaseModel):
    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)
    subject: str
    note: str


class BudgetStatus(BaseModel):
    daily_usd: float = 0.0
    monthly_usd: float = 0.0
    max_daily_usd: float = 0.35
    max_monthly_usd: float = 8.00
    daily_tokens_in: int = 0
    daily_tokens_out: int = 0
    daily_calls: int = 0
    brl_exchange_rate: float = 5.80
    is_throttled: bool = False
    warning_reached: bool = False

    @property
    def daily_brl(self) -> float:
        return round(self.daily_usd * self.brl_exchange_rate, 4)

    @property
    def monthly_brl(self) -> float:
        return round(self.monthly_usd * self.brl_exchange_rate, 2)


class SessionStats(BaseModel):
    session_id: str = Field(default_factory=new_id)
    started_at: datetime = Field(default_factory=utc_now)
    study_goal: str = "Estudo para Concurso Publico"
    current_subject: str = "Geral"
    current_state: FocusState = FocusState.PAUSED
    focus_seconds: float = 0.0
    distraction_seconds: float = 0.0
    grace_seconds: float = 0.0
    break_seconds: float = 0.0
    focus_ratio_pct: float = 100.0
    interventions_count: int = 0
    phone_alerts_count: int = 0
    drowsiness_alerts_count: int = 0
    camera_unseen_seconds: float = 0.0
    escalation_level: InterventionLevel = InterventionLevel.NONE
    active_break_remaining_sec: float = 0.0
    recent_verdicts: list[SemanticVerdict] = Field(default_factory=list)


class SystemStatusSnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    running: bool = False
    privacy_mode: bool = False
    api_key_configured: bool = False
    camera_available: bool = False
    mic_available: bool = False
    coach_persona: CoachPersona = CoachPersona.FIRME
    session: SessionStats = Field(default_factory=SessionStats)
    budget: BudgetStatus = Field(default_factory=BudgetStatus)
    latest_os: OSContextEvent | None = None
    latest_vision: VisionSignalEvent | None = None
    active_allowances: list[TemporaryAllowance] = Field(default_factory=list)
