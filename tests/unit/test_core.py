"""Unit tests for Wave 0 frozen core contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from focus_guard.core import (
    AIConfig,
    BudgetGuard,
    EventBus,
    OSContextEvent,
    PrivacyConfig,
    PrivacyShield,
    load_config,
    save_api_key_to_env,
)
from focus_guard.core.ports import (
    CoachAgentPort,
    FocusEnginePort,
    InterventionDispatcherPort,
    MicRecorderPort,
    OSSensorPort,
    ScreenCapturePort,
    SemanticJudgePort,
    StoragePort,
    TTSPort,
    VisionSensorPort,
)
from tests.fakes import (
    FakeCoachAgent,
    FakeFocusEngine,
    FakeInterventionDispatcher,
    FakeMicRecorder,
    FakeOSSensor,
    FakeScreenCapture,
    FakeSemanticJudge,
    FakeStorage,
    FakeTTS,
    FakeVisionSensor,
)


def test_load_default_config() -> None:
    cfg = load_config()
    assert cfg.user_name == "Pedro"
    assert cfg.coach_persona == "firme"
    assert cfg.ai.primary_model == "gemini-3.5-flash-lite"
    assert cfg.ai.max_monthly_usd == 8.0


def test_save_api_key_to_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n", encoding="utf-8")
    save_api_key_to_env("test-secret-key", env_file)
    text = env_file.read_text(encoding="utf-8")
    assert "FOO=bar" in text
    assert "GEMINI_API_KEY=test-secret-key" in text


def test_privacy_shield_blocks_banks_and_incognito() -> None:
    shield = PrivacyShield(
        PrivacyConfig(
            blocked_process_names=["bitwarden.exe"],
            blocked_title_keywords=["anonima", "nubank", "itau", "pix"],
        )
    )
    assert shield.is_sensitive("Bitwarden.exe", "Vault")[0] is True
    assert shield.is_sensitive("chrome.exe", "Nova guia anônima - Google Chrome")[0] is True
    assert shield.is_sensitive("msedge.exe", "Banco Itaú - Internet Banking")[0] is True
    assert shield.is_sensitive("chrome.exe", "Tec Concursos - Direito Penal")[0] is False


def test_budget_guard_enforces_hard_cap() -> None:
    guard = BudgetGuard(AIConfig(max_daily_usd=0.01, max_monthly_usd=1.0))
    assert guard.can_spend() is True
    # Record heavy usage exceeding daily $0.01 cap
    cost = guard.record_usage("gemini-3.5-flash-lite", tokens_in=50_000, tokens_out=2_000)
    assert cost > 0.01
    assert guard.can_spend() is False
    status = guard.get_status()
    assert status.is_throttled is True
    assert status.warning_reached is True


@pytest.mark.asyncio
async def test_event_bus_isolates_handler_errors() -> None:
    bus = EventBus()
    received: list[str] = []

    async def bad_handler(_ev: OSContextEvent) -> None:
        raise RuntimeError("boom")

    async def good_handler(ev: OSContextEvent) -> None:
        received.append(ev.window_title)

    bus.subscribe(OSContextEvent, bad_handler)
    bus.subscribe(OSContextEvent, good_handler)

    await bus.publish(OSContextEvent(process_name="code.exe", window_title="FocusGuard"))
    assert received == ["FocusGuard"]


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeOSSensor(), OSSensorPort)
    assert isinstance(FakeScreenCapture(), ScreenCapturePort)
    assert isinstance(FakeVisionSensor(), VisionSensorPort)
    assert isinstance(FakeSemanticJudge(), SemanticJudgePort)
    assert isinstance(FakeTTS(), TTSPort)
    assert isinstance(FakeMicRecorder(), MicRecorderPort)
    assert isinstance(FakeCoachAgent(), CoachAgentPort)
    assert isinstance(FakeStorage(), StoragePort)
    assert isinstance(FakeInterventionDispatcher(), InterventionDispatcherPort)
    assert isinstance(FakeFocusEngine(), FocusEnginePort)
