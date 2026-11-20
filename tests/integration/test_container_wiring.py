"""Integration tests verifying AppContainer wiring and FastAPI startup/shutdown."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from focus_guard.app.container import AppContainer
from focus_guard.core.config import AppConfig
from focus_guard.core.events import FocusState


def test_app_container_wiring_and_api_lifecycle(tmp_path: Path) -> None:
    cfg = AppConfig(
        gemini_api_key="fake-integration-key",
        data_dir=tmp_path / "data",
    )
    cfg.vision.enabled = False
    cfg.server.open_browser_on_start = False

    container = AppContainer(
        config=cfg,
        enable_tray=False,
        enable_ui=False,
        enable_camera=False,
        auto_start_session=True,
    )

    assert container.semantic_judge.is_configured() is True
    assert container.coach.is_configured() is True

    import asyncio

    async def _run() -> None:
        await container.start()
        try:
            status = container.engine.get_status()
            assert status.running is True
            assert status.api_key_configured is True
            assert status.session.current_state == FocusState.FOCUSED
        finally:
            await container.stop()

    asyncio.run(_run())

    with TestClient(container.api_app) as client:
        res = client.get("/api/status")
        assert res.status_code == 200
        payload = res.json()
        assert payload["status"]["api_key_configured"] is True
        assert payload["status"]["coach_persona"] == "firme"

        res_html = client.get("/")
        assert res_html.status_code == 200
        assert "FocusGuard" in res_html.text

        res_audio = client.get("/api/audio/devices")
        assert res_audio.status_code == 200
        audio_payload = res_audio.json()
        assert "output_devices" in audio_payload
        assert "input_devices" in audio_payload

        res_preview = client.get("/api/vision/preview.jpg")
        assert res_preview.status_code in (200, 204)
