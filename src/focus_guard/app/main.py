"""CLI entrypoint and uvicorn lifecycle runner for FocusGuard."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
from collections.abc import Sequence

import uvicorn

from focus_guard.app.container import AppContainer
from focus_guard.core.config import load_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="focus-guard",
        description="FocusGuard — Monitor multimodal de foco para concurso público.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host do servidor local (padrão: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Porta do dashboard local (padrão: 8765).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Não abrir o navegador automaticamente ao iniciar.",
    )
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Desativar ícone na bandeja do sistema e hotkeys globais.",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Rodar sem captura de webcam (somente telemetria do SO + tela + voz).",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Iniciar em estado PAUSED em vez de iniciar sessão automaticamente.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nível de logging (padrão: INFO).",
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep noisy third-party loggers tidy
    for noisy in ("httpx", "httpcore", "uvicorn.access", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def run_app(args: argparse.Namespace) -> None:
    config = load_config()
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.no_browser:
        config.server.open_browser_on_start = False

    container = AppContainer(
        config=config,
        enable_tray=not args.no_tray,
        enable_ui=True,
        enable_camera=False if args.no_camera else None,
        auto_start_session=not args.paused,
    )

    uv_config = uvicorn.Config(
        app=container.api_app,
        host=config.server.host,
        port=config.server.port,
        log_level=args.log_level.lower(),
        access_log=False,
    )
    server = uvicorn.Server(uv_config)
    container.set_exit_callback(lambda: setattr(server, "should_exit", True))

    browser_task: asyncio.Task[None] | None = None

    async def _open_browser_soon() -> None:
        await asyncio.sleep(1.2)
        container.open_dashboard()

    await container.start()
    try:
        if config.server.open_browser_on_start:
            browser_task = asyncio.create_task(_open_browser_soon())
        await server.serve()
    finally:
        if browser_task is not None and not browser_task.done():
            browser_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await browser_task
        await container.stop()


def cli(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_logging(args.log_level)
    try:
        asyncio.run(run_app(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(cli())
