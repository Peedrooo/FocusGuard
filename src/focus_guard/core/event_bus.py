"""Typed async in-memory EventBus for loose coupling between modules."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
EventHandler = Callable[[Any], Awaitable[None] | None]


class EventBus:
    """Async pub/sub bus with fault isolation so one bad handler never breaks others."""

    def __init__(self) -> None:
        self._subscribers: dict[type[Any], list[EventHandler]] = defaultdict(list)
        self._wildcard_subscribers: list[EventHandler] = []

    def subscribe(self, event_type: type[T], handler: Callable[[T], Awaitable[None] | None]) -> None:
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        if handler not in self._wildcard_subscribers:
            self._wildcard_subscribers.append(handler)

    def unsubscribe(self, event_type: type[T], handler: Callable[[T], Awaitable[None] | None]) -> None:
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    def unsubscribe_all(self, handler: EventHandler) -> None:
        if handler in self._wildcard_subscribers:
            self._wildcard_subscribers.remove(handler)

    async def publish(self, event: Any) -> None:
        handlers: list[EventHandler] = list(self._subscribers.get(type(event), []))
        handlers.extend(self._wildcard_subscribers)
        if not handlers:
            return

        async def _invoke(fn: EventHandler) -> None:
            try:
                res = fn(event)
                if inspect.isawaitable(res):
                    await res
            except Exception:
                logger.exception("EventBus handler %r failed for %s", fn, type(event).__name__)

        await asyncio.gather(*(_invoke(h) for h in handlers))
