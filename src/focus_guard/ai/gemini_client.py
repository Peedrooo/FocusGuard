"""Async-safe Gemini client wrapper with model fallback chain and BudgetGuard enforcement."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from focus_guard.core.budget import BudgetGuard
from focus_guard.core.config import AppConfig
from focus_guard.core.ports import StoragePort

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class BudgetExceededError(RuntimeError):
    """Raised when BudgetGuard throttles further Gemini API calls."""


class GeminiNotConfiguredError(RuntimeError):
    """Raised when a Gemini call is attempted without a configured API key or client."""


def extract_response_text(response: Any) -> str:
    """Safely extract concatenated non-thought text from a Gemini response or test double."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response

    try:
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text
    except Exception:
        pass

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        chunks: list[str] = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and not getattr(part, "thought", False):
                chunks.append(part_text)
        if chunks:
            return "".join(chunks)

    return ""


def parse_json_object(raw_text: str) -> dict[str, Any] | None:
    """Parse a JSON object from LLM text, stripping markdown fences or surrounding prose."""
    text = (raw_text or "").strip()
    if not text:
        return None

    candidates: list[str] = [text]
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        candidates.insert(0, fence_match.group(1).strip())

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx > start_idx:
        candidates.append(text[start_idx : end_idx + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


class GeminiClient:
    """Thread-offloaded Gemini wrapper supporting fallback models and BudgetGuard."""

    def __init__(
        self,
        config: AppConfig,
        budget: BudgetGuard,
        storage: StoragePort | None = None,
        *,
        client: Any | None = None,
        _raw_generate: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        self._budget = budget
        self._storage = storage
        self._api_key = (config.gemini_api_key or "").strip()
        self._raw_generate = _raw_generate
        self._client: Any | None = client

        if self._client is None and self._api_key:
            self._client = self._build_sdk_client(self._api_key)

    @property
    def budget(self) -> BudgetGuard:
        return self._budget

    @staticmethod
    def _build_sdk_client(api_key: str) -> Any | None:
        try:
            return genai.Client(api_key=api_key)
        except Exception:
            logger.exception("Failed to instantiate google.genai.Client.")
            return None

    def is_configured(self) -> bool:
        return bool(self._api_key or self._client is not None or self._raw_generate is not None)

    def reconfigure_api_key(self, api_key: str) -> None:
        clean_key = (api_key or "").strip()
        self._api_key = clean_key
        self._config.gemini_api_key = clean_key
        if not clean_key:
            self._client = None
            self._raw_generate = None
            return
        self._client = self._build_sdk_client(clean_key)

    def _model_chain(self) -> list[str]:
        ordered = [self._config.ai.primary_model, *self._config.ai.fallback_models]
        return list(dict.fromkeys(m.strip() for m in ordered if m and m.strip()))

    async def generate_content(
        self,
        contents: Any,
        system_instruction: str | None = None,
        response_mime_type: str | None = None,
        tools: list[Any] | None = None,
        temperature: float = 0.3,
    ) -> tuple[Any, str, int, int, float]:
        """Call Gemini across configured model fallback chain without blocking the event loop."""
        if not self._budget.can_spend():
            raise BudgetExceededError("BudgetGuard throttled Gemini API usage.")

        if not self.is_configured():
            raise GeminiNotConfiguredError("Gemini API key is not configured.")

        models = self._model_chain()
        if not models:
            raise RuntimeError("No Gemini models configured.")

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if system_instruction is not None:
            config_kwargs["system_instruction"] = system_instruction
        if response_mime_type is not None:
            config_kwargs["response_mime_type"] = response_mime_type
        if tools:
            config_kwargs["tools"] = tools

        gen_config = types.GenerateContentConfig(**config_kwargs)

        last_exc: Exception | None = None
        for model_name in models:
            try:
                if self._raw_generate is not None:
                    result = self._raw_generate(
                        model=model_name,
                        contents=contents,
                        config=gen_config,
                    )
                    response = await result if inspect.isawaitable(result) else result
                else:
                    if self._client is None:
                        raise GeminiNotConfiguredError("Gemini SDK client is unavailable.")
                    response = await asyncio.to_thread(
                        self._client.models.generate_content,
                        model=model_name,
                        contents=contents,
                        config=gen_config,
                    )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Gemini model %s failed (%s); falling back if another model is available.",
                    model_name,
                    exc,
                )
                continue

            usage = getattr(response, "usage_metadata", None)
            tokens_in = int(getattr(usage, "prompt_token_count", None) or 0)
            tokens_out = int(getattr(usage, "candidates_token_count", None) or 0)
            cached_tokens = int(getattr(usage, "cached_content_token_count", None) or 0)

            cost_usd = self._budget.record_usage(
                model=model_name,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cached_tokens=cached_tokens,
            )

            if self._storage is not None:
                try:
                    await self._storage.save_usage(model_name, tokens_in, tokens_out, cost_usd)
                except Exception:
                    logger.exception("Failed to save Gemini usage telemetry to storage.")

            return response, model_name, tokens_in, tokens_out, cost_usd

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("All Gemini fallback models failed.")
