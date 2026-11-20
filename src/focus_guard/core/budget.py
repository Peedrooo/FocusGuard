"""BudgetGuard — Hard financial circuit breaker for Gemini API usage."""

from __future__ import annotations

from datetime import UTC, datetime
from threading import Lock

from focus_guard.core.config import AIConfig
from focus_guard.core.events import BudgetStatus

# Official approximate rates in USD per 1,000,000 tokens (input, output, cached_input)
MODEL_PRICING_PER_1M: dict[str, tuple[float, float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03),
    "gemini-3.1-flash-lite": (0.25, 1.50, 0.025),
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01),
    "gemini-2.5-flash": (0.30, 2.50, 0.03),
    "gemini-2.0-flash": (0.10, 0.40, 0.025),
}
DEFAULT_PRICING = (0.30, 2.50, 0.03)


def estimate_cost_usd(
    model: str,
    tokens_in: int,
    tokens_out: int,
    cached_tokens: int = 0,
) -> float:
    in_rate, out_rate, cache_rate = MODEL_PRICING_PER_1M.get(model.lower(), DEFAULT_PRICING)
    non_cached_in = max(0, tokens_in - cached_tokens)
    cost = (
        (non_cached_in / 1_000_000.0) * in_rate
        + (cached_tokens / 1_000_000.0) * cache_rate
        + (max(0, tokens_out) / 1_000_000.0) * out_rate
    )
    return round(cost, 8)


class BudgetGuard:
    """Tracks daily/monthly API spend and enforces hard caps."""

    def __init__(self, ai_config: AIConfig) -> None:
        self._lock = Lock()
        self._max_daily_usd = ai_config.max_daily_usd
        self._max_monthly_usd = ai_config.max_monthly_usd
        self._brl_rate = ai_config.brl_exchange_rate

        now = datetime.now(UTC)
        self._current_day = now.strftime("%Y-%m-%d")
        self._current_month = now.strftime("%Y-%m")
        self._daily_usd = 0.0
        self._monthly_usd = 0.0
        self._daily_tokens_in = 0
        self._daily_tokens_out = 0
        self._daily_calls = 0

    def _rollover_if_needed(self, now: datetime | None = None) -> None:
        ref = now or datetime.now(UTC)
        day_str = ref.strftime("%Y-%m-%d")
        month_str = ref.strftime("%Y-%m")
        if month_str != self._current_month:
            self._current_month = month_str
            self._monthly_usd = 0.0
        if day_str != self._current_day:
            self._current_day = day_str
            self._daily_usd = 0.0
            self._daily_tokens_in = 0
            self._daily_tokens_out = 0
            self._daily_calls = 0

    def hydrate(
        self,
        daily_usd: float,
        monthly_usd: float,
        daily_tokens_in: int = 0,
        daily_tokens_out: int = 0,
        daily_calls: int = 0,
    ) -> None:
        with self._lock:
            self._rollover_if_needed()
            self._daily_usd = max(0.0, daily_usd)
            self._monthly_usd = max(0.0, monthly_usd)
            self._daily_tokens_in = max(0, daily_tokens_in)
            self._daily_tokens_out = max(0, daily_tokens_out)
            self._daily_calls = max(0, daily_calls)

    def can_spend(self, estimated_next_usd: float = 0.0) -> bool:
        with self._lock:
            self._rollover_if_needed()
            if (self._daily_usd + estimated_next_usd) > self._max_daily_usd:
                return False
            return (self._monthly_usd + estimated_next_usd) <= self._max_monthly_usd

    def record_usage(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cached_tokens: int = 0,
    ) -> float:
        cost = estimate_cost_usd(model, tokens_in, tokens_out, cached_tokens)
        with self._lock:
            self._rollover_if_needed()
            self._daily_usd += cost
            self._monthly_usd += cost
            self._daily_tokens_in += max(0, tokens_in)
            self._daily_tokens_out += max(0, tokens_out)
            self._daily_calls += 1
        return cost

    def update_limits(
        self,
        max_daily_usd: float | None = None,
        max_monthly_usd: float | None = None,
    ) -> None:
        with self._lock:
            if max_daily_usd is not None and max_daily_usd > 0:
                self._max_daily_usd = max_daily_usd
            if max_monthly_usd is not None and max_monthly_usd > 0:
                self._max_monthly_usd = max_monthly_usd

    def get_status(self) -> BudgetStatus:
        with self._lock:
            self._rollover_if_needed()
            throttled = (
                self._daily_usd >= self._max_daily_usd
                or self._monthly_usd >= self._max_monthly_usd
            )
            warning = (
                self._daily_usd >= (self._max_daily_usd * 0.8)
                or self._monthly_usd >= (self._max_monthly_usd * 0.8)
            )
            return BudgetStatus(
                daily_usd=round(self._daily_usd, 6),
                monthly_usd=round(self._monthly_usd, 6),
                max_daily_usd=self._max_daily_usd,
                max_monthly_usd=self._max_monthly_usd,
                daily_tokens_in=self._daily_tokens_in,
                daily_tokens_out=self._daily_tokens_out,
                daily_calls=self._daily_calls,
                brl_exchange_rate=self._brl_rate,
                is_throttled=throttled,
                warning_reached=warning,
            )
