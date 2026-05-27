"""Technical indicators — pure numpy, no TA-Lib dependency."""

from __future__ import annotations

import numpy as np


def compute_rsi(prices: list[float], period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def compute_sma(prices: list[float], period: int) -> float | None:
    """Simple moving average over the last `period` bars. Returns None if insufficient data."""
    if len(prices) < period:
        return None
    return float(np.mean(prices[-period:]))


def compute_52w_high(prices: list[float], bars_per_year: int = 1638) -> float | None:
    """52-week high using hourly bars (52 wks × 5 days × 6.3 h/day ≈ 1638 bars).

    Falls back to all available data if fewer than `bars_per_year` bars exist, so
    a fresh warmup still produces a useful (conservative) estimate.
    """
    if not prices:
        return None
    window = prices[-bars_per_year:] if len(prices) >= bars_per_year else prices
    return float(max(window))


def detect_rsi_divergence(
    prices: list[float],
    rsi_period: int = 14,
    lookback: int = 10,
) -> bool:
    """Bullish RSI divergence: price makes a lower low while RSI makes a higher low.

    This signals that downward momentum is *slowing* — a prerequisite for a real
    bounce rather than a continued slide.  Returns False when there is not enough
    data to compute the check (caller should treat this as "skip the guard").
    """
    min_len = rsi_period + 1 + lookback
    if len(prices) < min_len:
        return False
    rsi_now   = compute_rsi(prices,            rsi_period)
    rsi_prior = compute_rsi(prices[:-lookback], rsi_period)
    if rsi_now is None or rsi_prior is None:
        return False
    price_now   = prices[-1]
    price_prior = prices[-1 - lookback]
    # Price lower + RSI higher → bullish divergence
    return price_now < price_prior and rsi_now > rsi_prior


def compute_ema(prices: list[float], period: int) -> float | None:
    if len(prices) < period:
        return None
    k = 2 / (period + 1)
    ema = prices[-period]
    for p in prices[-period + 1:]:
        ema = p * k + ema * (1 - k)
    return ema


def compute_macd(
    prices: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float | None, float | None, float | None]:
    """Returns (macd_line, signal_line, histogram). All None if insufficient data."""
    if len(prices) < slow + signal:
        return None, None, None
    ema_fast = compute_ema(prices, fast)
    ema_slow = compute_ema(prices, slow)
    if ema_fast is None or ema_slow is None:
        return None, None, None
    macd_line = ema_fast - ema_slow

    macd_history = []
    for i in range(signal + 5):
        idx = len(prices) - (signal + 5 - i)
        if idx < slow:
            continue
        ef = compute_ema(prices[: idx + 1], fast)
        es = compute_ema(prices[: idx + 1], slow)
        if ef and es:
            macd_history.append(ef - es)

    if len(macd_history) < signal:
        return macd_line, None, None

    signal_line = compute_ema(macd_history, signal)
    if signal_line is None:
        return macd_line, None, None

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
