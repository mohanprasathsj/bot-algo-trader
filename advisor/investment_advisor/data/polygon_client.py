"""
Polygon.io client — fetches OHLCV price data and computes momentum signals.

Docs: https://polygon.io/docs
Free tier: unlimited calls (delayed 15 min), daily aggregate bars available.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"


class PolygonClient:
    """Thin wrapper around Polygon.io REST API v2/v3."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ["POLYGON_API_KEY"]
        self.session = requests.Session()
        self.session.params = {"apiKey": self.api_key}  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{POLYGON_BASE}{path}"
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    def get_daily_bars(
        self,
        symbol: str,
        days_back: int = 60,
    ) -> list[dict[str, Any]]:
        """Return daily OHLCV bars for the last *days_back* calendar days.

        Each bar: { date, open, high, low, close, volume, vwap }
        """
        today = datetime.utcnow().date()
        from_date = (today - timedelta(days=days_back)).isoformat()
        to_date = today.isoformat()

        path = f"/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"
        raw = self._get(path, params={"adjusted": "true", "sort": "asc", "limit": 120})

        results = raw.get("results", [])
        bars = []
        for r in results:
            ts = r.get("t", 0) / 1000  # Polygon uses ms timestamps
            bars.append(
                {
                    "date": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
                    "open": r.get("o"),
                    "high": r.get("h"),
                    "low": r.get("l"),
                    "close": r.get("c"),
                    "volume": r.get("v"),
                    "vwap": r.get("vw"),
                }
            )
        return bars

    # ------------------------------------------------------------------
    # Momentum & technical signals
    # ------------------------------------------------------------------

    @staticmethod
    def _sma(closes: list[float], window: int) -> float | None:
        """Simple moving average over the last *window* closes."""
        if len(closes) < window:
            return None
        return sum(closes[-window:]) / window

    @staticmethod
    def _pct_change(closes: list[float], lookback: int) -> float | None:
        """% price change over *lookback* periods."""
        if len(closes) <= lookback:
            return None
        old = closes[-(lookback + 1)]
        new = closes[-1]
        if old == 0:
            return None
        return round((new - old) / old * 100, 2)

    @staticmethod
    def _avg_volume(volumes: list[float], window: int = 20) -> float | None:
        if len(volumes) < window:
            return None
        return sum(volumes[-window:]) / window

    def compute_signals(self, bars: list[dict[str, Any]]) -> dict[str, Any]:
        """Derive momentum signals from a list of OHLCV bars.

        Returns:
            latest_close, sma_20, sma_50, price_vs_sma20_pct,
            return_1w, return_1m, return_3m,
            avg_volume_20d, latest_volume, volume_ratio
        """
        if not bars:
            return {}

        closes = [b["close"] for b in bars if b["close"] is not None]
        volumes = [b["volume"] for b in bars if b["volume"] is not None]

        if not closes:
            return {}

        latest_close = closes[-1]
        sma_20 = PolygonClient._sma(closes, 20)
        sma_50 = PolygonClient._sma(closes, 50)
        avg_vol = PolygonClient._avg_volume(volumes, 20)
        latest_vol = volumes[-1] if volumes else None

        price_vs_sma20 = (
            round((latest_close - sma_20) / sma_20 * 100, 2) if sma_20 else None
        )

        return {
            "latest_close": latest_close,
            "latest_date": bars[-1]["date"],
            "sma_20": round(sma_20, 2) if sma_20 else None,
            "sma_50": round(sma_50, 2) if sma_50 else None,
            "price_vs_sma20_pct": price_vs_sma20,
            "return_5d_pct": PolygonClient._pct_change(closes, 5),
            "return_20d_pct": PolygonClient._pct_change(closes, 20),
            "return_60d_pct": PolygonClient._pct_change(closes, 60),
            "avg_volume_20d": round(avg_vol) if avg_vol else None,
            "latest_volume": latest_vol,
            "volume_ratio": (
                round(latest_vol / avg_vol, 2)
                if (avg_vol and latest_vol and avg_vol > 0)
                else None
            ),
        }

    def collect_ticker_context(
        self,
        symbol: str,
        days_back: int = 65,
    ) -> dict[str, Any]:
        """Convenience: fetch bars and compute signals for one ticker."""
        bars = self.get_daily_bars(symbol, days_back=days_back)
        signals = self.compute_signals(bars)
        return {
            "symbol": symbol,
            "price_signals": signals,
            # keep last 5 bars for the prompt (avoid blowing context)
            "recent_bars": bars[-5:] if bars else [],
        }
