"""
Data aggregator — collects context from all three sources into a
unified dict per ticker, ready to be handed to the Claude advisor.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .data.finnhub_client import FinnhubClient
from .data.polygon_client import PolygonClient
from .data.fred_client import FredClient

logger = logging.getLogger(__name__)


class MarketDataAggregator:
    """Orchestrates Finnhub + Polygon + FRED into per-ticker context bundles."""

    def __init__(
        self,
        finnhub_client: FinnhubClient | None = None,
        polygon_client: PolygonClient | None = None,
        fred_client: FredClient | None = None,
    ) -> None:
        self.finnhub = finnhub_client or FinnhubClient()
        self.polygon = polygon_client or PolygonClient()
        self.fred = fred_client or FredClient()

    # ------------------------------------------------------------------
    # Macro context (fetched once, shared across all tickers)
    # ------------------------------------------------------------------

    def get_macro_context(self) -> dict[str, Any]:
        """Pull FRED macro snapshot (one API round-trip)."""
        logger.info("Fetching macro context from FRED…")
        try:
            return self.fred.get_macro_snapshot()
        except Exception as exc:
            logger.error("FRED fetch failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Per-ticker context
    # ------------------------------------------------------------------

    def get_ticker_context(
        self,
        symbol: str,
        news_days_back: int = 7,
        price_days_back: int = 65,
        rate_limit_sleep: float = 1.2,
    ) -> dict[str, Any]:
        """Fetch and merge Finnhub + Polygon data for a single ticker."""
        logger.info("Collecting context for %s…", symbol)

        # --- Polygon price data ---
        polygon_ctx: dict[str, Any] = {}
        try:
            polygon_ctx = self.polygon.collect_ticker_context(
                symbol, days_back=price_days_back
            )
        except Exception as exc:
            logger.warning("Polygon failed for %s: %s", symbol, exc)

        # --- Finnhub news + sentiment ---
        finnhub_ctx: dict[str, Any] = {}
        try:
            finnhub_ctx = self.finnhub.collect_ticker_context(
                symbol,
                days_back=news_days_back,
                rate_limit_sleep=rate_limit_sleep,
            )
        except Exception as exc:
            logger.warning("Finnhub failed for %s: %s", symbol, exc)

        return {
            "symbol": symbol,
            "price": polygon_ctx.get("price_signals", {}),
            "recent_bars": polygon_ctx.get("recent_bars", []),
            "sentiment": finnhub_ctx.get("sentiment", {}),
            "recent_news": finnhub_ctx.get("recent_news", []),
        }

    # ------------------------------------------------------------------
    # Batch collection
    # ------------------------------------------------------------------

    def collect_all(
        self,
        tickers: list[str],
        sleep_between: float = 1.5,
    ) -> dict[str, Any]:
        """Return a full context bundle for a list of tickers.

        Shape:
            {
                "macro": { ... },             # FRED snapshot
                "tickers": {
                    "XLK": { price, sentiment, news, ... },
                    ...
                }
            }
        """
        macro = self.get_macro_context()

        ticker_data: dict[str, Any] = {}
        for i, symbol in enumerate(tickers):
            ticker_data[symbol] = self.get_ticker_context(symbol)
            if i < len(tickers) - 1:
                time.sleep(sleep_between)

        return {
            "macro": macro,
            "tickers": ticker_data,
        }
