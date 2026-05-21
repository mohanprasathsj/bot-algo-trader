"""
Finnhub client — fetches company news and sentiment scores.

Docs: https://finnhub.io/docs/api
Free tier: 60 API calls/min, news + basic sentiment available.
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"


class FinnhubClient:
    """Thin wrapper around Finnhub REST API."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ["FINNHUB_API_KEY"]
        self.session = requests.Session()
        self.session.params = {"token": self.api_key}  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{FINNHUB_BASE}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_company_news(
        self,
        symbol: str,
        days_back: int = 7,
    ) -> list[dict[str, Any]]:
        """Return recent news articles for *symbol*.

        Args:
            symbol:    Ticker (e.g. "AAPL", "XLK").
            days_back: How many calendar days to look back (default 7).

        Returns:
            List of article dicts with keys: headline, summary, source,
            datetime (unix ts), url, sentiment (if available).
        """
        today = datetime.utcnow().date()
        from_date = (today - timedelta(days=days_back)).isoformat()
        to_date = today.isoformat()

        raw = self._get(
            "company-news",
            params={"symbol": symbol, "from": from_date, "to": to_date},
        )
        # Finnhub returns a list; guard against empty / error responses
        if not isinstance(raw, list):
            logger.warning("Unexpected Finnhub response for %s: %s", symbol, raw)
            return []

        articles = []
        for item in raw[:20]:  # cap at 20 most-recent per ticker
            articles.append(
                {
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "published_at": datetime.utcfromtimestamp(
                        item.get("datetime", 0)
                    ).strftime("%Y-%m-%d %H:%M UTC"),
                    "url": item.get("url", ""),
                }
            )
        return articles

    def get_sentiment(self, symbol: str) -> dict[str, Any]:
        """Return aggregated news sentiment for *symbol*.

        Returns a dict with:
            buzz        – relative news volume vs. historical average
            sentiment   – bullish/bearish score (-1 to +1)
            articles_in_week – raw article count
        """
        raw = self._get("news-sentiment", params={"symbol": symbol})

        if not isinstance(raw, dict) or "sentiment" not in raw:
            logger.warning("No sentiment data for %s", symbol)
            return {
                "buzz": None,
                "sentiment_score": None,
                "articles_in_week": None,
                "bearish_pct": None,
                "bullish_pct": None,
            }

        sentiment = raw.get("sentiment", {})
        buzz = raw.get("buzz", {})
        return {
            "buzz": buzz.get("buzz"),
            "articles_in_week": buzz.get("articlesInLastWeek"),
            "weekly_average": buzz.get("weeklyAverage"),
            "sentiment_score": sentiment.get("score"),  # -1 (bearish) to +1 (bullish)
            "bearish_pct": sentiment.get("bearishPercent"),
            "bullish_pct": sentiment.get("bullishPercent"),
        }

    def get_market_news(self, category: str = "general") -> list[dict[str, Any]]:
        """Return broad market news headlines.

        category: "general" | "forex" | "crypto" | "merger"
        """
        raw = self._get("news", params={"category": category})
        if not isinstance(raw, list):
            return []
        headlines = []
        for item in raw[:10]:
            headlines.append(
                {
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", "")[:300],
                    "source": item.get("source", ""),
                    "published_at": datetime.utcfromtimestamp(
                        item.get("datetime", 0)
                    ).strftime("%Y-%m-%d %H:%M UTC"),
                }
            )
        return headlines

    def collect_ticker_context(
        self,
        symbol: str,
        days_back: int = 7,
        rate_limit_sleep: float = 1.0,
    ) -> dict[str, Any]:
        """Convenience method: sentiment + recent headlines for one ticker."""
        sentiment = self.get_sentiment(symbol)
        time.sleep(rate_limit_sleep)
        news = self.get_company_news(symbol, days_back=days_back)
        return {
            "symbol": symbol,
            "sentiment": sentiment,
            "recent_news": news,
        }
