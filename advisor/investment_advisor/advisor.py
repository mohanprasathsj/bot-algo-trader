"""
Gemini advisor — builds a structured prompt from aggregated market data
and calls the Google Gemini API to produce Buy / Hold / Sell recommendations.

Model: gemini-2.0-flash (fast, cost-effective, great at analysis).
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a rigorous equity research analyst.  You receive structured market data
    for a set of tickers — price momentum, news sentiment, and macroeconomic context —
    and produce concise investment recommendations.

    For each ticker you MUST output a JSON object with exactly these keys:
      - "ticker":        string
      - "signal":        one of "BUY", "HOLD", or "SELL"
      - "confidence":    "HIGH", "MEDIUM", or "LOW"
      - "reasoning":     2-4 sentences explaining the signal, citing specific data points
      - "key_risks":     one sentence describing the primary risk to your thesis

    Return a JSON array of these objects, one per ticker.  No markdown, no prose outside
    the JSON.  If data is missing for a ticker, still output an entry with signal "HOLD"
    and explain the data gap in reasoning.

    Important: this is for educational / personal research purposes only.
    Do not provide advice that implies certainty about future prices.
""")


def _format_macro(macro: dict[str, Any]) -> str:
    """Convert macro snapshot dict into a readable block for the prompt."""
    lines = ["MACRO CONTEXT (FRED):"]
    for key, val in macro.items():
        if isinstance(val, dict):
            value = val.get("value", "N/A")
            date = val.get("date", "")
            interp = val.get("interpretation", "")
            line = f"  {key}: {value}"
            if date:
                line += f" (as of {date})"
            if interp:
                line += f" — {interp}"
            lines.append(line)
    return "\n".join(lines)


def _format_ticker(symbol: str, data: dict[str, Any]) -> str:
    """Format one ticker's context block for the prompt."""
    parts = [f"--- {symbol} ---"]

    price = data.get("price", {})
    if price:
        parts.append("Price signals:")
        parts.append(f"  Latest close:       {price.get('latest_close')} ({price.get('latest_date')})")
        parts.append(f"  SMA-20 / SMA-50:    {price.get('sma_20')} / {price.get('sma_50')}")
        parts.append(f"  Price vs SMA-20:    {price.get('price_vs_sma20_pct')}%")
        parts.append(f"  5d / 20d / 60d ret: {price.get('return_5d_pct')}% / {price.get('return_20d_pct')}% / {price.get('return_60d_pct')}%")
        parts.append(f"  Volume ratio (1d/20d avg): {price.get('volume_ratio')}")

    sentiment = data.get("sentiment", {})
    if sentiment:
        parts.append("Sentiment (Finnhub):")
        parts.append(f"  Score (-1 bearish → +1 bullish): {sentiment.get('sentiment_score')}")
        parts.append(f"  Bullish/Bearish %: {sentiment.get('bullish_pct')} / {sentiment.get('bearish_pct')}")
        parts.append(f"  Buzz (articles this week): {sentiment.get('articles_in_week')}")

    news = data.get("recent_news", [])
    if news:
        parts.append(f"Recent headlines (last {min(len(news), 5)}):")
        for article in news[:5]:
            parts.append(f"  [{article.get('published_at', '')}] {article.get('headline', '')}")

    return "\n".join(parts)


def _build_user_message(bundle: dict[str, Any]) -> str:
    """Assemble the full user message from a collected bundle."""
    sections: list[str] = []

    macro = bundle.get("macro", {})
    if macro:
        sections.append(_format_macro(macro))

    tickers = bundle.get("tickers", {})
    if tickers:
        sections.append("\nTICKER DATA:")
        for symbol, data in tickers.items():
            sections.append(_format_ticker(symbol, data))

    sections.append(
        "\nPlease analyse each ticker and return a JSON array of recommendations."
    )
    return "\n".join(sections)


@dataclass
class Recommendation:
    ticker: str
    signal: str          # BUY | HOLD | SELL
    confidence: str      # HIGH | MEDIUM | LOW
    reasoning: str
    key_risks: str
    raw: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        indicator = {"BUY": "▲", "SELL": "▼", "HOLD": "●"}.get(self.signal, "?")
        return (
            f"{indicator} {self.ticker} — {self.signal} ({self.confidence})\n"
            f"   {self.reasoning}\n"
            f"   Risk: {self.key_risks}"
        )


class InvestmentAdvisor:
    """Wraps the Gemini client and produces structured recommendations."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ) -> None:
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model_name = model

    # ------------------------------------------------------------------
    # Core recommendation method
    # ------------------------------------------------------------------

    def recommend(self, bundle: dict[str, Any]) -> list[Recommendation]:
        """Given an aggregated market bundle, return a list of Recommendations.

        Args:
            bundle: Output of MarketDataAggregator.collect_all()

        Returns:
            List of Recommendation dataclass instances.
        """
        user_message = _build_user_message(bundle)
        logger.info("Sending %d chars to Gemini…", len(user_message))

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=2048,
            ),
        )

        raw_text = response.text.strip()
        logger.debug("Gemini raw response: %s", raw_text[:500])

        # Parse JSON array
        try:
            # Gemini may occasionally wrap in markdown fences — strip them
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            recs_json: list[dict] = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini response as JSON: %s", exc)
            logger.error("Raw response was: %s", raw_text)
            raise

        recommendations = []
        for item in recs_json:
            recommendations.append(
                Recommendation(
                    ticker=item.get("ticker", "?"),
                    signal=item.get("signal", "HOLD").upper(),
                    confidence=item.get("confidence", "LOW").upper(),
                    reasoning=item.get("reasoning", ""),
                    key_risks=item.get("key_risks", ""),
                    raw=item,
                )
            )
        return recommendations

    def recommend_single(
        self, symbol: str, bundle: dict[str, Any]
    ) -> Recommendation | None:
        """Convenience: get recommendation for one ticker from a full bundle."""
        recs = self.recommend(bundle)
        for r in recs:
            if r.ticker.upper() == symbol.upper():
                return r
        return None
