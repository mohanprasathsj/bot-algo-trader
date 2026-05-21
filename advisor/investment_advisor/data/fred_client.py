"""
FRED client — fetches macroeconomic indicator series from the Federal Reserve.

Docs: https://fred.stlouisfed.org/docs/api/fred/
Free API key: https://fredaccount.stlouisfed.org/apikeys
"""

from __future__ import annotations

import os
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred"

# Key macro series to pull.  Format: { label: series_id }
DEFAULT_SERIES: dict[str, str] = {
    "fed_funds_rate": "DFF",          # Effective Federal Funds Rate (daily)
    "cpi_yoy": "CPIAUCSL",            # CPI All Urban Consumers (monthly)
    "unemployment_rate": "UNRATE",    # Civilian Unemployment Rate (monthly)
    "treasury_10y": "GS10",           # 10-Year Treasury Constant Maturity (monthly)
    "treasury_2y": "GS2",             # 2-Year Treasury (monthly)
    "vix": "VIXCLS",                  # CBOE Volatility Index (daily)
    "gdp_growth": "A191RL1Q225SBEA",  # Real GDP % Change QoQ (quarterly)
    "pce": "PCE",                     # Personal Consumption Expenditures (monthly)
    "breakeven_10y": "T10YIE",        # 10-Year Breakeven Inflation Rate (daily)
}


class FredClient:
    """Thin wrapper around FRED REST API."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ["FRED_API_KEY"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_series_latest(self, series_id: str, limit: int = 3) -> list[dict]:
        """Fetch the *limit* most-recent observations for a series."""
        url = f"{FRED_BASE}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("observations", [])
        # Filter out missing values (".")
        return [o for o in obs if o.get("value", ".") != "."]

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_latest_value(self, series_id: str) -> dict[str, Any] | None:
        """Return the single most-recent valid observation for *series_id*."""
        obs = self._get_series_latest(series_id, limit=3)
        if not obs:
            return None
        latest = obs[0]
        return {
            "series_id": series_id,
            "date": latest["date"],
            "value": float(latest["value"]),
        }

    def get_macro_snapshot(
        self,
        series: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Pull the latest reading for each series in *series* dict.

        Returns a flat dict: { label: { date, value } }
        Plus a derived 'yield_curve_spread' (10Y - 2Y).
        """
        series = series or DEFAULT_SERIES
        snapshot: dict[str, Any] = {}

        for label, sid in series.items():
            try:
                result = self.get_latest_value(sid)
                if result:
                    snapshot[label] = {
                        "value": result["value"],
                        "date": result["date"],
                    }
            except Exception as exc:
                logger.warning("FRED fetch failed for %s (%s): %s", label, sid, exc)
                snapshot[label] = None

        # Derived: yield-curve spread (recession signal)
        t10 = snapshot.get("treasury_10y")
        t2 = snapshot.get("treasury_2y")
        if t10 and t2:
            spread = round(t10["value"] - t2["value"], 3)
            snapshot["yield_curve_spread_10y2y"] = {
                "value": spread,
                "interpretation": (
                    "inverted (recession warning)" if spread < 0 else "normal"
                ),
            }

        return snapshot

    def summarize_macro(self) -> str:
        """Return a short human-readable macro summary string for prompts."""
        snap = self.get_macro_snapshot()
        lines = ["=== Macroeconomic Context ==="]

        def fmt(label: str, unit: str = "") -> str:
            entry = snap.get(label)
            if entry is None:
                return f"  {label}: N/A"
            return f"  {label}: {entry['value']}{unit} (as of {entry['date']})"

        lines.append(fmt("fed_funds_rate", "%"))
        lines.append(fmt("cpi_yoy", " index"))
        lines.append(fmt("unemployment_rate", "%"))
        lines.append(fmt("treasury_10y", "%"))
        lines.append(fmt("treasury_2y", "%"))
        lines.append(fmt("vix"))
        lines.append(fmt("gdp_growth", "%"))
        lines.append(fmt("breakeven_10y", "%"))

        yc = snap.get("yield_curve_spread_10y2y")
        if yc:
            lines.append(
                f"  yield_curve_spread (10Y-2Y): {yc['value']}% — {yc['interpretation']}"
            )

        return "\n".join(lines)
