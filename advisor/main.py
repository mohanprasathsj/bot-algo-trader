#!/usr/bin/env python3
"""
Investment Advisor — main entry point.

Usage:
    python main.py                         # analyse all sector ETFs
    python main.py XLK XLF NVDA AAPL      # analyse specific tickers
    python main.py --output recs.json      # save results to JSON

Environment:
    Copy .env.example → .env and fill in your API keys, then:
        pip install python-dotenv
    The script auto-loads .env if present.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Auto-load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from investment_advisor import MarketDataAggregator, InvestmentAdvisor

# ---------------------------------------------------------------------------
# Sector ETF universe (11 SPDR Select Sector ETFs + broad market)
# ---------------------------------------------------------------------------
SECTOR_ETFS = [
    "SPY",   # S&P 500 (broad market)
    "XLK",   # Technology
    "XLF",   # Financials
    "XLV",   # Health Care
    "XLE",   # Energy
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

SIGNAL_COLORS = {
    "BUY":  "\033[92m",   # green
    "SELL": "\033[91m",   # red
    "HOLD": "\033[93m",   # yellow
}
RESET = "\033[0m"


def print_recommendation(rec) -> None:
    color = SIGNAL_COLORS.get(rec.signal, "")
    indicator = {"BUY": "▲", "SELL": "▼", "HOLD": "●"}.get(rec.signal, "?")
    print(
        f"\n{color}{indicator} {rec.ticker:<6} {rec.signal:<4}  [{rec.confidence}]{RESET}"
    )
    print(f"   {rec.reasoning}")
    print(f"   \033[2mRisk: {rec.key_risks}\033[0m")


def print_summary_table(recs) -> None:
    print("\n" + "=" * 60)
    print(f"{'TICKER':<8} {'SIGNAL':<6} {'CONF':<8} REASONING (truncated)")
    print("=" * 60)
    for r in sorted(recs, key=lambda x: ["BUY", "HOLD", "SELL"].index(x.signal)):
        snippet = r.reasoning[:55] + "…" if len(r.reasoning) > 55 else r.reasoning
        color = SIGNAL_COLORS.get(r.signal, "")
        print(f"{color}{r.ticker:<8} {r.signal:<6}{RESET} {r.confidence:<8} {snippet}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Investment Advisor")
    parser.add_argument(
        "tickers",
        nargs="*",
        default=[],
        help="Tickers to analyse (default: all 12 sector ETFs)",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Write recommendations to a JSON file",
    )
    parser.add_argument(
        "--news-days",
        type=int,
        default=7,
        metavar="N",
        help="Days of news to pull from Finnhub (default: 7)",
    )
    parser.add_argument(
        "--price-days",
        type=int,
        default=65,
        metavar="N",
        help="Days of price history from Polygon (default: 65)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Gemini model to use (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tickers = [t.upper() for t in args.tickers] if args.tickers else SECTOR_ETFS

    print(f"\n{'='*60}")
    print(f"  Investment Advisor — {len(tickers)} ticker(s)")
    print(f"  Model: {args.model}")
    print(f"{'='*60}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"{'='*60}\n")

    # --- Collect data ---
    aggregator = MarketDataAggregator()
    logger.info("Collecting market data for %d tickers…", len(tickers))
    bundle = aggregator.collect_all(tickers)

    # --- Get Gemini recommendations ---
    advisor = InvestmentAdvisor(model=args.model)
    logger.info("Requesting recommendations from Gemini…")
    recs = advisor.recommend(bundle)

    # --- Display ---
    for rec in recs:
        print_recommendation(rec)

    print_summary_table(recs)

    # --- Optional JSON output ---
    if args.output:
        out_path = Path(args.output)
        payload = [
            {
                "ticker": r.ticker,
                "signal": r.signal,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "key_risks": r.key_risks,
            }
            for r in recs
        ]
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\n✓ Recommendations saved to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
