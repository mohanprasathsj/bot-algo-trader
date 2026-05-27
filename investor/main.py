from __future__ import annotations

"""
Robo Investor — Small Account Edition ($5,000 or under)
========================================================
Two-bucket approach (hourly bars):

  TACTICAL     (RSI Mean Reversion, 10% SL / 30% TP)
  EXPERIMENTAL (MACD Momentum + Volume, 2% SL / 5% TP)

Usage (run from project root):
    python -m investor.main --mode backtest
    python -m investor.main --mode paper
    python -m investor.main --mode live          # dry-run hourly loop (no real orders)
    python -m investor.main --mode live-ibkr     # real orders via IB Gateway (paper by default)
    python -m investor.main --mode live-ibkr --ibkr-live  # REAL MONEY — use with caution
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from .config import (
    AccountConfig, BrokerageConfig, ExperimentalConfig,
    IBKRConfig, TacticalConfig, TaxConfig,
)
from .engine import run_backtest, run_live, run_live_ibkr, run_paper


def _setup_logging(log_dir: str | None, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(Path(log_dir) / "robo_investor.log")
        )
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=handlers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robo Investor — Small Account Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live", "live-ibkr"],
        default="backtest",
        help=(
            "backtest   = historical hourly replay\n"
            "paper      = recent 300-day replay (no orders) — use --lookback to override\n"
            "live       = hourly dry-run loop (logs signals, no orders)\n"
            "live-ibkr  = hourly loop with real IBKR order execution"
        ),
    )
    parser.add_argument("--capital",    type=float, default=2_000.0,      help="Total bot capital in USD")
    parser.add_argument("--tactical",   type=float, default=1.0,          help="Tactical bucket fraction (0–1)")
    parser.add_argument("--experimental", type=float, default=0.0,        help="Experimental bucket fraction (0–1)")
    parser.add_argument("--start",      default="2022-01-01",             help="Backtest start date")
    parser.add_argument("--end",        default="2023-01-01",             help="Backtest end date")
    parser.add_argument(
        "--lookback", type=int, default=300,
        help=(
            "Calendar days of history for --mode paper (default 300). "
            "Needs ≥ 300 days for the 52W-high and 200-day MA guards to work correctly. "
            "Smaller values speed up the run but degrade the falling-knife protection."
        ),
    )

    # IBKR connection
    parser.add_argument("--ibkr-host",  default=os.getenv("IBKR_HOST", "127.0.0.1"))
    parser.add_argument("--ibkr-port",  type=int, default=int(os.getenv("IBKR_PORT", "4002")))
    parser.add_argument("--ibkr-id",    type=int, default=int(os.getenv("IBKR_CLIENT_ID", "1")))
    parser.add_argument(
        "--ibkr-live",
        action="store_true",
        default=False,
        help="Connect to live IBKR account (real money). Default is paper trading.",
    )

    # Paths
    parser.add_argument("--state-file", default="investor/data/state.json", help="Path to state JSON file")
    parser.add_argument("--log-dir",    default="investor/logs",            help="Directory for log files")
    parser.add_argument("--verbose", "-v", action="store_true",             help="Debug-level logging")

    args = parser.parse_args()

    _setup_logging(args.log_dir if args.mode in ("live", "live-ibkr") else None, args.verbose)

    account_cfg      = AccountConfig(
        total_capital=args.capital,
        tactical_pct=args.tactical,
        experimental_pct=args.experimental,
    )
    tactical_cfg     = TacticalConfig()
    experimental_cfg = ExperimentalConfig()
    brokerage_cfg    = BrokerageConfig()
    tax_cfg          = TaxConfig()

    if args.mode == "backtest":
        run_backtest(account_cfg, tactical_cfg, experimental_cfg, brokerage_cfg, tax_cfg,
                     start=args.start, end=args.end)

    elif args.mode == "paper":
        run_paper(account_cfg, tactical_cfg, experimental_cfg, brokerage_cfg, tax_cfg,
                  lookback_days=args.lookback)

    elif args.mode == "live":
        run_live(account_cfg, tactical_cfg, experimental_cfg, brokerage_cfg, tax_cfg)

    elif args.mode == "live-ibkr":
        if args.ibkr_live:
            confirm = input(
                "\n⚠  You are about to trade with REAL MONEY via IBKR.\n"
                "   Type 'yes' to confirm: "
            ).strip().lower()
            if confirm != "yes":
                print("Aborted.")
                sys.exit(0)

        ibkr_cfg = IBKRConfig(
            host=args.ibkr_host,
            port=args.ibkr_port if not args.ibkr_live else int(os.getenv("IBKR_LIVE_PORT", "4001")),
            client_id=args.ibkr_id,
            paper_trading=not args.ibkr_live,
        )
        run_live_ibkr(
            account_cfg, tactical_cfg, experimental_cfg,
            ibkr_cfg, brokerage_cfg, tax_cfg,
            state_path=args.state_file,
            log_dir=args.log_dir,
        )
