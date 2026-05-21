"""
Robo Investor — Small Account Edition ($5,000 or under)
========================================================
Two-bucket approach (hourly bars):

  TACTICAL  (70%) — RSI Mean Reversion, 10% SL / 30% TP
  EXPERIMENTAL (30%) — MACD Momentum + Volume, 2% SL / 5% TP

Usage:
    python3 robo_invester.py --mode backtest
    python3 robo_invester.py --mode paper
    python3 robo_invester.py --mode live      # runs every hour during US market hours
"""

import argparse
import logging

from config import AccountConfig, BrokerageConfig, ExperimentalConfig, TacticalConfig, TaxConfig
from engine import run_backtest, run_live, run_paper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robo Investor — Small Account Edition")
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        default="backtest",
        help="backtest = historical hourly | paper = recent 60-day replay | live = hourly loop",
    )
    parser.add_argument("--capital", type=float, default=2_000.0, help="Total bot capital in USD")
    parser.add_argument("--start",   default="2022-01-01",         help="Backtest start date")
    parser.add_argument("--end",     default="2023-01-01",         help="Backtest end date")
    args = parser.parse_args()

    account_cfg      = AccountConfig(total_capital=args.capital)
    tactical_cfg     = TacticalConfig()
    experimental_cfg = ExperimentalConfig()
    brokerage_cfg    = BrokerageConfig()
    tax_cfg          = TaxConfig()          # 0% CGT by default (Singapore retail investor)

    if args.mode == "backtest":
        run_backtest(account_cfg, tactical_cfg, experimental_cfg, brokerage_cfg, tax_cfg,
                     start=args.start, end=args.end)
    elif args.mode == "paper":
        run_paper(account_cfg, tactical_cfg, experimental_cfg, brokerage_cfg, tax_cfg)
    else:
        run_live(account_cfg, tactical_cfg, experimental_cfg, brokerage_cfg, tax_cfg)
