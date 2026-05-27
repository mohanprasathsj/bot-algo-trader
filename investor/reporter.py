"""Reporting for the live IBKR trading engine.

Generates three types of output every run:
  1. Per-run risk report  — circuit breaker status, PDT check, open positions
  2. Trade log            — one JSON line per trade appended to trades.jsonl
  3. Daily summary        — end-of-day P&L report matching backtest format
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .config import AccountConfig, TaxConfig
from .models import CircuitBreaker, TradeRecord
from .state import AppState

logger = logging.getLogger(__name__)


class Reporter:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._trade_log = self.log_dir / "trades.jsonl"

    # ------------------------------------------------------------------ #
    #  Trade log (JSONL — one record per executed trade)                   #
    # ------------------------------------------------------------------ #

    def log_trade(self, record: TradeRecord) -> None:
        line = json.dumps({
            "timestamp": record.timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
            "symbol": record.symbol,
            "side": record.side,
            "bucket": record.bucket,
            "shares": record.shares,
            "price": record.price,
            "fee": record.fee,
            "spread_cost": record.spread_cost,
            "pnl": record.pnl,
            "reason": record.reason,
            "ibkr_order_id": record.ibkr_order_id,
        })
        with self._trade_log.open("a") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------ #
    #  Per-run risk report                                                  #
    # ------------------------------------------------------------------ #

    def risk_report(
        self,
        state: AppState,
        circuit: CircuitBreaker,
        actions: list[dict],
        now: datetime,
    ) -> str:
        pdt_warnings = self.check_pdt(state.trade_history)

        t = state.tactical
        e = state.experimental
        total_equity = t.equity + e.equity
        drawdown = 0.0
        if circuit._peak and circuit._peak > 0:
            drawdown = (circuit._peak - total_equity) / circuit._peak

        cb_status = (
            f"TRIPPED ⚠  (drawdown={drawdown:.1%}  peak=${circuit._peak:,.0f})"
            if circuit.tripped
            else f"OK  (peak=${circuit._peak or total_equity:,.0f}  drawdown={drawdown:.1%})"
        )

        lines = [
            f"\n{'='*60}",
            f"  RISK REPORT  —  {now.strftime('%d-%b-%Y  %H:%M')} ET",
            f"{'='*60}",
            f"  Portfolio",
            f"    Total equity    : ${total_equity:>10,.2f}",
            f"    Circuit breaker : {cb_status}",
            f"",
            f"  Tactical Bucket (RSI swing)",
            f"    Cash            : ${t.cash:>10,.2f}",
            f"    Open positions  : {len(t.positions)}",
        ]
        for sym, pos in t.positions.items():
            unrealized = (pos.entry_price - pos.entry_price) * pos.shares  # placeholder
            lines.append(
                f"      {sym:<6}  {pos.shares:>5} shares @ ${pos.entry_price:.2f}"
                f"  held={pos.bars_held}h"
                f"  stop=${pos.stop_price:.2f}  target=${pos.target_price:.2f}"
            )

        lines += [
            f"",
            f"  Experimental Bucket (MACD momentum)",
            f"    Cash            : ${e.cash:>10,.2f}",
            f"    Open positions  : {len(e.positions)}",
        ]
        for sym, pos in e.positions.items():
            lines.append(
                f"      {sym:<6}  {pos.shares:>5} shares @ ${pos.entry_price:.2f}"
                f"  held={pos.bars_held}h"
                f"  stop=${pos.stop_price:.2f}  target=${pos.target_price:.2f}"
            )

        lines += [f"", f"  PDT Check"]
        day_trades_this_week = self._count_day_trades_last_5d(state.trade_history)
        lines.append(f"    Day trades (rolling 5 days): {day_trades_this_week}")
        if pdt_warnings:
            lines.append(f"    ⚠  {len(pdt_warnings)} PDT violation window(s) detected")
            for w in pdt_warnings:
                lines.append(f"       {w}")
        else:
            lines.append(f"    No PDT violations detected")

        if actions:
            lines += [f"", f"  Actions this hour: {len(actions)} order(s)"]
            for a in actions:
                lines.append(
                    f"    {a['action'].upper():<4}  {a['symbol']:<6}  "
                    f"{a['shares']} shares @ ${a.get('fill_price', a.get('price', 0)):.2f}"
                    f"  [{a['reason']}]"
                )
        else:
            lines += [f"", f"  Actions this hour: no signals"]

        lines.append(f"{'='*60}\n")
        report = "\n".join(lines)

        daily_file = self.log_dir / f"risk_{now.strftime('%Y-%m-%d')}.log"
        with daily_file.open("a") as f:
            f.write(report + "\n")

        return report

    # ------------------------------------------------------------------ #
    #  Daily summary (end-of-day, matches backtest _print_report format)   #
    # ------------------------------------------------------------------ #

    def daily_summary(
        self,
        state: AppState,
        account_cfg: AccountConfig,
        tax_cfg: TaxConfig,
        date: datetime,
    ) -> str:
        t = state.tactical
        e = state.experimental
        total_start = account_cfg.total_capital
        total_end = t.equity + e.equity
        total_return = (total_end - total_start) / total_start if total_start > 0 else 0.0

        def _ret(equity, capital):
            return (equity - capital) / capital if capital > 0 else None

        t_ret = _ret(t.equity, account_cfg.tactical_capital)
        e_ret = _ret(e.equity, account_cfg.experimental_capital)
        fmt_ret = lambda r: f"{r:>7.1%}" if r is not None else "    N/A"

        today_trades = [
            tr for tr in state.trade_history
            if tr.timestamp.date() == date.date()
        ]
        today_buys = [tr for tr in today_trades if tr.side == "buy"]
        today_sells = [tr for tr in today_trades if tr.side == "sell"]

        total_fees = t.total_fees_paid + e.total_fees_paid
        total_spread = t.total_spread_cost + e.total_spread_cost
        total_realized = t.realized_pnl + e.realized_pnl
        effective_tax = max(tax_cfg.capital_gains_rate, tax_cfg.income_tax_rate)
        tax_owed = max(0.0, total_realized * effective_tax)

        lines = [
            f"\n{'='*60}",
            f"  DAILY SUMMARY  —  {date.strftime('%d-%b-%Y')}",
            f"{'='*60}",
            f"  {'':30s}  {'START':>10}  {'NOW':>10}  {'RETURN':>8}",
            f"  {'-'*58}",
            f"  {'TACTICAL (swing, RSI)':30s}  ${account_cfg.tactical_capital:>9,.0f}  ${t.equity:>9,.0f}  {fmt_ret(t_ret)}",
            f"  {'EXPERIMENTAL (daily, MACD)':30s}  ${account_cfg.experimental_capital:>9,.0f}  ${e.equity:>9,.0f}  {fmt_ret(e_ret)}",
            f"  {'-'*58}",
            f"  {'TOTAL BOT PORTFOLIO':30s}  ${total_start:>9,.0f}  ${total_end:>9,.0f}  {total_return:>7.1%}",
            f"{'='*60}",
            f"  Today's trades     : {len(today_buys)} buys, {len(today_sells)} sells",
            f"  All-time trades    : {len([t for t in state.trade_history if t.side == 'buy'])} buys, "
            f"{len([t for t in state.trade_history if t.side == 'sell'])} sells",
            f"",
            f"  TRADING COSTS (cumulative)",
            f"  {'IBKR commissions':25s}: ${total_fees:>8,.2f}  ({total_fees/total_start:.2%} of capital)",
            f"  {'Bid-ask spread':25s}: ${total_spread:>8,.2f}  ({total_spread/total_start:.2%} of capital)",
            f"  {'Total cost drag':25s}: ${total_fees+total_spread:>8,.2f}  ({(total_fees+total_spread)/total_start:.2%} of capital)",
            f"",
            f"  TAX (Singapore)",
            f"  {'Realized P&L (closed)':25s}: ${total_realized:>+8,.2f}",
        ]
        if effective_tax > 0:
            lines += [
                f"  {'Tax rate applied':25s}:  {effective_tax:.0%}",
                f"  {'Estimated tax owed':25s}: ${tax_owed:>8,.2f}",
            ]
        else:
            lines.append(f"  No capital gains tax (Singapore retail investor)")

        if t.positions or e.positions:
            lines.append(f"")
            lines.append(f"  OPEN POSITIONS")
            for sym, pos in t.positions.items():
                lines.append(f"    [T] {sym:<6}  {pos.shares:>5} shares @ ${pos.entry_price:.2f}  held={pos.bars_held}h")
            for sym, pos in e.positions.items():
                lines.append(f"    [E] {sym:<6}  {pos.shares:>5} shares @ ${pos.entry_price:.2f}  held={pos.bars_held}h")

        lines.append(f"{'='*60}\n")
        summary = "\n".join(lines)

        out_path = self.log_dir / f"daily_{date.strftime('%Y-%m-%d')}.txt"
        out_path.write_text(summary)
        logger.info(f"Daily summary saved → {out_path}")
        return summary

    # ------------------------------------------------------------------ #
    #  PDT helpers                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_pdt(trade_history: list[TradeRecord]) -> list[str]:
        """Return warning strings for any rolling 5-day windows with 4+ day trades."""
        trades_by_day: dict = defaultdict(lambda: defaultdict(set))
        for t in trade_history:
            day = t.timestamp.date()
            trades_by_day[day][t.symbol].add(t.side)

        day_trade_events = []
        for day in sorted(trades_by_day):
            for _, actions in trades_by_day[day].items():
                if "buy" in actions and "sell" in actions:
                    day_trade_events.append(day)
                    break

        warnings = []
        for i, day in enumerate(day_trade_events):
            window_start = day - timedelta(days=4)
            count = sum(1 for d in day_trade_events if window_start <= d <= day)
            if count >= 4 and (i == 0 or day_trade_events[i - 1] < day):
                warnings.append(
                    f"{day}  —  {count} day trades in 5-day window "
                    f"(PDT flags at 4+ for US margin accounts)"
                )
        return warnings

    def _count_day_trades_last_5d(self, trade_history: list[TradeRecord]) -> int:
        cutoff = datetime.utcnow().date() - timedelta(days=4)
        trades_by_day: dict = defaultdict(lambda: defaultdict(set))
        for t in trade_history:
            if t.timestamp.date() >= cutoff:
                trades_by_day[t.timestamp.date()][t.symbol].add(t.side)
        count = 0
        for day_symbols in trades_by_day.values():
            for actions in day_symbols.values():
                if "buy" in actions and "sell" in actions:
                    count += 1
        return count
