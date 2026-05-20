import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from config import AccountConfig, BrokerageConfig, ExperimentalConfig, TacticalConfig, TaxConfig
from models import CircuitBreaker
from strategies import ExperimentalStrategy, TacticalStrategy

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
WARMUP_DAYS = 60


def is_market_open() -> bool:
    """Returns True if US market is currently open (9:30–16:00 ET, Mon–Fri)."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_time  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_time <= now < close_time


def _fetch_hourly(tickers: list[str], days: int = WARMUP_DAYS):
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("Run: pip install yfinance")
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    return yf.download(tickers, start=start, end=end, interval="1h", progress=False, auto_adjust=True)


def _run_bars(raw, tactical, experimental, tactical_cfg, experimental_cfg, all_tickers, circuit):
    """Replay all bars in `raw` through both strategies. Returns list of orders."""
    all_orders = []
    dates = raw.index.tolist()

    for date in dates:
        today = date.to_pydatetime() if hasattr(date, "to_pydatetime") else datetime.fromisoformat(str(date))
        total_equity = tactical.equity + experimental.equity

        if not circuit.update(total_equity):
            logger.warning(f"[{today.strftime('%d-%b-%y %H:%M')}] Circuit breaker active — skipping.")
            continue

        for symbol in tactical_cfg.tickers:
            try:
                close = float(raw["Close"][symbol][date]) if len(all_tickers) > 1 else float(raw["Close"][date])
                if np.isnan(close):
                    continue
                all_orders.extend(tactical.on_bar(symbol, close, today))
            except (KeyError, TypeError):
                continue

        for symbol in experimental_cfg.tickers:
            try:
                if len(all_tickers) > 1:
                    close  = float(raw["Close"][symbol][date])
                    volume = float(raw["Volume"][symbol][date])
                else:
                    close  = float(raw["Close"][date])
                    volume = float(raw["Volume"][date])
                if np.isnan(close) or np.isnan(volume):
                    continue
                all_orders.extend(experimental.on_bar(symbol, close, volume, today))
            except (KeyError, TypeError):
                continue

    return all_orders


def _check_pdt(all_orders: list[dict]) -> list[str]:
    """
    Detects Pattern Day Trader violations: 4+ day trades in any rolling 5-business-day window.
    A day trade = buying AND selling the same symbol on the same calendar day.

    Note: PDT is a US FINRA rule for margin accounts with <$25k.
    IBKR Singapore (non-US residents) are typically exempt, but this flags
    the activity so you can assess your account type.
    """
    # Collect day-trade events: (date, symbol)
    trades_by_day: dict = defaultdict(lambda: defaultdict(set))
    for o in all_orders:
        day = o["date"].date()
        trades_by_day[day][o["symbol"]].add(o["action"])

    day_trade_events = []
    for day in sorted(trades_by_day):
        for _, actions in trades_by_day[day].items():
            if "buy" in actions and "sell" in actions:
                day_trade_events.append(day)
                break  # count max 1 day-trade event per calendar day

    if not day_trade_events:
        return []

    warnings = []
    for i, day in enumerate(day_trade_events):
        window_start = day - timedelta(days=4)
        count = sum(1 for d in day_trade_events if window_start <= d <= day)
        if count >= 4 and (i == 0 or day_trade_events[i - 1] < day):
            warnings.append(
                f"  ⚠  {day}  —  {count} day trades in 5-day window  "
                f"(PDT flags at 4+ in a US margin account)"
            )

    return warnings


def run_backtest(
    account_cfg: AccountConfig,
    tactical_cfg: TacticalConfig,
    experimental_cfg: ExperimentalConfig,
    brokerage_cfg: BrokerageConfig = None,
    tax_cfg: TaxConfig = None,
    start: str = "2022-01-01",
    end: str   = "2023-01-01",
):
    if brokerage_cfg is None:
        brokerage_cfg = BrokerageConfig()
    if tax_cfg is None:
        tax_cfg = TaxConfig()

    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("Run: pip install yfinance")

    logger.info(
        f"\n{'='*60}\n"
        f"  ROBO INVESTOR — BACKTEST (hourly)\n"
        f"  Period   : {start}  →  {end}\n"
        f"  Capital  : ${account_cfg.total_capital:,.0f}  "
        f"(tactical=${account_cfg.tactical_capital:,.0f}  "
        f"experimental=${account_cfg.experimental_capital:,.0f})\n"
        f"  Brokerage: IBKR Singapore  "
        f"(${brokerage_cfg.fee_per_share}/share, min ${brokerage_cfg.min_fee}, "
        f"spread={brokerage_cfg.spread_pct:.2%})\n"
        f"  Tax      : CGT={tax_cfg.capital_gains_rate:.0%}  "
        f"Income={tax_cfg.income_tax_rate:.0%}  (Singapore defaults)\n"
        f"{'='*60}"
    )

    all_tickers = list(set(tactical_cfg.tickers + experimental_cfg.tickers))
    logger.info(f"Downloading hourly data for: {all_tickers}")
    raw = yf.download(all_tickers, start=start, end=end, interval="1h", progress=False, auto_adjust=True)

    if raw.empty:
        raise SystemExit("No data returned from yfinance.")

    tactical     = TacticalStrategy(tactical_cfg, account_cfg.tactical_capital, brokerage_cfg)
    experimental = ExperimentalStrategy(experimental_cfg, account_cfg.experimental_capital, brokerage_cfg)
    circuit      = CircuitBreaker(account_cfg.max_total_drawdown)

    all_orders = _run_bars(raw, tactical, experimental, tactical_cfg, experimental_cfg, all_tickers, circuit)
    pdt_warnings = _check_pdt(all_orders)
    _print_report(account_cfg, tactical, experimental, circuit, all_orders, pdt_warnings, tax_cfg, start, end)

    buys  = [o for o in all_orders if o["action"] == "buy"]
    sells = [o for o in all_orders if o["action"] == "sell"]
    total_start      = account_cfg.total_capital
    total_end        = tactical.equity + experimental.equity
    total_realized   = tactical.realized_pnl + experimental.realized_pnl
    effective_tax    = max(tax_cfg.capital_gains_rate, tax_cfg.income_tax_rate)
    tax_owed         = max(0.0, total_realized * effective_tax)
    return {
        "total_return":        (total_end - total_start) / total_start,
        "tactical_return":     (tactical.equity - account_cfg.tactical_capital) / account_cfg.tactical_capital,
        "experimental_return": (experimental.equity - account_cfg.experimental_capital) / account_cfg.experimental_capital,
        "total_trades":        len(buys) + len(sells),
        "total_fees_paid":     tactical.total_fees_paid + experimental.total_fees_paid,
        "total_spread_cost":   tactical.total_spread_cost + experimental.total_spread_cost,
        "realized_pnl":        total_realized,
        "tax_owed":            tax_owed,
        "pdt_violations":      len(pdt_warnings),
        "circuit_tripped":     circuit.tripped,
    }


def run_paper(
    account_cfg: AccountConfig,
    tactical_cfg: TacticalConfig,
    experimental_cfg: ExperimentalConfig,
    brokerage_cfg: BrokerageConfig = None,
    tax_cfg: TaxConfig = None,
    lookback_days: int = 60,
):
    """Replay the last N days of hourly data as a paper trade."""
    if brokerage_cfg is None:
        brokerage_cfg = BrokerageConfig()
    if tax_cfg is None:
        tax_cfg = TaxConfig()
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    logger.info(f"Paper trading mode: replaying {lookback_days} days of hourly data ({start} → {end})")
    return run_backtest(account_cfg, tactical_cfg, experimental_cfg, brokerage_cfg, tax_cfg, start=start, end=end)


def run_live(
    account_cfg: AccountConfig,
    tactical_cfg: TacticalConfig,
    experimental_cfg: ExperimentalConfig,
    brokerage_cfg: BrokerageConfig = None,
    tax_cfg: TaxConfig = None,
):
    """
    Live hourly loop — wakes up every hour during US market hours (9:30–16:00 ET).
    Singapore time reference: market runs ~21:30–04:00 SGT (summer) or ~22:30–05:00 SGT (winter).
    """
    if brokerage_cfg is None:
        brokerage_cfg = BrokerageConfig()
    if tax_cfg is None:
        tax_cfg = TaxConfig()

    all_tickers = list(set(tactical_cfg.tickers + experimental_cfg.tickers))
    circuit     = CircuitBreaker(account_cfg.max_total_drawdown)

    logger.info("Live mode started. Waiting for US market hours (9:30–16:00 ET).")

    while True:
        now_et = datetime.now(ET)

        if not is_market_open():
            next_check = now_et.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            wait_secs  = (next_check - now_et).total_seconds()
            logger.info(
                f"Market closed ({now_et.strftime('%d-%b-%y %H:%M')} ET). "
                f"Next check in {int(wait_secs // 60)} min."
            )
            time.sleep(wait_secs)
            continue

        logger.info(f"Market open — fetching hourly data ({now_et.strftime('%d-%b-%y %H:%M')} ET)")

        try:
            raw = _fetch_hourly(all_tickers, days=WARMUP_DAYS)
        except Exception as e:
            logger.error(f"Data fetch failed: {e}. Retrying next hour.")
            time.sleep(3600)
            continue

        if raw.empty:
            logger.warning("Empty data returned. Retrying next hour.")
            time.sleep(3600)
            continue

        tactical_tmp     = TacticalStrategy(tactical_cfg, account_cfg.tactical_capital, brokerage_cfg)
        experimental_tmp = ExperimentalStrategy(experimental_cfg, account_cfg.experimental_capital, brokerage_cfg)
        orders = _run_bars(raw, tactical_tmp, experimental_tmp, tactical_cfg, experimental_cfg, all_tickers, circuit)

        if orders:
            logger.info(f"Signals this tick: {len(orders)} order(s)")
            for o in orders:
                logger.info(
                    f"  → {o['action'].upper():4s} {o['symbol']} | {o['shares']} shares "
                    f"@ ${o['price']:.2f} | fee=${o['fee']:.2f} | spread=${o['spread_cost']:.2f}"
                )
        else:
            logger.info("No signals this hour.")

        next_hour = now_et.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        wait_secs = (next_hour - datetime.now(ET)).total_seconds()
        logger.info(f"Next tick in {int(wait_secs // 60)}m {int(wait_secs % 60)}s.")
        time.sleep(max(wait_secs, 1))


def _print_report(account_cfg, tactical, experimental, circuit, all_orders, pdt_warnings, tax_cfg, start, end):
    total_start  = account_cfg.total_capital
    total_end    = tactical.equity + experimental.equity
    total_return = (total_end - total_start) / total_start

    tactical_return     = (tactical.equity    - account_cfg.tactical_capital)    / account_cfg.tactical_capital
    experimental_return = (experimental.equity - account_cfg.experimental_capital) / account_cfg.experimental_capital

    buys  = [o for o in all_orders if o["action"] == "buy"]
    sells = [o for o in all_orders if o["action"] == "sell"]

    total_fees        = tactical.total_fees_paid   + experimental.total_fees_paid
    total_spread      = tactical.total_spread_cost + experimental.total_spread_cost
    total_realized    = tactical.realized_pnl      + experimental.realized_pnl
    effective_tax_rate = max(tax_cfg.capital_gains_rate, tax_cfg.income_tax_rate)
    tax_owed          = max(0.0, total_realized * effective_tax_rate)

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS  ({start} → {end})  [hourly bars]")
    print(f"{'='*60}")
    print(f"  {'':30s}  {'START':>10}  {'END':>10}  {'RETURN':>8}")
    print(f"  {'-'*58}")
    print(f"  {'TACTICAL (swing, RSI)':30s}  ${account_cfg.tactical_capital:>9,.0f}  ${tactical.equity:>9,.0f}  {tactical_return:>7.1%}")
    print(f"  {'EXPERIMENTAL (daily, MACD)':30s}  ${account_cfg.experimental_capital:>9,.0f}  ${experimental.equity:>9,.0f}  {experimental_return:>7.1%}")
    print(f"  {'-'*58}")
    print(f"  {'TOTAL BOT PORTFOLIO':30s}  ${total_start:>9,.0f}  ${total_end:>9,.0f}  {total_return:>7.1%}")
    print(f"{'='*60}")
    print(f"  Total trades       : {len(buys)} buys, {len(sells)} sells")
    print(f"")
    print(f"  TRADING COSTS")
    print(f"  {'IBKR commissions':25s}: ${total_fees:>8,.2f}  ({total_fees/total_start:.2%} of capital)")
    print(f"  {'Bid-ask spread':25s}: ${total_spread:>8,.2f}  ({total_spread/total_start:.2%} of capital)")
    print(f"  {'Total cost drag':25s}: ${total_fees+total_spread:>8,.2f}  ({(total_fees+total_spread)/total_start:.2%} of capital)")
    print(f"")
    print(f"  TAX (Singapore)")
    print(f"  {'Realized P&L (closed)':25s}: ${total_realized:>+8,.2f}")
    if effective_tax_rate > 0:
        print(f"  {'Tax rate applied':25s}:  {effective_tax_rate:.0%}")
        print(f"  {'Estimated tax owed':25s}: ${tax_owed:>8,.2f}")
    else:
        print(f"  No capital gains tax (Singapore retail investor)")
        print(f"  Set income_tax_rate in TaxConfig if IRAS classifies you as a trader")
    print(f"")
    print(f"  RISK")
    print(f"  Circuit breaker    : {'TRIPPED ⚠' if circuit.tripped else 'Not tripped'}")

    if pdt_warnings:
        print(f"")
        print(f"  PATTERN DAY TRADER WARNINGS  ({len(pdt_warnings)} window(s) flagged)")
        print(f"  Note: PDT is a US FINRA rule — likely exempt via IBKR Singapore,")
        print(f"  but check your account type (cash vs margin) to confirm.")
        for w in pdt_warnings:
            print(w)
    else:
        print(f"  PDT rule           : No violations detected")

    if tactical.positions:
        print(f"\n  Open tactical positions:")
        for s, p in tactical.positions.items():
            print(f"    {s:<8} {p.shares:>5} shares @ ${p.entry_price:.2f}  (held {p.bars_held}h)")

    if experimental.positions:
        print(f"\n  Open experimental positions:")
        for s, p in experimental.positions.items():
            print(f"    {s:<8} {p.shares:>5} shares @ ${p.entry_price:.2f}  (held {p.bars_held}h)")

    print()
