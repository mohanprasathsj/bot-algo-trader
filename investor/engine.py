from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .config import (
    AccountConfig, BrokerageConfig, ExperimentalConfig,
    IBKRConfig, TacticalConfig, TaxConfig,
)
from .models import Bucket, CircuitBreaker, Position, TradeRecord
from .state import AppState, StrategyState
from .strategies import ExperimentalStrategy, TacticalStrategy

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# Hourly lookback for RSI / volume / divergence warm-up only.
# 52W-high and 200-day MA now use a separate daily data stream,
# so hourly data no longer needs to stretch 250+ days.
WARMUP_DAYS       = 60
DAILY_WARMUP_DAYS = 300   # calendar days of daily history pre-loaded before any bar


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


def _to_multi_ticker(ticker_dfs: dict) -> "pd.DataFrame":
    """Combine per-ticker DataFrames into yfinance-style MultiIndex format.

    yfinance multi-ticker output:  raw["Close"]["AAPL"]  →  Series
    We reconstruct that exact shape from individually-cached DataFrames so the
    rest of the engine needs no changes.
    """
    import pandas as pd
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    frames = []
    for ticker in sorted(ticker_dfs):
        df  = ticker_dfs[ticker]
        cols = [c for c in ohlcv_cols if c in df.columns]
        sub  = df[cols].copy()
        sub.columns = pd.MultiIndex.from_tuples([(col, ticker) for col in cols])
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=1)
    combined.sort_index(inplace=True)
    combined.sort_index(axis=1, inplace=True)
    return combined


def _fetch_hourly_cached(tickers: list[str], days: int = WARMUP_DAYS) -> "pd.DataFrame":
    """Cache-aware hourly fetch.

    Each ticker is loaded from its local Parquet file; only the missing tail
    (usually 1 trading day) is downloaded from yfinance.  Falls back to the
    original batch download if the cache layer fails entirely.
    """
    from .cache import get_ticker_ohlcv
    end   = datetime.today()
    start = end - timedelta(days=days)

    ticker_dfs: dict = {}
    failed: list[str] = []
    for ticker in tickers:
        df = get_ticker_ohlcv(ticker, "1h", start, end)
        if df is not None and not df.empty:
            ticker_dfs[ticker] = df
        else:
            failed.append(ticker)

    if failed:
        logger.warning(
            f"[cache] Hourly cache miss for {failed} — "
            "falling back to direct yfinance batch download"
        )
        return _fetch_hourly(tickers, days)

    return _to_multi_ticker(ticker_dfs)


def _fetch_daily_cached(tickers: list[str], start: str, end: str) -> "pd.DataFrame":
    """Cache-aware daily fetch.

    Each tactical ticker is stored in its own Parquet file.  Only the delta
    since the last cached bar is fetched; the full multi-ticker DataFrame is
    reconstructed in the same shape as `_fetch_daily` so callers need no change.
    Falls back to `_fetch_daily` if the cache layer errors.
    """
    from .cache import get_ticker_ohlcv
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d") + timedelta(days=1)  # end is exclusive

    ticker_dfs: dict = {}
    failed: list[str] = []
    for ticker in tickers:
        df = get_ticker_ohlcv(ticker, "1d", start_dt, end_dt)
        if df is not None and not df.empty:
            ticker_dfs[ticker] = df
        else:
            failed.append(ticker)

    if failed:
        logger.warning(
            f"[cache] Daily cache miss for {failed} — "
            "falling back to _fetch_daily"
        )
        if ticker_dfs:
            # Merge what we have from cache with a fresh batch call for the failures
            fallback = _fetch_daily(failed, start, end)
            # Pull per-ticker frames from the fallback and add to ticker_dfs
            import pandas as pd
            if not fallback.empty:
                try:
                    close_block = fallback["Close"]
                    for t in failed:
                        if isinstance(close_block, pd.DataFrame) and t in close_block.columns:
                            sub_df = fallback.xs(t, axis=1, level=1)
                            ticker_dfs[t] = sub_df
                except Exception:
                    pass
        else:
            return _fetch_daily(tickers, start, end)

    return _to_multi_ticker(ticker_dfs)


def _fetch_daily(tickers: list[str], start: str, end: str):
    """Fetch daily OHLCV data from yfinance for the knife-guard indicators.

    yfinance 1.2.x has an intermittent bug where some tickers in a batch
    download fail with ``TypeError("'NoneType' object is not subscriptable")``
    due to missing metadata.  After the initial batch we detect any missing
    tickers and retry them one at a time, then merge the results back.
    This keeps the fast path (one request) while recovering stragglers.
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        raise SystemExit("Run: pip install yfinance")

    raw = yf.download(tickers, start=start, end=end, interval="1d", progress=False, auto_adjust=True)

    # Detect tickers that are missing or entirely NaN in the batch result.
    missing: list[str] = []
    if not raw.empty:
        try:
            close_block = raw["Close"]
            present = set(close_block.columns) if isinstance(close_block, pd.DataFrame) else set()
            for t in tickers:
                if t not in present or close_block[t].dropna().empty:
                    missing.append(t)
        except (KeyError, TypeError):
            missing = list(tickers)   # fallback: retry everything
    else:
        missing = list(tickers)

    if not missing:
        return raw

    # Per-ticker retry for stragglers — avoids the batch metadata bug.
    logger.debug(f"Retrying {len(missing)} tickers individually: {missing}")
    retry_frames: list = []
    for t in missing:
        try:
            single = yf.download(t, start=start, end=end, interval="1d",
                                 progress=False, auto_adjust=True)
            if not single.empty:
                # Wrap into multi-level columns to match the batch format.
                single.columns = pd.MultiIndex.from_tuples(
                    [(col, t) for col in single.columns]
                )
                retry_frames.append(single)
                logger.debug(f"  ✓ {t}: {len(single)} rows recovered")
            else:
                logger.warning(f"  ✗ {t}: still empty after individual retry — knife guards will skip it")
        except Exception as e:
            logger.warning(f"  ✗ {t}: individual retry failed ({e}) — knife guards will skip it")

    if retry_frames:
        try:
            recovered = pd.concat(retry_frames, axis=1)
            raw = pd.concat([raw, recovered], axis=1).sort_index()
        except Exception as e:
            logger.warning(f"Could not merge retry data: {e}")

    return raw


def _build_daily_close_map(daily_raw, symbol: str, all_tickers: list[str]) -> dict:
    """Return ``{datetime.date: float}`` of daily closes for *symbol*.

    Sorted chronologically; NaN values are dropped.  Safe to call when the
    ticker is missing from ``daily_raw`` — returns an empty dict.

    yfinance returns different shapes depending on ticker count:
    - multi-ticker  : ``daily_raw["Close"]`` is a DataFrame; access by column name
    - single-ticker : ``daily_raw["Close"]`` may be a Series OR a 1-col DataFrame
    We handle both by always trying the symbol lookup first, then falling back
    to squeezing a single-column DataFrame to a Series.
    """
    import pandas as pd
    try:
        close_block = daily_raw["Close"]
        # Multi-ticker path: columns are labelled by ticker symbol
        if isinstance(close_block, pd.DataFrame) and symbol in close_block.columns:
            series = close_block[symbol].dropna()
        elif isinstance(close_block, pd.DataFrame):
            # Single-ticker: one unnamed column — squeeze to Series
            series = close_block.squeeze().dropna()
        else:
            # Already a Series (single-ticker some versions)
            series = close_block.dropna()
        return {pd.Timestamp(ts).date(): float(v) for ts, v in series.items()}
    except (KeyError, TypeError, ValueError):
        return {}


def _run_bars(
    raw,
    tactical,
    experimental,
    tactical_cfg,
    experimental_cfg,
    all_tickers,
    circuit,
    daily_close_maps: dict | None = None,
):
    """Replay all hourly bars through both strategies.

    ``daily_close_maps`` is ``{symbol: {datetime.date: close}}`` and drives the
    falling-knife guards (200-day MA, 52W high).  On each trading-day boundary
    the *previous* day's close is injected into the tactical strategy so there
    is zero look-ahead — when the 09:30 bar of Tuesday is processed, only
    Monday's (and earlier) daily closes are in memory.
    """
    all_orders = []
    dates = raw.index.tolist()
    daily_close_maps = daily_close_maps or {}

    prev_bar_date = None   # tracks the last-seen calendar date

    for date in dates:
        today = date.to_pydatetime() if hasattr(date, "to_pydatetime") else datetime.fromisoformat(str(date))
        bar_date = today.date()

        # ── Day transition: feed the COMPLETED day's close ───────────── #
        # Injected at the first bar of the NEXT day → strictly no look-ahead.
        if prev_bar_date is not None and bar_date > prev_bar_date:
            for symbol in tactical_cfg.tickers:
                dc = daily_close_maps.get(symbol, {}).get(prev_bar_date)
                if dc is not None:
                    tactical.on_daily_bar(symbol, dc)
        prev_bar_date = bar_date

        total_equity = tactical.equity + experimental.equity

        if not circuit.update(total_equity):
            logger.warning(f"[{today.strftime('%d-%b-%y %H:%M')}] Circuit breaker active — skipping.")
            continue

        for symbol in tactical_cfg.tickers:
            try:
                close  = float(raw["Close"][symbol][date])  if len(all_tickers) > 1 else float(raw["Close"][date])
                volume = float(raw["Volume"][symbol][date]) if len(all_tickers) > 1 else float(raw["Volume"][date])
                if np.isnan(close):
                    continue
                all_orders.extend(tactical.on_bar(symbol, close, today, volume=volume if not np.isnan(volume) else 0.0))
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
    trades_by_day: dict = defaultdict(lambda: defaultdict(set))
    for o in all_orders:
        day = o["date"].date()
        trades_by_day[day][o["symbol"]].add(o["action"])

    day_trade_events = []
    for day in sorted(trades_by_day):
        for _, actions in trades_by_day[day].items():
            if "buy" in actions and "sell" in actions:
                day_trade_events.append(day)
                break

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

    from .cache import cache_summary
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
        f"  Local cache:\n{cache_summary()}\n"
        f"{'='*60}"
    )

    all_tickers = list(set(tactical_cfg.tickers + experimental_cfg.tickers))

    # ── Hourly data (RSI, volume, divergence) — served from local cache ── #
    # Each ticker's Parquet file is updated with only the missing tail;
    # a cold-start ticker downloads its full history once and is fast after.
    logger.info(f"Loading hourly data      : {all_tickers}")
    raw = _fetch_hourly_cached(all_tickers, days=(datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days + 1)
    if raw.empty:
        raise SystemExit("No hourly data returned (cache + yfinance both failed).")

    # ── Daily data (200-day MA, 52W high) — with pre-backtest warm-up ─── #
    # Fetch DAILY_WARMUP_DAYS before `start` so the 200-day MA and 52W high
    # have proper context from the very first bar of the backtest.
    start_dt    = datetime.strptime(start, "%Y-%m-%d")
    daily_start = (start_dt - timedelta(days=DAILY_WARMUP_DAYS)).strftime("%Y-%m-%d")
    logger.info(f"Loading daily data       : {tactical_cfg.tickers} ({daily_start} → {end})")
    daily_raw = _fetch_daily_cached(tactical_cfg.tickers, daily_start, end)

    tactical     = TacticalStrategy(tactical_cfg, account_cfg.tactical_capital, brokerage_cfg)
    experimental = ExperimentalStrategy(experimental_cfg, account_cfg.experimental_capital, brokerage_cfg)
    circuit      = CircuitBreaker(account_cfg.max_total_drawdown)

    # Pre-load daily closes from the warm-up window (before backtest start).
    # These are historical — no look-ahead risk.
    backtest_start_date = start_dt.date()
    daily_close_maps: dict[str, dict] = {}
    if not daily_raw.empty:
        for symbol in tactical_cfg.tickers:
            dmap = _build_daily_close_map(daily_raw, symbol, tactical_cfg.tickers)
            for day in sorted(dmap):
                if day < backtest_start_date:
                    tactical.on_daily_bar(symbol, dmap[day])    # pre-load warmup
            # In-window data is fed progressively (no look-ahead) by _run_bars
            daily_close_maps[symbol] = {d: v for d, v in dmap.items() if d >= backtest_start_date}

    all_orders = _run_bars(raw, tactical, experimental, tactical_cfg, experimental_cfg, all_tickers, circuit, daily_close_maps)
    pdt_warnings = _check_pdt(all_orders)
    _print_report(account_cfg, tactical, experimental, circuit, all_orders, pdt_warnings, tax_cfg, start, end)

    buys  = [o for o in all_orders if o["action"] == "buy"]
    sells = [o for o in all_orders if o["action"] == "sell"]
    total_start      = account_cfg.total_capital
    total_end        = tactical.equity + experimental.equity
    total_realized   = tactical.realized_pnl + experimental.realized_pnl
    effective_tax    = max(tax_cfg.capital_gains_rate, tax_cfg.income_tax_rate)
    tax_owed         = max(0.0, total_realized * effective_tax)
    def _return(equity, capital):
        return (equity - capital) / capital if capital > 0 else None

    return {
        "total_return":        (total_end - total_start) / total_start,
        "tactical_return":     _return(tactical.equity, account_cfg.tactical_capital),
        "experimental_return": _return(experimental.equity, account_cfg.experimental_capital),
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
    lookback_days: int = 300,
):
    """Replay the last N days of hourly data as a paper trade.

    Why 300 days?  The falling-knife guards need real historical context:
      • 200-day MA  : 200 trading days × 6.5 h ≈ 1 300 bars  (≥ 220 calendar days)
      • 52-week high: 52 wks × 5 days × 6.5 h ≈ 1 638 bars  (≥ 252 calendar days)
      • Warm-up gate: first `min_history_bars` bars are skipped for buy signals
    300 calendar days ≈ 1 450–1 650 hourly bars — sufficient for all three.
    Pass a smaller value only for quick smoke-tests; the guards will degrade.
    """
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
    """Dry-run hourly loop (no IBKR). Logs signals but does not execute orders.
    Use run_live_ibkr() for real order execution.
    """
    if brokerage_cfg is None:
        brokerage_cfg = BrokerageConfig()
    if tax_cfg is None:
        tax_cfg = TaxConfig()

    all_tickers = list(set(tactical_cfg.tickers + experimental_cfg.tickers))
    circuit     = CircuitBreaker(account_cfg.max_total_drawdown)

    logger.info("Live (dry-run) mode started. Waiting for US market hours (9:30–16:00 ET).")

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

        logger.info(f"Market open — loading hourly + daily data ({now_et.strftime('%d-%b-%y %H:%M')} ET)")

        try:
            raw = _fetch_hourly_cached(all_tickers, days=WARMUP_DAYS)
        except Exception as e:
            logger.error(f"Data load failed: {e}. Retrying next hour.")
            time.sleep(3600)
            continue

        if raw.empty:
            logger.warning("Empty data returned. Retrying next hour.")
            time.sleep(3600)
            continue

        # Load daily data for knife-guard indicators (cache-aware)
        try:
            daily_end   = datetime.today().strftime("%Y-%m-%d")
            daily_start = (datetime.today() - timedelta(days=DAILY_WARMUP_DAYS)).strftime("%Y-%m-%d")
            daily_raw   = _fetch_daily_cached(tactical_cfg.tickers, daily_start, daily_end)
        except Exception as e:
            logger.warning(f"Daily data load failed: {e} — knife guards may be skipped.")
            daily_raw = None

        tactical_tmp     = TacticalStrategy(tactical_cfg, account_cfg.tactical_capital, brokerage_cfg)
        experimental_tmp = ExperimentalStrategy(experimental_cfg, account_cfg.experimental_capital, brokerage_cfg)

        # Pre-load all available daily history into the fresh strategy object
        if daily_raw is not None and not daily_raw.empty:
            daily_close_maps_live: dict[str, dict] = {}
            for symbol in tactical_cfg.tickers:
                dmap = _build_daily_close_map(daily_raw, symbol, tactical_cfg.tickers)
                # All daily history is "past" — pre-load everything except today
                today_date = datetime.today().date()
                for day in sorted(dmap):
                    if day < today_date:
                        tactical_tmp.on_daily_bar(symbol, dmap[day])
                # Today's bar (if present) will be fed at day-boundary in _run_bars
                daily_close_maps_live[symbol] = {d: v for d, v in dmap.items() if d >= today_date}
        else:
            daily_close_maps_live = {}

        orders = _run_bars(raw, tactical_tmp, experimental_tmp, tactical_cfg, experimental_cfg, all_tickers, circuit, daily_close_maps_live)

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


# ======================================================================= #
#  IBKR Live Trading Engine                                                 #
# ======================================================================= #

def _price_series(raw, symbol: str, all_tickers: list[str]) -> list[float]:
    """Extract close price list for a symbol from a yfinance DataFrame."""
    try:
        if len(all_tickers) > 1:
            col = raw["Close"][symbol]
        else:
            col = raw["Close"]
        return [float(v) for v in col.dropna().tolist()]
    except (KeyError, TypeError):
        return []


def _volume_series(raw, symbol: str, all_tickers: list[str]) -> list[float]:
    try:
        if len(all_tickers) > 1:
            col = raw["Volume"][symbol]
        else:
            col = raw["Volume"]
        return [float(v) for v in col.dropna().tolist()]
    except (KeyError, TypeError):
        return []


def _knife_guard_live(
    symbol: str,
    close: float,
    hourly_prices: list[float],
    daily_prices: list[float],
    hourly_volumes: list[float],
    cfg: TacticalConfig,
    state: StrategyState,
    now: datetime,
) -> tuple[bool, str]:
    """Falling-knife protection for the live IBKR path.

    Mirrors ``TacticalStrategy._knife_guard`` but operates on pre-built
    series from the tick's yfinance data rather than in-process accumulators.

    Data split (same as the backtest path):
      hourly_prices / hourly_volumes  →  RSI divergence (rule 5), volume (rule 4)
      daily_prices                    →  200-day MA (rule 2), 52W high (rule 3)
    """
    from .indicators import compute_sma, detect_rsi_divergence

    # Rule 0 — daily warm-up gate
    if len(daily_prices) < cfg.min_daily_history:
        return True, (
            f"building daily history "
            f"({len(daily_prices)}/{cfg.min_daily_history} trading days)"
        )

    # Rule 2 — 200-day MA from daily closes
    if cfg.require_price_above_200ma and len(daily_prices) >= 200:
        ma200 = compute_sma(daily_prices, 200)
        if ma200 is not None and close < ma200:
            return True, f"below 200-day MA (${close:.2f} < ${ma200:.2f})"

    # Rule 3 — 52W high from daily closes (last 252 trading-day bars)
    window_52w = daily_prices[-252:] if len(daily_prices) >= 252 else daily_prices
    high_52w   = float(max(window_52w))
    drop       = (close - high_52w) / high_52w
    if drop < -cfg.max_drop_from_52w_high:
        return True, (
            f"≥{cfg.max_drop_from_52w_high:.0%} below 52W high "
            f"(${close:.2f} vs ${high_52w:.2f}, {drop:.1%})"
        )

    # Rule 4 — volume confirmation from hourly bars
    if cfg.require_volume_increase and len(hourly_volumes) >= 20:
        avg_vol     = float(np.mean(hourly_volumes[-20:]))
        current_vol = hourly_volumes[-1]
        threshold   = avg_vol * cfg.volume_increase_multiplier
        if avg_vol > 0 and current_vol < threshold:
            return True, (
                f"volume below threshold "
                f"({current_vol:,.0f} < {threshold:,.0f} = "
                f"{cfg.volume_increase_multiplier:.1f}× avg)"
            )

    # Rule 5 — bullish RSI divergence from hourly bars
    if cfg.require_rsi_divergence:
        min_len = cfg.rsi_period + 1 + cfg.rsi_divergence_lookback
        if len(hourly_prices) >= min_len:
            if not detect_rsi_divergence(hourly_prices, cfg.rsi_period, cfg.rsi_divergence_lookback):
                return True, "no bullish RSI divergence"

    # Rule 6a — cooldown after stop-loss
    last_sl_str = state.last_stop_loss.get(symbol)
    if last_sl_str:
        try:
            last_sl = datetime.strptime(last_sl_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            last_sl = None
        if last_sl is not None:
            days_since = (now.replace(tzinfo=None) - last_sl).days
            if days_since < cfg.re_entry_cooldown_days:
                remaining = cfg.re_entry_cooldown_days - days_since
                return True, (
                    f"stop-loss cooldown ({days_since}d elapsed, "
                    f"{remaining}d remaining)"
                )

    # Rule 6b — consecutive stop-loss streak exceeds limit
    streak = state.stop_loss_streak.get(symbol, 0)
    if streak > cfg.max_re_entries_per_stock:
        return True, (
            f"max re-entries exceeded "
            f"({streak} consecutive stop-losses, limit={cfg.max_re_entries_per_stock})"
        )

    return False, ""


def _tactical_signals(
    state: StrategyState,
    cfg: TacticalConfig,
    brokerage: BrokerageConfig,
    raw,
    current_prices: dict[str, float],
    all_tickers: list[str],
    now: datetime,
    daily_close_maps: dict | None = None,
) -> list[dict]:
    """Compute RSI-based entry/exit signals for the tactical bucket.

    ``daily_close_maps`` is ``{symbol: {date: close}}`` and drives rules 2 & 3
    of the knife guard.  When omitted the daily guards are silently skipped.
    """
    from .indicators import compute_rsi

    daily_close_maps = daily_close_maps or {}
    orders = []

    for symbol in cfg.tickers:
        prices  = _price_series(raw, symbol, all_tickers)
        volumes = _volume_series(raw, symbol, all_tickers)
        current = current_prices.get(symbol)
        if not prices or current is None:
            continue
        hourly_prices  = prices + [current]
        hourly_volumes = volumes  # live bar volume not reliably available intra-hour

        # Build ordered daily close list for this symbol
        dmap          = daily_close_maps.get(symbol, {})
        daily_prices  = [v for _, v in sorted(dmap.items())]

        if symbol in state.positions:
            pos = state.positions[symbol]
            pos.bars_held += 1
            exit_flag, reason = pos.should_exit(current)

            rsi = compute_rsi(hourly_prices, cfg.rsi_period)
            if rsi and rsi > cfg.rsi_overbought and pos.bars_held >= cfg.min_hold_hours:
                exit_flag, reason = True, f"RSI overbought ({rsi:.0f})"

            if exit_flag:
                sell_price  = brokerage.effective_sell_price(current)
                fee         = brokerage.calculate_fee(pos.shares, sell_price)
                spread_cost = pos.shares * current * brokerage.spread_pct / 2
                orders.append({
                    "action": "sell", "symbol": symbol, "shares": pos.shares,
                    "price": sell_price, "fee": fee, "spread_cost": spread_cost,
                    "reason": reason, "bucket": "tactical", "date": now,
                })

        elif len(state.positions) < cfg.max_positions:
            rsi = compute_rsi(hourly_prices, cfg.rsi_period)
            if rsi is not None and rsi < cfg.rsi_oversold:

                # ── Falling knife protection — all 6 guards ──────── #
                blocked, block_reason = _knife_guard_live(
                    symbol, current,
                    hourly_prices, daily_prices, hourly_volumes,
                    cfg, state, now,
                )
                if blocked:
                    logger.debug(
                        f"[TACTICAL] SKIP  {symbol:6s} | {now.strftime('%d-%b-%y %H:%M')} | "
                        f"RSI={rsi:.0f} oversold but blocked: {block_reason}"
                    )
                    continue

                buy_price = brokerage.effective_buy_price(current)
                max_spend = state.cash * cfg.max_position_pct
                available = max(0.0, state.cash - state.cash * 0.10)
                spend     = min(max_spend, available)
                shares    = int(spend / buy_price)
                if shares >= 1:
                    fee         = brokerage.calculate_fee(shares, buy_price)
                    spread_cost = shares * current * brokerage.spread_pct / 2
                    orders.append({
                        "action": "buy", "symbol": symbol, "shares": shares,
                        "price": buy_price, "fee": fee, "spread_cost": spread_cost,
                        "reason": f"RSI oversold ({rsi:.0f})", "bucket": "tactical", "date": now,
                    })
    return orders


def _experimental_signals(
    state: StrategyState,
    cfg: ExperimentalConfig,
    brokerage: BrokerageConfig,
    raw,
    current_prices: dict[str, float],
    all_tickers: list[str],
    now: datetime,
) -> list[dict]:
    """Compute MACD+volume entry/exit signals for the experimental bucket."""
    from .indicators import compute_macd

    orders = []
    for symbol in cfg.tickers:
        prices  = _price_series(raw, symbol, all_tickers)
        volumes = _volume_series(raw, symbol, all_tickers)
        current = current_prices.get(symbol)
        if not prices or current is None:
            continue
        all_prices  = prices + [current]
        all_volumes = volumes  # current bar volume not reliably available intra-hour

        _, _, histogram = compute_macd(all_prices, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        prev_hist = state.prev_histogram.get(symbol)

        if symbol in state.positions:
            pos = state.positions[symbol]
            pos.bars_held += 1
            exit_flag, reason = pos.should_exit(current)

            if histogram is not None and histogram < 0 and pos.bars_held >= 1:
                exit_flag, reason = True, "MACD momentum lost"

            if exit_flag:
                sell_price  = brokerage.effective_sell_price(current)
                fee         = brokerage.calculate_fee(pos.shares, sell_price)
                spread_cost = pos.shares * current * brokerage.spread_pct / 2
                orders.append({
                    "action": "sell", "symbol": symbol, "shares": pos.shares,
                    "price": sell_price, "fee": fee, "spread_cost": spread_cost,
                    "reason": reason, "bucket": "experimental", "date": now,
                })

        elif len(state.positions) < cfg.max_positions:
            avg_vol = np.mean(all_volumes[-20:]) if len(all_volumes) >= 20 else None
            current_vol = all_volumes[-1] if all_volumes else 0
            volume_ok = avg_vol and current_vol > avg_vol * cfg.volume_multiplier
            macd_bullish = (
                histogram is not None and prev_hist is not None
                and prev_hist <= 0 and histogram > 0
            )
            if macd_bullish and volume_ok:
                buy_price = brokerage.effective_buy_price(current)
                max_spend = state.cash * cfg.max_position_pct
                available = max(0.0, state.cash - state.cash * 0.10)
                spend     = min(max_spend, available)
                shares    = int(spend / buy_price)
                if shares >= 1:
                    fee         = brokerage.calculate_fee(shares, buy_price)
                    spread_cost = shares * current * brokerage.spread_pct / 2
                    vol_ratio   = current_vol / avg_vol
                    orders.append({
                        "action": "buy", "symbol": symbol, "shares": shares,
                        "price": buy_price, "fee": fee, "spread_cost": spread_cost,
                        "reason": f"MACD bullish + vol={vol_ratio:.1f}x",
                        "bucket": "experimental", "date": now,
                    })

        if histogram is not None:
            state.prev_histogram[symbol] = histogram

    return orders


def _apply_order(
    state: StrategyState,
    order: dict,
    fill_price: float,
    brokerage: BrokerageConfig,
    cfg,
    now: datetime,
    ibkr_order_id: int | None = None,
) -> TradeRecord:
    """Apply a filled order to strategy state and return a TradeRecord."""
    symbol = order["symbol"]
    shares = order["shares"]
    bucket = order["bucket"]

    if order["action"] == "buy":
        fee         = brokerage.calculate_fee(shares, fill_price)
        spread_cost = shares * fill_price * brokerage.spread_pct / 2
        cost        = shares * fill_price + fee
        state.cash -= cost
        state.total_fees_paid   += fee
        state.total_spread_cost += spread_cost

        bucket_enum = Bucket.TACTICAL if bucket == "tactical" else Bucket.EXPERIMENTAL
        pos = Position(
            symbol=symbol, bucket=bucket_enum,
            entry_price=fill_price, shares=shares, entry_date=now,
            stop_loss_pct=cfg.stop_loss_pct, take_profit_pct=cfg.take_profit_pct,
            max_hold_hours=cfg.max_hold_hours, min_hold_hours=cfg.min_hold_hours,
        )
        state.positions[symbol] = pos
        logger.info(
            f"[{bucket.upper()}] ENTRY {symbol}  {shares} shares @ ${fill_price:.2f}"
            f"  fee=${fee:.2f}  cash=${state.cash:,.0f}"
        )
        return TradeRecord(
            timestamp=now, symbol=symbol, side="buy", bucket=bucket,
            shares=shares, price=fill_price, fee=fee,
            spread_cost=spread_cost, pnl=0.0,
            reason=order["reason"], ibkr_order_id=ibkr_order_id,
        )

    else:  # sell
        pos = state.positions.get(symbol)
        if pos is None:
            logger.warning(f"Sell signal for {symbol} but no open position — skipping.")
            return None
        fee         = brokerage.calculate_fee(shares, fill_price)
        spread_cost = shares * fill_price * brokerage.spread_pct / 2
        proceeds    = shares * fill_price - fee
        pnl         = proceeds - (shares * pos.entry_price)
        state.cash += proceeds
        state.total_fees_paid   += fee
        state.total_spread_cost += spread_cost
        state.realized_pnl      += pnl
        del state.positions[symbol]

        # ── Update falling-knife cooldown state (tactical bucket only) ─── #
        if bucket == "tactical":
            reason = order.get("reason", "")
            if "stop-loss" in reason:
                state.last_stop_loss[symbol]   = now.strftime("%Y-%m-%dT%H:%M:%S")
                state.stop_loss_streak[symbol] = state.stop_loss_streak.get(symbol, 0) + 1
                streak_note = f"  [stop streak={state.stop_loss_streak[symbol]}]"
            else:
                state.stop_loss_streak[symbol] = 0   # profitable / time exit resets streak
                streak_note = ""
        else:
            streak_note = ""

        logger.info(
            f"[{bucket.upper()}] EXIT  {symbol}  {shares} shares @ ${fill_price:.2f}"
            f"  PnL=${pnl:+.2f}  fee=${fee:.2f}  cash=${state.cash:,.0f}{streak_note}"
        )
        return TradeRecord(
            timestamp=now, symbol=symbol, side="sell", bucket=bucket,
            shares=shares, price=fill_price, fee=fee,
            spread_cost=spread_cost, pnl=pnl,
            reason=order["reason"], ibkr_order_id=ibkr_order_id,
        )


def run_live_ibkr(
    account_cfg: AccountConfig,
    tactical_cfg: TacticalConfig,
    experimental_cfg: ExperimentalConfig,
    ibkr_cfg: IBKRConfig = None,
    brokerage_cfg: BrokerageConfig = None,
    tax_cfg: TaxConfig = None,
    state_path: str = "investor/data/state.json",
    log_dir: str = "investor/logs",
):
    """
    Live IBKR trading loop — runs every hour during US market hours.

    Each tick:
      1. Fetch 60-day hourly bars from yfinance (indicator warmup)
      2. Get latest prices from IBKR (or fall back to yfinance last close)
      3. Compute tactical (RSI) and experimental (MACD) signals
      4. Execute real market orders via IB Gateway
      5. Update and persist state to JSON
      6. Print a risk report (circuit breaker, PDT, positions)
      7. At market close (~16:00 ET): write daily summary

    Requires IB Gateway running locally (paper port 4002 by default).
    """
    from .broker import IBKRBroker
    from .reporter import Reporter

    if ibkr_cfg is None:
        ibkr_cfg = IBKRConfig()
    if brokerage_cfg is None:
        brokerage_cfg = BrokerageConfig()
    if tax_cfg is None:
        tax_cfg = TaxConfig()

    all_tickers = list(set(tactical_cfg.tickers + experimental_cfg.tickers))
    reporter    = Reporter(log_dir)

    state = AppState.load(state_path)
    if state is None:
        logger.info("No saved state found — starting fresh.")
        state = AppState.new(account_cfg, tactical_cfg, experimental_cfg)

    broker = IBKRBroker(ibkr_cfg)
    connected = broker.connect()
    if not connected:
        logger.error(
            "Could not connect to IB Gateway. "
            "Ensure IB Gateway is running on "
            f"{ibkr_cfg.host}:{ibkr_cfg.port} and API access is enabled."
        )
        raise SystemExit(1)

    mode = "PAPER" if ibkr_cfg.paper_trading else "LIVE  ⚠  REAL MONEY"
    logger.info(
        f"\n{'='*60}\n"
        f"  ROBO INVESTOR — IBKR LIVE  [{mode}]\n"
        f"  Capital   : ${account_cfg.total_capital:,.0f}\n"
        f"  Tactical  : {tactical_cfg.tickers}\n"
        f"  Experimental: {experimental_cfg.tickers}\n"
        f"  State file: {state_path}\n"
        f"  Log dir   : {log_dir}\n"
        f"{'='*60}"
    )

    daily_summary_sent = False

    try:
        while True:
            now_et = datetime.now(ET)

            if not is_market_open():
                if not daily_summary_sent and now_et.weekday() < 5 and now_et.hour >= 16:
                    print(reporter.daily_summary(state, account_cfg, tax_cfg, now_et))
                    daily_summary_sent = True

                next_check = now_et.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                wait_secs  = (next_check - now_et).total_seconds()
                logger.info(
                    f"Market closed ({now_et.strftime('%d-%b-%y %H:%M')} ET). "
                    f"Next check in {int(wait_secs // 60)} min."
                )
                time.sleep(max(wait_secs, 60))
                continue

            daily_summary_sent = False
            logger.info(f"=== Tick {now_et.strftime('%d-%b-%Y  %H:%M')} ET ===")

            # ── 1. Load historical hourly bars (cache-aware) ──────────── #
            try:
                raw = _fetch_hourly_cached(all_tickers, days=WARMUP_DAYS)
            except Exception as e:
                logger.error(f"Hourly data load failed: {e}. Retrying next hour.")
                time.sleep(3600)
                continue

            if raw.empty:
                logger.warning("Empty hourly data. Retrying next hour.")
                time.sleep(3600)
                continue

            # ── 1b. Load daily bars for knife-guard indicators (cache-aware) #
            try:
                daily_end_str   = datetime.today().strftime("%Y-%m-%d")
                daily_start_str = (datetime.today() - timedelta(days=DAILY_WARMUP_DAYS)).strftime("%Y-%m-%d")
                daily_raw_tick  = _fetch_daily_cached(tactical_cfg.tickers, daily_start_str, daily_end_str)
                tick_daily_maps: dict[str, dict] = {}
                if not daily_raw_tick.empty:
                    today_date = now_et.date()
                    for sym in tactical_cfg.tickers:
                        dmap = _build_daily_close_map(daily_raw_tick, sym, tactical_cfg.tickers)
                        # Exclude today's incomplete bar (market is open)
                        tick_daily_maps[sym] = {d: v for d, v in dmap.items() if d < today_date}
            except Exception as e:
                logger.warning(f"Daily data load failed: {e} — 52W/200MA guards may be skipped.")
                tick_daily_maps = {}

            # ── 2. Get latest prices ──────────────────────────────────── #
            if not broker.is_connected():
                logger.warning("IBKR connection lost — attempting reconnect.")
                if not broker.reconnect():
                    logger.error("Reconnect failed. Retrying next hour.")
                    time.sleep(3600)
                    continue

            current_prices: dict[str, float] = {}
            for sym in all_tickers:
                price = broker.get_latest_price(sym)
                if price:
                    current_prices[sym] = price
                else:
                    # Fall back to last close from yfinance
                    series = _price_series(raw, sym, all_tickers)
                    if series:
                        current_prices[sym] = series[-1]
                        logger.debug(f"IBKR price unavailable for {sym} — using yfinance last close")

            # ── 3. Circuit breaker check ──────────────────────────────── #
            circuit = CircuitBreaker(account_cfg.max_total_drawdown)
            circuit._peak    = state.circuit_breaker_peak
            circuit.tripped  = state.circuit_breaker_tripped
            total_equity = state.tactical.equity + state.experimental.equity
            safe_to_trade = circuit.update(total_equity)
            state.circuit_breaker_peak    = circuit._peak
            state.circuit_breaker_tripped = circuit.tripped

            # ── 4. Generate signals ───────────────────────────────────── #
            actions: list[dict] = []

            if safe_to_trade:
                t_orders = _tactical_signals(
                    state.tactical, tactical_cfg, brokerage_cfg,
                    raw, current_prices, all_tickers, now_et,
                    daily_close_maps=tick_daily_maps,
                )
                e_orders = _experimental_signals(
                    state.experimental, experimental_cfg, brokerage_cfg,
                    raw, current_prices, all_tickers, now_et,
                )
            else:
                t_orders, e_orders = [], []
                logger.critical("Circuit breaker ACTIVE — no new orders this tick.")

            # ── 5. Execute orders and update state ────────────────────── #
            for order in t_orders:
                sym = order["symbol"]
                try:
                    if order["action"] == "buy":
                        fill = broker.place_market_buy(sym, order["shares"])
                    else:
                        fill = broker.place_market_sell(sym, order["shares"])
                    fill_price = fill["price"] if fill["price"] > 0 else order["price"]
                    order["fill_price"] = fill_price
                    ibkr_id = fill.get("order_id")
                except Exception as ex:
                    logger.error(f"Order execution failed for {sym}: {ex}")
                    order["fill_price"] = order["price"]
                    ibkr_id = None

                record = _apply_order(
                    state.tactical, order, order["fill_price"],
                    brokerage_cfg, tactical_cfg, now_et, ibkr_id,
                )
                if record:
                    state.trade_history.append(record)
                    reporter.log_trade(record)
                    actions.append(order)

            for order in e_orders:
                sym = order["symbol"]
                try:
                    if order["action"] == "buy":
                        fill = broker.place_market_buy(sym, order["shares"])
                    else:
                        fill = broker.place_market_sell(sym, order["shares"])
                    fill_price = fill["price"] if fill["price"] > 0 else order["price"]
                    order["fill_price"] = fill_price
                    ibkr_id = fill.get("order_id")
                except Exception as ex:
                    logger.error(f"Order execution failed for {sym}: {ex}")
                    order["fill_price"] = order["price"]
                    ibkr_id = None

                record = _apply_order(
                    state.experimental, order, order["fill_price"],
                    brokerage_cfg, experimental_cfg, now_et, ibkr_id,
                )
                if record:
                    state.trade_history.append(record)
                    reporter.log_trade(record)
                    actions.append(order)

            # ── 6. Persist state ──────────────────────────────────────── #
            state.last_run = now_et
            state.save(state_path)

            # ── 7. Risk report ────────────────────────────────────────── #
            print(reporter.risk_report(state, circuit, actions, now_et))

            # ── 8. Sleep until next hour boundary ────────────────────── #
            next_hour = now_et.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            wait_secs = (next_hour - datetime.now(ET)).total_seconds()
            logger.info(f"Next tick in {int(wait_secs // 60)}m {int(wait_secs % 60)}s.")
            time.sleep(max(wait_secs, 1))

    except KeyboardInterrupt:
        logger.info("Shutdown requested — saving state and disconnecting.")
    finally:
        state.save(state_path)
        broker.disconnect()


def _print_report(account_cfg, tactical, experimental, circuit, all_orders, pdt_warnings, tax_cfg, start, end):
    total_start  = account_cfg.total_capital
    total_end    = tactical.equity + experimental.equity
    total_return = (total_end - total_start) / total_start

    def _return(equity, capital):
        return (equity - capital) / capital if capital > 0 else None

    tactical_return     = _return(tactical.equity,    account_cfg.tactical_capital)
    experimental_return = _return(experimental.equity, account_cfg.experimental_capital)

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
    fmt_ret = lambda r: f"{r:>7.1%}" if r is not None else "    N/A"
    print(f"  {'TACTICAL (swing, RSI)':30s}  ${account_cfg.tactical_capital:>9,.0f}  ${tactical.equity:>9,.0f}  {fmt_ret(tactical_return)}")
    print(f"  {'EXPERIMENTAL (daily, MACD)':30s}  ${account_cfg.experimental_capital:>9,.0f}  ${experimental.equity:>9,.0f}  {fmt_ret(experimental_return)}")
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
