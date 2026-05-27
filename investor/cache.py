"""Local OHLCV disk cache — per-ticker Parquet files.

On the first run for a ticker the full history is downloaded (slow, one-off).
Every subsequent run only fetches the missing delta — usually 1 trading day —
and merges it into the cached file before returning.

Cache layout
------------
    investor/data/cache/hourly/<TICKER>.parquet   ← timezone-aware index
    investor/data/cache/daily/<TICKER>.parquet    ← date-indexed (tz-naive)

The cache grows monotonically: new rows are merged with a 2-day backward
overlap (to catch any late yfinance adjustments) and deduplicated by
timestamp (newest value wins).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_ROOT = Path(__file__).parent / "data" / "cache"


# ── Private helpers ──────────────────────────────────────────────────────── #

def _cache_path(ticker: str, interval: str) -> Path:
    subdir = "hourly" if interval == "1h" else "daily"
    return _CACHE_ROOT / subdir / f"{ticker}.parquet"


def _load(ticker: str, interval: str) -> pd.DataFrame | None:
    """Return the cached DataFrame, or None on miss / read error."""
    path = _cache_path(ticker, interval)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        return df if not df.empty else None
    except Exception as exc:
        logger.warning(f"[cache] Read failed for {path.name}: {exc} — will re-download")
        return None


def _save(ticker: str, interval: str, df: pd.DataFrame) -> None:
    """Atomic write: write to .tmp, then rename — avoids corrupted files."""
    path = _cache_path(ticker, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    try:
        df.to_parquet(tmp)
        tmp.rename(path)
        logger.debug(f"[cache] Saved {len(df)} rows → {path.name}")
    except Exception as exc:
        logger.warning(f"[cache] Write failed for {path.name}: {exc}")
        if tmp.exists():
            tmp.unlink()


def _download_single(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame | None:
    """Download one ticker from yfinance and flatten any MultiIndex columns."""
    try:
        import yfinance as yf
        df = yf.download(
            ticker, start=start, end=end,
            interval=interval, progress=False, auto_adjust=True,
        )
        if df is None or df.empty:
            return None
        # yfinance ≥ 0.2 wraps single-ticker downloads in a MultiIndex too.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df
    except Exception as exc:
        logger.warning(f"[cache] yfinance download failed ({ticker}, {interval}): {exc}")
        return None


def _merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Concatenate, deduplicate by index (newer value wins), sort."""
    merged = pd.concat([existing, new])
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.sort_index(inplace=True)
    return merged


def _normalise_ts(ts: datetime | pd.Timestamp, ref_index: pd.Index) -> pd.Timestamp:
    """Return a Timestamp with the same timezone as ref_index (or tz-naive)."""
    ts = pd.Timestamp(ts)
    if hasattr(ref_index, "tz") and ref_index.tz is not None:
        return ts.tz_localize(ref_index.tz) if ts.tzinfo is None else ts.tz_convert(ref_index.tz)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


# ── Public API ───────────────────────────────────────────────────────────── #

def get_ticker_ohlcv(
    ticker: str,
    interval: str,          # "1h" or "1d"
    start: datetime,
    end: datetime,
) -> pd.DataFrame | None:
    """Return OHLCV DataFrame for *ticker* covering [start, end).

    Cache miss  →  full download, cached, returned.
    Cache hit   →  only the missing tail is fetched, merged, re-saved, sliced.

    The function also extends the cache backwards if *start* predates the
    earliest cached bar (e.g. a backtest requesting older history).
    """
    cached = _load(ticker, interval)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    # ── Cold start: nothing in cache yet ────────────────────────────────── #
    if cached is None:
        logger.info(
            f"[cache] COLD  {ticker:6s} ({interval}): "
            f"downloading {start_str} → {end_str}"
        )
        df = _download_single(ticker, start_str, end_str, interval)
        if df is not None:
            _save(ticker, interval, df)
        return df

    cache_start = cached.index.min()
    cache_end   = cached.index.max()

    # ── Extend backward if start predates the earliest cached bar ────────── #
    # Skip if the gap is ≤ 5 calendar days — almost certainly just a
    # weekend or market holiday that yfinance would return nothing for anyway.
    req_start_ts = _normalise_ts(pd.Timestamp(start).normalize(), cached.index)
    gap_days = (pd.Timestamp(cache_start).normalize() - req_start_ts).days
    if gap_days > 5:
        hist_end = (pd.Timestamp(cache_start).normalize() + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(
            f"[cache] BACK  {ticker:6s} ({interval}): "
            f"extending history {start_str} → {hist_end}"
        )
        historical = _download_single(ticker, start_str, hist_end, interval)
        if historical is not None and not historical.empty:
            cached = _merge(cached, historical)
            _save(ticker, interval, cached)
            cache_end = cached.index.max()

    # ── Extend forward: fetch the tail since the last cached bar ─────────── #
    # Use a 2-day overlap to catch any retroactive yfinance split/dividend adjustments.
    fetch_from     = (pd.Timestamp(cache_end).normalize() - timedelta(days=2))
    fetch_from_str = fetch_from.strftime("%Y-%m-%d")
    req_end_ts     = _normalise_ts(end, cached.index)

    if fetch_from < req_end_ts:
        logger.info(
            f"[cache] DELTA {ticker:6s} ({interval}): "
            f"fetching {fetch_from_str} → {end_str}"
        )
        delta = _download_single(ticker, fetch_from_str, end_str, interval)
        if delta is not None and not delta.empty:
            cached = _merge(cached, delta)
            _save(ticker, interval, cached)
        else:
            logger.debug(f"[cache] {ticker} ({interval}): no new bars returned by yfinance")
    else:
        logger.debug(f"[cache] HIT   {ticker:6s} ({interval}): fully up-to-date, no fetch needed")

    # ── Slice to the requested window ────────────────────────────────────── #
    # Always floor to day boundaries so that a time component in `start`/`end`
    # (e.g. datetime.today() = "2024-05-27 10:30") doesn't accidentally exclude
    # bars whose index timestamp is at midnight on the same calendar day.
    #   s_ts  = midnight of start date         (inclusive)
    #   e_ts  = midnight of end date + 1 day   (exclusive — covers the full end day)
    idx   = cached.index
    s_ts  = _normalise_ts(pd.Timestamp(start).normalize(),                      idx)
    e_ts  = _normalise_ts(pd.Timestamp(end).normalize() + pd.Timedelta(days=1), idx)
    sliced = cached.loc[(idx >= s_ts) & (idx < e_ts)]
    return sliced if not sliced.empty else None


def cache_summary() -> str:
    """One-line summary of cached files (shown at engine startup)."""
    lines = []
    for interval in ("hourly", "daily"):
        subdir = _CACHE_ROOT / interval
        if not subdir.exists():
            continue
        files = sorted(subdir.glob("*.parquet"))
        if not files:
            continue
        total_rows = 0
        tickers    = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                total_rows += len(df)
                tickers.append(f.stem)
            except Exception:
                pass
        lines.append(
            f"  {interval:7s}: {len(files)} ticker(s) cached "
            f"({total_rows:,} rows total) — {', '.join(tickers)}"
        )
    return "\n".join(lines) if lines else "  (no cache yet — first run will be slow)"
