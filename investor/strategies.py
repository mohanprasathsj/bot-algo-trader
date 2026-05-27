from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from .config import BrokerageConfig, ExperimentalConfig, TacticalConfig
from .indicators import (
    compute_macd,
    compute_rsi,
    compute_sma,
    detect_rsi_divergence,
)
from .models import Bucket, Position

logger = logging.getLogger(__name__)


class TacticalStrategy:
    """
    RSI Mean Reversion swing trader with falling-knife protection.

    Two data streams feed the six entry guards:
      hourly bars → RSI signal (rule 1), volume confirmation (rule 4),
                    RSI divergence (rule 5), cooldown tracking (rule 6)
      daily  bars → 200-day MA (rule 2), 52-week high (rule 3)

    ENTRY : ALL six guards must pass —
              1. RSI < oversold threshold               (beaten-down signal)
              2. Price > 200-day MA  [daily close]      (still in uptrend)
              3. < 20 % below 52-week high  [daily]     (not in free-fall)
              4. Volume ≥ rolling-20 avg  [hourly]      (real buyers present)
              5. Bullish RSI divergence  [hourly]       (momentum slowing)
              6. Cooldown after any stop-loss           (no knife-catching loop)
    EXIT  : RSI > overbought, OR stop-loss, OR take-profit, OR max hold
    """

    def __init__(self, config: TacticalConfig, capital: float, brokerage: BrokerageConfig):
        self.cfg = config
        self.capital = capital
        self.brokerage = brokerage
        # hourly bars — RSI, volume, divergence
        self.prices:        dict[str, list[float]] = {t: [] for t in config.tickers}
        self.volumes:       dict[str, list[float]] = {t: [] for t in config.tickers}
        # daily bars — 200-day MA and 52W high (fed via on_daily_bar)
        self.daily_prices:  dict[str, list[float]] = {t: [] for t in config.tickers}
        self.positions: dict[str, Position] = {}
        self.cash = capital
        self.total_fees_paid:   float = 0.0
        self.total_spread_cost: float = 0.0
        self.realized_pnl:      float = 0.0

        # Falling knife protection state — persisted via StrategyState in live mode
        self.last_stop_loss:   dict[str, datetime | None] = {t: None for t in config.tickers}
        self.stop_loss_streak: dict[str, int]             = {t: 0    for t in config.tickers}

    def on_daily_bar(self, symbol: str, close: float) -> None:
        """Record one end-of-day close for the 200-day MA and 52W-high guards.

        Called by the engine once per trading day (at the start of the next
        trading day's first hourly bar) so there is zero look-ahead bias.
        """
        if symbol in self.daily_prices:
            self.daily_prices[symbol].append(close)

    # ------------------------------------------------------------------ #
    #  Falling knife guard                                                 #
    # ------------------------------------------------------------------ #

    def _knife_guard(
        self, symbol: str, close: float, today: datetime
    ) -> tuple[bool, str]:
        """Return (True, reason) when the trade should be skipped.

        Rules 2 & 3 use daily_prices (calendar-aligned); rules 4 & 5 use the
        hourly prices/volumes.  Any check that lacks enough data is silently
        skipped so the strategy degrades gracefully during warm-up.
        """
        prices = self.prices[symbol]          # hourly
        vols   = self.volumes[symbol]         # hourly
        daily  = self.daily_prices[symbol]    # daily (fed by engine via on_daily_bar)

        # Rule 0 — Daily warm-up gate.
        # 52W-high and 200-day MA are meaningless on a handful of days.
        # The engine pre-loads 300 calendar days of daily history before the
        # first bar, so this gate passes once ~60 trading days have been seen.
        if len(daily) < self.cfg.min_daily_history:
            return True, (
                f"building daily history "
                f"({len(daily)}/{self.cfg.min_daily_history} trading days)"
            )

        # Rule 2 — 200-day MA from daily closes (uptrend filter).
        # Silently skipped when fewer than 200 daily bars are available.
        if self.cfg.require_price_above_200ma and len(daily) >= 200:
            ma200 = compute_sma(daily, 200)
            if ma200 is not None and close < ma200:
                return True, f"below 200-day MA (${close:.2f} < ${ma200:.2f})"

        # Rule 3 — Not more than N % below 52-week high (daily closes, last 252 bars).
        window_52w = daily[-252:] if len(daily) >= 252 else daily
        high_52w   = float(max(window_52w))
        drop       = (close - high_52w) / high_52w
        if drop < -self.cfg.max_drop_from_52w_high:
            return True, (
                f"≥{self.cfg.max_drop_from_52w_high:.0%} below 52W high "
                f"(${close:.2f} vs ${high_52w:.2f}, {drop:.1%})"
            )

        # Rule 4 — Volume must confirm the dip (buyers present, not just sellers).
        # Uses hourly bars — same stream as RSI.
        if self.cfg.require_volume_increase and len(vols) >= 20:
            avg_vol     = float(np.mean(vols[-20:]))
            current_vol = vols[-1]
            threshold   = avg_vol * self.cfg.volume_increase_multiplier
            if avg_vol > 0 and current_vol < threshold:
                return True, (
                    f"volume below threshold "
                    f"({current_vol:,.0f} < {threshold:,.0f} = "
                    f"{self.cfg.volume_increase_multiplier:.1f}× avg)"
                )

        # Rule 5 — Bullish RSI divergence on hourly bars (momentum slowing).
        if self.cfg.require_rsi_divergence:
            min_len = self.cfg.rsi_period + 1 + self.cfg.rsi_divergence_lookback
            if len(prices) >= min_len:
                if not detect_rsi_divergence(
                    prices, self.cfg.rsi_period, self.cfg.rsi_divergence_lookback
                ):
                    return True, "no bullish RSI divergence"

        # Rule 6a — Cooldown: wait N calendar days after a stop-loss exit.
        last_sl = self.last_stop_loss.get(symbol)
        if last_sl is not None:
            days_since = (today - last_sl).days
            if days_since < self.cfg.re_entry_cooldown_days:
                remaining = self.cfg.re_entry_cooldown_days - days_since
                return True, (
                    f"stop-loss cooldown ({days_since}d elapsed, "
                    f"{remaining}d remaining)"
                )

        # Rule 6b — Block re-entry when consecutive stop-loss streak is exhausted.
        streak = self.stop_loss_streak.get(symbol, 0)
        if streak > self.cfg.max_re_entries_per_stock:
            return True, (
                f"max re-entries exceeded "
                f"({streak} consecutive stop-losses, "
                f"limit={self.cfg.max_re_entries_per_stock})"
            )

        return False, ""

    # ------------------------------------------------------------------ #
    #  Main bar handler                                                    #
    # ------------------------------------------------------------------ #

    def on_bar(
        self, symbol: str, close: float, today: datetime, volume: float = 0.0
    ) -> list[dict]:
        """Process one hourly bar.

        `volume` defaults to 0.0 for callers that do not supply it; the
        volume guard is effectively skipped when no volumes have been recorded.
        """
        if symbol not in self.prices:
            return []

        self.prices[symbol].append(close)
        self.volumes[symbol].append(volume)
        orders = []

        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.bars_held += 1
            exit_flag, reason = pos.should_exit(close)

            rsi = compute_rsi(self.prices[symbol], self.cfg.rsi_period)
            if rsi and rsi > self.cfg.rsi_overbought and pos.bars_held >= self.cfg.min_hold_hours:
                exit_flag, reason = True, f"RSI overbought ({rsi:.0f})"

            if exit_flag:
                sell_price  = self.brokerage.effective_sell_price(close)
                fee         = self.brokerage.calculate_fee(pos.shares, sell_price)
                spread_cost = pos.shares * close * self.brokerage.spread_pct / 2
                proceeds    = pos.shares * sell_price - fee
                self.cash  += proceeds
                self.total_fees_paid   += fee
                self.total_spread_cost += spread_cost
                pnl = proceeds - (pos.shares * pos.entry_price)
                self.realized_pnl += pnl

                # ── Track stop-loss streak for re-entry cooldown ── #
                if "stop-loss" in reason:
                    self.last_stop_loss[symbol]   = today
                    self.stop_loss_streak[symbol] = self.stop_loss_streak.get(symbol, 0) + 1
                    streak_note = f" [stop streak={self.stop_loss_streak[symbol]}]"
                else:
                    self.stop_loss_streak[symbol] = 0   # profitable / time exit resets streak
                    streak_note = ""

                logger.info(
                    f"[TACTICAL] EXIT  {symbol:6s} | {today.strftime('%d-%b-%y %H:%M')} | {reason:25s} | "
                    f"held={pos.bars_held}h | PnL=${pnl:+.2f} | fee=${fee:.2f} | spread=${spread_cost:.2f} | "
                    f"cash=${self.cash:,.0f}{streak_note}"
                )
                orders.append({"action": "sell", "symbol": symbol, "date": today,
                                "shares": pos.shares, "price": sell_price, "bucket": "tactical",
                                "fee": fee, "spread_cost": spread_cost, "pnl": pnl})
                del self.positions[symbol]
                self.capital = self.equity

        elif len(self.positions) < self.cfg.max_positions:
            rsi = compute_rsi(self.prices[symbol], self.cfg.rsi_period)
            if rsi is not None and rsi < self.cfg.rsi_oversold:

                # ── Falling knife protection — all 6 guards ─────── #
                blocked, block_reason = self._knife_guard(symbol, close, today)
                if blocked:
                    logger.debug(
                        f"[TACTICAL] SKIP  {symbol:6s} | {today.strftime('%d-%b-%y %H:%M')} | "
                        f"RSI={rsi:.0f} oversold but blocked: {block_reason}"
                    )
                else:
                    max_spend   = self.capital * self.cfg.max_position_pct
                    reserve     = self.capital * 0.10
                    available   = max(0.0, self.cash - reserve)
                    spend       = min(max_spend, available)
                    buy_price   = self.brokerage.effective_buy_price(close)
                    shares      = int(spend / buy_price)

                    if shares >= 1:
                        fee         = self.brokerage.calculate_fee(shares, buy_price)
                        spread_cost = shares * close * self.brokerage.spread_pct / 2
                        cost        = shares * buy_price + fee
                        self.cash  -= cost
                        self.total_fees_paid   += fee
                        self.total_spread_cost += spread_cost
                        pos = Position(
                            symbol=symbol, bucket=Bucket.TACTICAL,
                            entry_price=buy_price, shares=shares, entry_date=today,
                            stop_loss_pct=self.cfg.stop_loss_pct,
                            take_profit_pct=self.cfg.take_profit_pct,
                            max_hold_hours=self.cfg.max_hold_hours,
                            min_hold_hours=self.cfg.min_hold_hours,
                        )
                        self.positions[symbol] = pos
                        logger.info(
                            f"[TACTICAL] ENTRY {symbol:6s} | {today.strftime('%d-%b-%y %H:%M')} | "
                            f"RSI={rsi:.0f} (oversold) | {shares} shares @ ${buy_price:.2f} | "
                            f"fee=${fee:.2f} | spread=${spread_cost:.2f} | cash=${self.cash:,.0f}"
                        )
                        orders.append({"action": "buy", "symbol": symbol, "date": today,
                                       "shares": shares, "price": buy_price, "bucket": "tactical",
                                       "fee": fee, "spread_cost": spread_cost, "pnl": 0.0})

        return orders

    @property
    def equity(self) -> float:
        pos_value = sum(p.shares * p.entry_price for p in self.positions.values())
        return self.cash + pos_value


class ExperimentalStrategy:
    """
    MACD Momentum with volume confirmation.

    ENTRY : MACD histogram turns positive AND volume > 30% above 20-day avg
    EXIT  : MACD histogram turns negative, OR stop-loss/take-profit/max hold
    """

    def __init__(self, config: ExperimentalConfig, capital: float, brokerage: BrokerageConfig):
        self.cfg = config
        self.capital = capital
        self.brokerage = brokerage
        self.prices: dict[str, list[float]]  = {t: [] for t in config.tickers}
        self.volumes: dict[str, list[float]] = {t: [] for t in config.tickers}
        self.prev_histogram: dict[str, float | None] = {t: None for t in config.tickers}
        self.positions: dict[str, Position] = {}
        self.cash = capital
        self.total_fees_paid: float = 0.0
        self.total_spread_cost: float = 0.0
        self.realized_pnl: float = 0.0

    def on_bar(self, symbol: str, close: float, volume: float, today: datetime) -> list[dict]:
        if symbol not in self.prices:
            return []

        self.prices[symbol].append(close)
        self.volumes[symbol].append(volume)
        orders = []

        macd, signal_line, histogram = compute_macd(
            self.prices[symbol],
            self.cfg.macd_fast, self.cfg.macd_slow, self.cfg.macd_signal,
        )

        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.bars_held += 1
            exit_flag, reason = pos.should_exit(close)

            if histogram is not None and histogram < 0 and pos.bars_held >= 1:
                exit_flag, reason = True, "MACD momentum lost"

            if exit_flag:
                sell_price  = self.brokerage.effective_sell_price(close)
                fee         = self.brokerage.calculate_fee(pos.shares, sell_price)
                spread_cost = pos.shares * close * self.brokerage.spread_pct / 2
                proceeds    = pos.shares * sell_price - fee
                self.cash  += proceeds
                self.total_fees_paid   += fee
                self.total_spread_cost += spread_cost
                pnl = proceeds - (pos.shares * pos.entry_price)
                self.realized_pnl += pnl
                logger.info(
                    f"[EXPERIM] EXIT  {symbol:6s} | {today.strftime('%d-%b-%y %H:%M')} | {reason:25s} | "
                    f"held={pos.bars_held}h | PnL=${pnl:+.2f} | fee=${fee:.2f} | spread=${spread_cost:.2f} | cash=${self.cash:,.0f}"
                )
                orders.append({"action": "sell", "symbol": symbol, "date": today,
                                "shares": pos.shares, "price": sell_price, "bucket": "experimental",
                                "fee": fee, "spread_cost": spread_cost, "pnl": pnl})
                del self.positions[symbol]
                self.capital = self.equity

        elif len(self.positions) < self.cfg.max_positions:
            prev_hist = self.prev_histogram.get(symbol)

            vols = self.volumes[symbol]
            avg_volume = np.mean(vols[-20:]) if len(vols) >= 20 else None
            volume_confirmed = (
                avg_volume is not None and volume > avg_volume * self.cfg.volume_multiplier
            )
            macd_bullish = (
                histogram is not None and
                prev_hist is not None and
                prev_hist <= 0 and histogram > 0
            )

            if macd_bullish and volume_confirmed:
                max_spend = self.capital * self.cfg.max_position_pct
                reserve   = self.capital * 0.10
                available = max(0.0, self.cash - reserve)
                spend     = min(max_spend, available)
                buy_price = self.brokerage.effective_buy_price(close)
                shares    = int(spend / buy_price)

                if shares >= 1:
                    fee         = self.brokerage.calculate_fee(shares, buy_price)
                    spread_cost = shares * close * self.brokerage.spread_pct / 2
                    cost        = shares * buy_price + fee
                    self.cash  -= cost
                    self.total_fees_paid   += fee
                    self.total_spread_cost += spread_cost
                    pos = Position(
                        symbol=symbol, bucket=Bucket.EXPERIMENTAL,
                        entry_price=buy_price, shares=shares, entry_date=today,
                        stop_loss_pct=self.cfg.stop_loss_pct,
                        take_profit_pct=self.cfg.take_profit_pct,
                        max_hold_hours=self.cfg.max_hold_hours,
                        min_hold_hours=self.cfg.min_hold_hours,
                    )
                    self.positions[symbol] = pos
                    vol_ratio = volume / avg_volume
                    logger.info(
                        f"[EXPERIM] ENTRY {symbol:6s} | {today.strftime('%d-%b-%y %H:%M')} | MACD bullish + vol={vol_ratio:.1f}x avg | "
                        f"{shares} shares @ ${buy_price:.2f} | fee=${fee:.2f} | spread=${spread_cost:.2f} | cash=${self.cash:,.0f}"
                    )
                    orders.append({"action": "buy", "symbol": symbol, "date": today,
                                   "shares": shares, "price": buy_price, "bucket": "experimental",
                                   "fee": fee, "spread_cost": spread_cost, "pnl": 0.0})

        if histogram is not None:
            self.prev_histogram[symbol] = histogram

        return orders

    @property
    def equity(self) -> float:
        pos_value = sum(p.shares * p.entry_price for p in self.positions.values())
        return self.cash + pos_value
