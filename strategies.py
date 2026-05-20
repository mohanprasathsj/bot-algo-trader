from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from config import BrokerageConfig, ExperimentalConfig, TacticalConfig
from indicators import compute_macd, compute_rsi
from models import Bucket, Position

logger = logging.getLogger(__name__)


class TacticalStrategy:
    """
    RSI Mean Reversion swing trader.

    ENTRY : RSI < oversold threshold  →  stock is beaten down, expect bounce
    EXIT  : RSI > overbought threshold, OR stop-loss, OR take-profit, OR max hold
    """

    def __init__(self, config: TacticalConfig, capital: float, brokerage: BrokerageConfig):
        self.cfg = config
        self.capital = capital
        self.brokerage = brokerage
        self.prices: dict[str, list[float]] = {t: [] for t in config.tickers}
        self.positions: dict[str, Position] = {}
        self.cash = capital
        self.total_fees_paid: float = 0.0
        self.total_spread_cost: float = 0.0
        self.realized_pnl: float = 0.0

    def on_bar(self, symbol: str, close: float, today: datetime) -> list[dict]:
        if symbol not in self.prices:
            return []

        self.prices[symbol].append(close)
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
                logger.info(
                    f"[TACTICAL] EXIT  {symbol:6s} | {today.strftime('%d-%b-%y %H:%M')} | {reason:25s} | "
                    f"held={pos.bars_held}h | PnL=${pnl:+.2f} | fee=${fee:.2f} | spread=${spread_cost:.2f} | cash=${self.cash:,.0f}"
                )
                orders.append({"action": "sell", "symbol": symbol, "date": today,
                                "shares": pos.shares, "price": sell_price, "bucket": "tactical",
                                "fee": fee, "spread_cost": spread_cost, "pnl": pnl})
                del self.positions[symbol]
                self.capital = self.equity

        elif len(self.positions) < self.cfg.max_positions:
            rsi = compute_rsi(self.prices[symbol], self.cfg.rsi_period)
            if rsi is not None and rsi < self.cfg.rsi_oversold:
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
                        f"[TACTICAL] ENTRY {symbol:6s} | {today.strftime('%d-%b-%y %H:%M')} | RSI={rsi:.0f} (oversold) | "
                        f"{shares} shares @ ${buy_price:.2f} | fee=${fee:.2f} | spread=${spread_cost:.2f} | cash=${self.cash:,.0f}"
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
