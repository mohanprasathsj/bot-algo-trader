from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class Bucket(Enum):
    TACTICAL     = "tactical"
    EXPERIMENTAL = "experimental"


@dataclass
class Position:
    symbol: str
    bucket: Bucket
    entry_price: float
    shares: int
    entry_date: datetime
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_hours: int
    min_hold_hours: int = 0
    bars_held: int = field(default=0, init=False)  # incremented each hourly bar

    @property
    def stop_price(self) -> float:
        return self.entry_price * (1 - self.stop_loss_pct)

    @property
    def target_price(self) -> float:
        return self.entry_price * (1 + self.take_profit_pct)

    def should_exit(self, current_price: float) -> tuple[bool, str]:
        if current_price <= self.stop_price:
            loss_pct = (current_price - self.entry_price) / self.entry_price
            return True, f"stop-loss ({loss_pct:.1%})"

        if current_price >= self.target_price:
            gain_pct = (current_price - self.entry_price) / self.entry_price
            return True, f"take-profit ({gain_pct:.1%})"

        if self.bars_held >= self.max_hold_hours:
            return True, f"max hold ({self.bars_held}h)"

        return False, ""


@dataclass
class TradeRecord:
    """Immutable record of a single executed trade (buy or sell)."""
    timestamp: datetime
    symbol: str
    side: str            # "buy" | "sell"
    bucket: str          # "tactical" | "experimental"
    shares: int
    price: float         # actual fill price
    fee: float
    spread_cost: float
    pnl: float           # realized PnL (0.0 for buys)
    reason: str          # signal reason, e.g. "RSI oversold (32.1)"
    ibkr_order_id: int | None = None


class CircuitBreaker:
    """
    Trips if total equity drops max_drawdown from its peak.
    Once tripped, no new trades are entered.
    """

    def __init__(self, max_drawdown: float):
        self.max_drawdown = max_drawdown
        self._peak: float | None = None
        self.tripped = False

    def update(self, equity: float) -> bool:
        """Returns True if safe to trade."""
        if self.tripped:
            return False
        if self._peak is None or equity > self._peak:
            self._peak = equity
        drawdown = (self._peak - equity) / self._peak
        if drawdown >= self.max_drawdown:
            logger.critical(
                f"CIRCUIT BREAKER TRIPPED — drawdown={drawdown:.1%} "
                f"(peak=${self._peak:,.0f} → now=${equity:,.0f}). "
                "No new trades will open."
            )
            self.tripped = True
        return not self.tripped
