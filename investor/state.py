"""Persistent state for the live IBKR trading engine.

State is stored as a JSON file between hourly runs so positions, cash
balances, and trade history survive process restarts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import AccountConfig, ExperimentalConfig, TacticalConfig
from .models import Bucket, Position, TradeRecord

logger = logging.getLogger(__name__)

_DT_FMT = "%Y-%m-%dT%H:%M:%S"


def _dt(s: str | None) -> datetime | None:
    return datetime.strptime(s, _DT_FMT) if s else None


def _dts(dt: datetime | None) -> str | None:
    return dt.strftime(_DT_FMT) if dt else None


def _pos_to_dict(pos: Position) -> dict:
    return {
        "symbol": pos.symbol,
        "bucket": pos.bucket.value,
        "entry_price": pos.entry_price,
        "shares": pos.shares,
        "entry_date": _dts(pos.entry_date),
        "stop_loss_pct": pos.stop_loss_pct,
        "take_profit_pct": pos.take_profit_pct,
        "max_hold_hours": pos.max_hold_hours,
        "min_hold_hours": pos.min_hold_hours,
        "bars_held": pos.bars_held,
    }


def _pos_from_dict(d: dict) -> Position:
    p = Position(
        symbol=d["symbol"],
        bucket=Bucket(d["bucket"]),
        entry_price=d["entry_price"],
        shares=d["shares"],
        entry_date=_dt(d["entry_date"]) or datetime.utcnow(),
        stop_loss_pct=d["stop_loss_pct"],
        take_profit_pct=d["take_profit_pct"],
        max_hold_hours=d["max_hold_hours"],
        min_hold_hours=d["min_hold_hours"],
    )
    p.bars_held = d.get("bars_held", 0)
    return p


def _trade_to_dict(t: TradeRecord) -> dict:
    return {
        "timestamp": _dts(t.timestamp),
        "symbol": t.symbol,
        "side": t.side,
        "bucket": t.bucket,
        "shares": t.shares,
        "price": t.price,
        "fee": t.fee,
        "spread_cost": t.spread_cost,
        "pnl": t.pnl,
        "reason": t.reason,
        "ibkr_order_id": t.ibkr_order_id,
    }


def _trade_from_dict(d: dict) -> TradeRecord:
    return TradeRecord(
        timestamp=_dt(d["timestamp"]) or datetime.utcnow(),
        symbol=d["symbol"],
        side=d["side"],
        bucket=d["bucket"],
        shares=d["shares"],
        price=d["price"],
        fee=d["fee"],
        spread_cost=d["spread_cost"],
        pnl=d["pnl"],
        reason=d["reason"],
        ibkr_order_id=d.get("ibkr_order_id"),
    )


@dataclass
class StrategyState:
    cash: float
    realized_pnl: float = 0.0
    total_fees_paid: float = 0.0
    total_spread_cost: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    prev_histogram: dict[str, float | None] = field(default_factory=dict)

    # ── Falling knife protection — persisted across restarts ────────────── #
    # last_stop_loss  : ISO-formatted datetime string of the most recent
    #                   stop-loss exit per ticker, or None if never stopped out.
    # stop_loss_streak: consecutive stop-loss count per ticker; resets to 0 on
    #                   any profitable or time-limit exit.
    last_stop_loss: dict[str, str | None] = field(default_factory=dict)
    stop_loss_streak: dict[str, int] = field(default_factory=dict)

    @property
    def equity(self) -> float:
        pos_value = sum(p.shares * p.entry_price for p in self.positions.values())
        return self.cash + pos_value

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "total_fees_paid": self.total_fees_paid,
            "total_spread_cost": self.total_spread_cost,
            "positions": {s: _pos_to_dict(p) for s, p in self.positions.items()},
            "prev_histogram": self.prev_histogram,
            "last_stop_loss": self.last_stop_loss,
            "stop_loss_streak": self.stop_loss_streak,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StrategyState:
        s = cls(cash=d["cash"])
        s.realized_pnl = d.get("realized_pnl", 0.0)
        s.total_fees_paid = d.get("total_fees_paid", 0.0)
        s.total_spread_cost = d.get("total_spread_cost", 0.0)
        s.positions = {sym: _pos_from_dict(p) for sym, p in d.get("positions", {}).items()}
        s.prev_histogram = d.get("prev_histogram", {})
        s.last_stop_loss = d.get("last_stop_loss", {})
        s.stop_loss_streak = d.get("stop_loss_streak", {})
        return s


@dataclass
class AppState:
    tactical: StrategyState
    experimental: StrategyState
    circuit_breaker_peak: float | None = None
    circuit_breaker_tripped: bool = False
    trade_history: list[TradeRecord] = field(default_factory=list)
    last_run: datetime | None = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_run": _dts(self.last_run),
            "circuit_breaker_peak": self.circuit_breaker_peak,
            "circuit_breaker_tripped": self.circuit_breaker_tripped,
            "tactical": self.tactical.to_dict(),
            "experimental": self.experimental.to_dict(),
            "trade_history": [_trade_to_dict(t) for t in self.trade_history],
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)
        logger.debug(f"State saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> AppState | None:
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls(
                tactical=StrategyState.from_dict(data["tactical"]),
                experimental=StrategyState.from_dict(data["experimental"]),
                circuit_breaker_peak=data.get("circuit_breaker_peak"),
                circuit_breaker_tripped=data.get("circuit_breaker_tripped", False),
                trade_history=[_trade_from_dict(t) for t in data.get("trade_history", [])],
                last_run=_dt(data.get("last_run")),
            )
        except Exception as e:
            logger.error(f"Failed to load state from {path}: {e}. Starting fresh.")
            return None

    @classmethod
    def new(
        cls,
        account_cfg: AccountConfig,
        tactical_cfg: TacticalConfig,
        experimental_cfg: ExperimentalConfig,
    ) -> AppState:
        return cls(
            tactical=StrategyState(
                cash=account_cfg.tactical_capital,
                prev_histogram={},
            ),
            experimental=StrategyState(
                cash=account_cfg.experimental_capital,
                prev_histogram={t: None for t in experimental_cfg.tickers},
            ),
        )
