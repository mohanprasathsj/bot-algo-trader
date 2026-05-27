"""IBKR broker wrapper using ib_insync.

Connects to IB Gateway (or TWS) and executes market orders.
Designed for use in the live hourly loop — connect once, reuse.

IB Gateway ports:
  Paper : 4002 (default)
  Live  : 4001

TWS ports:
  Paper : 7497
  Live  : 7496
"""
from __future__ import annotations

import logging
import time

from .config import IBKRConfig

logger = logging.getLogger(__name__)

_FILL_WAIT = 5   # seconds to wait for market order fill
_SNAP_WAIT = 2   # seconds to wait for price snapshot


class IBKRBroker:
    """Thin synchronous wrapper around ib_insync for market order execution."""

    def __init__(self, config: IBKRConfig):
        try:
            import ib_insync as ibi
            self._ibi = ibi
        except ImportError:
            raise SystemExit("Run: pip install ib_insync")

        self.cfg = config
        self.ib = ibi.IB()
        ibi.util.logToConsole(logging.WARNING)
        self._connected = False

    # ------------------------------------------------------------------ #
    #  Connection                                                           #
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        try:
            self.ib.connect(
                self.cfg.host,
                self.cfg.port,
                clientId=self.cfg.client_id,
                timeout=self.cfg.timeout,
            )
            self._connected = True
            mode = "PAPER" if self.cfg.paper_trading else "LIVE"
            logger.info(
                f"IBKR connected — {self.cfg.host}:{self.cfg.port} "
                f"clientId={self.cfg.client_id} [{mode}]"
            )
            return True
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("IBKR disconnected")

    def is_connected(self) -> bool:
        return self._connected and self.ib.isConnected()

    def reconnect(self) -> bool:
        self.disconnect()
        time.sleep(2)
        return self.connect()

    # ------------------------------------------------------------------ #
    #  Account data                                                         #
    # ------------------------------------------------------------------ #

    def get_net_liquidation(self) -> float:
        """Total account value in USD."""
        for val in self.ib.accountValues():
            if val.tag == "NetLiquidation" and val.currency == "USD":
                return float(val.value)
        return 0.0

    def get_cash_balance(self) -> float:
        """Available cash (TotalCashValue) in USD."""
        for val in self.ib.accountValues():
            if val.tag == "TotalCashValue" and val.currency == "USD":
                return float(val.value)
        return 0.0

    def get_positions(self) -> dict[str, tuple[int, float]]:
        """Returns {symbol: (shares, avg_cost)} for all long stock positions."""
        result: dict[str, tuple[int, float]] = {}
        for pos in self.ib.positions():
            c = pos.contract
            if c.secType == "STK" and pos.position > 0:
                result[c.symbol] = (int(pos.position), float(pos.avgCost))
        return result

    # ------------------------------------------------------------------ #
    #  Market data                                                          #
    # ------------------------------------------------------------------ #

    def get_latest_price(self, symbol: str) -> float | None:
        """Fetch a one-shot price snapshot for a US stock. Returns last/close, or None."""
        contract = self._ibi.Stock(symbol, "SMART", "USD")
        try:
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, "", True, False)
            self.ib.sleep(_SNAP_WAIT)
            self.ib.cancelMktData(contract)

            price = ticker.last or ticker.close or ticker.bid or None
            if price and price > 0:
                return float(price)
        except Exception as e:
            logger.warning(f"Price snapshot failed for {symbol}: {e}")
        return None

    # ------------------------------------------------------------------ #
    #  Order execution                                                      #
    # ------------------------------------------------------------------ #

    def place_market_buy(self, symbol: str, shares: int) -> dict:
        """Place a market BUY. Waits up to _FILL_WAIT seconds for a fill.

        Returns dict with: symbol, side, shares, price, order_id, status.
        """
        return self._execute("BUY", symbol, shares)

    def place_market_sell(self, symbol: str, shares: int) -> dict:
        """Place a market SELL. Waits up to _FILL_WAIT seconds for a fill."""
        return self._execute("SELL", symbol, shares)

    def _execute(self, side: str, symbol: str, shares: int) -> dict:
        contract = self._ibi.Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        order = self._ibi.MarketOrder(side, shares)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(_FILL_WAIT)

        fill_price = 0.0
        filled_shares = 0
        for fill in trade.fills:
            fill_price = fill.execution.price
            filled_shares += int(fill.execution.shares)

        result = {
            "symbol": symbol,
            "side": side.lower(),
            "shares": filled_shares if filled_shares else shares,
            "price": fill_price,
            "order_id": trade.order.orderId,
            "status": trade.orderStatus.status,
        }
        logger.info(
            f"ORDER {side} {symbol}: {result['shares']} shares "
            f"@ ${result['price']:.2f}  orderId={result['order_id']}  "
            f"status={result['status']}"
        )
        return result
