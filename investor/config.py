from dataclasses import dataclass, field


@dataclass
class BrokerageConfig:
    """Interactive Brokers Singapore — US stock fee model.

    Fee : $0.005/share, min $1.00, max 1% of trade value.
    Spread: 0.05% is typical for liquid large-caps (AAPL, MSFT).
            Increase to 0.10–0.30% for volatile names (COIN, MSTR).
    """
    fee_per_share: float = 0.005
    min_fee: float = 1.00
    max_fee_pct: float = 0.01
    spread_pct: float = 0.0005      # 0.05% round-trip half-spread per side

    def calculate_fee(self, shares: int, price: float) -> float:
        trade_value = shares * price
        fee = self.fee_per_share * shares
        fee = max(fee, self.min_fee)
        fee = min(fee, trade_value * self.max_fee_pct)
        return round(fee, 2)

    def effective_buy_price(self, mid: float) -> float:
        """Ask price — you pay slightly above mid when buying."""
        return mid * (1 + self.spread_pct / 2)

    def effective_sell_price(self, mid: float) -> float:
        """Bid price — you receive slightly below mid when selling."""
        return mid * (1 - self.spread_pct / 2)


@dataclass
class TaxConfig:
    """Tax settings for Singapore-based retail investor trading US stocks.

    Singapore has NO capital gains tax for retail investors.
    Set income_tax_rate > 0 only if IRAS classifies your trading as a business.
    """
    capital_gains_rate: float = 0.0    # Singapore: no CGT
    income_tax_rate: float = 0.0       # Set if IRAS treats profits as business income


@dataclass
class AccountConfig:
    """Overall account settings."""
    total_capital: float = 2_000.0
    tactical_pct: float = 0.80
    experimental_pct: float = 0.20
    max_total_drawdown: float = 0.20
    min_cash_reserve_pct: float = 0.10

    @property
    def tactical_capital(self) -> float:
        return self.total_capital * self.tactical_pct

    @property
    def experimental_capital(self) -> float:
        return self.total_capital * self.experimental_pct


@dataclass
class TacticalConfig:
    """Swing trading bucket — RSI Mean Reversion. Hold times in trading hours (1 day = 6.5h)."""
    tickers: list = field(default_factory=lambda: [
        # "TSLA", "MSFT", "COIN", "AMZN", "JPM", "NVDA", "TSLA", "AMD", "META", "COIN", "MSTR", "SNDK", "INTC", "TPL", "STX", "ON", "JBL"
        "NVDA", "TSM", "APH", "AEM", "AGI", "AU", "GFI", "PAAS", "PODD", "SCHW", "ADBE", "ACMR", "ADI", "AVGO"
    ])
    rsi_period: int = 14
    rsi_oversold: int = 35
    rsi_overbought: int = 65
    max_hold_hours: int = 7118      # 1095 trading days × 6.5h
    min_hold_hours: int = 2373      # 365 trading days × 6.5h
    stop_loss_pct: float = 0.20
    take_profit_pct: float = 0.30
    max_position_pct: float = 0.25
    max_positions: int = 4

    # ── Falling knife protection ───────────────────────────────────────── #
    # All six guards must pass before a buy signal is executed.
    # Set a bool flag to False to disable that individual guard.
    #
    # Data sources (two parallel streams):
    #   hourly bars  →  RSI (rules 1, 5), volume (rule 4)
    #   daily  bars  →  200-day MA (rule 2), 52-week high (rule 3)
    #
    # This keeps the precision indicators on the right timeframe: RSI/volume
    # respond quickly to intraday conditions; the trend/crash filters use the
    # canonical calendar-aligned daily closes that every trader looks at.

    # Warm-up gate (daily bars) — no buys until this many trading-day closes
    # are in memory.  Prevents cold-start entries where the 52W-high is
    # computed from only a handful of days and gives a misleading reading
    # (e.g. PODD looked -6% below "peak" on 60-day data; true drop was -56%).
    # 60 trading days ≈ 3 months — sufficient once the engine pre-loads 300
    # calendar days of daily history before the backtest/paper window starts.
    min_daily_history: int = 60

    # Rule 2 — 200-day MA uses daily closes (exactly 200 trading-day bars).
    # Guard is silently skipped when fewer than 200 daily bars are in memory.
    require_price_above_200ma: bool = True

    # Rule 3 — Skip if the stock is down > N % from its 52-week high.
    # Computed from daily closes, last 252 trading-day bars.
    max_drop_from_52w_high: float = 0.20

    # Rule 4 — Volume must confirm the dip (real buyers present, not just sellers).
    # Current bar's volume must be ≥ multiplier × 20-bar rolling average.
    # Uses hourly bars (same stream as RSI).
    require_volume_increase: bool = True
    volume_increase_multiplier: float = 1.0   # 1.0 = at least average volume

    # Rule 5 — Bullish RSI divergence required (momentum slowing before entry).
    # Price makes a lower low while RSI makes a higher low over `lookback` bars.
    # Uses hourly bars.
    require_rsi_divergence: bool = True
    rsi_divergence_lookback: int = 10          # hourly bars to compare

    # Rule 6 — Re-entry cooldown after a stop-loss exit.
    # After being stopped out, wait at least `re_entry_cooldown_days` calendar
    # days before re-entering the same ticker.
    # After `max_re_entries_per_stock` consecutive stop-losses the ticker is
    # blocked from new entries until a profitable exit resets the streak.
    max_re_entries_per_stock: int = 1
    re_entry_cooldown_days: int = 10


@dataclass
class ExperimentalConfig:
    """Daily momentum bucket — MACD + Volume. Hold times in trading hours (1 day = 6.5h)."""
    tickers: list = field(default_factory=lambda: [
        "NOW", "KSPI", "PAYX", "ZBRA", "ADSK", "APH", "SCHW", "FIVE", "PODD"
    ])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_multiplier: float = 1.3
    max_hold_hours: int = 2373      # 365 trading days × 6.5h
    min_hold_hours: int = 65
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    max_position_pct: float = 0.33
    max_positions: int = 3


@dataclass
class IBKRConfig:
    """Interactive Brokers Gateway / TWS connection settings.

    IB Gateway  — paper: port 4002  |  live: port 4001
    TWS         — paper: port 7497  |  live: port 7496
    """
    host: str = "127.0.0.1"
    port: int = 4002           # IB Gateway paper trading default
    client_id: int = 1
    timeout: int = 30          # seconds to wait for connection
    paper_trading: bool = True  # safety flag — set False only for live money
