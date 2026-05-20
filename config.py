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
    tactical_pct: float = 0.70
    experimental_pct: float = 0.30
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
        "AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "NVDA", "TSLA", "AMD", "META", "COIN", "MSTR"
    ])
    rsi_period: int = 14
    rsi_oversold: int = 35
    rsi_overbought: int = 65
    max_hold_hours: int = 7118      # 1095 trading days × 6.5h
    min_hold_hours: int = 2373      # 365 trading days × 6.5h
    stop_loss_pct: float = 0.10
    take_profit_pct: float = 0.30
    max_position_pct: float = 0.25
    max_positions: int = 4


@dataclass
class ExperimentalConfig:
    """Daily momentum bucket — MACD + Volume. Hold times in trading hours (1 day = 6.5h)."""
    tickers: list = field(default_factory=lambda: [
        "NVDA", "TSLA", "AMD", "META", "COIN", "MSTR"
    ])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_multiplier: float = 1.3
    max_hold_hours: int = 2373      # 365 trading days × 6.5h
    min_hold_hours: int = 0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    max_position_pct: float = 0.33
    max_positions: int = 3
