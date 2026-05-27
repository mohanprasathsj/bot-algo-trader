# RoboAdvisor

A personal learning project for exploring trading concepts and identifying investment opportunities. It contains two independent modules:

- **`investor/`** — Automated trading bot: backtests and runs RSI + MACD strategies on US stocks using hourly bars
- **`advisor/`** — AI recommendation engine: fetches live market data and asks Google Gemini for Buy / Hold / Sell signals

---

## Setup

**Requirements:** Python 3.10+

```bash
# 1. Clone and enter the project
cd RoboAdvisor

# 2. Create the virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and fill in your keys (see API Keys section below)
```

---

## Module 1 — Investor (Backtester / Live Bot)

Runs two parallel strategies on hourly OHLCV data fetched from Yahoo Finance (no API key needed).

### Strategies

| Bucket | Allocation | Strategy | Entry | Exit |
|---|---|---|---|---|
| Tactical | 70% of capital | RSI Mean Reversion | RSI < 35 (oversold) | RSI > 65, stop-loss, take-profit, or max hold |
| Experimental | 30% of capital | MACD + Volume Momentum | MACD histogram turns positive + volume spike | MACD turns negative, stop-loss, take-profit |

### How to Run

```bash
# Backtest (default: 2022-01-01 → 2023-01-01)
python -m investor.main --mode backtest

# Backtest with custom date range and capital
python -m investor.main --mode backtest --start 2023-01-01 --end 2024-01-01 --capital 5000

# Paper trade (replay last 60 days of real data)
python -m investor.main --mode paper

# Live trading loop (fires every hour during US market hours)
python -m investor.main --mode live
```

### Configuration — `investor/config.py`

#### `AccountConfig`
| Parameter | Default | Description |
|---|---|---|
| `total_capital` | `2000.0` | Total USD allocated to the bot |
| `tactical_pct` | `0.70` | Fraction of capital for the tactical bucket (70%) |
| `experimental_pct` | `0.30` | Fraction of capital for the experimental bucket (30%) |
| `max_total_drawdown` | `0.20` | Circuit-breaker trips if portfolio drops 20% from peak |
| `min_cash_reserve_pct` | `0.10` | Always keep 10% of capital as cash |

#### `TacticalConfig` (RSI Swing Trader)
| Parameter | Default | Description |
|---|---|---|
| `tickers` | AAPL, MSFT, GOOGL, AMZN, JPM, JNJ, NVDA, TSLA, AMD, META, COIN, MSTR | Watchlist |
| `rsi_period` | `14` | RSI lookback period |
| `rsi_oversold` | `35` | Buy signal threshold |
| `rsi_overbought` | `65` | Sell signal threshold |
| `stop_loss_pct` | `0.10` | Exit if price drops 10% from entry |
| `take_profit_pct` | `0.30` | Exit if price rises 30% from entry |
| `max_position_pct` | `0.25` | Max 25% of tactical capital per position |
| `max_positions` | `4` | Max concurrent open positions |
| `max_hold_hours` | `7118` | Force exit after ~1095 trading days |
| `min_hold_hours` | `2373` | Don't exit RSI signal before ~365 trading days |

#### `ExperimentalConfig` (MACD Momentum)
| Parameter | Default | Description |
|---|---|---|
| `tickers` | NVDA, TSLA, AMD, META, COIN, MSTR | High-volatility watchlist |
| `macd_fast` | `12` | MACD fast EMA period |
| `macd_slow` | `26` | MACD slow EMA period |
| `macd_signal` | `9` | MACD signal line period |
| `volume_multiplier` | `1.3` | Entry requires volume > 1.3× 20-day average |
| `stop_loss_pct` | `0.02` | Tight 2% stop-loss |
| `take_profit_pct` | `0.05` | 5% take-profit target |
| `max_position_pct` | `0.33` | Max 33% of experimental capital per position |
| `max_positions` | `3` | Max concurrent open positions |
| `max_hold_hours` | `2373` | Force exit after ~365 trading days |
| `min_hold_hours` | `65` | Min hold of ~10 trading days |

#### `BrokerageConfig` (IBKR Singapore)
| Parameter | Default | Description |
|---|---|---|
| `fee_per_share` | `$0.005` | Commission per share |
| `min_fee` | `$1.00` | Minimum commission per trade |
| `max_fee_pct` | `0.01` | Commission capped at 1% of trade value |
| `spread_pct` | `0.0005` | 0.05% bid-ask half-spread per side |

#### `TaxConfig` (Singapore defaults)
| Parameter | Default | Description |
|---|---|---|
| `capital_gains_rate` | `0.0` | Singapore has no capital gains tax |
| `income_tax_rate` | `0.0` | Set > 0 if IRAS classifies your trading as a business |

---

## Module 2 — Advisor (AI Recommendations)

Fetches live price data (Polygon), news sentiment (Finnhub), and macro context (FRED), then asks Google Gemini to produce structured Buy / Hold / Sell recommendations.

### How to Run

```bash
# Analyse all 12 sector ETFs (default)
python -m advisor.main

# Analyse specific tickers
python -m advisor.main NVDA AAPL MSFT

# Save results to a JSON file
python -m advisor.main NVDA AAPL --output recs.json

# Use a different Gemini model
python -m advisor.main NVDA --model gemini-2.0-pro

# Verbose / debug output
python -m advisor.main NVDA -v
```

### Output

Each ticker gets a structured recommendation:

```
▲ NVDA   BUY   [HIGH]
   Strong momentum with price 8.2% above SMA-20; MACD bullish cross confirmed
   by 1.4× average volume. Positive macro backdrop (Fed pause, low VIX).
   Risk: Stretched valuation; any guidance miss could trigger sharp reversal.
```

---

## API Keys

Add these to your `.env` file at the project root:

| Key | Used By | Where to Get |
|---|---|---|
| `GEMINI_API_KEY` | advisor | [aistudio.google.com](https://aistudio.google.com) → Get API Key |
| `FINNHUB_API_KEY` | advisor | [finnhub.io](https://finnhub.io) → Dashboard → API Key (free tier) |
| `POLYGON_API_KEY` | advisor | [polygon.io](https://polygon.io) → Dashboard → API Keys (free tier) |
| `FRED_API_KEY` | advisor | [fredaccount.stlouisfed.org/apikeys](https://fredaccount.stlouisfed.org/apikeys) (free) |

The `investor` module uses **yfinance** which requires no API key.

---

## Project Structure

```
RoboAdvisor/
├── .env                          # Your API keys (git-ignored)
├── .env.example                  # Key template
├── requirements.txt              # All dependencies
├── investor/                     # Trading bot module
│   ├── config.py                 # All tuneable parameters
│   ├── models.py                 # Position, CircuitBreaker dataclasses
│   ├── indicators.py             # RSI, EMA, MACD (pure numpy)
│   ├── strategies.py             # TacticalStrategy, ExperimentalStrategy
│   ├── engine.py                 # Backtest / paper / live runners
│   └── main.py                   # Entry point
└── advisor/
    ├── main.py                   # Entry point
    └── investment_advisor/
        ├── advisor.py            # Gemini API integration
        ├── aggregator.py         # Combines all data sources
        └── data/
            ├── polygon_client.py # Price & momentum signals
            ├── finnhub_client.py # News & sentiment
            └── fred_client.py    # Macro indicators (Fed rate, CPI, VIX…)
```

---

> **Disclaimer:** This project is for personal learning and research only. Nothing here constitutes financial advice.
