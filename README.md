# quaver

Standalone Python library for quantitative trading signal generation and walk-forward backtesting.

## Installation

```bash
pip install pyquaver
```

With optional extras:

```bash
# Interactive notebooks with yfinance data
pip install pyquaver[notebooks]

# Development tools (pytest, ruff, mypy)
pip install pyquaver[dev]

# Sphinx documentation
pip install pyquaver[docs]
```

## Quick Start

### Single-asset backtest

```python
import pandas as pd
from quaver.backtest import run_backtest

# Load your OHLCV data (must have columns: ts, open, high, low, close, volume)
candles = pd.read_csv("data.csv")

result = run_backtest(
    engine_name="mean_reversion",
    parameters={"fast_period": 20, "slow_period": 50, "threshold": 0.02},
    candles=candles,
    instrument_id="AAPL",
    initial_capital=10_000.0,
)

print(result.summary())
```

### Multi-asset pairs backtest

```python
from quaver.backtest import run_multi_asset_backtest

results = run_multi_asset_backtest(
    engine_name="pairs_mean_reversion",
    parameters={
        "instrument_a": "AAPL",
        "instrument_b": "MSFT",
        "spread_window": 60,
        "entry_z": 2.0,
        "exit_z": 0.5,
    },
    candles_map={"AAPL": candles_aapl, "MSFT": candles_msft},
    initial_capital=10_000.0,
    allow_shorting=True,
)

for iid, r in results.items():
    print(f"{iid}: {r.summary()}")
```

### Discover strategies

```python
from quaver.strategies.registry import StrategyRegistry
import quaver.strategies  # auto-registers all built-in engines

print(StrategyRegistry.list_engines())
# ['mean_reversion', 'pairs_mean_reversion', 'regime_mean_reversion', 'vsa_stopping_volume']
```

## Built-in Strategies

| Strategy | Type | Description |
|---|---|---|
| `mean_reversion` | single | Dual moving-average mean reversion |
| `regime_mean_reversion` | single | Regime-based probabilistic mean reversion |
| `vsa_stopping_volume` | single | VSA stopping-volume reversal pattern |
| `pairs_mean_reversion` | multi | Statistical arbitrage pairs trading |

## Requirements

- Python >= 3.12
- numpy >= 2.4.2
- pandas >= 3.0.1

## License

MIT
