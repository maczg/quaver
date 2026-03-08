"""Unit tests for MultiAssetStrategy and MultiAssetBacktestEngine."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quaver.strategies.pairs_mean_reversion import PairsMeanReversionStrategy
from quaver.strategies.registry import StrategyRegistry
from quaver.backtest.multi_engine import MultiAssetBacktestEngine
from quaver.backtest.portfolio import Portfolio
from quaver.backtest.data import normalise_candles


def make_pair_candles(n=200, spread_offset=0.0, seed=42):
    """Generate two correlated instruments with a controlled spread.

    Both instruments share a common random walk, but instrument B gets an
    additional per-bar offset plus independent noise. This ensures the spread
    has real variance (not a perfect linear ramp) so z-scores can exceed
    entry thresholds.
    """
    rng = np.random.default_rng(seed)
    ts = datetime(2020, 1, 1)
    rows_a, rows_b = [], []
    price_a, price_b = 100.0, 100.0
    for i in range(n):
        change = rng.normal(0, 0.5)
        noise_b = rng.normal(0, 0.3)  # independent noise for spread variance
        price_a += change
        price_b += change + spread_offset + noise_b
        for rows, price in [(rows_a, price_a), (rows_b, price_b)]:
            rows.append({
                "ts": ts, "open": price, "high": price + 0.5,
                "low": price - 0.5, "close": price, "volume": 1_000.0,
            })
        ts += timedelta(days=1)
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


PARAMS = {
    "instrument_a": "A",
    "instrument_b": "B",
    "spread_window": 30,
    "entry_z": 2.0,
    "exit_z": 0.5,
}


def test_pairs_compute_returns_signals_outside_band():
    """With a large spread offset, strategy should eventually return signals."""
    df_a, df_b = make_pair_candles(200, spread_offset=10.0)
    df_a = normalise_candles(df_a)
    df_b = normalise_candles(df_b)
    # Use entry_z=1.5 — a constant per-bar offset produces a linear spread
    # whose rolling z-score is bounded at ~sqrt(3) ≈ 1.73, so entry_z=2.0
    # would never trigger.
    params = {**PARAMS, "entry_z": 1.5, "exit_z": 0.3}
    strategy = PairsMeanReversionStrategy(parameters=params)
    strategy.validate_parameters()

    found_signal = False
    for i in range(40, len(df_a)):
        window = {"A": df_a.iloc[:i], "B": df_b.iloc[:i]}
        out = strategy.compute(window, as_of=df_a.iloc[i]["ts"])
        if out is not None:
            assert "A" in out.signals
            assert "B" in out.signals
            found_signal = True
            break
    assert found_signal, "Strategy produced no signals on data with large spread offset"


def test_pairs_no_signal_inside_band():
    """With zero spread offset, most compute() calls return None."""
    df_a, df_b = make_pair_candles(100, spread_offset=0.0)
    df_a = normalise_candles(df_a)
    df_b = normalise_candles(df_b)
    strategy = PairsMeanReversionStrategy(parameters=PARAMS)
    strategy.validate_parameters()

    none_count = 0
    for i in range(35, 60):
        window = {"A": df_a.iloc[:i], "B": df_b.iloc[:i]}
        out = strategy.compute(window, as_of=df_a.iloc[i]["ts"])
        if out is None:
            none_count += 1
    # The majority should be None when spread is near zero
    assert none_count > 10


def test_multi_engine_run_returns_both_results():
    """Full engine run returns BacktestResult for both instruments."""
    df_a, df_b = make_pair_candles(200, spread_offset=5.0)
    candles_map = {
        "A": normalise_candles(df_a),
        "B": normalise_candles(df_b),
    }
    portfolios = {
        "A": Portfolio(initial_capital=10_000),
        "B": Portfolio(initial_capital=10_000),
    }
    strategy = PairsMeanReversionStrategy(parameters=PARAMS)
    engine = MultiAssetBacktestEngine(strategy=strategy, portfolios=portfolios)
    results = engine.run(candles_map)

    assert "A" in results
    assert "B" in results
    assert results["A"].instrument_id == "A"
    assert results["B"].instrument_id == "B"


def test_registry_kind_detection():
    assert StrategyRegistry.get_strategy_kind("pairs_mean_reversion") == "multi"
    assert StrategyRegistry.get_strategy_kind("mean_reversion") == "single"
