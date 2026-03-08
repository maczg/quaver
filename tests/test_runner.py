"""Integration tests for run_backtest and run_multi_asset_backtest."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quaver.backtest.runner import run_backtest, run_multi_asset_backtest
from quaver.strategies.registry import EngineNotFoundError


def make_candles(n=200, seed=42):
    rng = np.random.default_rng(seed)
    ts = datetime(2020, 1, 1)
    rows = []
    price = 100.0
    for _ in range(n):
        price += rng.normal(0, 1)
        rows.append({
            "ts": ts, "open": price, "high": price + 1,
            "low": price - 1, "close": price, "volume": 1000.0,
        })
        ts += timedelta(days=1)
    return pd.DataFrame(rows)


def test_run_backtest_end_to_end():
    candles = make_candles(200)
    result = run_backtest(
        engine_name="mean_reversion",
        parameters={"fast_period": 10, "slow_period": 30, "threshold": 0.01},
        candles=candles,
        instrument_id="TEST",
        initial_capital=10_000.0,
    )
    assert result.instrument_id == "TEST"
    assert result.initial_capital == 10_000.0
    assert isinstance(result.total_trades, int)
    assert result.total_trades >= 0
    s = result.summary()
    assert "total_return_pct" in s
    assert "win_rate_pct" in s


def test_unknown_engine_raises():
    with pytest.raises(EngineNotFoundError):
        run_backtest(
            engine_name="nonexistent_engine",
            parameters={},
            candles=make_candles(),
            instrument_id="X",
        )


def test_run_multi_with_single_asset_engine_raises():
    with pytest.raises(TypeError, match="single-asset"):
        run_multi_asset_backtest(
            engine_name="mean_reversion",
            parameters={"fast_period": 10, "slow_period": 30, "threshold": 0.01},
            candles_map={"A": make_candles()},
        )


def test_run_backtest_with_multi_asset_engine_raises():
    with pytest.raises(TypeError, match="MultiAssetStrategy"):
        run_backtest(
            engine_name="pairs_mean_reversion",
            parameters={
                "instrument_a": "A", "instrument_b": "B",
                "spread_window": 30, "entry_z": 2.0, "exit_z": 0.5,
            },
            candles=make_candles(),
            instrument_id="A",
        )


def test_missing_instrument_raises():
    with pytest.raises(ValueError, match="missing"):
        run_multi_asset_backtest(
            engine_name="pairs_mean_reversion",
            parameters={
                "instrument_a": "A", "instrument_b": "B",
                "spread_window": 30, "entry_z": 2.0, "exit_z": 0.5,
            },
            candles_map={"A": make_candles()},  # B is missing
        )


def test_insufficient_candles_raises():
    with pytest.raises(ValueError, match="Insufficient"):
        run_backtest(
            engine_name="mean_reversion",
            parameters={"fast_period": 10, "slow_period": 30, "threshold": 0.01},
            candles=make_candles(10),   # way below required
            instrument_id="X",
        )


def test_run_multi_asset_end_to_end():
    from datetime import timedelta
    import numpy as np

    rng = np.random.default_rng(0)
    ts = datetime(2020, 1, 1)
    rows_a, rows_b = [], []
    pa, pb = 100.0, 100.0
    for _ in range(200):
        c = rng.normal(0, 0.5)
        pa += c; pb += c + 0.2
        for rows, p in [(rows_a, pa), (rows_b, pb)]:
            rows.append({"ts": ts, "open": p, "high": p+0.5,
                         "low": p-0.5, "close": p, "volume": 1000.0})
        ts += timedelta(days=1)

    results = run_multi_asset_backtest(
        engine_name="pairs_mean_reversion",
        parameters={
            "instrument_a": "A", "instrument_b": "B",
            "spread_window": 30, "entry_z": 2.0, "exit_z": 0.5,
        },
        candles_map={"A": pd.DataFrame(rows_a), "B": pd.DataFrame(rows_b)},
        initial_capital=10_000.0,
        allow_shorting=True,
    )
    assert "A" in results
    assert "B" in results
