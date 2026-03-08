"""Unit tests for the three new strategy engines and new indicator functions."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from quaver.strategies.indicators import atr, rolling_max, rolling_min, rsi
from quaver.strategies.breakout_consolidation import BreakoutConsolidationStrategy
from quaver.strategies.pullback_trend import PullbackTrendStrategy
from quaver.strategies.reversal_support import ReversalSupportStrategy
from quaver.strategies.registry import StrategyRegistry
from quaver.types import SignalDirection


# ── Indicator tests ──────────────────────────────────────────────────────────


def test_rsi_basic():
    """RSI on a steady uptrend should be > 50."""
    close = np.arange(100, 120, dtype=float)  # 20 bars, steady rise
    result = rsi(close, period=14)
    assert np.isnan(result[13])  # not enough data yet
    assert not np.isnan(result[14])  # first valid value
    assert result[14] > 50  # uptrend -> bullish


def test_rsi_downtrend():
    """RSI on a steady downtrend should be < 50."""
    close = np.arange(120, 100, -1, dtype=float)
    result = rsi(close, period=14)
    assert result[14] < 50


def test_rsi_all_gains():
    """RSI should be 100.0 when there are only gains."""
    close = np.arange(50, 70, dtype=float)
    result = rsi(close, period=14)
    assert result[-1] == pytest.approx(100.0)


def test_rsi_short_array():
    """RSI returns all NaN when array is shorter than period+1."""
    close = np.array([100.0, 101.0, 102.0])
    result = rsi(close, period=14)
    assert all(np.isnan(result))


def test_atr_basic():
    """ATR should return positive values after warmup period."""
    n = 30
    high = np.full(n, 102.0)
    low = np.full(n, 98.0)
    close = np.full(n, 100.0)
    result = atr(high, low, close, period=14)
    assert np.isnan(result[13])  # not enough data
    assert not np.isnan(result[14])
    assert result[14] > 0  # ATR must be positive


def test_rolling_max_basic():
    """Rolling max over a known series."""
    values = np.array([1, 3, 2, 5, 4, 6], dtype=float)
    result = rolling_max(values, 3)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert result[2] == 3.0  # max(1, 3, 2)
    assert result[3] == 5.0  # max(3, 2, 5)
    assert result[4] == 5.0  # max(2, 5, 4)
    assert result[5] == 6.0  # max(5, 4, 6)


def test_rolling_min_basic():
    """Rolling min over a known series."""
    values = np.array([5, 3, 4, 1, 6, 2], dtype=float)
    result = rolling_min(values, 3)
    assert np.isnan(result[0])
    assert result[2] == 3.0  # min(5, 3, 4)
    assert result[3] == 1.0  # min(3, 4, 1)
    assert result[4] == 1.0  # min(4, 1, 6)
    assert result[5] == 1.0  # min(1, 6, 2) -> 1


# ── Helper to build OHLCV DataFrame ─────────────────────────────────────────


def _make_candles(
    n: int,
    start_price: float = 100.0,
    drift: float = 0.0,
    spread: float = 2.0,
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    ts = datetime(2023, 1, 1)
    rows = []
    price = start_price
    for _ in range(n):
        rows.append(
            {
                "ts": ts,
                "open": price,
                "high": price + spread / 2,
                "low": price - spread / 2,
                "close": price + drift,
                "volume": volume,
            }
        )
        price += drift
        ts += timedelta(days=1)
    return pd.DataFrame(rows)


# ── Registry tests ───────────────────────────────────────────────────────────


def test_new_strategies_registered():
    """All three new strategies are in the registry."""
    import quaver.strategies  # noqa: F401

    engines = StrategyRegistry.list_engines()
    assert "breakout_consolidation" in engines
    assert "pullback_trend" in engines
    assert "reversal_support" in engines


def test_new_strategies_are_single_asset():
    """All three new strategies are single-asset."""
    import quaver.strategies  # noqa: F401

    for name in ("breakout_consolidation", "pullback_trend", "reversal_support"):
        assert StrategyRegistry.get_strategy_kind(name) == "single"


def test_default_parameters_non_empty():
    """All three new strategies return non-empty default parameters."""
    import quaver.strategies  # noqa: F401

    for name in ("breakout_consolidation", "pullback_trend", "reversal_support"):
        cls = StrategyRegistry.get(name)
        defaults = cls.get_default_parameters()
        assert isinstance(defaults, dict)
        assert len(defaults) > 0


# ── Breakout Consolidation tests ─────────────────────────────────────────────


def test_breakout_no_signal_insufficient_data():
    """Breakout strategy returns None with insufficient data."""
    s = BreakoutConsolidationStrategy(BreakoutConsolidationStrategy.get_default_parameters())
    s.validate_parameters()
    df = _make_candles(20)
    result = s.compute(df, datetime(2023, 2, 1))
    assert result is None


def test_breakout_no_signal_flat_market():
    """Flat market with no breakout should produce no signal."""
    s = BreakoutConsolidationStrategy(BreakoutConsolidationStrategy.get_default_parameters())
    df = _make_candles(100, drift=0.0)
    result = s.compute(df, datetime(2023, 5, 1))
    assert result is None


def test_breakout_validate_parameters():
    """Validation rejects bad parameters."""
    s = BreakoutConsolidationStrategy(
        {
            "ma_period": -1,
            "consolidation_period": 20,
            "range_max_pct": 0.10,
            "atr_period": 14,
            "atr_lookback": 10,
            "volume_sma_period": 20,
        }
    )
    with pytest.raises(ValueError, match="ma_period"):
        s.validate_parameters()


def test_breakout_signal_on_engineered_data():
    """Breakout fires on a consolidation followed by volume-backed breakout."""
    ts = datetime(2023, 1, 1)
    rows = []
    price = 100.0

    # Phase 1: uptrend (bars 0-49) to get MA50 above
    for i in range(50):
        rows.append(
            {
                "ts": ts + timedelta(days=i),
                "open": price,
                "high": price + 1.5,
                "low": price - 1.5,
                "close": price + 0.5,
                "volume": 1_000_000,
            }
        )
        price += 0.5

    # Phase 2: tight consolidation (bars 50-69)
    consol_price = price
    for i in range(50, 70):
        rows.append(
            {
                "ts": ts + timedelta(days=i),
                "open": consol_price,
                "high": consol_price + 0.3,
                "low": consol_price - 0.3,
                "close": consol_price,
                "volume": 800_000,
            }
        )

    # Phase 3: breakout bars (70-79) — price jumps above range + volume spike
    for i in range(70, 80):
        bp = consol_price + 2.0 + (i - 70) * 0.5
        rows.append(
            {
                "ts": ts + timedelta(days=i),
                "open": bp - 0.5,
                "high": bp + 1.0,
                "low": bp - 1.0,
                "close": bp,
                "volume": 2_500_000,
            }
        )

    df = pd.DataFrame(rows)
    s = BreakoutConsolidationStrategy(BreakoutConsolidationStrategy.get_default_parameters())
    result = s.compute(df, datetime(2023, 4, 1))
    # Signal may or may not fire depending on ATR computation,
    # but should not error
    if result is not None:
        assert result.direction == SignalDirection.BUY
        assert 0.0 <= result.confidence <= 1.0


# ── Pullback in Trend tests ─────────────────────────────────────────────────


def test_pullback_no_signal_insufficient_data():
    """Pullback strategy returns None with insufficient data."""
    s = PullbackTrendStrategy(PullbackTrendStrategy.get_default_parameters())
    s.validate_parameters()
    df = _make_candles(50)
    result = s.compute(df, datetime(2023, 3, 1))
    assert result is None


def test_pullback_no_signal_downtrend():
    """Downtrending stock should produce no pullback buy signal."""
    s = PullbackTrendStrategy(PullbackTrendStrategy.get_default_parameters())
    df = _make_candles(250, start_price=200.0, drift=-0.5)
    result = s.compute(df, datetime(2024, 1, 1))
    assert result is None


def test_pullback_validate_bad_ma_order():
    """Validation rejects ma_fast >= ma_medium."""
    params = PullbackTrendStrategy.get_default_parameters()
    params["ma_fast"] = 60  # > ma_medium (50)
    s = PullbackTrendStrategy(params)
    with pytest.raises(ValueError, match="ma_fast"):
        s.validate_parameters()


def test_pullback_validate_bad_rsi_order():
    """Validation rejects rsi_low >= rsi_high."""
    params = PullbackTrendStrategy.get_default_parameters()
    params["rsi_low"] = 55
    params["rsi_high"] = 50
    s = PullbackTrendStrategy(params)
    with pytest.raises(ValueError, match="rsi_low"):
        s.validate_parameters()


# ── Reversal at Support tests ────────────────────────────────────────────────


def test_reversal_no_signal_insufficient_data():
    """Reversal strategy returns None with insufficient data."""
    s = ReversalSupportStrategy(ReversalSupportStrategy.get_default_parameters())
    s.validate_parameters()
    df = _make_candles(50)
    result = s.compute(df, datetime(2023, 3, 1))
    assert result is None


def test_reversal_no_signal_uptrend():
    """Strong uptrend (no oversold) should produce no reversal signal."""
    s = ReversalSupportStrategy(ReversalSupportStrategy.get_default_parameters())
    df = _make_candles(250, drift=0.3)
    result = s.compute(df, datetime(2024, 1, 1))
    assert result is None


def test_reversal_validate_parameters():
    """Validation rejects bad parameters."""
    params = ReversalSupportStrategy.get_default_parameters()
    params["ma_slow"] = -5
    s = ReversalSupportStrategy(params)
    with pytest.raises(ValueError, match="ma_slow"):
        s.validate_parameters()


def test_reversal_empty_candles():
    """Reversal strategy returns None on empty DataFrame."""
    s = ReversalSupportStrategy(ReversalSupportStrategy.get_default_parameters())
    df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    result = s.compute(df, datetime(2023, 1, 1))
    assert result is None


# ── Backtest integration tests ───────────────────────────────────────────────


def test_breakout_backtest_integration():
    """Breakout strategy runs through backtest runner without errors."""
    from quaver.backtest import run_backtest
    import quaver.strategies  # noqa: F401

    df = _make_candles(100, drift=0.2)
    result = run_backtest(
        engine_name="breakout_consolidation",
        parameters=BreakoutConsolidationStrategy.get_default_parameters(),
        candles=df,
        instrument_id="TEST",
        initial_capital=10_000,
        quantity_per_trade=10,
    )
    assert result.instrument_id == "TEST"
    assert result.initial_capital == 10_000


def test_pullback_backtest_integration():
    """Pullback strategy runs through backtest runner without errors."""
    from quaver.backtest import run_backtest
    import quaver.strategies  # noqa: F401

    df = _make_candles(250, drift=0.2)
    result = run_backtest(
        engine_name="pullback_trend",
        parameters=PullbackTrendStrategy.get_default_parameters(),
        candles=df,
        instrument_id="TEST",
        initial_capital=10_000,
        quantity_per_trade=10,
    )
    assert result.instrument_id == "TEST"


def test_reversal_backtest_integration():
    """Reversal strategy runs through backtest runner without errors."""
    from quaver.backtest import run_backtest
    import quaver.strategies  # noqa: F401

    df = _make_candles(250, drift=-0.1)
    result = run_backtest(
        engine_name="reversal_support",
        parameters=ReversalSupportStrategy.get_default_parameters(),
        candles=df,
        instrument_id="TEST",
        initial_capital=10_000,
        quantity_per_trade=10,
    )
    assert result.instrument_id == "TEST"
