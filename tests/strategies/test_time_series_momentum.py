"""Behavioural tests for :class:`TimeSeriesMomentumStrategy`.

The strategy emits BUY when the trailing-window return is positive and SELL
when it is negative.  A flat market should produce no signal.  An uptrend
followed by a sharp recent drop should still emit BUY when the drop falls
inside the *skip_period* and the longer trailing window is still positive
(this is the whole point of the 12-1 momentum convention).
"""

from datetime import datetime

import pandas as pd
import pytest

from quaver.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from quaver.types import SignalDirection

from tests.strategies.conftest import flat_market, linear_trend


PARAMS = {"lookback_period": 60, "skip_period": 5, "threshold": 0.0}


def _make(**overrides) -> TimeSeriesMomentumStrategy:
    params = {**PARAMS, **overrides}
    s = TimeSeriesMomentumStrategy(parameters=params)
    s.validate_parameters()
    return s


def test_buy_on_sustained_uptrend():
    """A long uptrend produces a positive trailing return → BUY."""
    candles = linear_trend(n=120, start_price=100.0, drift=0.5)
    s = _make()
    out = s.compute(candles, as_of=datetime(2024, 1, 1))
    assert out is not None, "Expected BUY signal on sustained uptrend"
    assert out.direction == SignalDirection.BUY
    assert 0.0 < out.confidence <= 1.0
    assert out.metadata["trailing_return"] > 0


def test_sell_on_sustained_downtrend():
    """A long downtrend produces a negative trailing return → SELL."""
    candles = linear_trend(n=120, start_price=200.0, drift=-0.5)
    s = _make()
    out = s.compute(candles, as_of=datetime(2024, 1, 1))
    assert out is not None, "Expected SELL signal on sustained downtrend"
    assert out.direction == SignalDirection.SELL
    assert out.metadata["trailing_return"] < 0


def test_no_signal_on_flat_market():
    """A perfectly flat market has zero trailing return → no signal."""
    candles = flat_market(n=120, price=100.0)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_skip_period_ignores_recent_reversal():
    """An uptrend followed by a recent drop within *skip_period* should still
    emit BUY: the skip window is precisely designed to neutralise short-horizon
    reversals (the "12-1" convention)."""
    uptrend = linear_trend(n=120, start_price=100.0, drift=0.5)
    # Recent sharp drop, length == skip_period so it is fully skipped.
    drop = linear_trend(n=5, start_price=160.0, drift=-3.0)
    candles = pd.concat([uptrend, drop], ignore_index=True)

    s = _make(lookback_period=60, skip_period=5)
    out = s.compute(candles, as_of=datetime(2024, 1, 1))
    assert out is not None, "Skip period should hide the recent drop"
    assert out.direction == SignalDirection.BUY


def test_threshold_suppresses_weak_trends():
    """A barely-positive trailing return below *threshold* should not fire."""
    # Very small drift → trailing return well under 5%.
    candles = linear_trend(n=120, start_price=100.0, drift=0.005)
    s = _make(threshold=0.05)
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_when_insufficient_data():
    """compute() returns None when fewer than lookback+skip+1 bars are present."""
    candles = flat_market(n=10, price=100.0)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


@pytest.mark.parametrize(
    "bad",
    [
        {"lookback_period": 1, "skip_period": 0, "threshold": 0.0},
        {"lookback_period": 60, "skip_period": -1, "threshold": 0.0},
        {"lookback_period": 60, "skip_period": 60, "threshold": 0.0},
        {"lookback_period": 60, "skip_period": 5, "threshold": -0.01},
    ],
)
def test_validate_parameters_rejects_invalid(bad):
    s = TimeSeriesMomentumStrategy(parameters=bad)
    with pytest.raises(ValueError):
        s.validate_parameters()
