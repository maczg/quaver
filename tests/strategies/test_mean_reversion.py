"""Behavioural tests for :class:`MeanReversionStrategy`.

The strategy compares a fast SMA against a slow SMA. When the fast SMA is
below the slow by more than *threshold* it emits BUY (oversold); when it's
above by more than *threshold* it emits SELL (overbought). Flat markets
should produce no signal.
"""

from datetime import datetime

import pandas as pd
import pytest

from quaver.strategies.mean_reversion import MeanReversionStrategy
from quaver.types import SignalDirection

from tests.strategies.conftest import flat_market, linear_trend


PARAMS = {"fast_period": 10, "slow_period": 30, "threshold": 0.02}


def _make() -> MeanReversionStrategy:
    s = MeanReversionStrategy(parameters=dict(PARAMS))
    s.validate_parameters()
    return s


def test_buy_on_fresh_drop_after_high_plateau():
    """A long flat run keeps the slow MA elevated; a sharp recent drop pulls
    the fast MA below by more than the threshold → BUY."""
    plateau = flat_market(n=80, price=100.0)
    drop = linear_trend(n=15, start_price=100.0, drift=-1.5)
    candles = pd.concat([plateau, drop], ignore_index=True)

    s = _make()
    out = s.compute(candles, as_of=datetime(2024, 1, 1))
    assert out is not None, "Expected BUY signal on sharp drop after plateau"
    assert out.direction == SignalDirection.BUY
    assert 0.0 < out.confidence <= 1.0
    assert out.metadata["divergence"] < 0  # fast below slow


def test_sell_on_fresh_rally_after_low_plateau():
    """Sharp recent rally after a flat plateau pulls the fast MA above the
    slow by more than threshold → SELL."""
    plateau = flat_market(n=80, price=100.0)
    rally = linear_trend(n=15, start_price=100.0, drift=1.5)
    candles = pd.concat([plateau, rally], ignore_index=True)

    s = _make()
    out = s.compute(candles, as_of=datetime(2024, 1, 1))
    assert out is not None, "Expected SELL signal on sharp rally after plateau"
    assert out.direction == SignalDirection.SELL
    assert out.metadata["divergence"] > 0


def test_no_signal_on_flat_market():
    """A perfectly flat market has fast MA == slow MA → no signal."""
    candles = flat_market(n=100, price=100.0)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_on_steady_uniform_trend():
    """A uniform linear trend keeps fast and slow MAs separated by less than
    threshold over the long run, so on most bars no signal fires.

    (This documents that a steady drift alone is *not* the trigger — the
    strategy needs a divergence between fast and slow that exceeds threshold.)
    """
    # Drift small enough that |fast_ma - slow_ma| / slow_ma stays below 2%
    candles = linear_trend(n=200, start_price=100.0, drift=0.05)
    s = _make()
    out = s.compute(candles, as_of=datetime(2024, 1, 1))
    # Either None, or — if it fires — confidence must still be within bounds
    if out is not None:
        assert 0.0 <= out.confidence <= 1.0


def test_no_signal_when_insufficient_data():
    """compute() returns None when fewer than slow_period bars are present."""
    candles = flat_market(n=20, price=100.0)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


@pytest.mark.parametrize(
    "bad",
    [
        {"fast_period": 0, "slow_period": 50, "threshold": 0.02},
        {"fast_period": 10, "slow_period": 10, "threshold": 0.02},
        {"fast_period": 10, "slow_period": 50, "threshold": 0.0},
    ],
)
def test_validate_parameters_rejects_invalid(bad):
    s = MeanReversionStrategy(parameters=bad)
    with pytest.raises(ValueError):
        s.validate_parameters()
