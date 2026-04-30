"""Behavioural tests for :class:`ReversalSupportStrategy`.

The BUY pattern is a counter-trend mean reversion: not in a structural
downtrend, RSI deeply oversold, price near the rolling-low support, and a
bullish reversal candle (close > prior bar's high).
"""

from datetime import datetime

import pytest

from quaver.strategies.reversal_support import ReversalSupportStrategy
from quaver.types import SignalDirection

from tests.strategies.conftest import (
    downtrend_into_support,
    flat_market,
    linear_trend,
)


def _make(**overrides) -> ReversalSupportStrategy:
    params = ReversalSupportStrategy.get_default_parameters()
    params.update(overrides)
    s = ReversalSupportStrategy(parameters=params)
    s.validate_parameters()
    return s


def test_buy_on_engineered_reversal_at_support():
    """Plateau (so MA200 stays elevated) → sharp drop into a fresh rolling-low
    → bullish reversal candle. Default RSI threshold is 30; we relax MA200
    distance because the drop pulls close ~10% off MA200."""
    candles = downtrend_into_support(
        plateau_bars=220,
        plateau_price=100.0,
        drop_bars=14,
        drop_per_bar=0.8,
        rebound_jump=2.0,
    )
    s = _make(max_dist_ma200=0.30)
    out = s.compute(candles, as_of=datetime(2024, 1, 1))
    assert out is not None, "Expected reversal-at-support BUY signal"
    assert out.direction == SignalDirection.BUY
    assert 0.0 < out.confidence <= 1.0
    md = out.metadata
    assert md["rsi"] < 30.0
    assert md["close"] >= md["support_level"]


def test_no_signal_on_steady_uptrend():
    """Uptrend keeps RSI well above the oversold threshold."""
    candles = linear_trend(n=250, start_price=100.0, drift=0.3)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_on_flat_market():
    """Flat market: RSI sits near 50, no oversold trigger."""
    candles = flat_market(n=250, price=100.0)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_on_structural_downtrend():
    """Strategy explicitly excludes structural downtrends — close must stay
    within *max_dist_ma200* of the slow MA. A persistent downtrend pulls close
    >20% below MA200, so the filter rejects."""
    candles = linear_trend(n=250, start_price=200.0, drift=-0.5)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_when_insufficient_data():
    candles = flat_market(n=50, price=100.0)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


@pytest.mark.parametrize(
    "bad_overrides",
    [
        {"ma_slow": 0},
        {"rsi_threshold": -1},
        {"max_dist_ma200": 0.0},
        {"support_period": 0},
    ],
)
def test_validate_parameters_rejects_invalid(bad_overrides):
    params = ReversalSupportStrategy.get_default_parameters()
    params.update(bad_overrides)
    s = ReversalSupportStrategy(parameters=params)
    with pytest.raises(ValueError):
        s.validate_parameters()
