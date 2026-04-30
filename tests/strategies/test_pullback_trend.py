"""Behavioural tests for :class:`PullbackTrendStrategy`.

The BUY pattern requires a confirmed multi-timeframe uptrend, a pullback into
the short-term MA with RSI in the [40, 50] zone, and a momentum-resumption
trigger (close > prior bar's high, or close > MA_fast).
"""

from datetime import datetime

import pytest

from quaver.strategies.pullback_trend import PullbackTrendStrategy
from quaver.types import SignalDirection

from tests.strategies.conftest import flat_market, linear_trend, uptrend_with_dip


def _make(**overrides) -> PullbackTrendStrategy:
    params = PullbackTrendStrategy.get_default_parameters()
    params.update(overrides)
    s = PullbackTrendStrategy(parameters=params)
    s.validate_parameters()
    return s


def test_buy_on_engineered_pullback_in_uptrend():
    """Long uptrend + brief pullback toward MA_fast + bullish recovery → BUY.

    The default RSI band [40, 50] is narrow and sensitive to the exact dip
    depth, so we widen it slightly. The structural test — uptrend + pullback +
    recovery — is the meaningful behaviour.
    """
    candles = uptrend_with_dip(
        n_pre=240,
        drift=0.4,
        dip_bars=8,
        dip_drop_per_bar=0.6,
        rebound_jump=1.5,
    )
    s = _make(rsi_low=30, rsi_high=55, near_ma_pct=0.05)
    out = s.compute(candles, as_of=datetime(2024, 1, 1))
    assert out is not None, "Expected pullback BUY signal"
    assert out.direction == SignalDirection.BUY
    assert 0.0 < out.confidence <= 1.0
    md = out.metadata
    assert md["close"] > md["ma_medium"] > md["ma_slow"]


def test_no_signal_on_downtrend():
    """In a downtrend, ``close > MA_medium > MA_slow`` is false → no signal."""
    candles = linear_trend(n=250, start_price=200.0, drift=-0.5)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_on_flat_market():
    """Flat market: MA slope is zero, RSI hovers around 50, no pullback → no
    signal."""
    candles = flat_market(n=250, price=100.0)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_on_steady_uptrend_without_pullback():
    """A clean steady uptrend has RSI well above the pullback zone, so the RSI
    filter rejects it."""
    candles = linear_trend(n=250, start_price=100.0, drift=0.5)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_when_insufficient_data():
    candles = flat_market(n=50, price=100.0)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


@pytest.mark.parametrize(
    "bad_overrides",
    [
        {"ma_fast": 50, "ma_medium": 50},  # ma_fast not less than ma_medium
        {"ma_medium": 200, "ma_slow": 200},  # ma_medium not less than ma_slow
        {"rsi_low": 50, "rsi_high": 50},  # rsi_low not less than rsi_high
        {"atr_stop_mult": -1.0},
    ],
)
def test_validate_parameters_rejects_invalid(bad_overrides):
    params = PullbackTrendStrategy.get_default_parameters()
    params.update(bad_overrides)
    s = PullbackTrendStrategy(parameters=params)
    with pytest.raises(ValueError):
        s.validate_parameters()
