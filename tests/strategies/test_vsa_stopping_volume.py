"""Behavioural tests for :class:`VSAStoppingVolumeStrategy`.

The BUY pattern fires when, after a local downtrend, a bear bar appears with
unusually high volume but a narrow spread and a close *not* on the lows of
the bar — the textbook "stopping volume" signature.

The SELL pattern is the symmetric mirror in an uptrend.
"""

from datetime import datetime

import pytest

from quaver.strategies.vsa_stopping_volume import VSAStoppingVolumeStrategy
from quaver.types import SignalDirection

from tests.strategies.conftest import (
    flat_market,
    linear_trend,
    vsa_stopping_volume_pattern,
)


def _make(**overrides) -> VSAStoppingVolumeStrategy:
    params = VSAStoppingVolumeStrategy.get_default_parameters()
    params.update(overrides)
    s = VSAStoppingVolumeStrategy(parameters=params)
    s.validate_parameters()
    return s


def test_buy_on_engineered_stopping_volume_pattern():
    """The conftest builder is engineered to satisfy every BUY condition with
    default parameters → BUY signal expected."""
    candles = vsa_stopping_volume_pattern()
    s = _make()
    out = s.compute(candles, as_of=datetime(2024, 1, 1))
    assert out is not None, "Expected stopping-volume BUY signal"
    assert out.direction == SignalDirection.BUY
    assert 0.0 < out.confidence <= 1.0
    md = out.metadata
    assert md["vol_rel"] > 2.0
    assert md["spread_rel"] < 0.7
    assert 0.4 < md["close_position"] <= 1.0


def test_no_signal_on_flat_market():
    """Flat market: no relative volume spike, no narrow-spread bear bar → no signal."""
    candles = flat_market(n=80, price=100.0)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_on_plain_downtrend_without_volume_spike():
    """Steady downtrend with constant volume — relative volume sits near 1.0,
    so the stopping-volume threshold is never breached."""
    candles = linear_trend(n=80, start_price=100.0, drift=-0.5)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_buy_disabled_returns_none():
    """When enable_buy=False the engineered pattern is suppressed."""
    candles = vsa_stopping_volume_pattern()
    s = _make(enable_buy=False, enable_sell=False)
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_when_insufficient_data():
    candles = flat_market(n=10, price=100.0)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


@pytest.mark.parametrize(
    "bad_overrides",
    [
        {"sma_window": 0},
        {"vol_high": -1.0},
        {"buy_close_pos_min": 1.5},
        {"enable_buy": "yes"},
    ],
)
def test_validate_parameters_rejects_invalid(bad_overrides):
    params = VSAStoppingVolumeStrategy.get_default_parameters()
    params.update(bad_overrides)
    s = VSAStoppingVolumeStrategy(parameters=params)
    with pytest.raises(ValueError):
        s.validate_parameters()
