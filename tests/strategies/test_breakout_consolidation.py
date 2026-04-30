"""Behavioural tests for :class:`BreakoutConsolidationStrategy`.

The BUY pattern requires a textbook setup:

  * close above the medium-term SMA (trend filter)
  * tight 20-bar range as a fraction of price
  * declining ATR (volatility compression)
  * close exceeds the 20-bar high (breakout trigger)
  * volume above its SMA (confirmation)
"""

from datetime import datetime

import pytest

from quaver.strategies.breakout_consolidation import BreakoutConsolidationStrategy
from quaver.types import SignalDirection

from tests.strategies.conftest import (
    flat_market,
    linear_trend,
    trend_then_consolidation_then_breakout,
)


def _make(**overrides) -> BreakoutConsolidationStrategy:
    params = BreakoutConsolidationStrategy.get_default_parameters()
    params.update(overrides)
    s = BreakoutConsolidationStrategy(parameters=params)
    s.validate_parameters()
    return s


def test_buy_on_engineered_breakout():
    """Trend → tight consolidation → high-volume breakout above range → BUY.

    *atr_lookback* is set to 30 so the ATR-decline check spans from the
    high-volatility trend phase (wide bars) into the low-volatility
    consolidation (tight bars). The default of 10 falls entirely inside
    consolidation, so it can never actually detect compression in this kind of
    setup — that's a real quirk of the strategy worth documenting via the test.
    """
    candles = trend_then_consolidation_then_breakout(
        trend_bars=80,
        trend_drift=0.6,
        consol_bars=25,
        consol_spread=0.15,
        breakout_jump=2.0,
        breakout_volume_mult=3.0,
    )
    s = _make(atr_lookback=30)
    out = s.compute(candles, as_of=datetime(2024, 1, 1))
    assert out is not None, "Expected BUY signal on engineered breakout"
    assert out.direction == SignalDirection.BUY
    assert 0.0 < out.confidence <= 1.0
    md = out.metadata
    assert md["close"] > md["prior_high_max"]
    assert md["vol_rel"] > 1.0
    assert md["atr"] < md["atr_prev"]


def test_no_signal_on_flat_market():
    """Flat market: no breakout candle exists."""
    candles = flat_market(n=120, price=100.0)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_on_steady_trend_without_consolidation():
    """A steady, wide-bodied uptrend doesn't form a tight consolidation, so the
    range filter rejects it."""
    # Wide swings → 20-bar range is much greater than the 10% threshold
    candles = linear_trend(n=120, start_price=100.0, drift=0.6, spread=2.0)
    s = _make(range_max_pct=0.05)  # tighter range filter
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_when_volume_does_not_confirm():
    """Override the volume so the breakout bar has *low* volume — the volume
    filter must reject the otherwise-valid setup."""
    candles = trend_then_consolidation_then_breakout(breakout_volume_mult=0.5)
    s = _make(atr_lookback=30)
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_when_insufficient_data():
    candles = flat_market(n=15, price=100.0)
    s = _make()
    assert s.compute(candles, as_of=datetime(2024, 1, 1)) is None


@pytest.mark.parametrize(
    "bad_overrides",
    [
        {"ma_period": 0},
        {"consolidation_period": -1},
        {"range_max_pct": 0.0},
        {"atr_period": 0},
    ],
)
def test_validate_parameters_rejects_invalid(bad_overrides):
    params = BreakoutConsolidationStrategy.get_default_parameters()
    params.update(bad_overrides)
    s = BreakoutConsolidationStrategy(parameters=params)
    with pytest.raises(ValueError):
        s.validate_parameters()
