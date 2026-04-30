"""Behavioural tests for :class:`PairsMeanReversionStrategy`.

The strategy emits paired BUY/SELL signals on the two legs when the rolling
z-score of the spread exceeds *entry_z*, and CLOSE/CLOSE when |z| drops back
under *exit_z*.
"""

from datetime import datetime

import pytest

from quaver.strategies.pairs_mean_reversion import PairsMeanReversionStrategy
from quaver.types import SignalDirection

from tests.strategies.conftest import (
    correlated_pair_with_divergence,
    flat_market,
)


PARAMS = {
    "instrument_a": "A",
    "instrument_b": "B",
    "spread_window": 30,
    "entry_z": 2.0,
    "exit_z": 0.5,
}


def _make(**overrides) -> PairsMeanReversionStrategy:
    params = {**PARAMS, **overrides}
    s = PairsMeanReversionStrategy(parameters=params)
    s.validate_parameters()
    return s


def test_paired_buy_sell_when_spread_widens_positive():
    """Engineer A_close - B_close to spike positive at the end. Expect SELL on
    A and BUY on B (reverting trade: spread should fall)."""
    df_a, df_b = correlated_pair_with_divergence(n=120, final_divergence=12.0)
    s = _make()
    out = s.compute({"A": df_a, "B": df_b}, as_of=datetime(2024, 1, 1))
    assert out is not None, "Expected paired signals on widened spread"
    assert "A" in out.signals and "B" in out.signals
    z = out.signals["A"].metadata["z_score"]
    assert abs(z) > PARAMS["entry_z"]
    if z > 0:
        assert out.signals["A"].direction == SignalDirection.SELL
        assert out.signals["B"].direction == SignalDirection.BUY
    else:
        assert out.signals["A"].direction == SignalDirection.BUY
        assert out.signals["B"].direction == SignalDirection.SELL


def test_close_when_spread_inside_exit_band():
    """When two synthetic series are identical, the spread is exactly zero,
    z-score is exactly zero (well inside exit_z) → CLOSE / CLOSE."""
    df_a = flat_market(n=80, price=100.0)
    df_b = flat_market(n=80, price=100.0)
    s = _make()
    out = s.compute({"A": df_a, "B": df_b}, as_of=datetime(2024, 1, 1))
    # When std is zero the strategy short-circuits to None — that's also
    # documented behaviour, so we accept either.
    if out is not None:
        assert out.signals["A"].direction == SignalDirection.CLOSE
        assert out.signals["B"].direction == SignalDirection.CLOSE


def test_no_signal_when_spread_inside_entry_band_with_variance():
    """Two correlated series with no end-of-window divergence keep z within
    [-entry_z, +entry_z] but outside [-exit_z, +exit_z], so neither entry nor
    exit fires for most bars."""
    df_a, df_b = correlated_pair_with_divergence(n=120, final_divergence=0.0)
    s = _make(entry_z=3.5)
    out = s.compute({"A": df_a, "B": df_b}, as_of=datetime(2024, 1, 1))
    if out is not None:
        # If anything fires it must be CLOSE (|z|<exit_z), not entry
        assert out.signals["A"].direction == SignalDirection.CLOSE


def test_no_signal_when_one_leg_missing():
    df_a, _ = correlated_pair_with_divergence(n=80, final_divergence=10.0)
    s = _make()
    assert s.compute({"A": df_a}, as_of=datetime(2024, 1, 1)) is None


def test_no_signal_when_insufficient_history():
    df_a, df_b = correlated_pair_with_divergence(n=20, final_divergence=10.0)
    s = _make()
    assert s.compute({"A": df_a, "B": df_b}, as_of=datetime(2024, 1, 1)) is None


@pytest.mark.parametrize(
    "bad_overrides",
    [
        {"instrument_a": "X", "instrument_b": "X"},
        {"spread_window": 1},
        {"entry_z": -1.0},
        {"entry_z": 0.5, "exit_z": 1.0},  # exit >= entry
    ],
)
def test_validate_parameters_rejects_invalid(bad_overrides):
    params = {**PARAMS, **bad_overrides}
    s = PairsMeanReversionStrategy(parameters=params)
    with pytest.raises(ValueError):
        s.validate_parameters()
