"""Unit tests for risk-based position sizing."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pytest

from quaver.backtest.portfolio import Portfolio
from quaver.backtest.sizing import size_by_risk
from quaver.strategies.base import SignalOutput
from quaver.types import SignalDirection


# ── size_by_risk ─────────────────────────────────────────────────────────────


class TestSizeByRisk:
    def test_basic_calculation(self) -> None:
        # Risk 2% of 10000 = 200. Entry 100, stop 95 → risk/unit = 5.
        # qty = 200 / 5 = 40
        qty = size_by_risk(10_000, 0.02, 100.0, 95.0)
        assert qty == pytest.approx(40.0)

    def test_short_stop_above_entry(self) -> None:
        # Short: entry 100, stop 105 → risk/unit = 5
        qty = size_by_risk(10_000, 0.02, 100.0, 105.0)
        assert qty == pytest.approx(40.0)

    def test_min_quantity_floor(self) -> None:
        # Tiny account → computed qty < min_quantity
        qty = size_by_risk(100, 0.01, 100.0, 95.0, min_quantity=1.0)
        # 100 * 0.01 / 5 = 0.2 → floored to 1.0
        assert qty == pytest.approx(1.0)

    def test_custom_min_quantity(self) -> None:
        qty = size_by_risk(100, 0.01, 100.0, 95.0, min_quantity=5.0)
        assert qty == pytest.approx(5.0)

    def test_same_entry_and_stop_raises(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            size_by_risk(10_000, 0.02, 100.0, 100.0)

    def test_large_risk_pct(self) -> None:
        qty = size_by_risk(10_000, 0.10, 100.0, 90.0)
        # 1000 / 10 = 100
        assert qty == pytest.approx(100.0)


# ── Portfolio integration with sizing_fn ─────────────────────────────────────


def _make_signal(direction: SignalDirection = SignalDirection.BUY) -> SignalOutput:
    return SignalOutput(direction=direction, confidence=0.8)


TS = datetime(2020, 1, 1)
TS2 = datetime(2020, 1, 2)


def _make_sizing_fn(risk_pct: float, stop_loss: float) -> "Callable[[float, float], float]":
    """Create a sizing_fn closure compatible with Portfolio's (account_value, entry_price) signature."""

    def fn(account_value: float, entry_price: float) -> float:
        return size_by_risk(account_value, risk_pct, entry_price, stop_loss)

    return fn


def test_portfolio_sizing_fn_overrides_quantity() -> None:
    """When sizing_fn is provided, quantity_per_trade is ignored."""
    fn = _make_sizing_fn(risk_pct=0.02, stop_loss=95.0)
    p = Portfolio(initial_capital=10_000, quantity_per_trade=999.0, sizing_fn=fn)
    p.open_long("AAPL", TS, 100.0, _make_signal())
    pos = p._open_position
    assert pos is not None
    # size_by_risk(10_000, 0.02, 100.0, 95.0) → 200/5 = 40
    assert pos.quantity == pytest.approx(40.0)


def test_portfolio_sizing_fn_none_uses_fixed_qty() -> None:
    """Without sizing_fn, quantity_per_trade is used."""
    p = Portfolio(initial_capital=10_000, quantity_per_trade=5.0)
    p.open_long("AAPL", TS, 100.0, _make_signal())
    pos = p._open_position
    assert pos is not None
    assert pos.quantity == pytest.approx(5.0)


def test_sizing_fn_with_short() -> None:
    fn = _make_sizing_fn(risk_pct=0.01, stop_loss=105.0)
    p = Portfolio(initial_capital=20_000, quantity_per_trade=1.0, sizing_fn=fn)
    p.open_short("AAPL", TS, 100.0, _make_signal(SignalDirection.SELL))
    pos = p._open_position
    assert pos is not None
    # size_by_risk(20_000, 0.01, 100.0, 105.0) → 200/5 = 40
    assert pos.quantity == pytest.approx(40.0)
