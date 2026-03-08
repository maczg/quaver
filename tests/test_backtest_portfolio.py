"""Unit tests for Portfolio."""

import pytest
from datetime import datetime
from quaver.backtest.portfolio import Portfolio
from quaver.strategies.base import SignalOutput
from quaver.types import SignalDirection


def make_signal(direction=SignalDirection.BUY):
    return SignalOutput(direction=direction, confidence=0.8)


TS = datetime(2020, 1, 1)
TS2 = datetime(2020, 1, 2)


def test_open_close_long_pnl():
    p = Portfolio(initial_capital=10_000, quantity_per_trade=1.0)
    p.open_long("AAPL", TS, price=100.0, signal=make_signal(SignalDirection.BUY))
    record = p.close_long(TS2, price=110.0, signal=make_signal(SignalDirection.SELL))
    assert record.pnl == pytest.approx(10.0)
    assert record.direction == SignalDirection.BUY


def test_open_close_short_pnl():
    p = Portfolio(initial_capital=10_000, quantity_per_trade=1.0)
    p.open_short("AAPL", TS, price=100.0, signal=make_signal(SignalDirection.SELL))
    record = p.close_short(TS2, price=90.0, signal=make_signal(SignalDirection.BUY))
    assert record.pnl == pytest.approx(10.0)
    assert record.direction == SignalDirection.SELL


def test_no_double_open_long(caplog):
    import logging
    p = Portfolio(initial_capital=10_000, quantity_per_trade=1.0)
    p.open_long("AAPL", TS, 100.0, make_signal())
    with caplog.at_level(logging.WARNING):
        p.open_long("AAPL", TS2, 110.0, make_signal())
    assert len(p._closed_trades) == 0
    assert p._open_position.entry_price == 100.0   # unchanged


def test_reset_restores_state():
    p = Portfolio(initial_capital=5_000, quantity_per_trade=2.0)
    p.open_long("AAPL", TS, 50.0, make_signal())
    p.close_long(TS2, 60.0, make_signal(SignalDirection.SELL))
    p.reset()
    assert p.cash == 5_000
    assert p.is_flat()
    assert p.closed_trades == []


def test_cash_decreases_on_buy():
    p = Portfolio(initial_capital=10_000, quantity_per_trade=3.0)
    p.open_long("AAPL", TS, 100.0, make_signal())
    assert p.cash == pytest.approx(10_000 - 100.0 * 3.0)


def test_close_long_without_open_raises():
    p = Portfolio(initial_capital=10_000)
    with pytest.raises(RuntimeError):
        p.close_long(TS, 100.0, signal=None)


def test_close_short_without_open_raises():
    p = Portfolio(initial_capital=10_000)
    with pytest.raises(RuntimeError):
        p.close_short(TS, 100.0, signal=None)
