"""Unit tests for BacktestResult."""

import pytest
import math
from datetime import datetime
from quaver.backtest.result import BacktestResult
from quaver.backtest.portfolio import TradeRecord
from quaver.strategies.base import SignalOutput
from quaver.types import SignalDirection


def make_trade(pnl: float, direction=SignalDirection.BUY) -> TradeRecord:
    sig = SignalOutput(direction=direction, confidence=0.5)
    return TradeRecord(
        instrument_id="X",
        entry_ts=datetime(2020, 1, 1),
        exit_ts=datetime(2020, 1, 2),
        entry_price=100.0,
        exit_price=100.0 + pnl,
        quantity=1.0,
        direction=direction,
        pnl=pnl,
        entry_signal=sig,
        exit_signal=None,
    )


def make_result(pnls: list[float], initial=10_000.0) -> BacktestResult:
    trades = [make_trade(p) for p in pnls]
    final = initial + sum(pnls)
    return BacktestResult(
        instrument_id="X",
        initial_capital=initial,
        final_cash=final,
        trades=trades,
    )


def test_total_return():
    r = make_result([100.0, -50.0], initial=1_000.0)
    assert r.total_return == pytest.approx(0.05)


def test_win_rate():
    r = make_result([10.0, -5.0, 10.0, -5.0])
    assert r.win_rate == pytest.approx(0.5)


def test_max_drawdown():
    # cumulative pnl: [10, -10, 5] => running: [10, 0, 5]
    # peak=10 at idx 0, trough=0 at idx 1 → dd = (0-10)/initial = -10/10000
    r = make_result([10.0, -10.0, 5.0], initial=10_000.0)
    assert r.max_drawdown == pytest.approx(-10.0 / 10_000.0, abs=1e-6)


def test_max_drawdown_zero_with_few_trades():
    r = make_result([10.0])
    assert r.max_drawdown == 0.0


def test_sharpe_ratio_zero_with_one_trade():
    r = make_result([10.0])
    assert r.sharpe_ratio == 0.0


def test_sharpe_ratio_nonzero():
    import numpy as np
    pnls = [10.0, -5.0, 8.0, -3.0, 12.0]
    r = make_result(pnls)
    arr = [10.0, -5.0, 8.0, -3.0, 12.0]
    expected = float(np.mean(arr)) / float(np.std(arr, ddof=1)) * math.sqrt(252)
    assert r.sharpe_ratio == pytest.approx(expected, rel=1e-3)


def test_profit_factor():
    r = make_result([10.0, 5.0, -3.0, -2.0])
    assert r.profit_factor == pytest.approx(15.0 / 5.0)


def test_profit_factor_no_losses():
    r = make_result([10.0, 5.0])
    assert r.profit_factor == float("inf")


def test_summary_keys():
    r = make_result([10.0, -5.0])
    s = r.summary()
    required_keys = {
        "instrument_id", "initial_capital", "final_cash",
        "total_return_pct", "total_trades", "winning_trades",
        "losing_trades", "win_rate_pct", "avg_pnl",
        "profit_factor", "sharpe_ratio", "max_drawdown_pct",
    }
    assert required_keys.issubset(s.keys())


def test_empty_result():
    r = make_result([])
    assert r.total_trades == 0
    assert r.win_rate == 0.0
    assert r.max_drawdown == 0.0
    assert r.sharpe_ratio == 0.0
    assert r.profit_factor == float("inf")
