"""Unit tests for BacktestEngine."""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from quaver.backtest.engine import BacktestEngine
from quaver.backtest.portfolio import Portfolio
from quaver.backtest.data import normalise_candles
from quaver.strategies.base import BaseStrategy, SignalOutput
from quaver.types import SignalDirection


class StubStrategy(BaseStrategy):
    """Returns BUY at bar index `buy_bar`, SELL at `sell_bar`, None otherwise."""

    def __init__(self, buy_bar: int, sell_bar: int):
        super().__init__(parameters={})
        self.buy_bar = buy_bar
        self.sell_bar = sell_bar
        self._call_count = 0
        self._window_sizes = []

    def validate_parameters(self) -> None:
        pass

    def get_required_candle_count(self) -> int:
        return 10

    def compute(self, candles: pd.DataFrame, as_of: datetime) -> SignalOutput | None:
        self._window_sizes.append(len(candles))
        self._call_count += 1
        n = len(candles)
        if n == self.buy_bar:
            return SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        if n == self.sell_bar:
            return SignalOutput(direction=SignalDirection.SELL, confidence=0.9)
        return None


def make_candles_df(n=100, start_price=100.0):
    rows = []
    ts = datetime(2020, 1, 1)
    price = start_price
    for _ in range(n):
        rows.append(
            {
                "ts": ts,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1000.0,
            }
        )
        price += 0.5
        ts += timedelta(days=1)
    return pd.DataFrame(rows)


def test_no_lookahead_bias():
    """Window passed to compute() must never contain the current bar."""
    strategy = StubStrategy(buy_bar=999, sell_bar=999)  # never fires
    portfolio = Portfolio(initial_capital=10_000)
    engine = BacktestEngine(strategy, portfolio, "TEST")
    df = normalise_candles(make_candles_df(50))
    engine.run(df)
    # Each window size must equal its call index (0-based from required start)
    required = strategy.get_required_candle_count()
    for idx, ws in enumerate(strategy._window_sizes):
        expected = required + idx
        assert ws == expected, f"Call {idx}: expected window {expected}, got {ws}"


def test_single_trade_cycle():
    """BUY followed by SELL produces exactly one TradeRecord."""
    strategy = StubStrategy(buy_bar=20, sell_bar=40)
    portfolio = Portfolio(initial_capital=10_000)
    engine = BacktestEngine(strategy, portfolio, "TEST")
    df = normalise_candles(make_candles_df(60))
    result = engine.run(df)
    assert result.total_trades == 1


def test_force_close_at_end():
    """Open position with no SELL signal is force-closed at the last bar."""
    strategy = StubStrategy(buy_bar=20, sell_bar=9999)  # SELL never fires
    portfolio = Portfolio(initial_capital=10_000)
    engine = BacktestEngine(strategy, portfolio, "TEST")
    df = normalise_candles(make_candles_df(50))
    result = engine.run(df)
    assert result.total_trades == 1
    assert portfolio.is_flat()


def test_empty_candles_zero_trades():
    """Fewer bars than required returns a result with zero trades."""
    strategy = StubStrategy(buy_bar=5, sell_bar=8)
    portfolio = Portfolio(initial_capital=10_000)
    engine = BacktestEngine(strategy, portfolio, "TEST")
    df = normalise_candles(make_candles_df(5))  # less than required (10)
    result = engine.run(df)
    assert result.total_trades == 0


def test_none_signal_ignored():
    """compute() returning None never opens a position."""
    strategy = StubStrategy(buy_bar=9999, sell_bar=9999)
    portfolio = Portfolio(initial_capital=10_000)
    engine = BacktestEngine(strategy, portfolio, "TEST")
    df = normalise_candles(make_candles_df(50))
    result = engine.run(df)
    assert result.total_trades == 0


def test_sell_from_flat_ignored_without_allow_shorting():
    """SELL from a flat portfolio is ignored when allow_shorting=False."""
    strategy = StubStrategy(buy_bar=9999, sell_bar=20)  # only SELL fires
    portfolio = Portfolio(initial_capital=10_000)
    engine = BacktestEngine(strategy, portfolio, "TEST", allow_shorting=False)
    df = normalise_candles(make_candles_df(50))
    result = engine.run(df)
    assert result.total_trades == 0


def test_sell_from_flat_opens_short_with_allow_shorting():
    """SELL from a flat portfolio opens a short when allow_shorting=True."""
    strategy = StubStrategy(buy_bar=9999, sell_bar=20)
    portfolio = Portfolio(initial_capital=10_000)
    engine = BacktestEngine(strategy, portfolio, "TEST", allow_shorting=True)
    df = normalise_candles(make_candles_df(50))
    result = engine.run(df)
    # force-close at end counts as one trade
    assert result.total_trades == 1


def test_run_is_idempotent():
    """Running the engine twice on the same candles produces identical results."""
    strategy = StubStrategy(buy_bar=20, sell_bar=40)
    portfolio = Portfolio(initial_capital=10_000)
    engine = BacktestEngine(strategy, portfolio, "TEST")
    df = normalise_candles(make_candles_df(60))
    r1 = engine.run(df)
    r2 = engine.run(df)
    assert r1.total_trades == r2.total_trades
    assert r1.final_cash == pytest.approx(r2.final_cash)
