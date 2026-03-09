"""Tests for stop-loss, take-profit, and trailing-stop exit rules."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from quaver.backtest.engine import BacktestEngine
from quaver.backtest.multi_engine import MultiAssetBacktestEngine
from quaver.backtest.portfolio import (
    CommissionConfig,
    ExitRules,
    Portfolio,
    SlippageConfig,
)
from quaver.strategies.base import (
    BaseStrategy,
    MultiAssetStrategy,
    MultiAssetStrategyOutput,
    SignalOutput,
)
from quaver.types import ExitReason, SignalDirection


# ── Helpers ──────────────────────────────────────────────────────────────────


BASE_TS = datetime(2020, 1, 1)


def ts(i: int) -> datetime:
    return BASE_TS + timedelta(days=i)


def make_candles(
    bars: list[tuple[float, float, float, float]],
    start: int = 0,
) -> pd.DataFrame:
    """Create a candles DataFrame from (open, high, low, close) tuples."""
    rows = []
    for idx, (o, h, lo, c) in enumerate(bars):
        rows.append(
            {"ts": ts(start + idx), "open": o, "high": h, "low": lo, "close": c, "volume": 100}
        )
    return pd.DataFrame(rows)


class StubStrategy(BaseStrategy):
    """Strategy that emits a BUY on first call and HOLD thereafter."""

    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        *,
        signals: list[SignalOutput | None] | None = None,
    ) -> None:
        super().__init__(parameters or {})
        self._signals = signals or []
        self._call_count = 0

    def validate_parameters(self) -> None:
        pass

    def get_required_candle_count(self) -> int:
        return 1

    def compute(self, candles: pd.DataFrame, as_of: datetime) -> SignalOutput | None:
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._signals):
            return self._signals[idx]
        return None

    @property
    def call_count(self) -> int:
        return self._call_count


class StubMultiStrategy(MultiAssetStrategy):
    """Multi-asset strategy that emits prescribed signals."""

    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        *,
        signal_sequence: list[MultiAssetStrategyOutput | None] | None = None,
    ) -> None:
        super().__init__(parameters or {})
        self._signal_sequence = signal_sequence or []
        self._call_count = 0

    def validate_parameters(self) -> None:
        pass

    def get_required_candle_count(self) -> int:
        return 1

    def compute(
        self,
        candles_map: dict[str, pd.DataFrame],
        as_of: datetime,
    ) -> MultiAssetStrategyOutput | None:
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._signal_sequence):
            return self._signal_sequence[idx]
        return None


# ── Test cases ───────────────────────────────────────────────────────────────


class TestStopLossLong:
    def test_sl_long_triggers_on_low(self) -> None:
        """SL long triggers when low <= stop_loss_price, fills at SL price."""
        buy_signal = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        strategy = StubStrategy(signals=[buy_signal])
        rules = ExitRules(stop_loss_pct=0.05)  # 5% SL
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules)
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        # Bar 0: skipped (required=1)
        # Bar 1: strategy emits BUY, entry at close=100
        # Bar 2: low=94 breaches SL at 95, should exit at 95
        candles = make_candles(
            [
                (100, 105, 95, 100),  # bar 0 (warmup)
                (100, 105, 99, 100),  # bar 1 → BUY at 100
                (100, 102, 94, 101),  # bar 2 → SL triggers (low=94 <= 95)
                (101, 110, 100, 105),  # bar 3 → should not reach strategy
            ]
        )
        result = engine.run(candles)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == ExitReason.STOP_LOSS
        assert trade.exit_price == pytest.approx(95.0)  # SL price, not close
        assert trade.pnl == pytest.approx(-5.0)  # (95 - 100) * 1

    def test_sl_long_no_trigger_above(self) -> None:
        """SL does not trigger when low stays above SL price."""
        buy_signal = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        strategy = StubStrategy(signals=[buy_signal])
        rules = ExitRules(stop_loss_pct=0.05)
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules)
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (100, 102, 96, 101),  # low=96 > 95, no trigger
            ]
        )
        result = engine.run(candles)

        # Position force-closed at end of data
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == ExitReason.END_OF_DATA


class TestStopLossShort:
    def test_sl_short_triggers_on_high(self) -> None:
        """SL short triggers when high >= stop_loss_price."""
        sell_signal = SignalOutput(direction=SignalDirection.SELL, confidence=0.9)
        strategy = StubStrategy(signals=[sell_signal])
        rules = ExitRules(stop_loss_pct=0.05)  # SL for short = entry * 1.05 = 105
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules)
        engine = BacktestEngine(
            strategy=strategy,
            portfolio=portfolio,
            instrument_id="TEST",
            allow_shorting=True,
        )

        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 102, 98, 100),  # SELL (short) at 100
                (101, 106, 99, 103),  # high=106 >= 105, SL triggers
            ]
        )
        result = engine.run(candles)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == ExitReason.STOP_LOSS
        assert trade.exit_price == pytest.approx(105.0)
        assert trade.pnl == pytest.approx(-5.0)  # (100 - 105) * 1


class TestTakeProfitLong:
    def test_tp_long_triggers_on_high(self) -> None:
        """TP long triggers when high >= take_profit_price."""
        buy_signal = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        strategy = StubStrategy(signals=[buy_signal])
        rules = ExitRules(take_profit_pct=0.10)  # TP at 110
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules)
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (102, 112, 101, 108),  # high=112 >= 110, TP triggers
            ]
        )
        result = engine.run(candles)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == ExitReason.TAKE_PROFIT
        assert trade.exit_price == pytest.approx(110.0)
        assert trade.pnl == pytest.approx(10.0)


class TestTakeProfitShort:
    def test_tp_short_triggers_on_low(self) -> None:
        """TP short triggers when low <= take_profit_price."""
        sell_signal = SignalOutput(direction=SignalDirection.SELL, confidence=0.9)
        strategy = StubStrategy(signals=[sell_signal])
        rules = ExitRules(take_profit_pct=0.10)  # TP at 90 for short
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules)
        engine = BacktestEngine(
            strategy=strategy,
            portfolio=portfolio,
            instrument_id="TEST",
            allow_shorting=True,
        )

        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 102, 98, 100),  # SELL (short) at 100
                (97, 99, 88, 92),  # low=88 <= 90, TP triggers
            ]
        )
        result = engine.run(candles)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == ExitReason.TAKE_PROFIT
        assert trade.exit_price == pytest.approx(90.0)
        assert trade.pnl == pytest.approx(10.0)


class TestSameBarSLAndTP:
    def test_sl_wins_over_tp_pessimistic(self) -> None:
        """When both SL and TP trigger on the same bar, SL wins (pessimistic)."""
        buy_signal = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        strategy = StubStrategy(signals=[buy_signal])
        rules = ExitRules(stop_loss_pct=0.05, take_profit_pct=0.10)
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules)
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        # Entry at 100, SL=95, TP=110
        # Bar with low=94 and high=112 → both trigger, SL should win
        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (100, 112, 94, 105),  # both SL and TP trigger
            ]
        )
        result = engine.run(candles)

        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == ExitReason.STOP_LOSS


class TestTrailingStopLong:
    def test_trailing_stop_long(self) -> None:
        """Trailing stop tracks highest high for long, triggers on pullback."""
        buy_signal = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        strategy = StubStrategy(signals=[buy_signal])
        rules = ExitRules(trailing_stop_pct=0.05)
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules)
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        # Entry at 100, trailing extreme starts at 100
        # Bar 2: high=110, extreme→110, trail=104.5, low=108 → no trigger
        # Bar 3: high=120, extreme→120, trail=114, low=115 → no trigger
        # Bar 4: high=121, extreme→121, trail=114.95, low=113 → trigger! fill at 114.95
        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (102, 110, 108, 109),  # extreme=110, trail=104.5
                (110, 120, 115, 118),  # extreme=120, trail=114
                (118, 121, 113, 115),  # extreme=121, trail=114.95, low=113 triggers
            ]
        )
        result = engine.run(candles)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == ExitReason.TRAILING_STOP
        assert trade.exit_price == pytest.approx(121 * 0.95)


class TestTrailingStopShort:
    def test_trailing_stop_short(self) -> None:
        """Trailing stop tracks lowest low for short, triggers on bounce."""
        sell_signal = SignalOutput(direction=SignalDirection.SELL, confidence=0.9)
        strategy = StubStrategy(signals=[sell_signal])
        rules = ExitRules(trailing_stop_pct=0.05)
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules)
        engine = BacktestEngine(
            strategy=strategy,
            portfolio=portfolio,
            instrument_id="TEST",
            allow_shorting=True,
        )

        # Entry at 100, extreme starts at 100
        # Bar 2: low=90, extreme→90, trail=94.5, high=92 → no trigger
        # Bar 3: low=80, extreme→80, trail=84, high=82 → no trigger
        # Bar 4: low=79, extreme→79, trail=82.95, high=83 → trigger
        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 102, 98, 100),  # SELL (short) at 100
                (98, 92, 90, 91),  # extreme=90, trail=94.5
                (91, 82, 80, 81),  # extreme=80, trail=84
                (81, 83, 79, 82),  # extreme=79, trail=82.95, high=83 triggers
            ]
        )
        result = engine.run(candles)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == ExitReason.TRAILING_STOP
        assert trade.exit_price == pytest.approx(79 * 1.05)


class TestPerTradeMetadataOverride:
    def test_metadata_stop_loss_overrides_global(self) -> None:
        """Per-trade metadata stop_loss overrides global stop_loss_pct."""
        # Global SL at 5% → 95, but metadata SL at 92
        buy_signal = SignalOutput(
            direction=SignalDirection.BUY,
            confidence=0.9,
            metadata={"stop_loss": 92.0},
        )
        strategy = StubStrategy(signals=[buy_signal])
        rules = ExitRules(stop_loss_pct=0.05)
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules)
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (100, 102, 93, 99),  # low=93, global SL=95 would trigger but override SL=92 doesn't
                (99, 101, 91, 93),  # low=91, SL=92 triggers
            ]
        )
        result = engine.run(candles)

        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == ExitReason.STOP_LOSS
        assert result.trades[0].exit_price == pytest.approx(92.0)

    def test_metadata_take_profit_overrides_global(self) -> None:
        """Per-trade metadata take_profit overrides global take_profit_pct."""
        buy_signal = SignalOutput(
            direction=SignalDirection.BUY,
            confidence=0.9,
            metadata={"take_profit": 115.0},
        )
        strategy = StubStrategy(signals=[buy_signal])
        rules = ExitRules(take_profit_pct=0.05)  # global TP = 105
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules)
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (
                    102,
                    108,
                    101,
                    107,
                ),  # high=108, global TP=105 would trigger but override TP=115 doesn't
                (107, 116, 106, 115),  # high=116, TP=115 triggers
            ]
        )
        result = engine.run(candles)

        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == ExitReason.TAKE_PROFIT
        assert result.trades[0].exit_price == pytest.approx(115.0)

    def test_metadata_trailing_stop_pct_overrides_global(self) -> None:
        """Per-trade metadata trailing_stop_pct overrides global."""
        buy_signal = SignalOutput(
            direction=SignalDirection.BUY,
            confidence=0.9,
            metadata={"trailing_stop_pct": 0.10},  # 10% trail vs global 5%
        )
        strategy = StubStrategy(signals=[buy_signal])
        rules = ExitRules(trailing_stop_pct=0.05)
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules)
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (102, 110, 101, 109),  # extreme=110, 5% trail=104.5, 10% trail=99
                (
                    108,
                    109,
                    103,
                    106,
                ),  # 5% trail would trigger (low=103<104.5), 10% trail=99 doesn't
            ]
        )
        result = engine.run(candles)

        # Should NOT trigger with 10% trail; force-closed at EOD
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == ExitReason.END_OF_DATA


class TestNoExitRulesBackwardCompat:
    def test_no_exit_rules_no_triggers(self) -> None:
        """Without exit_rules, no triggers fire — backward compatible."""
        buy_signal = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        strategy = StubStrategy(signals=[buy_signal])
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0)
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (100, 200, 50, 100),  # huge range, but no rules → no trigger
            ]
        )
        result = engine.run(candles)

        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == ExitReason.END_OF_DATA


class TestFillAtTriggerPrice:
    def test_fill_at_sl_price_not_close(self) -> None:
        """Exit fills at the trigger price level, not at bar close."""
        buy_signal = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        strategy = StubStrategy(signals=[buy_signal])
        rules = ExitRules(stop_loss_pct=0.05)
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules)
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (99, 101, 90, 92),  # SL=95, close=92, should fill at 95 not 92
            ]
        )
        result = engine.run(candles)

        assert result.trades[0].exit_price == pytest.approx(95.0)
        assert result.trades[0].exit_price != 92.0


class TestCommissionSlippageOnExitRule:
    def test_commission_applied_on_sl_exit(self) -> None:
        """Commission and slippage apply on SL/TP exit."""
        buy_signal = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        strategy = StubStrategy(signals=[buy_signal])
        rules = ExitRules(stop_loss_pct=0.05)
        comm = CommissionConfig(fixed_per_trade=1.0)
        slip = SlippageConfig(slippage_pct=0.01)
        portfolio = Portfolio(
            initial_capital=100_000,
            quantity_per_trade=1.0,
            commission=comm,
            slippage=slip,
            exit_rules=rules,
        )
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100 (fill at 101 due to slippage)
                (99, 101, 90, 92),  # SL at 95
            ]
        )
        result = engine.run(candles)

        trade = result.trades[0]
        assert trade.commission > 0
        assert trade.slippage_cost > 0


class TestForceCloseExitReason:
    def test_force_close_eod_exit_reason(self) -> None:
        """Force-close at end of data gets exit_reason=END_OF_DATA."""
        buy_signal = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        strategy = StubStrategy(signals=[buy_signal])
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0)
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (100, 105, 99, 102),  # no signal, position stays open
            ]
        )
        result = engine.run(candles)

        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == ExitReason.END_OF_DATA


class TestExitBeforeCompute:
    def test_exit_fires_before_strategy_compute(self) -> None:
        """When an exit triggers, strategy.compute is NOT called for that bar."""
        buy_signal = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        strategy = StubStrategy(signals=[buy_signal])
        rules = ExitRules(stop_loss_pct=0.05)
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules)
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        candles = make_candles(
            [
                (100, 105, 95, 100),  # bar 0 (warmup)
                (100, 105, 99, 100),  # bar 1 → compute called (BUY)
                (99, 101, 90, 92),  # bar 2 → SL triggers, compute NOT called
                (92, 95, 90, 93),  # bar 3 → compute called (no signal)
            ]
        )
        engine.run(candles)

        # strategy.compute should be called for bars 1 and 3, NOT bar 2
        # bar 2 triggers SL before compute, then bar 3 calls compute
        assert strategy.call_count == 2


class TestSignalExitReason:
    def test_signal_close_gets_signal_exit_reason(self) -> None:
        """A CLOSE signal exit gets exit_reason=SIGNAL."""
        buy_signal = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        close_signal = SignalOutput(direction=SignalDirection.CLOSE, confidence=0.9)
        strategy = StubStrategy(signals=[buy_signal, close_signal])
        portfolio = Portfolio(initial_capital=10_000, quantity_per_trade=1.0)
        engine = BacktestEngine(strategy=strategy, portfolio=portfolio, instrument_id="TEST")

        candles = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY
                (102, 108, 101, 105),  # CLOSE
            ]
        )
        result = engine.run(candles)

        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == ExitReason.SIGNAL


class TestMultiAssetExitRules:
    def test_multi_asset_sl_triggers(self) -> None:
        """Exit rules work in multi-asset engine."""
        buy_a = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        buy_b = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        output = MultiAssetStrategyOutput(signals={"A": buy_a, "B": buy_b})
        strategy = StubMultiStrategy(signal_sequence=[output])
        rules = ExitRules(stop_loss_pct=0.05)

        portfolios = {
            "A": Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules),
            "B": Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules),
        }
        engine = MultiAssetBacktestEngine(
            strategy=strategy,
            portfolios=portfolios,
            allow_shorting=True,
        )

        candles_a = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (99, 101, 90, 92),  # SL at 95, low=90 triggers
            ]
        )
        candles_b = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (100, 108, 99, 105),  # no trigger
            ]
        )

        results = engine.run({"A": candles_a, "B": candles_b})

        # A should have SL exit
        assert len(results["A"].trades) == 1
        assert results["A"].trades[0].exit_reason == ExitReason.STOP_LOSS
        assert results["A"].trades[0].exit_price == pytest.approx(95.0)

        # B should be force-closed at EOD
        assert len(results["B"].trades) == 1
        assert results["B"].trades[0].exit_reason == ExitReason.END_OF_DATA

    def test_multi_asset_triggered_instruments_skip_signals(self) -> None:
        """Instruments that exit via trigger skip strategy signals for that bar."""
        buy_a = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        buy_b = SignalOutput(direction=SignalDirection.BUY, confidence=0.9)
        output1 = MultiAssetStrategyOutput(signals={"A": buy_a, "B": buy_b})
        # Second output tries to CLOSE A, but A already exited via SL
        close_a = SignalOutput(direction=SignalDirection.CLOSE, confidence=0.9)
        output2 = MultiAssetStrategyOutput(signals={"A": close_a})

        strategy = StubMultiStrategy(signal_sequence=[output1, output2])
        rules = ExitRules(stop_loss_pct=0.05)

        portfolios = {
            "A": Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules),
            "B": Portfolio(initial_capital=10_000, quantity_per_trade=1.0, exit_rules=rules),
        }
        engine = MultiAssetBacktestEngine(
            strategy=strategy,
            portfolios=portfolios,
        )

        candles_a = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (99, 101, 90, 92),  # SL triggers
            ]
        )
        candles_b = make_candles(
            [
                (100, 105, 95, 100),
                (100, 105, 99, 100),  # BUY at 100
                (100, 108, 99, 105),  # no trigger
            ]
        )

        results = engine.run({"A": candles_a, "B": candles_b})

        # A has exactly 1 trade (SL), not a RuntimeError from double-close
        assert len(results["A"].trades) == 1
        assert results["A"].trades[0].exit_reason == ExitReason.STOP_LOSS
