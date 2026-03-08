"""Single-asset walk-forward backtest engine."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from quaver.strategies.base import BaseStrategy, SignalOutput
from quaver.types import SignalDirection
from quaver.backtest.portfolio import Portfolio
from quaver.backtest.result import BacktestResult

log = logging.getLogger(__name__)


class BacktestEngine:
    """
    Replays a single instrument's candle history through a BaseStrategy.

    Design invariants:
        - NO look-ahead bias: window passed to compute() = candles.iloc[:i],
          which excludes the bar at index i (the "current" bar).
        - Input DataFrame is never mutated.
        - Exceptions from strategy.compute() are NOT caught — they propagate.
        - HOLD signals are logged at DEBUG level and treated as no-ops.
        - SELL signals:
            * If a long position is open  → close_long.
            * If flat AND allow_shorting  → open_short.
            * If flat AND NOT allow_shorting → logged and skipped.
        - portfolio.reset() is called at the start of each run() call so that
          running the engine twice produces identical results.

    Args:
        strategy: Instantiated and validated BaseStrategy.
        portfolio: Portfolio instance (will be reset on each run()).
        instrument_id: Identifier used in TradeRecord and BacktestResult.
        ts_column: Name of the timestamp column (default "ts").
        allow_shorting: If True, SELL signals from a flat portfolio open a short.
                        Default False (most mean-reversion strategies are long-only).
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        instrument_id: str,
        ts_column: str = "ts",
        allow_shorting: bool = False,
    ) -> None:
        self.strategy = strategy
        self.portfolio = portfolio
        self.instrument_id = instrument_id
        self.ts_column = ts_column
        self.allow_shorting = allow_shorting

    def run(self, candles: pd.DataFrame) -> BacktestResult:
        """
        Run the backtest over the full candles history.

        Args:
            candles: Normalised OHLCV DataFrame (output of normalise_candles).

        Returns:
            BacktestResult summarising all trades and metrics.
        """
        self.portfolio.reset()

        required = self.strategy.get_required_candle_count()
        n = len(candles)
        last_signal: SignalOutput | None = None

        for i in range(required, n):
            window = candles.iloc[:i]               # excludes bar i
            current = candles.iloc[i]
            as_of: datetime = current[self.ts_column]
            price = float(current["close"])

            signal = self.strategy.compute(window, as_of)
            if signal is None:
                continue

            last_signal = signal
            self._apply_signal(signal, as_of, price)

        # Force-close any remaining open position at the final bar
        if not self.portfolio.is_flat():
            last = candles.iloc[-1]
            last_ts: datetime = last[self.ts_column]
            last_price = float(last["close"])
            log.debug(
                "Force-closing open position at end of data: ts=%s price=%.4f",
                last_ts,
                last_price,
            )
            pos = self.portfolio._open_position
            if pos is not None and pos.direction == SignalDirection.BUY:
                self.portfolio.close_long(last_ts, last_price, signal=None)
            else:
                self.portfolio.close_short(last_ts, last_price, signal=None)

        return BacktestResult.from_portfolio(
            self.portfolio, candles, self.instrument_id
        )

    def _apply_signal(
        self,
        signal: SignalOutput,
        as_of: datetime,
        price: float,
    ) -> None:
        """Translate a SignalOutput into portfolio operations."""
        direction = signal.direction

        if direction == SignalDirection.BUY:
            if self.portfolio.is_flat():
                self.portfolio.open_long(self.instrument_id, as_of, price, signal)
            else:
                log.debug("BUY signal ignored: position already open")

        elif direction == SignalDirection.SELL:
            pos = self.portfolio._open_position
            if pos is not None and pos.direction == SignalDirection.BUY:
                # Close an existing long
                self.portfolio.close_long(as_of, price, signal)
            elif self.portfolio.is_flat():
                if self.allow_shorting:
                    self.portfolio.open_short(self.instrument_id, as_of, price, signal)
                else:
                    log.debug(
                        "SELL signal ignored: flat portfolio and allow_shorting=False"
                    )
            else:
                # Already short — ignore
                log.debug("SELL signal ignored: short position already open")

        elif direction == SignalDirection.CLOSE:
            if not self.portfolio.is_flat():
                pos = self.portfolio._open_position
                if pos is not None and pos.direction == SignalDirection.BUY:
                    self.portfolio.close_long(as_of, price, signal)
                else:
                    self.portfolio.close_short(as_of, price, signal)

        elif direction == SignalDirection.HOLD:
            log.debug("HOLD signal at %s — no action", as_of)
