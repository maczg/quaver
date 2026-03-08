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
    """Replays a single instrument's candle history through a
    :class:`~quaver.strategies.base.BaseStrategy`.

    **Design invariants**

    - **No look-ahead bias**: the window passed to
      :meth:`~quaver.strategies.base.BaseStrategy.compute` is
      ``candles.iloc[:i]``, which excludes the bar at index *i* (the
      "current" bar).
    - The input DataFrame is **never** mutated.
    - Exceptions raised by
      :meth:`~quaver.strategies.base.BaseStrategy.compute` are **not**
      caught -- they propagate to the caller.
    - ``HOLD`` signals are logged at ``DEBUG`` level and treated as no-ops.
    - ``SELL`` signal routing:

      - If a long position is open -> :meth:`~Portfolio.close_long`.
      - If flat **and** ``allow_shorting`` is ``True`` ->
        :meth:`~Portfolio.open_short`.
      - If flat **and** ``allow_shorting`` is ``False`` -> logged and
        skipped.

    - :meth:`~Portfolio.reset` is called at the start of each
      :meth:`run` invocation so that running the engine twice produces
      identical results.

    :param strategy: Instantiated and validated
        :class:`~quaver.strategies.base.BaseStrategy`.
    :type strategy: BaseStrategy
    :param portfolio: :class:`~quaver.backtest.portfolio.Portfolio` instance
        (will be reset automatically at the start of each :meth:`run` call).
    :type portfolio: Portfolio
    :param instrument_id: Identifier embedded in
        :class:`~quaver.backtest.portfolio.TradeRecord` and
        :class:`~quaver.backtest.result.BacktestResult` objects.
    :type instrument_id: str
    :param ts_column: Name of the timestamp column in the candles DataFrame.
        Defaults to ``"ts"``.
    :type ts_column: str
    :param allow_shorting: When ``True``, ``SELL`` signals received while the
        portfolio is flat will open a short position.  Defaults to ``False``
        (suitable for most long-only mean-reversion strategies).
    :type allow_shorting: bool
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
        """Run the backtest over the full candles history.

        The portfolio is reset before iteration begins.  Any position that
        remains open at the end of the data is force-closed at the final
        bar's close price with ``signal=None``.

        :param candles: Normalised OHLCV DataFrame (output of
            :func:`~quaver.backtest.data.normalise_candles`).
        :type candles: pandas.DataFrame
        :returns: :class:`~quaver.backtest.result.BacktestResult` summarising
            all trades and computed performance metrics.
        :rtype: BacktestResult
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
        """Translate a :class:`~quaver.strategies.base.SignalOutput` into
        portfolio operations.

        Routing rules:

        - ``BUY``: open a long position if the portfolio is flat; otherwise
          log at ``DEBUG`` and ignore.
        - ``SELL``: close an existing long, or (if flat and
          ``allow_shorting``) open a short; otherwise log and ignore.
        - ``CLOSE``: close whichever position is currently open (long or
          short); no-op if flat.
        - ``HOLD``: log at ``DEBUG`` level; no portfolio action taken.

        :param signal: Signal emitted by the strategy for the current bar.
        :type signal: SignalOutput
        :param as_of: Timestamp of the current bar.
        :type as_of: datetime
        :param price: Close price of the current bar used as the execution
            price.
        :type price: float
        :returns: None
        :rtype: None
        """
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
