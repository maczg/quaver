"""Multi-asset walk-forward backtest engine."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from quaver.strategies.base import MultiAssetStrategy, SignalOutput
from quaver.types import ExitReason, SignalDirection
from quaver.backtest.portfolio import Portfolio
from quaver.backtest.result import BacktestResult

log = logging.getLogger(__name__)


class MultiAssetBacktestEngine:
    """Replays multiple instruments' candle histories through a
    :class:`~quaver.strategies.base.MultiAssetStrategy`.

    **Alignment**

    All instruments are aligned to the sorted intersection of their
    timestamp sets.  If the intersection discards more than
    ``timestamp_overlap_warn_threshold`` of any single instrument's
    timestamps a ``WARNING`` is logged.

    **Atomicity**

    All signals emitted in a single
    :class:`~quaver.strategies.base.MultiAssetStrategyOutput` are applied
    within the same iteration step, before advancing to the next timestamp.

    **Position management**

    Each instrument has its own independent
    :class:`~quaver.backtest.portfolio.Portfolio`.

    :param strategy: Instantiated and validated
        :class:`~quaver.strategies.base.MultiAssetStrategy`.
    :type strategy: MultiAssetStrategy
    :param portfolios: Mapping of ``instrument_id`` to
        :class:`~quaver.backtest.portfolio.Portfolio`, one entry per
        instrument.
    :type portfolios: dict[str, Portfolio]
    :param ts_column: Timestamp column name shared by all candle DataFrames.
        Defaults to ``"ts"``.
    :type ts_column: str
    :param allow_shorting: When ``True``, ``SELL`` signals from flat
        portfolios open short positions.  Defaults to ``False``.
    :type allow_shorting: bool
    :param timestamp_overlap_warn_threshold: Fraction of timestamps that may
        be dropped by the intersection before a ``WARNING`` is emitted.
        Defaults to ``0.05`` (5%).
    :type timestamp_overlap_warn_threshold: float
    """

    def __init__(
        self,
        strategy: MultiAssetStrategy,
        portfolios: dict[str, Portfolio],
        ts_column: str = "ts",
        allow_shorting: bool = False,
        timestamp_overlap_warn_threshold: float = 0.05,
    ) -> None:
        self.strategy = strategy
        self.portfolios = portfolios
        self.ts_column = ts_column
        self.allow_shorting = allow_shorting
        self.timestamp_overlap_warn_threshold = timestamp_overlap_warn_threshold

    def run(self, candles_map: dict[str, pd.DataFrame]) -> dict[str, BacktestResult]:
        """Run the multi-asset backtest.

        All portfolios are reset before iteration begins.  Instruments are
        iterated over a shared timestamp intersection.  Any positions that
        remain open at the end of the data are force-closed at each
        instrument's final bar close price with ``signal=None``.

        :param candles_map: Mapping of ``instrument_id`` to a normalised
            OHLCV DataFrame (output of
            :func:`~quaver.backtest.data.normalise_candles`).
        :type candles_map: dict[str, pandas.DataFrame]
        :returns: Mapping of ``instrument_id`` to
            :class:`~quaver.backtest.result.BacktestResult`.
        :rtype: dict[str, BacktestResult]
        """
        # Reset all portfolios
        for p in self.portfolios.values():
            p.reset()

        # Build shared timeline (sorted intersection of all timestamps)
        ts_sets = {iid: set(df[self.ts_column].tolist()) for iid, df in candles_map.items()}
        shared_ts = sorted(set.intersection(*ts_sets.values()))

        # Warn if intersection is significantly smaller than any instrument's history
        for iid, ts_set in ts_sets.items():
            if len(ts_set) == 0:
                continue
            drop_fraction = 1.0 - len(shared_ts) / len(ts_set)
            if drop_fraction > self.timestamp_overlap_warn_threshold:
                log.warning(
                    "Timestamp intersection dropped %.1f%% of '%s' candles "
                    "(%d -> %d). Check for missing data.",
                    drop_fraction * 100,
                    iid,
                    len(ts_set),
                    len(shared_ts),
                )

        # Build fast lookup: instrument_id -> {ts -> row index}
        ts_index: dict[str, dict[object, int]] = {
            iid: {ts: idx for idx, ts in enumerate(df[self.ts_column].tolist())}
            for iid, df in candles_map.items()
        }

        required = self.strategy.get_required_candle_count()

        for step, ts in enumerate(shared_ts):
            if step < required:
                continue

            # Check exit triggers for all instruments before strategy.compute
            triggered_instruments: set[str] = set()
            for iid, portfolio in self.portfolios.items():
                if portfolio.is_flat():
                    continue
                hl = self._get_high_low_at(candles_map[iid], ts)
                if hl is None:
                    continue
                high, low = hl
                trigger = portfolio.check_exit_triggers(ts, high, low)
                if trigger is not None:
                    reason, fill_price = trigger
                    self._close_position(portfolio, ts, fill_price, signal=None, exit_reason=reason)
                    triggered_instruments.add(iid)

            # Build window_map: each instrument's history up to (not including) ts
            window_map: dict[str, pd.DataFrame] = {}
            for iid, df in candles_map.items():
                row_idx = ts_index[iid].get(ts)
                if row_idx is None or row_idx == 0:
                    window_map[iid] = df.iloc[:0]  # empty
                else:
                    window_map[iid] = df.iloc[:row_idx]

            output = self.strategy.compute(window_map, as_of=ts)
            if output is None:
                continue

            # Apply all signals atomically (skip instruments that already exited)
            for instrument_id, signal in output.signals.items():
                if instrument_id in triggered_instruments:
                    continue
                if instrument_id not in candles_map:
                    log.warning("Signal for unknown instrument '%s' — skipping", instrument_id)
                    continue
                price = self._get_price_at(candles_map[instrument_id], ts)
                if price is None:
                    log.warning("No price for '%s' at %s — signal skipped", instrument_id, ts)
                    continue
                portfolio = self.portfolios[instrument_id]
                self._apply_signal(portfolio, instrument_id, signal, ts, price)

        # Force-close all remaining open positions at last available bar
        for iid, portfolio in self.portfolios.items():
            if not portfolio.is_flat():
                df = candles_map[iid]
                last = df.iloc[-1]
                last_ts: datetime = last[self.ts_column]
                last_price = float(last["close"])
                log.debug("Force-closing open position for '%s' at end of data", iid)
                self._close_position(
                    portfolio,
                    last_ts,
                    last_price,
                    signal=None,
                    exit_reason=ExitReason.END_OF_DATA,
                )

        return {
            iid: BacktestResult.from_portfolio(p, candles_map[iid], iid)
            for iid, p in self.portfolios.items()
        }

    def _get_high_low_at(self, df: pd.DataFrame, ts: datetime) -> tuple[float, float] | None:
        """Return ``(high, low)`` for ``df`` at timestamp ``ts``, or ``None``."""
        rows = df[df[self.ts_column] == ts]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return (float(row["high"]), float(row["low"]))

    def _close_position(
        self,
        portfolio: Portfolio,
        ts: datetime,
        price: float,
        signal: SignalOutput | None,
        exit_reason: ExitReason | None = None,
    ) -> None:
        """Close the currently open position in the given portfolio."""
        pos = portfolio._open_position
        if pos is None:
            return
        if pos.direction == SignalDirection.BUY:
            portfolio.close_long(ts, price, signal, exit_reason=exit_reason)
        else:
            portfolio.close_short(ts, price, signal, exit_reason=exit_reason)

    def _get_price_at(self, df: pd.DataFrame, ts: datetime) -> float | None:
        """Return the close price of ``df`` at timestamp ``ts``.

        :param df: Normalised OHLCV DataFrame for a single instrument.
        :type df: pandas.DataFrame
        :param ts: Timestamp to look up.
        :type ts: datetime
        :returns: Close price at ``ts``, or ``None`` if the timestamp is not
            present in ``df``.
        :rtype: float or None
        """
        rows = df[df[self.ts_column] == ts]
        if rows.empty:
            return None
        return float(rows.iloc[0]["close"])

    def _apply_signal(
        self,
        portfolio: Portfolio,
        instrument_id: str,
        signal: SignalOutput,
        ts: datetime,
        price: float,
    ) -> None:
        """Apply a single signal to a single instrument's portfolio.

        Routing rules mirror those of
        :meth:`~quaver.backtest.engine.BacktestEngine._apply_signal`:

        - ``BUY``: open a long if flat; otherwise log at ``DEBUG`` and ignore.
        - ``SELL``: close an existing long, or (if flat and
          ``allow_shorting``) open a short; otherwise log and ignore.
        - ``CLOSE``: close whichever position is open; no-op if flat.
        - ``HOLD``: log at ``DEBUG`` level; no portfolio action taken.

        :param portfolio: The portfolio for this instrument.
        :type portfolio: Portfolio
        :param instrument_id: Identifier of the instrument being traded.
        :type instrument_id: str
        :param signal: Signal emitted by the strategy for the current bar.
        :type signal: SignalOutput
        :param ts: Timestamp of the current bar.
        :type ts: datetime
        :param price: Close price of the current bar used as the execution
            price.
        :type price: float
        :returns: None
        :rtype: None
        """
        direction = signal.direction

        if direction == SignalDirection.BUY:
            if portfolio.is_flat():
                portfolio.open_long(instrument_id, ts, price, signal)
            else:
                pos = portfolio._open_position
                if pos is not None and pos.direction == SignalDirection.SELL:
                    portfolio.close_short(ts, price, signal, exit_reason=ExitReason.SIGNAL)
                    portfolio.open_long(instrument_id, ts, price, signal)
                else:
                    log.debug("BUY ignored for '%s': long position already open", instrument_id)

        elif direction == SignalDirection.SELL:
            pos = portfolio._open_position
            if pos is not None and pos.direction == SignalDirection.BUY:
                portfolio.close_long(ts, price, signal, exit_reason=ExitReason.SIGNAL)
            elif portfolio.is_flat():
                if self.allow_shorting:
                    portfolio.open_short(instrument_id, ts, price, signal)
                else:
                    log.debug(
                        "SELL ignored for '%s': flat and allow_shorting=False",
                        instrument_id,
                    )
            else:
                log.debug("SELL ignored for '%s': short already open", instrument_id)

        elif direction == SignalDirection.CLOSE:
            if not portfolio.is_flat():
                self._close_position(portfolio, ts, price, signal, exit_reason=ExitReason.SIGNAL)

        elif direction == SignalDirection.HOLD:
            log.debug("HOLD for '%s' at %s — no action", instrument_id, ts)
