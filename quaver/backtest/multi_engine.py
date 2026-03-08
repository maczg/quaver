"""Multi-asset walk-forward backtest engine."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from quaver.strategies.base import MultiAssetStrategy, SignalOutput
from quaver.types import SignalDirection
from quaver.backtest.portfolio import Portfolio
from quaver.backtest.result import BacktestResult

log = logging.getLogger(__name__)


class MultiAssetBacktestEngine:
    """
    Replays multiple instruments' candle histories through a MultiAssetStrategy.

    Alignment:
        All instruments are aligned to a shared timestamp intersection.
        If the intersection drops more than `timestamp_overlap_warn_threshold`
        of any single instrument's timestamps, a WARNING is logged.

    Atomicity:
        All signals emitted in a single MultiAssetStrategyOutput are applied
        in the same iteration step, before moving to the next timestamp.

    Position management:
        Each instrument has its own Portfolio.

    Args:
        strategy: Instantiated and validated MultiAssetStrategy.
        portfolios: Dict of instrument_id -> Portfolio (one per instrument).
        ts_column: Timestamp column name (default "ts").
        allow_shorting: If True, SELL signals from flat portfolios open shorts.
        timestamp_overlap_warn_threshold: Warn if intersection drops > this
            fraction of an instrument's timestamps. Default 0.05.
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

    def run(
        self, candles_map: dict[str, pd.DataFrame]
    ) -> dict[str, BacktestResult]:
        """
        Run the multi-asset backtest.

        Args:
            candles_map: Dict of instrument_id -> normalised OHLCV DataFrame.

        Returns:
            Dict of instrument_id -> BacktestResult.
        """
        # Reset all portfolios
        for p in self.portfolios.values():
            p.reset()

        # Build shared timeline (sorted intersection of all timestamps)
        ts_sets = {
            iid: set(df[self.ts_column].tolist())
            for iid, df in candles_map.items()
        }
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
        ts_index: dict[str, dict] = {
            iid: {ts: idx for idx, ts in enumerate(df[self.ts_column].tolist())}
            for iid, df in candles_map.items()
        }

        required = self.strategy.get_required_candle_count()

        for step, ts in enumerate(shared_ts):
            if step < required:
                continue

            # Build window_map: each instrument's history up to (not including) ts
            window_map: dict[str, pd.DataFrame] = {}
            for iid, df in candles_map.items():
                row_idx = ts_index[iid].get(ts)
                if row_idx is None or row_idx == 0:
                    window_map[iid] = df.iloc[:0]   # empty
                else:
                    window_map[iid] = df.iloc[:row_idx]

            output = self.strategy.compute(window_map, as_of=ts)
            if output is None:
                continue

            # Apply all signals atomically
            for instrument_id, signal in output.signals.items():
                if instrument_id not in candles_map:
                    log.warning(
                        "Signal for unknown instrument '%s' — skipping", instrument_id
                    )
                    continue
                price = self._get_price_at(candles_map[instrument_id], ts)
                if price is None:
                    log.warning(
                        "No price for '%s' at %s — signal skipped", instrument_id, ts
                    )
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
                log.debug(
                    "Force-closing open position for '%s' at end of data", iid
                )
                pos = portfolio._open_position
                if pos is not None and pos.direction == SignalDirection.BUY:
                    portfolio.close_long(last_ts, last_price, signal=None)
                else:
                    portfolio.close_short(last_ts, last_price, signal=None)

        return {
            iid: BacktestResult.from_portfolio(p, candles_map[iid], iid)
            for iid, p in self.portfolios.items()
        }

    def _get_price_at(
        self, df: pd.DataFrame, ts: datetime
    ) -> float | None:
        """Return the close price of df at timestamp ts, or None if not found."""
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
        """Apply a single signal to a single instrument's portfolio."""
        direction = signal.direction

        if direction == SignalDirection.BUY:
            if portfolio.is_flat():
                portfolio.open_long(instrument_id, ts, price, signal)
            else:
                log.debug("BUY ignored for '%s': position already open", instrument_id)

        elif direction == SignalDirection.SELL:
            pos = portfolio._open_position
            if pos is not None and pos.direction == SignalDirection.BUY:
                portfolio.close_long(ts, price, signal)
            elif portfolio.is_flat():
                if self.allow_shorting:
                    portfolio.open_short(instrument_id, ts, price, signal)
                else:
                    log.debug(
                        "SELL ignored for '%s': flat and allow_shorting=False",
                        instrument_id,
                    )
            else:
                log.debug(
                    "SELL ignored for '%s': short already open", instrument_id
                )

        elif direction == SignalDirection.CLOSE:
            if not portfolio.is_flat():
                pos = portfolio._open_position
                if pos is not None and pos.direction == SignalDirection.BUY:
                    portfolio.close_long(ts, price, signal)
                else:
                    portfolio.close_short(ts, price, signal)

        elif direction == SignalDirection.HOLD:
            log.debug("HOLD for '%s' at %s — no action", instrument_id, ts)
