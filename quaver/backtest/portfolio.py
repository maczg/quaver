"""Portfolio: tracks cash, open positions, and closed trades."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from quaver.types import SignalDirection
from quaver.strategies.base import SignalOutput

log = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    """A single open position held by the portfolio.

    :param instrument_id: Identifier of the traded instrument.
    :type instrument_id: str
    :param entry_ts: Timestamp at which the position was opened.
    :type entry_ts: datetime
    :param entry_price: Execution price at entry.
    :type entry_price: float
    :param quantity: Number of units held.
    :type quantity: float
    :param direction: ``BUY`` for a long position, ``SELL`` for a short
        position.
    :type direction: SignalDirection
    :param entry_signal: The :class:`~quaver.strategies.base.SignalOutput`
        that triggered the entry.
    :type entry_signal: SignalOutput
    """

    instrument_id: str
    entry_ts: datetime
    entry_price: float
    quantity: float
    direction: SignalDirection  # BUY (long) or SELL (short)
    entry_signal: SignalOutput


@dataclass
class TradeRecord:
    """A completed round-trip trade.

    **P&L formula**

    - Long:  ``pnl = (exit_price - entry_price) * quantity``
    - Short: ``pnl = (entry_price - exit_price) * quantity``

    :param instrument_id: Identifier of the traded instrument.
    :type instrument_id: str
    :param entry_ts: Timestamp at which the position was opened.
    :type entry_ts: datetime
    :param exit_ts: Timestamp at which the position was closed.
    :type exit_ts: datetime
    :param entry_price: Execution price at entry.
    :type entry_price: float
    :param exit_price: Execution price at exit.
    :type exit_price: float
    :param quantity: Number of units traded.
    :type quantity: float
    :param direction: Opening direction -- ``BUY`` for long, ``SELL`` for
        short.
    :type direction: SignalDirection
    :param pnl: Realised profit and loss for this trade (see formula above).
    :type pnl: float
    :param entry_signal: Signal that triggered the entry.
    :type entry_signal: SignalOutput
    :param exit_signal: Signal that triggered the exit, or ``None`` when the
        position was force-closed at end-of-data.
    :type exit_signal: SignalOutput or None
    """

    instrument_id: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    quantity: float
    direction: SignalDirection  # opening direction: BUY=long, SELL=short
    pnl: float  # see formula below
    entry_signal: SignalOutput
    exit_signal: SignalOutput | None


class Portfolio:
    """Tracks cash balance, one open position at a time, and closed trades.

    **P&L conventions**

    - Long:  ``pnl = (exit_price - entry_price) * quantity``
    - Short: ``pnl = (entry_price - exit_price) * quantity``

    **Cash conventions**

    - ``open_long``:   ``cash -= entry_price * quantity``
    - ``close_long``:  ``cash += exit_price * quantity``
    - ``open_short``:  ``cash += entry_price * quantity``  (proceeds received)
    - ``close_short``: ``cash -= exit_price * quantity``   (buy back to cover)

    Cash CAN go negative if sizing is misconfigured.  No clamping is applied,
    but a ``WARNING`` is logged whenever cash drops below zero.

    Only ONE open position is allowed at a time.  Attempting to open a second
    position while one is already open logs a ``WARNING`` and is a no-op.

    :param initial_capital: Starting cash balance.
    :type initial_capital: float
    :param quantity_per_trade: Fixed number of units used for every trade.
        Defaults to ``1.0``.
    :type quantity_per_trade: float
    """

    def __init__(
        self,
        initial_capital: float,
        quantity_per_trade: float = 1.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.quantity_per_trade = quantity_per_trade
        self._cash: float = initial_capital
        self._open_position: OpenPosition | None = None
        self._closed_trades: list[TradeRecord] = []

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        """Current cash balance.

        :returns: Current cash balance, which may be negative if sizing is
            misconfigured.
        :rtype: float
        """
        return self._cash

    @property
    def closed_trades(self) -> list[TradeRecord]:
        """Snapshot list of all completed trades.

        :returns: A shallow copy of the internal closed-trades list so that
            callers cannot mutate portfolio state.
        :rtype: list[TradeRecord]
        """
        return list(self._closed_trades)

    def is_flat(self) -> bool:
        """Return ``True`` if no position is currently open.

        :returns: ``True`` when the portfolio holds no open position,
            ``False`` otherwise.
        :rtype: bool
        """
        return self._open_position is None

    # ── Long ────────────────────────────────────────────────────────────────

    def open_long(
        self,
        instrument_id: str,
        ts: datetime,
        price: float,
        signal: SignalOutput,
    ) -> None:
        """Open a long position.

        No-op (with ``WARNING``) if a position is already open.

        Cash effect: ``cash -= price * quantity_per_trade``.  A further
        ``WARNING`` is emitted if cash becomes negative after the deduction.

        :param instrument_id: Identifier of the instrument to buy.
        :type instrument_id: str
        :param ts: Timestamp of the entry bar.
        :type ts: datetime
        :param price: Execution (close) price.
        :type price: float
        :param signal: The :class:`~quaver.strategies.base.SignalOutput` that
            triggered this entry.
        :type signal: SignalOutput
        :returns: None
        :rtype: None
        """
        if not self.is_flat():
            log.warning(
                "open_long ignored: position already open for %s",
                self._open_position.instrument_id if self._open_position else "?",
            )
            return
        self._cash -= price * self.quantity_per_trade
        if self._cash < 0:
            log.warning(
                "Cash is negative (%.2f) after open_long at price=%.4f qty=%.4f",
                self._cash,
                price,
                self.quantity_per_trade,
            )
        self._open_position = OpenPosition(
            instrument_id=instrument_id,
            entry_ts=ts,
            entry_price=price,
            quantity=self.quantity_per_trade,
            direction=SignalDirection.BUY,
            entry_signal=signal,
        )

    def close_long(
        self,
        ts: datetime,
        price: float,
        signal: SignalOutput | None,
    ) -> TradeRecord:
        """Close an open long position.

        Cash effect: ``cash += price * quantity``.

        :param ts: Timestamp of the exit bar.
        :type ts: datetime
        :param price: Execution (close) price.
        :type price: float
        :param signal: The :class:`~quaver.strategies.base.SignalOutput` that
            triggered the exit, or ``None`` when force-closed at end-of-data.
        :type signal: SignalOutput or None
        :returns: The completed :class:`TradeRecord`.
        :rtype: TradeRecord
        :raises RuntimeError: If no position is currently open.
        """
        if self._open_position is None:
            raise RuntimeError("close_long called with no open position")
        pos = self._open_position
        self._cash += price * pos.quantity
        pnl = (price - pos.entry_price) * pos.quantity
        record = TradeRecord(
            instrument_id=pos.instrument_id,
            entry_ts=pos.entry_ts,
            exit_ts=ts,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            direction=SignalDirection.BUY,
            pnl=pnl,
            entry_signal=pos.entry_signal,
            exit_signal=signal,
        )
        self._closed_trades.append(record)
        self._open_position = None
        return record

    # ── Short ────────────────────────────────────────────────────────────────

    def open_short(
        self,
        instrument_id: str,
        ts: datetime,
        price: float,
        signal: SignalOutput,
    ) -> None:
        """Open a short position.

        No-op (with ``WARNING``) if a position is already open.

        Cash effect: ``cash += price * quantity_per_trade``  (short proceeds
        are received immediately).

        :param instrument_id: Identifier of the instrument to sell short.
        :type instrument_id: str
        :param ts: Timestamp of the entry bar.
        :type ts: datetime
        :param price: Execution (close) price.
        :type price: float
        :param signal: The :class:`~quaver.strategies.base.SignalOutput` that
            triggered this entry.
        :type signal: SignalOutput
        :returns: None
        :rtype: None
        """
        if not self.is_flat():
            log.warning(
                "open_short ignored: position already open for %s",
                self._open_position.instrument_id if self._open_position else "?",
            )
            return
        self._cash += price * self.quantity_per_trade  # receive short proceeds
        self._open_position = OpenPosition(
            instrument_id=instrument_id,
            entry_ts=ts,
            entry_price=price,
            quantity=self.quantity_per_trade,
            direction=SignalDirection.SELL,
            entry_signal=signal,
        )

    def close_short(
        self,
        ts: datetime,
        price: float,
        signal: SignalOutput | None,
    ) -> TradeRecord:
        """Close an open short position.

        Cash effect: ``cash -= price * quantity``  (buy back to cover).

        :param ts: Timestamp of the exit bar.
        :type ts: datetime
        :param price: Execution (close) price.
        :type price: float
        :param signal: The :class:`~quaver.strategies.base.SignalOutput` that
            triggered the exit, or ``None`` when force-closed at end-of-data.
        :type signal: SignalOutput or None
        :returns: The completed :class:`TradeRecord`.
        :rtype: TradeRecord
        :raises RuntimeError: If no position is currently open.
        """
        if self._open_position is None:
            raise RuntimeError("close_short called with no open position")
        pos = self._open_position
        self._cash -= price * pos.quantity  # buy back to cover
        pnl = (pos.entry_price - price) * pos.quantity
        record = TradeRecord(
            instrument_id=pos.instrument_id,
            entry_ts=pos.entry_ts,
            exit_ts=ts,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            direction=SignalDirection.SELL,
            pnl=pnl,
            entry_signal=pos.entry_signal,
            exit_signal=signal,
        )
        self._closed_trades.append(record)
        self._open_position = None
        return record

    def reset(self) -> None:
        """Restore the portfolio to its initial state.

        .. important::
            Call this before re-running a backtest engine to ensure
            deterministic, reproducible results.  :meth:`BacktestEngine.run`
            calls this automatically at the start of each invocation.

        :returns: None
        :rtype: None
        """
        self._cash = self.initial_capital
        self._open_position = None
        self._closed_trades = []
