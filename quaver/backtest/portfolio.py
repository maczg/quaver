"""Portfolio: tracks cash, open positions, and closed trades."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from quaver.types import ExitReason, SignalDirection
from quaver.strategies.base import SignalOutput

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommissionConfig:
    """Commission parameters applied to every open/close event.

    :param fixed_per_trade: Flat dollar amount charged on each open and each
        close. Defaults to ``0.0``.
    :type fixed_per_trade: float
    :param pct_of_notional: Fraction of ``price * quantity`` charged on each
        open and each close. Defaults to ``0.0``.
    :type pct_of_notional: float
    """

    fixed_per_trade: float = 0.0
    pct_of_notional: float = 0.0

    def calc(self, price: float, quantity: float) -> float:
        """Return commission for a single event (open or close)."""
        return self.fixed_per_trade + self.pct_of_notional * price * quantity


@dataclass(frozen=True)
class SlippageConfig:
    """Slippage parameters applied to every fill.

    Slippage shifts the fill price adversely:

    - Buys fill at ``price * (1 + slippage_pct)``
    - Sells fill at ``price * (1 - slippage_pct)``

    :param slippage_pct: Adverse price shift as a fraction. Defaults to
        ``0.0``.
    :type slippage_pct: float
    """

    slippage_pct: float = 0.0

    def buy_price(self, price: float) -> float:
        """Return the adverse fill price for a buy."""
        return price * (1.0 + self.slippage_pct)

    def sell_price(self, price: float) -> float:
        """Return the adverse fill price for a sell."""
        return price * (1.0 - self.slippage_pct)


@dataclass(frozen=True)
class ExitRules:
    """Global exit rules applied to all trades.

    All values are fractions (e.g. ``0.02`` = 2%).  ``None`` disables the rule.

    :param stop_loss_pct: Adverse move fraction triggering a stop-loss exit.
    :type stop_loss_pct: float or None
    :param take_profit_pct: Favorable move fraction triggering a take-profit exit.
    :type take_profit_pct: float or None
    :param trailing_stop_pct: Trailing distance fraction for a trailing-stop exit.
    :type trailing_stop_pct: float or None
    """

    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None


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
    :param entry_commission: Commission charged on entry. Defaults to ``0.0``.
    :type entry_commission: float
    :param entry_slippage_cost: Dollar slippage cost on entry. Defaults to
        ``0.0``.
    :type entry_slippage_cost: float
    """

    instrument_id: str
    entry_ts: datetime
    entry_price: float
    quantity: float
    direction: SignalDirection  # BUY (long) or SELL (short)
    entry_signal: SignalOutput
    entry_commission: float = 0.0
    entry_slippage_cost: float = 0.0
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    trailing_stop_pct: float | None = None
    trailing_stop_extreme: float | None = None


@dataclass
class TradeRecord:
    """A completed round-trip trade.

    **P&L formula**

    - Long:  ``pnl = (exit_price - entry_price) * quantity``
    - Short: ``pnl = (entry_price - exit_price) * quantity``

    ``pnl`` reflects the *ideal* P&L using intended (pre-slippage) prices.
    ``net_pnl`` subtracts commission and slippage costs.

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
    :param commission: Total commission (entry + exit). Defaults to ``0.0``.
    :type commission: float
    :param slippage_cost: Total dollar slippage cost (entry + exit). Defaults
        to ``0.0``.
    :type slippage_cost: float
    """

    instrument_id: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    quantity: float
    direction: SignalDirection  # opening direction: BUY=long, SELL=short
    pnl: float  # ideal P&L from intended prices
    entry_signal: SignalOutput
    exit_signal: SignalOutput | None
    commission: float = 0.0
    slippage_cost: float = 0.0
    exit_reason: ExitReason | None = None

    @property
    def net_pnl(self) -> float:
        """P&L after deducting commission and slippage costs."""
        return self.pnl - self.commission - self.slippage_cost


class Portfolio:
    """Tracks cash balance, one open position at a time, and closed trades.

    **P&L conventions**

    - Long:  ``pnl = (exit_price - entry_price) * quantity``
    - Short: ``pnl = (entry_price - exit_price) * quantity``

    **Cash conventions**

    When commission and slippage are configured, cash accounting uses the
    actual (slipped) fill prices and deducts commissions.

    Cash CAN go negative if sizing is misconfigured.  No clamping is applied,
    but a ``WARNING`` is logged whenever cash drops below zero.

    Only ONE open position is allowed at a time.  Attempting to open a second
    position while one is already open logs a ``WARNING`` and is a no-op.

    :param initial_capital: Starting cash balance.
    :type initial_capital: float
    :param quantity_per_trade: Fixed number of units used for every trade.
        Defaults to ``1.0``.
    :type quantity_per_trade: float
    :param commission: Commission configuration. ``None`` means zero cost.
    :type commission: CommissionConfig or None
    :param slippage: Slippage configuration. ``None`` means zero slippage.
    :type slippage: SlippageConfig or None
    :param sizing_fn: Optional callable ``(account_value, entry_price) ->
        quantity``.  When provided, overrides *quantity_per_trade*.
    :type sizing_fn: Callable[[float, float], float] or None
    """

    def __init__(
        self,
        initial_capital: float,
        quantity_per_trade: float = 1.0,
        commission: CommissionConfig | None = None,
        slippage: SlippageConfig | None = None,
        sizing_fn: Callable[[float, float], float] | None = None,
        exit_rules: ExitRules | None = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.quantity_per_trade = quantity_per_trade
        self._commission = commission or CommissionConfig()
        self._slippage = slippage or SlippageConfig()
        self._sizing_fn = sizing_fn
        self._exit_rules = exit_rules
        self._cash: float = initial_capital
        self._open_position: OpenPosition | None = None
        self._closed_trades: list[TradeRecord] = []

    def _resolve_quantity(self, price: float) -> float:
        """Return the trade quantity, using sizing_fn if configured."""
        if self._sizing_fn is not None:
            return self._sizing_fn(self._cash, price)
        return self.quantity_per_trade

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

        Cash is debited at the slipped fill price plus commission.

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
        qty = self._resolve_quantity(price)
        fill_price = self._slippage.buy_price(price)
        comm = self._commission.calc(fill_price, qty)
        slippage_cost = abs(fill_price - price) * qty

        self._cash -= fill_price * qty + comm
        if self._cash < 0:
            log.warning(
                "Cash is negative (%.2f) after open_long at price=%.4f qty=%.4f",
                self._cash,
                price,
                qty,
            )
        self._open_position = OpenPosition(
            instrument_id=instrument_id,
            entry_ts=ts,
            entry_price=price,
            quantity=qty,
            direction=SignalDirection.BUY,
            entry_signal=signal,
            entry_commission=comm,
            entry_slippage_cost=slippage_cost,
        )
        self._resolve_exit_levels()

    def close_long(
        self,
        ts: datetime,
        price: float,
        signal: SignalOutput | None,
        exit_reason: ExitReason | None = None,
    ) -> TradeRecord:
        """Close an open long position.

        Cash is credited at the slipped fill price minus commission.

        :param ts: Timestamp of the exit bar.
        :type ts: datetime
        :param price: Execution (close) price.
        :type price: float
        :param signal: The :class:`~quaver.strategies.base.SignalOutput` that
            triggered the exit, or ``None`` when force-closed at end-of-data.
        :type signal: SignalOutput or None
        :param exit_reason: Why the position was closed.
        :type exit_reason: ExitReason or None
        :returns: The completed :class:`TradeRecord`.
        :rtype: TradeRecord
        :raises RuntimeError: If no position is currently open.
        """
        if self._open_position is None:
            raise RuntimeError("close_long called with no open position")
        pos = self._open_position
        fill_price = self._slippage.sell_price(price)
        exit_comm = self._commission.calc(fill_price, pos.quantity)
        exit_slippage_cost = abs(price - fill_price) * pos.quantity

        self._cash += fill_price * pos.quantity - exit_comm

        pnl = (price - pos.entry_price) * pos.quantity
        total_commission = pos.entry_commission + exit_comm
        total_slippage = pos.entry_slippage_cost + exit_slippage_cost

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
            commission=total_commission,
            slippage_cost=total_slippage,
            exit_reason=exit_reason,
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

        Cash is credited at the slipped fill price minus commission.

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
        qty = self._resolve_quantity(price)
        fill_price = self._slippage.sell_price(price)
        comm = self._commission.calc(fill_price, qty)
        slippage_cost = abs(price - fill_price) * qty

        self._cash += fill_price * qty - comm
        self._open_position = OpenPosition(
            instrument_id=instrument_id,
            entry_ts=ts,
            entry_price=price,
            quantity=qty,
            direction=SignalDirection.SELL,
            entry_signal=signal,
            entry_commission=comm,
            entry_slippage_cost=slippage_cost,
        )
        self._resolve_exit_levels()

    def close_short(
        self,
        ts: datetime,
        price: float,
        signal: SignalOutput | None,
        exit_reason: ExitReason | None = None,
    ) -> TradeRecord:
        """Close an open short position.

        Cash is debited at the slipped fill price plus commission.

        :param ts: Timestamp of the exit bar.
        :type ts: datetime
        :param price: Execution (close) price.
        :type price: float
        :param signal: The :class:`~quaver.strategies.base.SignalOutput` that
            triggered the exit, or ``None`` when force-closed at end-of-data.
        :type signal: SignalOutput or None
        :param exit_reason: Why the position was closed.
        :type exit_reason: ExitReason or None
        :returns: The completed :class:`TradeRecord`.
        :rtype: TradeRecord
        :raises RuntimeError: If no position is currently open.
        """
        if self._open_position is None:
            raise RuntimeError("close_short called with no open position")
        pos = self._open_position
        fill_price = self._slippage.buy_price(price)
        exit_comm = self._commission.calc(fill_price, pos.quantity)
        exit_slippage_cost = abs(fill_price - price) * pos.quantity

        self._cash -= fill_price * pos.quantity + exit_comm

        pnl = (pos.entry_price - price) * pos.quantity
        total_commission = pos.entry_commission + exit_comm
        total_slippage = pos.entry_slippage_cost + exit_slippage_cost

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
            commission=total_commission,
            slippage_cost=total_slippage,
            exit_reason=exit_reason,
        )
        self._closed_trades.append(record)
        self._open_position = None
        return record

    def _resolve_exit_levels(self) -> None:
        """Set stop-loss, take-profit, and trailing-stop levels on the open position.

        Per-trade metadata keys (``stop_loss``, ``take_profit``,
        ``trailing_stop_pct``) override global :class:`ExitRules`.
        """
        pos = self._open_position
        if pos is None:
            return

        meta = pos.entry_signal.metadata or {}
        entry = pos.entry_price
        is_long = pos.direction == SignalDirection.BUY
        rules = self._exit_rules

        # Stop-loss
        if "stop_loss" in meta:
            pos.stop_loss_price = float(meta["stop_loss"])
        elif rules and rules.stop_loss_pct is not None:
            if is_long:
                pos.stop_loss_price = entry * (1.0 - rules.stop_loss_pct)
            else:
                pos.stop_loss_price = entry * (1.0 + rules.stop_loss_pct)

        # Take-profit
        if "take_profit" in meta:
            pos.take_profit_price = float(meta["take_profit"])
        elif rules and rules.take_profit_pct is not None:
            if is_long:
                pos.take_profit_price = entry * (1.0 + rules.take_profit_pct)
            else:
                pos.take_profit_price = entry * (1.0 - rules.take_profit_pct)

        # Trailing stop
        if "trailing_stop_pct" in meta:
            pos.trailing_stop_pct = float(meta["trailing_stop_pct"])
        elif rules and rules.trailing_stop_pct is not None:
            pos.trailing_stop_pct = rules.trailing_stop_pct

        if pos.trailing_stop_pct is not None:
            pos.trailing_stop_extreme = entry

    def check_exit_triggers(
        self,
        ts: datetime,
        high: float,
        low: float,
    ) -> tuple[ExitReason, float] | None:
        """Check whether the current bar triggers a stop-loss, trailing stop, or take-profit exit.

        Priority (pessimistic): stop-loss > trailing stop > take-profit.

        :param ts: Timestamp of the current bar.
        :param high: High price of the current bar.
        :param low: Low price of the current bar.
        :returns: ``(ExitReason, fill_price)`` if triggered, else ``None``.
        """
        pos = self._open_position
        if pos is None:
            return None

        is_long = pos.direction == SignalDirection.BUY

        # 1. Stop-loss
        if pos.stop_loss_price is not None:
            if is_long and low <= pos.stop_loss_price:
                return (ExitReason.STOP_LOSS, pos.stop_loss_price)
            if not is_long and high >= pos.stop_loss_price:
                return (ExitReason.STOP_LOSS, pos.stop_loss_price)

        # 2. Trailing stop
        if pos.trailing_stop_pct is not None and pos.trailing_stop_extreme is not None:
            if is_long:
                pos.trailing_stop_extreme = max(pos.trailing_stop_extreme, high)
                trail_level = pos.trailing_stop_extreme * (1.0 - pos.trailing_stop_pct)
                if low <= trail_level:
                    return (ExitReason.TRAILING_STOP, trail_level)
            else:
                pos.trailing_stop_extreme = min(pos.trailing_stop_extreme, low)
                trail_level = pos.trailing_stop_extreme * (1.0 + pos.trailing_stop_pct)
                if high >= trail_level:
                    return (ExitReason.TRAILING_STOP, trail_level)

        # 3. Take-profit
        if pos.take_profit_price is not None:
            if is_long and high >= pos.take_profit_price:
                return (ExitReason.TAKE_PROFIT, pos.take_profit_price)
            if not is_long and low <= pos.take_profit_price:
                return (ExitReason.TAKE_PROFIT, pos.take_profit_price)

        return None

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
