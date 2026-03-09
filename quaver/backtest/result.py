"""BacktestResult: immutable summary of a completed backtest run."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quaver.backtest.portfolio import Portfolio, TradeRecord


@dataclass
class BacktestResult:
    """Immutable summary produced after :meth:`BacktestEngine.run` completes.

    All monetary values are expressed in the same currency unit as
    ``initial_capital``.

    **Drawdown convention**

    Drawdown is computed on the cumulative P&L series (*not* the equity
    curve) and expressed as a fraction of ``initial_capital``::

        max_drawdown = (trough - peak) / initial_capital

    The value is always ``<= 0``.  Returns ``0.0`` when fewer than 2 trades
    are present.

    **Sharpe ratio convention**

    Sharpe ratio uses per-trade P&L (not annualised returns), scaled by
    ``sqrt(252)`` as a conventional approximation.  Returns ``0.0`` when
    fewer than 2 trades are present or when ``std(pnl) == 0``.

    .. important::
        The Sharpe ratio should be interpreted only as a *relative ranking
        metric* between strategies run on the same dataset.  It is **not** a
        calendar-annualised Sharpe ratio.

    :param instrument_id: Identifier of the traded instrument.
    :type instrument_id: str
    :param initial_capital: Starting cash balance used in the backtest.
    :type initial_capital: float
    :param final_cash: Cash balance at the end of the backtest.
    :type final_cash: float
    :param trades: Ordered list of all completed :class:`~quaver.backtest.portfolio.TradeRecord` objects.
    :type trades: list[TradeRecord]
    """

    instrument_id: str
    initial_capital: float
    final_cash: float
    trades: list[TradeRecord]

    # ── Derived properties ──────────────────────────────────────────────────

    @property
    def total_return(self) -> float:
        """Fractional return over the full backtest period.

        Computed as ``(final_cash - initial_capital) / initial_capital``.
        Returns ``0.0`` when ``initial_capital`` is zero.

        :returns: Fractional total return (e.g. ``0.15`` means +15%).
        :rtype: float
        """
        if self.initial_capital == 0:
            return 0.0
        return (self.final_cash - self.initial_capital) / self.initial_capital

    @property
    def total_trades(self) -> int:
        """Total number of completed round-trip trades.

        :returns: Count of all trades recorded in :attr:`trades`.
        :rtype: int
        """
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        """Number of trades with a strictly positive P&L.

        :returns: Count of trades where ``pnl > 0``.
        :rtype: int
        """
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losing_trades(self) -> int:
        """Number of trades with a zero or negative P&L.

        :returns: Count of trades where ``pnl <= 0``.
        :rtype: int
        """
        return sum(1 for t in self.trades if t.pnl <= 0)

    @property
    def win_rate(self) -> float:
        """Fraction of trades that were profitable.

        Returns ``0.0`` when no trades have been recorded.

        :returns: ``winning_trades / total_trades`` in the range ``[0.0, 1.0]``.
        :rtype: float
        """
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_pnl(self) -> float:
        """Average P&L per trade.

        Returns ``0.0`` when no trades have been recorded.

        :returns: Arithmetic mean of per-trade P&L values.
        :rtype: float
        """
        if not self.trades:
            return 0.0
        return sum(t.pnl for t in self.trades) / len(self.trades)

    @property
    def pnl_series(self) -> list[float]:
        """Ordered list of per-trade P&L values.

        :returns: List of ``pnl`` values in trade-completion order.
        :rtype: list[float]
        """
        return [t.pnl for t in self.trades]

    @property
    def cumulative_pnl(self) -> list[float]:
        """Running cumulative sum of per-trade P&L values.

        :returns: List where element *i* is the sum of all P&L values up to
            and including trade *i*.
        :rtype: list[float]
        """
        result: list[float] = []
        running = 0.0
        for p in self.pnl_series:
            running += p
            result.append(running)
        return result

    @property
    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drop in cumulative P&L, as a fraction of
        ``initial_capital``.

        Formula::

            max_drawdown = (trough - peak) / initial_capital

        Always ``<= 0``.  Returns ``0.0`` when no trades are present
        or when ``initial_capital`` is zero.

        :returns: Maximum drawdown as a non-positive fraction (e.g. ``-0.12``
            means a 12% drawdown relative to initial capital).
        :rtype: float
        """
        cpnl = self.cumulative_pnl
        if not cpnl or self.initial_capital == 0:
            return 0.0
        arr = np.array(cpnl, dtype=float)
        peak = 0.0  # equity starts at zero P&L before any trade
        max_dd = 0.0
        for v in arr:
            if v > peak:
                peak = v
            dd = (v - peak) / self.initial_capital
            if dd < max_dd:
                max_dd = dd
        return round(float(max_dd), 6)

    @property
    def sharpe_ratio(self) -> float:
        """Per-trade Sharpe proxy scaled by ``sqrt(252)``.

        Formula::

            sharpe = mean(pnl) / std(pnl, ddof=1) * sqrt(252)

        Returns ``0.0`` when fewer than 2 trades are present or when
        ``std(pnl) == 0``.

        .. note::
            This is a relative comparison metric only -- it is **not** a
            calendar-annualised Sharpe ratio.

        :returns: Per-trade Sharpe proxy rounded to 4 decimal places.
        :rtype: float
        """
        if len(self.trades) < 2:
            return 0.0
        arr = np.array(self.pnl_series, dtype=float)
        std = float(np.std(arr, ddof=1))
        if std == 0.0:
            return 0.0
        return round(float(np.mean(arr)) / std * math.sqrt(252), 4)

    @property
    def profit_factor(self) -> float:
        """Ratio of gross profit to gross loss.

        Formula::

            profit_factor = sum(winning pnl) / abs(sum(losing pnl))

        Returns ``0.0`` when there are no trades, or ``float('inf')`` when
        all trades are profitable (no losing trades).

        :returns: Profit factor rounded to 4 decimal places, ``0.0`` when
            no trades exist, or ``inf`` when gross loss is zero.
        :rtype: float
        """
        if not self.trades:
            return 0.0
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float("inf")
        return round(gross_profit / gross_loss, 4)

    def summary(self) -> dict[str, object]:
        """Return all key metrics as a flat dictionary with rounded values.

        The dictionary contains the following keys:

        - ``instrument_id`` -- str
        - ``initial_capital`` -- float, rounded to 2 dp
        - ``final_cash`` -- float, rounded to 2 dp
        - ``total_return_pct`` -- float, percentage rounded to 2 dp
        - ``total_trades`` -- int
        - ``winning_trades`` -- int
        - ``losing_trades`` -- int
        - ``win_rate_pct`` -- float, percentage rounded to 2 dp
        - ``avg_pnl`` -- float, rounded to 4 dp
        - ``profit_factor`` -- float
        - ``sharpe_ratio`` -- float
        - ``max_drawdown_pct`` -- float, percentage rounded to 2 dp

        :returns: Flat dictionary of performance metrics.
        :rtype: dict
        """
        return {
            "instrument_id": self.instrument_id,
            "initial_capital": round(self.initial_capital, 2),
            "final_cash": round(self.final_cash, 2),
            "total_return_pct": round(self.total_return * 100, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": round(self.win_rate * 100, 2),
            "avg_pnl": round(self.avg_pnl, 4),
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
        }

    @classmethod
    def from_portfolio(
        cls,
        portfolio: Portfolio,
        candles: pd.DataFrame,
        instrument_id: str,
    ) -> BacktestResult:
        """Construct a :class:`BacktestResult` from a completed
        :class:`~quaver.backtest.portfolio.Portfolio`.

        :param portfolio: Portfolio instance after all trades have been
            applied and any open position has been force-closed.
        :type portfolio: Portfolio
        :param candles: Normalised OHLCV DataFrame used during the backtest
            (currently retained for future extension; not read here).
        :type candles: pandas.DataFrame
        :param instrument_id: Identifier to embed in the result.
        :type instrument_id: str
        :returns: Fully populated :class:`BacktestResult`.
        :rtype: BacktestResult
        """
        return cls(
            instrument_id=instrument_id,
            initial_capital=portfolio.initial_capital,
            final_cash=portfolio.cash,
            trades=portfolio.closed_trades,
        )
