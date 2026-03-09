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

    @property
    def max_consecutive_wins(self) -> int:
        """Longest streak of consecutive trades with ``pnl > 0``.

        :returns: Length of the longest winning streak, or ``0`` if no trades.
        :rtype: int
        """
        best = 0
        current = 0
        for t in self.trades:
            if t.pnl > 0:
                current += 1
                if current > best:
                    best = current
            else:
                current = 0
        return best

    @property
    def max_consecutive_losses(self) -> int:
        """Longest streak of consecutive trades with ``pnl <= 0``.

        :returns: Length of the longest losing streak, or ``0`` if no trades.
        :rtype: int
        """
        best = 0
        current = 0
        for t in self.trades:
            if t.pnl <= 0:
                current += 1
                if current > best:
                    best = current
            else:
                current = 0
        return best

    @property
    def avg_win(self) -> float:
        """Mean P&L of winning trades (``pnl > 0``).

        Returns ``0.0`` when there are no winning trades.

        :returns: Average winning P&L.
        :rtype: float
        """
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        if not wins:
            return 0.0
        return sum(wins) / len(wins)

    @property
    def avg_loss(self) -> float:
        """Mean P&L of losing trades (``pnl <= 0``), non-positive.

        Returns ``0.0`` when there are no losing trades.

        :returns: Average losing P&L (non-positive value).
        :rtype: float
        """
        losses = [t.pnl for t in self.trades if t.pnl <= 0]
        if not losses:
            return 0.0
        return sum(losses) / len(losses)

    @property
    def recovery_factor(self) -> float:
        """Total return divided by the absolute value of max drawdown.

        Returns ``0.0`` when there is no drawdown.

        :returns: Recovery factor.
        :rtype: float
        """
        dd = self.max_drawdown
        if dd == 0.0:
            return 0.0
        return self.total_return / abs(dd)

    @property
    def expectancy(self) -> float:
        """Expected P&L per trade based on win rate and average win/loss.

        Formula::

            expectancy = win_rate * avg_win + loss_rate * avg_loss

        :returns: Expected value per trade.
        :rtype: float
        """
        wr = self.win_rate
        return wr * self.avg_win + (1.0 - wr) * self.avg_loss

    @property
    def total_commission(self) -> float:
        """Sum of commissions across all trades.

        :returns: Total commission in dollars.
        :rtype: float
        """
        return sum(t.commission for t in self.trades)

    @property
    def total_slippage(self) -> float:
        """Sum of slippage costs across all trades.

        :returns: Total slippage cost in dollars.
        :rtype: float
        """
        return sum(t.slippage_cost for t in self.trades)

    def summary(self) -> dict[str, object]:
        """Return all key metrics as a flat dictionary with rounded values.

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
            "total_commission": round(self.total_commission, 4),
            "total_slippage": round(self.total_slippage, 4),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "recovery_factor": round(self.recovery_factor, 4),
            "expectancy": round(self.expectancy, 4),
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
