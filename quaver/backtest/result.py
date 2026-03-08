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
    """
    Immutable summary produced after BacktestEngine.run() completes.

    All monetary values are in the same currency unit as initial_capital.

    Drawdown is computed on the cumulative P&L series (NOT equity curve),
    expressed as a fraction of initial_capital:
        max_drawdown = (trough - peak) / initial_capital
    This is always <= 0. Returns 0.0 if fewer than 2 trades.

    Sharpe ratio uses per-trade P&L (not annualised returns), scaled by
    sqrt(252) as a conventional approximation. Returns 0.0 if fewer than
    2 trades or if std(pnl) == 0.
    IMPORTANT: interpret this only as a relative ranking metric between
    strategies run on the same dataset — it is NOT a calendar-annualised Sharpe.
    """

    instrument_id: str
    initial_capital: float
    final_cash: float
    trades: list[TradeRecord]

    # ── Derived properties ──────────────────────────────────────────────────

    @property
    def total_return(self) -> float:
        """(final_cash - initial_capital) / initial_capital."""
        if self.initial_capital == 0:
            return 0.0
        return (self.final_cash - self.initial_capital) / self.initial_capital

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl <= 0)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl for t in self.trades) / len(self.trades)

    @property
    def pnl_series(self) -> list[float]:
        return [t.pnl for t in self.trades]

    @property
    def cumulative_pnl(self) -> list[float]:
        result: list[float] = []
        running = 0.0
        for p in self.pnl_series:
            running += p
            result.append(running)
        return result

    @property
    def max_drawdown(self) -> float:
        """
        Max peak-to-trough drop in cumulative P&L, as a fraction of initial_capital.

        Formula: (trough - peak) / initial_capital
        Always <= 0. Returns 0.0 if fewer than 2 trades.
        """
        cpnl = self.cumulative_pnl
        if len(cpnl) < 2 or self.initial_capital == 0:
            return 0.0
        arr = np.array(cpnl, dtype=float)
        peak = arr[0]
        max_dd = 0.0
        for v in arr[1:]:
            if v > peak:
                peak = v
            dd = (v - peak) / self.initial_capital
            if dd < max_dd:
                max_dd = dd
        return round(max_dd, 6)

    @property
    def sharpe_ratio(self) -> float:
        """
        Per-trade Sharpe proxy: mean(pnl) / std(pnl) * sqrt(252).

        Returns 0.0 if fewer than 2 trades or std == 0.
        NOTE: This is a relative comparison metric only, not a
        calendar-annualised Sharpe ratio.
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
        """sum(winning pnl) / abs(sum(losing pnl)). Returns inf if no losses."""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float("inf")
        return round(gross_profit / gross_loss, 4)

    def summary(self) -> dict:
        """Return all key metrics as a flat dict with rounded values."""
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
        """Construct a BacktestResult from a completed Portfolio."""
        return cls(
            instrument_id=instrument_id,
            initial_capital=portfolio.initial_capital,
            final_cash=portfolio.cash,
            trades=portfolio.closed_trades,
        )
