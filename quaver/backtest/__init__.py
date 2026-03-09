"""
quaver.backtest -- walk-forward backtesting harness.

This package provides a complete, look-ahead-free backtesting framework for
single-asset and multi-asset strategies.

**Public API**

- :class:`~quaver.backtest.engine.BacktestEngine` -- single-instrument replay
  engine.
- :class:`~quaver.backtest.multi_engine.MultiAssetBacktestEngine` -- multi-
  instrument replay engine with shared timestamp alignment.
- :class:`~quaver.backtest.portfolio.Portfolio` -- cash and position tracker.
- :class:`~quaver.backtest.portfolio.TradeRecord` -- immutable record of a
  completed round-trip trade.
- :class:`~quaver.backtest.result.BacktestResult` -- immutable performance
  summary produced after a run.
- :func:`~quaver.backtest.runner.run_backtest` -- high-level convenience
  wrapper for single-asset runs.
- :func:`~quaver.backtest.runner.run_multi_asset_backtest` -- high-level
  convenience wrapper for multi-asset runs.
"""

from quaver.backtest.engine import BacktestEngine
from quaver.backtest.portfolio import (
    CommissionConfig,
    ExitRules,
    Portfolio,
    SlippageConfig,
    TradeRecord,
)
from quaver.backtest.result import BacktestResult
from quaver.backtest.runner import run_backtest, run_multi_asset_backtest
from quaver.backtest.multi_engine import MultiAssetBacktestEngine
from quaver.backtest.sizing import size_by_risk

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CommissionConfig",
    "ExitRules",
    "MultiAssetBacktestEngine",
    "Portfolio",
    "SlippageConfig",
    "TradeRecord",
    "run_backtest",
    "run_multi_asset_backtest",
    "size_by_risk",
]
