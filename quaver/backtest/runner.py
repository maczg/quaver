"""High-level convenience functions for running backtests."""

from __future__ import annotations

import logging

import pandas as pd

from quaver.strategies.base import MultiAssetStrategy
from quaver.strategies.registry import StrategyRegistry
from quaver.backtest.data import normalise_candles, validate_candles
from quaver.backtest.engine import BacktestEngine
from quaver.backtest.multi_engine import MultiAssetBacktestEngine
from quaver.backtest.portfolio import Portfolio
from quaver.backtest.result import BacktestResult

log = logging.getLogger(__name__)


def run_backtest(
    engine_name: str,
    parameters: dict,
    candles: pd.DataFrame,
    instrument_id: str,
    initial_capital: float = 10_000.0,
    quantity_per_trade: float = 1.0,
    ts_column: str = "ts",
    allow_shorting: bool = False,
) -> BacktestResult:
    """
    Run a single-asset walk-forward backtest.

    Args:
        engine_name: Registered strategy name (e.g. "mean_reversion").
        parameters: Strategy parameter dict.
        candles: Raw OHLCV DataFrame (will be normalised internally).
        instrument_id: Label used in results and trade records.
        initial_capital: Starting cash balance (default 10,000).
        quantity_per_trade: Fixed units per trade (default 1.0).
        ts_column: Name of the timestamp column (default "ts").
        allow_shorting: If True, SELL signals from flat portfolio open a short.

    Returns:
        BacktestResult with all trades and performance metrics.

    Raises:
        EngineNotFoundError: if engine_name is not in the registry.
        TypeError: if the engine is a MultiAssetStrategy (use run_multi_asset_backtest).
        ValueError: on invalid parameters or insufficient candles.
    """
    strategy_cls = StrategyRegistry.get(engine_name)

    if issubclass(strategy_cls, MultiAssetStrategy):
        raise TypeError(
            f"Engine '{engine_name}' is a MultiAssetStrategy. "
            "Use run_multi_asset_backtest() instead."
        )

    strategy = strategy_cls(parameters=parameters)
    strategy.validate_parameters()

    clean = normalise_candles(candles, ts_col=ts_column)
    validate_candles(clean, strategy.get_required_candle_count(), label=instrument_id)

    portfolio = Portfolio(
        initial_capital=initial_capital,
        quantity_per_trade=quantity_per_trade,
    )
    engine = BacktestEngine(
        strategy=strategy,
        portfolio=portfolio,
        instrument_id=instrument_id,
        ts_column=ts_column,
        allow_shorting=allow_shorting,
    )
    return engine.run(clean)


def run_multi_asset_backtest(
    engine_name: str,
    parameters: dict,
    candles_map: dict[str, pd.DataFrame],
    initial_capital: float = 10_000.0,
    quantity_per_trade: float = 1.0,
    ts_column: str = "ts",
    allow_shorting: bool = False,
    timestamp_overlap_warn_threshold: float = 0.05,
) -> dict[str, BacktestResult]:
    """
    Run a multi-asset walk-forward backtest.

    Args:
        engine_name: Registered MultiAssetStrategy name.
        parameters: Strategy parameter dict.
        candles_map: Dict of instrument_id -> raw OHLCV DataFrame.
        initial_capital: Starting cash per instrument portfolio (default 10,000).
        quantity_per_trade: Fixed units per trade per instrument (default 1.0).
        ts_column: Timestamp column name (default "ts").
        allow_shorting: Passed to each instrument's portfolio engine.
        timestamp_overlap_warn_threshold: If the shared timestamp intersection
            drops more than this fraction of timestamps from any instrument,
            a WARNING is logged. Default 0.05 (5%).

    Returns:
        Dict of instrument_id -> BacktestResult.

    Raises:
        TypeError: if engine_name resolves to a single-asset strategy.
        ValueError: if required instruments are missing from candles_map,
                    or if any instrument has insufficient candles.
    """
    strategy_cls = StrategyRegistry.get(engine_name)

    if not issubclass(strategy_cls, MultiAssetStrategy):
        raise TypeError(
            f"Engine '{engine_name}' is a single-asset BaseStrategy. "
            "Use run_backtest() instead."
        )

    strategy: MultiAssetStrategy = strategy_cls(parameters=parameters)
    strategy.validate_parameters()

    # Validate required instruments are present
    required_ids = strategy.get_required_instrument_ids()
    missing = set(required_ids) - set(candles_map)
    if missing:
        raise ValueError(
            f"Required instrument(s) missing from candles_map: {sorted(missing)}"
        )

    # Normalise all DataFrames
    clean_map = {
        iid: normalise_candles(df, ts_col=ts_column)
        for iid, df in candles_map.items()
    }

    # Validate minimum candle counts
    required_count = strategy.get_required_candle_count()
    for iid, df in clean_map.items():
        validate_candles(df, required_count, label=iid)

    # Build one Portfolio per instrument
    portfolios = {
        iid: Portfolio(
            initial_capital=initial_capital,
            quantity_per_trade=quantity_per_trade,
        )
        for iid in clean_map
    }

    engine = MultiAssetBacktestEngine(
        strategy=strategy,
        portfolios=portfolios,
        ts_column=ts_column,
        allow_shorting=allow_shorting,
        timestamp_overlap_warn_threshold=timestamp_overlap_warn_threshold,
    )
    return engine.run(clean_map)
