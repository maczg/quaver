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
    parameters: dict[str, object],
    candles: pd.DataFrame,
    instrument_id: str,
    initial_capital: float = 10_000.0,
    quantity_per_trade: float = 1.0,
    ts_column: str = "ts",
    allow_shorting: bool = False,
) -> BacktestResult:
    """Run a single-asset walk-forward backtest.

    Convenience wrapper that resolves the strategy from the registry,
    normalises the candle data, constructs a
    :class:`~quaver.backtest.portfolio.Portfolio` and
    :class:`~quaver.backtest.engine.BacktestEngine`, and returns the result.

    :param engine_name: Registered strategy name (e.g. ``"mean_reversion"``).
    :type engine_name: str
    :param parameters: Strategy-specific parameter dictionary passed to the
        strategy constructor.
    :type parameters: dict
    :param candles: Raw OHLCV DataFrame; normalisation is applied internally
        via :func:`~quaver.backtest.data.normalise_candles`.
    :type candles: pandas.DataFrame
    :param instrument_id: Label embedded in trade records and the result
        object.
    :type instrument_id: str
    :param initial_capital: Starting cash balance.  Defaults to ``10_000.0``.
    :type initial_capital: float
    :param quantity_per_trade: Fixed number of units per trade.  Defaults to
        ``1.0``.
    :type quantity_per_trade: float
    :param ts_column: Name of the timestamp column in ``candles``.  Defaults
        to ``"ts"``.
    :type ts_column: str
    :param allow_shorting: When ``True``, ``SELL`` signals from a flat
        portfolio open a short position.  Defaults to ``False``.
    :type allow_shorting: bool
    :returns: :class:`~quaver.backtest.result.BacktestResult` containing all
        trades and performance metrics.
    :rtype: BacktestResult
    :raises EngineNotFoundError: If ``engine_name`` is not present in the
        strategy registry.
    :raises TypeError: If ``engine_name`` resolves to a
        :class:`~quaver.strategies.base.MultiAssetStrategy`; use
        :func:`run_multi_asset_backtest` instead.
    :raises ValueError: If strategy parameters are invalid or if ``candles``
        contains fewer rows than required by the strategy.
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
    parameters: dict[str, object],
    candles_map: dict[str, pd.DataFrame],
    initial_capital: float = 10_000.0,
    quantity_per_trade: float = 1.0,
    ts_column: str = "ts",
    allow_shorting: bool = False,
    timestamp_overlap_warn_threshold: float = 0.05,
) -> dict[str, BacktestResult]:
    """Run a multi-asset walk-forward backtest.

    Convenience wrapper that resolves the strategy from the registry,
    normalises all candle DataFrames, builds one
    :class:`~quaver.backtest.portfolio.Portfolio` per instrument, constructs a
    :class:`~quaver.backtest.multi_engine.MultiAssetBacktestEngine`, and
    returns the per-instrument results.

    :param engine_name: Registered
        :class:`~quaver.strategies.base.MultiAssetStrategy` name.
    :type engine_name: str
    :param parameters: Strategy-specific parameter dictionary passed to the
        strategy constructor.
    :type parameters: dict
    :param candles_map: Mapping of ``instrument_id`` to a raw OHLCV
        DataFrame; normalisation is applied internally.
    :type candles_map: dict[str, pandas.DataFrame]
    :param initial_capital: Starting cash balance per instrument portfolio.
        Defaults to ``10_000.0``.
    :type initial_capital: float
    :param quantity_per_trade: Fixed number of units per trade per instrument.
        Defaults to ``1.0``.
    :type quantity_per_trade: float
    :param ts_column: Timestamp column name shared by all DataFrames.
        Defaults to ``"ts"``.
    :type ts_column: str
    :param allow_shorting: Passed through to each instrument's portfolio
        engine.  Defaults to ``False``.
    :type allow_shorting: bool
    :param timestamp_overlap_warn_threshold: If the shared timestamp
        intersection drops more than this fraction of any instrument's
        timestamps, a ``WARNING`` is logged.  Defaults to ``0.05`` (5%).
    :type timestamp_overlap_warn_threshold: float
    :returns: Mapping of ``instrument_id`` to
        :class:`~quaver.backtest.result.BacktestResult`.
    :rtype: dict[str, BacktestResult]
    :raises TypeError: If ``engine_name`` resolves to a single-asset
        :class:`~quaver.strategies.base.BaseStrategy`; use
        :func:`run_backtest` instead.
    :raises ValueError: If instruments required by the strategy are absent
        from ``candles_map``, or if any instrument has fewer candles than the
        strategy requires.
    """
    strategy_cls = StrategyRegistry.get(engine_name)

    if not issubclass(strategy_cls, MultiAssetStrategy):
        raise TypeError(
            f"Engine '{engine_name}' is a single-asset BaseStrategy. Use run_backtest() instead."
        )

    strategy: MultiAssetStrategy = strategy_cls(parameters=parameters)
    strategy.validate_parameters()

    # Validate required instruments are present
    required_ids = strategy.get_required_instrument_ids()
    missing = set(required_ids) - set(candles_map)
    if missing:
        raise ValueError(f"Required instrument(s) missing from candles_map: {sorted(missing)}")

    # Normalise all DataFrames
    clean_map = {iid: normalise_candles(df, ts_col=ts_column) for iid, df in candles_map.items()}

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
