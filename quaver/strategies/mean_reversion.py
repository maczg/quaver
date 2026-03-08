"""Mean reversion strategy engine — reference implementation.

This module provides :class:`MeanReversionStrategy`, a dual moving-average
mean-reversion strategy that emits BUY/SELL signals based on the relative
divergence between a fast and a slow simple moving average.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from quaver.types import SignalDirection
from quaver.strategies.base import BaseStrategy, SignalOutput
from quaver.strategies.registry import StrategyRegistry

log = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "fast_period": 20,
    "slow_period": 50,
    "threshold": 0.02,
}


@StrategyRegistry.register("mean_reversion")
class MeanReversionStrategy(BaseStrategy):
    """Dual moving-average mean reversion strategy.

    Computes a fast SMA and a slow SMA over closing prices and emits a signal
    whenever their relative divergence exceeds *threshold*.

    **Signal logic**

    * **BUY** when ``fast_ma < slow_ma`` by more than *threshold* (oversold).
    * **SELL** when ``fast_ma > slow_ma`` by more than *threshold* (overbought).
    * Confidence scales with divergence magnitude, capped at ``1.0``.

    .. note::

       SELL signals mean "overbought — expect mean reversion downward".
       Whether the backtest engine opens a short or simply ignores
       SELL-from-flat is controlled by the engine's ``allow_shorting`` flag
       (default ``False``).

    :param fast_period: Short MA window. Must be a positive integer less than
        *slow_period*. Defaults to ``20``.
    :type fast_period: int
    :param slow_period: Long MA window. Must be a positive integer greater than
        *fast_period*. Defaults to ``50``.
    :type slow_period: int
    :param threshold: Relative divergence threshold to trigger a signal
        (e.g. ``0.02`` = 2 %). Must be a positive number. Defaults to ``0.02``.
    :type threshold: float
    """

    display_name = "Mean Reversion"
    description = (
        "Dual moving-average mean reversion. BUY when fast MA is below slow MA "
        "by more than threshold (oversold). SELL when above (overbought)."
    )

    def validate_parameters(self) -> None:
        """Validate all strategy parameters.

        Checks that *fast_period* and *slow_period* are positive integers,
        that *fast_period* is strictly less than *slow_period*, and that
        *threshold* is a positive number.

        :raises ValueError: If any parameter fails its type or range check, or
            if ``fast_period >= slow_period``.
        """
        fast = self.parameters.get("fast_period")
        slow = self.parameters.get("slow_period")
        threshold = self.parameters.get("threshold")

        if not isinstance(fast, int) or fast < 1:
            raise ValueError(f"fast_period must be a positive integer, got {fast!r}")
        if not isinstance(slow, int) or slow < 1:
            raise ValueError(f"slow_period must be a positive integer, got {slow!r}")
        if fast >= slow:
            raise ValueError(f"fast_period ({fast}) must be less than slow_period ({slow})")
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError(f"threshold must be a positive number, got {threshold!r}")

    def get_required_candle_count(self) -> int:
        """Return the minimum number of historical candles required.

        The value is ``slow_period + 10`` to allow the slow MA to be computed
        with a small safety buffer.

        :returns: Minimum candle count needed before ``compute()`` will produce
            a signal.
        :rtype: int
        """
        return int(self.parameters.get("slow_period", _DEFAULTS["slow_period"])) + 10

    def compute(self, candles: pd.DataFrame, as_of: datetime) -> SignalOutput | None:
        """Run mean-reversion logic on a single listing's candles.

        Computes a fast SMA and a slow SMA from the closing prices in
        *candles*, then emits a BUY or SELL signal when their relative
        divergence exceeds the configured *threshold*.  Returns ``None`` when
        there are insufficient bars, when the slow MA is zero, or when the
        divergence is below the threshold.

        :param candles: OHLCV DataFrame ordered by timestamp ascending.
            Must contain at least a ``close`` column.  The current bar being
            evaluated is **not** included.
        :type candles: pandas.DataFrame
        :param as_of: Point-in-time timestamp of the current bar being
            evaluated.
        :type as_of: datetime.datetime
        :returns: A :class:`~quaver.strategies.base.SignalOutput` with
            ``direction``, ``confidence``, ``notes``, and ``metadata`` when a
            signal condition is met; ``None`` otherwise.
        :rtype: SignalOutput or None
        """
        fast_period: int = self.parameters["fast_period"]
        slow_period: int = self.parameters["slow_period"]
        threshold: float = self.parameters["threshold"]

        closes = candles["close"].astype(float).tolist()
        if len(closes) < slow_period:
            return None

        fast_ma = sum(closes[-fast_period:]) / fast_period
        slow_ma = sum(closes[-slow_period:]) / slow_period

        if slow_ma == 0:
            return None

        divergence = (fast_ma - slow_ma) / slow_ma
        if abs(divergence) < threshold:
            return None

        raw_confidence = min(abs(divergence) / threshold * 0.3, 1.0)
        direction = SignalDirection.BUY if divergence < -threshold else SignalDirection.SELL

        return SignalOutput(
            direction=direction,
            confidence=round(raw_confidence, 4),
            notes=(f"fast_ma={fast_ma:.4f} slow_ma={slow_ma:.4f} divergence={divergence:.4f}"),
            metadata={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "threshold": threshold,
                "fast_ma": round(fast_ma, 6),
                "slow_ma": round(slow_ma, 6),
                "divergence": round(divergence, 6),
            },
        )

    @classmethod
    def get_default_parameters(cls) -> dict[str, Any]:
        """Return a copy of the default parameter dictionary.

        :returns: Mapping of parameter names to their default values:
            ``fast_period=20``, ``slow_period=50``, ``threshold=0.02``.
        :rtype: dict[str, Any]
        """
        return dict(_DEFAULTS)
