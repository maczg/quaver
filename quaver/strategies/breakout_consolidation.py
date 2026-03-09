"""Breakout from Consolidation strategy engine.

Detects low-volatility consolidation phases followed by a high-volume
breakout above resistance, in the direction of the prevailing trend.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quaver.strategies.base import BaseStrategy, SignalOutput
from quaver.strategies.indicators import atr, rolling_max, rolling_min, sma, volume_relative
from quaver.strategies.registry import StrategyRegistry
from quaver.types import SignalDirection

log = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "ma_period": 50,
    "consolidation_period": 20,
    "range_max_pct": 0.10,
    "atr_period": 14,
    "atr_lookback": 10,
    "volume_sma_period": 20,
}


@StrategyRegistry.register("breakout_consolidation")
class BreakoutConsolidationStrategy(BaseStrategy):
    """Breakout from Consolidation strategy.

    Identifies stocks in a tight, low-volatility price range (consolidation)
    and generates a BUY signal when price breaks above the range ceiling
    with above-average volume and declining ATR.

    **Signal logic**

    All five conditions must be true on the latest bar:

    * **Trend filter** -- ``close > SMA(ma_period)`` ensures the breakout
      is in the direction of the medium-term trend.
    * **Consolidation** -- the 20-day price range (highest high minus lowest
      low) as a fraction of price must be <= *range_max_pct*.
    * **Volatility compression** -- ``ATR(atr_period)`` today must be lower
      than *atr_lookback* bars ago, confirming narrowing daily swings.
    * **Breakout trigger** -- today's close exceeds the highest high of
      the prior *consolidation_period* bars.
    * **Volume confirmation** -- today's volume exceeds the
      *volume_sma_period*-day average volume.

    Confidence scales with the volume surge above average, capped at 1.0.

    :param ma_period: Trend-filter SMA window. Defaults to ``50``.
    :type ma_period: int
    :param consolidation_period: Lookback for range and breakout ceiling.
        Defaults to ``20``.
    :type consolidation_period: int
    :param range_max_pct: Maximum allowed range as a fraction of price
        (e.g. ``0.10`` = 10 %). Defaults to ``0.10``.
    :type range_max_pct: float
    :param atr_period: ATR lookback period. Defaults to ``14``.
    :type atr_period: int
    :param atr_lookback: Number of bars to look back for ATR decline check.
        Defaults to ``10``.
    :type atr_lookback: int
    :param volume_sma_period: Volume SMA window for confirmation.
        Defaults to ``20``.
    :type volume_sma_period: int
    """

    display_name = "Breakout from Consolidation"
    description = (
        "Detects low-volatility consolidation phases and generates BUY signals "
        "when price breaks above the range ceiling with above-average volume "
        "and declining ATR, in the direction of the prevailing trend."
    )

    def validate_parameters(self) -> None:
        """Validate all strategy parameters.

        :raises ValueError: If any parameter fails its type or range check.
        """
        p = self.parameters

        def _pos_int(name: str) -> None:
            v = p.get(name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive integer, got {v!r}")

        def _pos_num(name: str) -> None:
            v = p.get(name)
            if not isinstance(v, (int, float)) or v <= 0:
                raise ValueError(f"{name} must be a positive number, got {v!r}")

        _pos_int("ma_period")
        _pos_int("consolidation_period")
        _pos_num("range_max_pct")
        _pos_int("atr_period")
        _pos_int("atr_lookback")
        _pos_int("volume_sma_period")

    def get_required_candle_count(self) -> int:
        """Return the minimum number of historical candles required.

        :returns: Minimum candle count.
        :rtype: int
        """
        ma = int(self.parameters.get("ma_period", _DEFAULTS["ma_period"]))
        atr_p = int(self.parameters.get("atr_period", _DEFAULTS["atr_period"]))
        atr_lb = int(self.parameters.get("atr_lookback", _DEFAULTS["atr_lookback"]))
        return max(ma, atr_p + atr_lb) + 10

    def compute(
        self,
        candles: pd.DataFrame,
        as_of: datetime,
    ) -> SignalOutput | None:
        """Run breakout-from-consolidation logic on a single listing's candles.

        :param candles: OHLCV DataFrame ordered by timestamp ascending.
        :type candles: pandas.DataFrame
        :param as_of: Point-in-time timestamp of the current bar.
        :type as_of: datetime.datetime
        :returns: A :class:`~quaver.strategies.base.SignalOutput` when all
            breakout conditions are met; ``None`` otherwise.
        :rtype: SignalOutput or None
        """
        if candles.empty:
            return None

        ma_period: int = self.parameters["ma_period"]
        cons_period: int = self.parameters["consolidation_period"]
        range_max: float = self.parameters["range_max_pct"]
        atr_period: int = self.parameters["atr_period"]
        atr_lb: int = self.parameters["atr_lookback"]
        vol_sma_period: int = self.parameters["volume_sma_period"]

        highs = candles["high"].to_numpy(dtype=float)
        lows = candles["low"].to_numpy(dtype=float)
        closes = candles["close"].to_numpy(dtype=float)
        volumes = candles["volume"].to_numpy(dtype=float)

        t = len(candles) - 1

        # Indicators
        ma = sma(closes, ma_period)
        atr_arr = atr(highs, lows, closes, atr_period)
        high_roll = rolling_max(highs, cons_period)
        low_roll = rolling_min(lows, cons_period)
        vol_rel = volume_relative(volumes, vol_sma_period)

        # Check NaN guards
        if any(np.isnan(x) for x in (ma[t], atr_arr[t], high_roll[t], low_roll[t], vol_rel[t])):
            return None

        # Also need ATR from atr_lookback bars ago
        atr_prev_idx = t - atr_lb
        if atr_prev_idx < 0 or np.isnan(atr_arr[atr_prev_idx]):
            return None

        close_t = closes[t]

        # 1. Trend filter: close > MA
        if close_t <= ma[t]:
            return None

        # 2. Consolidation: range as fraction of price <= threshold
        range_pct = (high_roll[t] - low_roll[t]) / close_t
        if range_pct > range_max:
            return None

        # 3. Volatility compression: ATR declining
        if atr_arr[t] >= atr_arr[atr_prev_idx]:
            return None

        # 4. Breakout: close > highest high of prior consolidation_period bars
        # (exclude current bar from the max)
        if t < cons_period:
            return None
        prior_high_max = float(np.nanmax(highs[t - cons_period : t]))
        if close_t <= prior_high_max:
            return None

        # 5. Volume confirmation: volume > SMA(volume)
        if vol_rel[t] <= 1.0:
            return None

        # Confidence: scale with volume surge
        confidence = min((float(vol_rel[t]) - 1.0) / 2.0 + 0.5, 1.0)

        return SignalOutput(
            direction=SignalDirection.BUY,
            confidence=round(confidence, 4),
            notes=(
                f"close={close_t:.4f} ma{ma_period}={ma[t]:.4f} "
                f"range_pct={range_pct:.4f} atr={atr_arr[t]:.4f} "
                f"vol_rel={vol_rel[t]:.3f}"
            ),
            metadata={
                "as_of": as_of.isoformat(),
                "close": round(close_t, 6),
                "ma": round(float(ma[t]), 6),
                "range_pct": round(float(range_pct), 6),
                "atr": round(float(atr_arr[t]), 6),
                "atr_prev": round(float(atr_arr[atr_prev_idx]), 6),
                "vol_rel": round(float(vol_rel[t]), 6),
                "prior_high_max": round(prior_high_max, 6),
                "stop_loss": round(float(low_roll[t]), 6),
            },
        )

    @classmethod
    def get_default_parameters(cls) -> dict[str, Any]:
        """Return a copy of the default parameter dictionary.

        :returns: Mapping of parameter names to their default values.
        :rtype: dict[str, Any]
        """
        return dict(_DEFAULTS)
