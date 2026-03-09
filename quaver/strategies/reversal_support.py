"""Reversal at Support (Mean Reversion Swing) strategy engine.

Counter-trend strategy that buys extreme oversold dips at key support
levels when the stock is NOT in a structural downtrend.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quaver.strategies.base import BaseStrategy, SignalOutput
from quaver.strategies.indicators import rolling_min, rsi, sma
from quaver.strategies.registry import StrategyRegistry
from quaver.types import SignalDirection

log = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "ma_fast": 20,
    "ma_medium": 50,
    "ma_slow": 200,
    "rsi_period": 14,
    "rsi_threshold": 30,
    "max_dist_ma200": 0.20,
    "support_period": 20,
    "support_tolerance": 0.03,
}


@StrategyRegistry.register("reversal_support")
class ReversalSupportStrategy(BaseStrategy):
    """Reversal at Support strategy (mean reversion swing).

    Identifies extreme oversold conditions near a support level and
    generates a BUY signal when a bullish reversal candle confirms
    buyer participation, provided the stock is not in a structural
    downtrend.

    **Signal logic**

    All conditions must be true on the latest bar:

    * **No structural downtrend** -- the absolute distance between close
      and MA(ma_slow) as a fraction of MA(ma_slow) must be less than
      *max_dist_ma200*.
    * **Extreme oversold** -- ``RSI(rsi_period) < rsi_threshold``,
      indicating the stock has fallen sharply enough for seller exhaustion.
    * **At or near support** -- ``close <= rolling_min(low, support_period)
      * (1 + support_tolerance)``, meaning price is near or at its lowest
      level over the lookback period.
    * **Bullish entry trigger** -- ``close > prior bar's high``, confirming
      real buying pressure has emerged.

    Confidence scales with how deeply oversold the RSI is below the
    threshold.

    :param ma_fast: Short-term MA window (used as target). Defaults to ``20``.
    :type ma_fast: int
    :param ma_medium: Medium-term MA window (used as target). Defaults to ``50``.
    :type ma_medium: int
    :param ma_slow: Long-term MA window for structural-health filter.
        Defaults to ``200``.
    :type ma_slow: int
    :param rsi_period: RSI lookback period. Defaults to ``14``.
    :type rsi_period: int
    :param rsi_threshold: RSI must be below this value for an oversold
        condition. Defaults to ``30``.
    :type rsi_threshold: int
    :param max_dist_ma200: Maximum allowed fractional distance from MA(ma_slow).
        Defaults to ``0.20`` (20 %).
    :type max_dist_ma200: float
    :param support_period: Lookback for the rolling low (support level).
        Defaults to ``20``.
    :type support_period: int
    :param support_tolerance: Maximum fraction above the rolling low to still
        be considered "near support". Defaults to ``0.03`` (3 %).
    :type support_tolerance: float
    """

    display_name = "Reversal at Support"
    description = (
        "Counter-trend mean reversion strategy. BUY when the stock is extremely "
        "oversold (RSI < 30), near a key support level, and NOT in a structural "
        "downtrend, confirmed by a bullish reversal candle (close > prior high)."
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

        _pos_int("ma_fast")
        _pos_int("ma_medium")
        _pos_int("ma_slow")
        _pos_int("rsi_period")
        _pos_int("rsi_threshold")
        _pos_num("max_dist_ma200")
        _pos_int("support_period")
        _pos_num("support_tolerance")

    def get_required_candle_count(self) -> int:
        """Return the minimum number of historical candles required.

        :returns: Minimum candle count (driven by the slowest MA).
        :rtype: int
        """
        return int(self.parameters.get("ma_slow", _DEFAULTS["ma_slow"])) + 10

    def compute(
        self,
        candles: pd.DataFrame,
        as_of: datetime,
    ) -> SignalOutput | None:
        """Run reversal-at-support logic on a single listing's candles.

        :param candles: OHLCV DataFrame ordered by timestamp ascending.
        :type candles: pandas.DataFrame
        :param as_of: Point-in-time timestamp of the current bar.
        :type as_of: datetime.datetime
        :returns: A :class:`~quaver.strategies.base.SignalOutput` when all
            reversal conditions are met; ``None`` otherwise.
        :rtype: SignalOutput or None
        """
        if candles.empty:
            return None

        ma_fast_p: int = self.parameters["ma_fast"]
        ma_med_p: int = self.parameters["ma_medium"]
        ma_slow_p: int = self.parameters["ma_slow"]
        rsi_period: int = self.parameters["rsi_period"]
        rsi_thresh: int = self.parameters["rsi_threshold"]
        max_dist: float = self.parameters["max_dist_ma200"]
        support_p: int = self.parameters["support_period"]
        support_tol: float = self.parameters["support_tolerance"]

        highs = candles["high"].to_numpy(dtype=float)
        lows = candles["low"].to_numpy(dtype=float)
        closes = candles["close"].to_numpy(dtype=float)

        t = len(candles) - 1
        if t < 1:
            return None

        # Indicators
        ma_f = sma(closes, ma_fast_p)
        ma_m = sma(closes, ma_med_p)
        ma_s = sma(closes, ma_slow_p)
        rsi_arr = rsi(closes, rsi_period)
        low_roll = rolling_min(lows, support_p)
        stop_low = rolling_min(lows, 5)

        # NaN guards
        if any(np.isnan(x) for x in (ma_s[t], rsi_arr[t], low_roll[t])):
            return None

        close_t = closes[t]

        # 1. No structural downtrend: |close - MA_slow| / MA_slow < max_dist
        if ma_s[t] == 0.0:
            return None
        dist = abs(close_t - ma_s[t]) / ma_s[t]
        if dist >= max_dist:
            return None

        # 2. Extreme oversold: RSI < threshold
        rsi_val = float(rsi_arr[t])
        if rsi_val >= rsi_thresh:
            return None

        # 3. Near support: close <= rolling_min(low, period) * (1 + tolerance)
        support_level = float(low_roll[t])
        if close_t > support_level * (1.0 + support_tol):
            return None

        # 4. Bullish entry trigger: close > prior bar's high
        if close_t <= highs[t - 1]:
            return None

        # Confidence: scale with how deeply oversold RSI is
        confidence = min(max((rsi_thresh - rsi_val) / rsi_thresh * 0.5 + 0.5, 0.0), 1.0)

        # Targets
        target1 = float(ma_f[t]) if not np.isnan(ma_f[t]) else None
        target2 = float(ma_m[t]) if not np.isnan(ma_m[t]) else None

        return SignalOutput(
            direction=SignalDirection.BUY,
            confidence=round(confidence, 4),
            notes=(
                f"close={close_t:.4f} rsi={rsi_val:.2f} "
                f"dist_ma{ma_slow_p}={dist:.4f} support={support_level:.4f}"
            ),
            metadata={
                "as_of": as_of.isoformat(),
                "close": round(close_t, 6),
                "rsi": round(rsi_val, 4),
                "dist_ma_slow": round(dist, 6),
                "support_level": round(support_level, 6),
                "stop_loss": round(float(stop_low[t]), 6) if not np.isnan(stop_low[t]) else None,
                "target_ma_fast": round(target1, 6) if target1 is not None else None,
                "target_ma_medium": round(target2, 6) if target2 is not None else None,
            },
        )

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        """Return a JSON Schema describing accepted parameters.

        :returns: JSON Schema object with parameter types, constraints, and
            defaults.
        :rtype: dict[str, Any]
        """
        return {
            "type": "object",
            "properties": {
                "ma_fast": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 20,
                    "description": "Short-term MA window (used as target).",
                },
                "ma_medium": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 50,
                    "description": "Medium-term MA window (used as target).",
                },
                "ma_slow": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 200,
                    "description": "Long-term MA window for structural-health filter.",
                },
                "rsi_period": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 14,
                    "description": "RSI lookback period.",
                },
                "rsi_threshold": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 30,
                    "description": "RSI must be below this value for an oversold condition.",
                },
                "max_dist_ma200": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "default": 0.20,
                    "description": "Maximum allowed fractional distance from MA(ma_slow) (e.g. 0.20 = 20%).",
                },
                "support_period": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 20,
                    "description": "Lookback for the rolling low (support level).",
                },
                "support_tolerance": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "default": 0.03,
                    "description": "Maximum fraction above the rolling low to be considered near support (e.g. 0.03 = 3%).",
                },
            },
            "required": list(_DEFAULTS.keys()),
        }

    @classmethod
    def get_default_parameters(cls) -> dict[str, Any]:
        """Return a copy of the default parameter dictionary.

        :returns: Mapping of parameter names to their default values.
        :rtype: dict[str, Any]
        """
        return dict(_DEFAULTS)
