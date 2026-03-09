"""Pullback in Trend (Trend Continuation) strategy engine.

Waits for a temporary price retrace toward the short-term moving average
within a confirmed multi-timeframe uptrend, then enters when momentum
resumes.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quaver.strategies.base import BaseStrategy, SignalOutput
from quaver.strategies.indicators import atr, rolling_min, rsi, sma
from quaver.strategies.registry import StrategyRegistry
from quaver.types import SignalDirection

log = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "ma_fast": 20,
    "ma_medium": 50,
    "ma_slow": 200,
    "rsi_period": 14,
    "rsi_low": 40,
    "rsi_high": 50,
    "atr_period": 14,
    "atr_stop_mult": 0.5,
    "slope_lookback": 5,
    "near_ma_pct": 0.02,
}


@StrategyRegistry.register("pullback_trend")
class PullbackTrendStrategy(BaseStrategy):
    """Pullback in Trend strategy (trend continuation).

    Identifies healthy pullbacks within a confirmed uptrend and generates a
    BUY signal when the price shows signs of resuming its upward move.

    **Signal logic**

    All conditions must be true on the latest bar:

    * **Multi-timeframe uptrend** -- ``close > MA(ma_medium)``,
      ``MA(ma_medium) > MA(ma_slow)``, and MA(ma_medium) slope is positive
      (today > *slope_lookback* bars ago).
    * **Near short-term MA** -- ``close <= MA(ma_fast) * (1 + near_ma_pct)``,
      indicating price has pulled back toward the short-term average.
    * **RSI pullback zone** -- ``RSI(rsi_period)`` is between *rsi_low* and
      *rsi_high*, confirming a healthy dip rather than a collapse.
    * **Entry trigger** -- ``close > prior bar's high`` **or**
      ``close > MA(ma_fast)``, indicating resumed upward momentum.

    Confidence scales with RSI proximity to the lower bound (more oversold
    = higher conviction).

    :param ma_fast: Short-term MA window. Defaults to ``20``.
    :type ma_fast: int
    :param ma_medium: Medium-term MA window. Defaults to ``50``.
    :type ma_medium: int
    :param ma_slow: Long-term MA window. Defaults to ``200``.
    :type ma_slow: int
    :param rsi_period: RSI lookback period. Defaults to ``14``.
    :type rsi_period: int
    :param rsi_low: Lower bound of the RSI pullback zone. Defaults to ``40``.
    :type rsi_low: int
    :param rsi_high: Upper bound of the RSI pullback zone. Defaults to ``50``.
    :type rsi_high: int
    :param atr_period: ATR lookback for stop-loss buffer. Defaults to ``14``.
    :type atr_period: int
    :param atr_stop_mult: Multiplier of ATR for stop-loss buffer below the
        pullback low. Defaults to ``0.5``.
    :type atr_stop_mult: float
    :param slope_lookback: Number of bars to look back for MA slope check.
        Defaults to ``5``.
    :type slope_lookback: int
    :param near_ma_pct: Maximum distance above MA(ma_fast) as a fraction of
        MA (e.g. ``0.02`` = 2 %). Defaults to ``0.02``.
    :type near_ma_pct: float
    """

    display_name = "Pullback in Trend"
    description = (
        "Trend continuation strategy. BUY when price pulls back to the short-term "
        "MA within a confirmed multi-timeframe uptrend and RSI indicates a healthy "
        "dip (40-50), with momentum resumption confirmed by a close above the prior "
        "bar's high or the short-term MA."
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
        _pos_int("rsi_low")
        _pos_int("rsi_high")
        _pos_int("atr_period")
        _pos_num("atr_stop_mult")
        _pos_int("slope_lookback")
        _pos_num("near_ma_pct")

        if p["ma_fast"] >= p["ma_medium"]:
            raise ValueError(
                f"ma_fast ({p['ma_fast']}) must be less than ma_medium ({p['ma_medium']})"
            )
        if p["ma_medium"] >= p["ma_slow"]:
            raise ValueError(
                f"ma_medium ({p['ma_medium']}) must be less than ma_slow ({p['ma_slow']})"
            )
        if p["rsi_low"] >= p["rsi_high"]:
            raise ValueError(
                f"rsi_low ({p['rsi_low']}) must be less than rsi_high ({p['rsi_high']})"
            )

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
        """Run pullback-in-trend logic on a single listing's candles.

        :param candles: OHLCV DataFrame ordered by timestamp ascending.
        :type candles: pandas.DataFrame
        :param as_of: Point-in-time timestamp of the current bar.
        :type as_of: datetime.datetime
        :returns: A :class:`~quaver.strategies.base.SignalOutput` when all
            pullback conditions are met; ``None`` otherwise.
        :rtype: SignalOutput or None
        """
        if candles.empty:
            return None

        ma_fast_p: int = self.parameters["ma_fast"]
        ma_med_p: int = self.parameters["ma_medium"]
        ma_slow_p: int = self.parameters["ma_slow"]
        rsi_period: int = self.parameters["rsi_period"]
        rsi_lo: int = self.parameters["rsi_low"]
        rsi_hi: int = self.parameters["rsi_high"]
        atr_period: int = self.parameters["atr_period"]
        atr_mult: float = self.parameters["atr_stop_mult"]
        slope_lb: int = self.parameters["slope_lookback"]
        near_pct: float = self.parameters["near_ma_pct"]

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
        atr_arr = atr(highs, lows, closes, atr_period)
        pullback_low = rolling_min(lows, 5)

        # NaN guards
        slope_idx = t - slope_lb
        if slope_idx < 0:
            return None
        if any(np.isnan(x) for x in (ma_f[t], ma_m[t], ma_s[t], rsi_arr[t], ma_m[slope_idx])):
            return None

        close_t = closes[t]

        # 1. Multi-timeframe uptrend: close > MA_medium > MA_slow, slope positive
        if not (close_t > ma_m[t] and ma_m[t] > ma_s[t]):
            return None
        if ma_m[t] <= ma_m[slope_idx]:
            return None

        # 2. Near short-term MA (pullback zone)
        if close_t > ma_f[t] * (1.0 + near_pct):
            return None

        # 3. RSI in pullback range
        rsi_val = float(rsi_arr[t])
        if not (rsi_lo <= rsi_val <= rsi_hi):
            return None

        # 4. Entry trigger: close > prior high OR close > MA_fast
        prior_high = highs[t - 1]
        if not (close_t > prior_high or close_t > ma_f[t]):
            return None

        # Confidence: scale with how oversold RSI is within the zone
        rsi_range = rsi_hi - rsi_lo
        confidence = 0.5 + 0.5 * (rsi_hi - rsi_val) / max(rsi_range, 1)
        confidence = min(max(confidence, 0.0), 1.0)

        # Stop loss
        stop = float("nan")
        if not np.isnan(pullback_low[t]) and not np.isnan(atr_arr[t]):
            stop = float(pullback_low[t]) - atr_mult * float(atr_arr[t])

        return SignalOutput(
            direction=SignalDirection.BUY,
            confidence=round(confidence, 4),
            notes=(
                f"close={close_t:.4f} ma{ma_fast_p}={ma_f[t]:.4f} "
                f"ma{ma_med_p}={ma_m[t]:.4f} ma{ma_slow_p}={ma_s[t]:.4f} "
                f"rsi={rsi_val:.2f}"
            ),
            metadata={
                "as_of": as_of.isoformat(),
                "close": round(close_t, 6),
                "ma_fast": round(float(ma_f[t]), 6),
                "ma_medium": round(float(ma_m[t]), 6),
                "ma_slow": round(float(ma_s[t]), 6),
                "rsi": round(rsi_val, 4),
                "stop_loss": round(stop, 6) if not np.isnan(stop) else None,
            },
        )

    @classmethod
    def get_default_parameters(cls) -> dict[str, Any]:
        """Return a copy of the default parameter dictionary.

        :returns: Mapping of parameter names to their default values.
        :rtype: dict[str, Any]
        """
        return dict(_DEFAULTS)
