"""Time-Series Momentum (TSMOM) strategy engine.

Reference implementation of the *time-series momentum* anomaly documented in
Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", *Journal of
Financial Economics*.

The rule is intentionally minimal: look at the cumulative return over a
trailing *lookback_period* (typically 12 months / ~252 trading days) and take
the **sign** of that return as the trade direction.

* trailing return > 0  →  BUY  (the asset has been trending up; expect it to
  keep trending up)
* trailing return < 0  →  SELL (the asset has been trending down; expect it
  to keep trending down — i.e. short or stay flat depending on the engine's
  ``allow_shorting`` flag)

This is the textbook *trend-following* primitive and is structurally
uncorrelated with mean-reversion engines: it makes money in persistent
directional regimes, where mean-reversion strategies bleed.

Optional *skip_period* (a "12-1" momentum convention popularised by Asness,
Moskowitz & Pedersen) drops the most recent *N* bars from the trailing
window to neutralise short-horizon reversal noise.  Set to ``0`` to
disable.
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
    "lookback_period": 252,
    "skip_period": 21,
    "threshold": 0.0,
}


@StrategyRegistry.register("time_series_momentum")
class TimeSeriesMomentumStrategy(BaseStrategy):
    """Time-Series Momentum (TSMOM) — trend-following by sign of trailing return.

    Computes the cumulative return over the last *lookback_period* bars
    (optionally skipping the most recent *skip_period* bars) and emits a
    signal in the direction of that return when its absolute value exceeds
    *threshold*.

    **Signal logic**

    Let ``r = close[-1 - skip_period] / close[-lookback_period - skip_period] - 1``
    be the trailing return computed over the window
    ``[-lookback_period - skip_period, -1 - skip_period]``.

    * **BUY**  when ``r > +threshold``  (asset is in an up-trend)
    * **SELL** when ``r < -threshold``  (asset is in a down-trend)
    * No signal when ``|r| <= threshold``.

    Confidence scales with the magnitude of the trailing return, capped at
    ``1.0``.

    .. note::

       SELL signals mean "trailing return is negative — expect the down-trend
       to persist".  Whether the backtest engine opens a short or simply stays
       flat is controlled by the engine's ``allow_shorting`` flag.

    :param lookback_period: Number of bars over which to measure the
        trailing return.  Defaults to ``252`` (≈ one trading year on daily
        bars).
    :type lookback_period: int
    :param skip_period: Number of most-recent bars to drop from the trailing
        window (the "12-1" convention skips the most recent month to avoid
        short-term reversal contamination).  Defaults to ``21`` (≈ one trading
        month).  Set to ``0`` to disable.
    :type skip_period: int
    :param threshold: Minimum absolute trailing return required to trigger a
        signal (e.g. ``0.0`` = any non-zero return; ``0.02`` = 2 %).  Must be
        non-negative.  Defaults to ``0.0``.
    :type threshold: float
    """

    display_name = "Time-Series Momentum"
    description = (
        "Trend-following by sign of trailing return. BUY when the trailing "
        "lookback return is positive, SELL when negative. Optional skip period "
        "implements the classic 12-1 momentum convention."
    )

    def validate_parameters(self) -> None:
        """Validate all strategy parameters.

        Checks that *lookback_period* is a positive integer, *skip_period* is a
        non-negative integer strictly less than *lookback_period*, and that
        *threshold* is a non-negative number.

        :raises ValueError: If any parameter fails its type or range check.
        """
        lookback = self.parameters.get("lookback_period")
        skip = self.parameters.get("skip_period")
        threshold = self.parameters.get("threshold")

        if not isinstance(lookback, int) or lookback < 2:
            raise ValueError(
                f"lookback_period must be an integer >= 2, got {lookback!r}"
            )
        if not isinstance(skip, int) or skip < 0:
            raise ValueError(
                f"skip_period must be a non-negative integer, got {skip!r}"
            )
        if skip >= lookback:
            raise ValueError(
                f"skip_period ({skip}) must be less than lookback_period ({lookback})"
            )
        if not isinstance(threshold, (int, float)) or threshold < 0:
            raise ValueError(
                f"threshold must be a non-negative number, got {threshold!r}"
            )

    def get_required_candle_count(self) -> int:
        """Return the minimum number of historical candles required.

        The value is ``lookback_period + skip_period + 1`` so that the
        trailing window plus the skip offset both fit in the available
        history with one bar of margin.

        :returns: Minimum candle count needed before ``compute()`` will
            produce a signal.
        :rtype: int
        """
        lookback = int(self.parameters.get("lookback_period", _DEFAULTS["lookback_period"]))
        skip = int(self.parameters.get("skip_period", _DEFAULTS["skip_period"]))
        return lookback + skip + 1

    def compute(self, candles: pd.DataFrame, as_of: datetime) -> SignalOutput | None:
        """Run TSMOM logic on a single listing's candles.

        Computes the trailing return over the window
        ``[-lookback_period - skip_period, -1 - skip_period]`` and emits a
        BUY or SELL signal in the direction of that return when its absolute
        value exceeds *threshold*.

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
        lookback: int = self.parameters["lookback_period"]
        skip: int = self.parameters["skip_period"]
        threshold: float = self.parameters["threshold"]

        closes = candles["close"].astype(float).tolist()
        if len(closes) < lookback + skip + 1:
            return None

        end_idx = len(closes) - 1 - skip
        start_idx = end_idx - lookback
        end_price = closes[end_idx]
        start_price = closes[start_idx]

        if start_price == 0:
            return None

        trailing_return = end_price / start_price - 1.0

        if abs(trailing_return) <= threshold:
            return None

        direction = (
            SignalDirection.BUY if trailing_return > 0 else SignalDirection.SELL
        )
        # Confidence: 10 % return → ~1.0; smaller returns scale linearly.
        raw_confidence = min(abs(trailing_return) / 0.10, 1.0)

        return SignalOutput(
            direction=direction,
            confidence=round(raw_confidence, 4),
            notes=(
                f"trailing_return={trailing_return:.4f} "
                f"lookback={lookback} skip={skip}"
            ),
            metadata={
                "lookback_period": lookback,
                "skip_period": skip,
                "threshold": threshold,
                "start_price": round(start_price, 6),
                "end_price": round(end_price, 6),
                "trailing_return": round(trailing_return, 6),
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
                "lookback_period": {
                    "type": "integer",
                    "minimum": 2,
                    "default": 252,
                    "description": (
                        "Number of bars over which to measure the trailing "
                        "return. 252 ≈ one trading year on daily bars."
                    ),
                },
                "skip_period": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 21,
                    "description": (
                        "Number of most-recent bars to drop from the trailing "
                        "window (12-1 momentum convention). 0 disables."
                    ),
                },
                "threshold": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.0,
                    "description": (
                        "Minimum absolute trailing return to trigger a signal "
                        "(e.g. 0.02 = 2%)."
                    ),
                },
            },
            "required": list(_DEFAULTS.keys()),
        }

    @classmethod
    def get_default_parameters(cls) -> dict[str, Any]:
        """Return a copy of the default parameter dictionary.

        :returns: Mapping of parameter names to their default values:
            ``lookback_period=252``, ``skip_period=21``, ``threshold=0.0``.
        :rtype: dict[str, Any]
        """
        return dict(_DEFAULTS)