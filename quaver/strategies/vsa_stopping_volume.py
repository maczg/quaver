"""VSA Stopping Volume strategy engine.

Implements a Volume Spread Analysis pattern:

- Quantitative features from OHLCV (spread, close_position, vol_rel, spread_rel)
- BUY (Stopping Volume) pattern in a local downtrend
- Optional symmetric SELL (distribution on highs)

Candles must be ordered by ts ASC.
Emits at most one signal per compute() call (latest candle).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quaver.types import SignalDirection
from quaver.strategies.base import BaseStrategy, SignalOutput
from quaver.strategies.indicators import sma, volume_relative
from quaver.strategies.registry import StrategyRegistry

log = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    # lookbacks
    "sma_window": 20,
    "trend_sma": 20,

    # matrix thresholds
    "vol_high": 1.5,
    "vol_low": 0.7,
    "spread_big": 1.3,
    "spread_small": 0.7,

    # stopping volume pattern thresholds
    "stopping_vol_rel": 2.0,
    "buy_close_pos_min": 0.4,
    "sell_close_pos_max": 0.6,

    # enable/disable patterns
    "enable_buy": True,
    "enable_sell": True,
}


@StrategyRegistry.register("vsa_stopping_volume")
class VSAStoppingVolumeStrategy(BaseStrategy):
    """VSA-style pattern engine using stopping-volume reversal heuristic.

    Computes four quantitative features from OHLCV data — candle spread,
    close position within the bar, relative volume, and relative spread —
    and applies a stopping-volume pattern to identify potential reversals.

    **Signal logic**

    * **BUY (Stopping Volume)** — fires when the market is in a local
      downtrend (``close < trend_sma``), the current candle is bearish,
      relative volume exceeds *stopping_vol_rel*, the spread is narrow
      (below *spread_small*), and the close is not near the lows of the
      bar (``close_position > buy_close_pos_min``).  This combination
      signals absorbed selling pressure.
    * **SELL (Distribution)** — symmetric mirror of the BUY pattern,
      active only when *enable_sell* is ``True``.  Fires in an uptrend
      with a bullish bar, high volume, narrow spread, and close near the
      bottom of the range (``close_position < sell_close_pos_max``).

    Confidence is derived from excess relative volume above
    *stopping_vol_rel*, with a bonus ``+0.1`` when the bar is also
    classified as an ``absorption_trap`` (high volume, narrow spread).

    :param sma_window: Lookback period for volume and spread normalisation
        SMAs. Defaults to ``20``.
    :type sma_window: int
    :param trend_sma: Lookback period for the closing-price SMA used to
        determine local trend direction. Defaults to ``20``.
    :type trend_sma: int
    :param vol_high: Relative-volume threshold above which volume is
        classified as "high" in the matrix state. Defaults to ``1.5``.
    :type vol_high: float
    :param vol_low: Relative-volume threshold below which volume is
        classified as "low" in the matrix state. Defaults to ``0.7``.
    :type vol_low: float
    :param spread_big: Relative-spread threshold above which a candle is
        considered wide. Defaults to ``1.3``.
    :type spread_big: float
    :param spread_small: Relative-spread threshold below which a candle is
        considered narrow (absorption candidate). Defaults to ``0.7``.
    :type spread_small: float
    :param stopping_vol_rel: Minimum relative volume for a stopping-volume
        event on either side. Defaults to ``2.0``.
    :type stopping_vol_rel: float
    :param buy_close_pos_min: Minimum close position (``[0, 1]``) for the
        BUY pattern to fire.  Values near ``1.0`` mean the close must be
        near the top of the bar. Defaults to ``0.4``.
    :type buy_close_pos_min: float
    :param sell_close_pos_max: Maximum close position (``[0, 1]``) for the
        SELL pattern to fire.  Values near ``0.0`` mean the close must be
        near the bottom of the bar. Defaults to ``0.6``.
    :type sell_close_pos_max: float
    :param enable_buy: Whether to evaluate the BUY stopping-volume pattern.
        Defaults to ``True``.
    :type enable_buy: bool
    :param enable_sell: Whether to evaluate the SELL symmetric pattern.
        Defaults to ``True``.
    :type enable_sell: bool
    """

    display_name = "VSA Stopping Volume"
    description = (
        "VSA-style pattern engine: computes spread/close-position and relative volume/spread. "
        "Generates BUY (and optional SELL) signals using a stopping-volume reversal heuristic "
        "with mandatory absorption (narrow spread)."
    )

    def validate_parameters(self) -> None:
        """Validate all strategy parameters.

        Checks integer parameters (*sma_window*, *trend_sma*) for positive
        ``int`` type, positive-number parameters for ``> 0`` numeric type,
        unit-interval parameters (*buy_close_pos_min*, *sell_close_pos_max*)
        for membership in ``[0, 1]``, and boolean flags for ``bool`` type.

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

        def _unit(name: str) -> None:
            v = p.get(name)
            if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v!r}")

        _pos_int("sma_window")
        _pos_int("trend_sma")
        _pos_num("vol_high")
        _pos_num("vol_low")
        _pos_num("spread_big")
        _pos_num("spread_small")
        _pos_num("stopping_vol_rel")
        _unit("buy_close_pos_min")
        _unit("sell_close_pos_max")

        if not isinstance(p.get("enable_buy"), bool):
            raise ValueError("enable_buy must be boolean")
        if not isinstance(p.get("enable_sell"), bool):
            raise ValueError("enable_sell must be boolean")

    def get_required_candle_count(self) -> int:
        """Return the minimum number of historical candles required.

        The value is ``max(sma_window, trend_sma) + 5`` to ensure both SMAs
        are warm with a small safety buffer.

        :returns: Minimum candle count needed before ``compute()`` will produce
            a signal.
        :rtype: int
        """
        n = int(self.parameters.get("sma_window", _DEFAULTS["sma_window"]))
        t = int(self.parameters.get("trend_sma", _DEFAULTS["trend_sma"]))
        return max(n, t) + 5

    def compute(
            self,
            candles: pd.DataFrame,
            as_of: datetime,
    ) -> SignalOutput | None:
        """Run VSA stopping-volume logic on a single listing's candles.

        Computes spread, close position, relative volume, and relative spread
        for the entire series, then evaluates the BUY and SELL pattern
        conditions on the latest bar only.  Returns ``None`` when the candle
        DataFrame is empty, required indicators are ``NaN``, or neither
        pattern fires.

        :param candles: OHLCV DataFrame ordered by timestamp ascending.
            Must contain ``open``, ``high``, ``low``, ``close``, and
            ``volume`` columns.  The current bar being evaluated is **not**
            included.
        :type candles: pandas.DataFrame
        :param as_of: Point-in-time timestamp of the current bar being
            evaluated.  Stored verbatim in the signal metadata.
        :type as_of: datetime.datetime
        :returns: A :class:`~quaver.strategies.base.SignalOutput` when the
            stopping-volume BUY or symmetric SELL pattern fires on the latest
            bar; ``None`` otherwise.
        :rtype: SignalOutput or None
        """
        if candles.empty:
            return None

        n_sma: int = self.parameters["sma_window"]
        trend_n: int = self.parameters["trend_sma"]

        # Extract OHLCV arrays
        opens = candles["open"].astype(float).values
        highs = candles["high"].astype(float).values
        lows = candles["low"].astype(float).values
        closes = candles["close"].astype(float).values
        vols = candles["volume"].astype(float).values

        # Spread series and relative indicators
        spreads = highs - lows
        spread_sma = sma(spreads, n_sma)
        vol_rel = volume_relative(vols, n_sma)
        trend_sma_arr = sma(closes, trend_n)

        # Close position: (close - low) / (high - low), in [0, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            close_pos = np.where(spreads != 0, (closes - lows) / spreads, np.nan)

        # Spread relative: spread / sma(spread)
        with np.errstate(divide="ignore", invalid="ignore"):
            spread_rel = np.where(
                (~np.isnan(spread_sma)) & (spread_sma != 0),
                spreads / spread_sma,
                np.nan,
            )

        # Latest bar
        t = len(candles) - 1
        cp = close_pos[t]
        vr = vol_rel[t]
        sr = spread_rel[t]
        ts = trend_sma_arr[t]

        if any(np.isnan(x) for x in (cp, vr, sr, ts)):
            return None

        o, c, spread = opens[t], closes[t], spreads[t]

        # Matrix label (for metadata)
        vol_high = float(self.parameters["vol_high"])
        vol_low = float(self.parameters["vol_low"])
        spread_big = float(self.parameters["spread_big"])
        spread_small = float(self.parameters["spread_small"])

        matrix_state: str | None = None
        if vr > vol_high and sr > spread_big:
            matrix_state = "healthy_move"
        elif vr > vol_high and sr < spread_small:
            matrix_state = "absorption_trap"
        elif vr < vol_low and sr > 0.8:
            matrix_state = "no_supply_or_demand"

        # Pattern thresholds
        stopping_vol_rel = float(self.parameters["stopping_vol_rel"])
        buy_close_pos_min = float(self.parameters["buy_close_pos_min"])
        sell_close_pos_max = float(self.parameters["sell_close_pos_max"])

        is_bear = c < o
        is_bull = c > o
        downtrend = c < ts
        uptrend = c > ts

        # Stopping Volume BUY: downtrend + bear candle + high volume + narrow spread + not closing on lows
        buy_ok = (
            bool(self.parameters["enable_buy"])
            and downtrend
            and is_bear
            and vr > stopping_vol_rel
            and sr < spread_small
            and cp > buy_close_pos_min
        )

        # Symmetric SELL: uptrend + bull candle + high volume + narrow spread + not closing on highs
        sell_ok = (
            bool(self.parameters["enable_sell"])
            and uptrend
            and is_bull
            and vr > stopping_vol_rel
            and sr < spread_small
            and cp < sell_close_pos_max
        )

        if not (buy_ok or sell_ok):
            return None

        direction = SignalDirection.BUY if buy_ok else SignalDirection.SELL

        # Confidence: scale with excess relative volume + absorption bonus
        vol_score = min(max((vr - stopping_vol_rel) / max(stopping_vol_rel, 1e-9), 0.0) / 2.0 + 0.5, 1.0)
        absorption_bonus = 0.1 if matrix_state == "absorption_trap" else 0.0
        confidence = min(vol_score + absorption_bonus, 1.0)

        notes = (
            f"vol_rel={vr:.3f} spread_rel={sr:.3f} close_pos={cp:.3f} "
            f"trend_sma={ts:.4f} state={matrix_state or 'n/a'}"
        )

        return SignalOutput(
            direction=direction,
            confidence=round(float(confidence), 4),
            notes=notes,
            metadata={
                "as_of": as_of.isoformat(),
                "spread": round(float(spread), 8),
                "close_position": round(float(cp), 6),
                "vol_rel": round(float(vr), 6),
                "spread_rel": round(float(sr), 6),
                "trend_sma": round(float(ts), 8),
                "matrix_state": matrix_state,
                "params": {
                    "sma_window": n_sma,
                    "trend_sma": trend_n,
                    "stopping_vol_rel": stopping_vol_rel,
                    "buy_close_pos_min": buy_close_pos_min,
                    "sell_close_pos_max": sell_close_pos_max,
                },
            },
        )

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        """Return a JSON Schema object describing all strategy parameters.

        :returns: A ``dict`` conforming to JSON Schema ``"type": "object"``
            with a ``"properties"`` key mapping each parameter name to its
            type, constraints, and human-readable description, and a
            ``"required"`` key listing all parameter names.
        :rtype: dict[str, Any]
        """
        return {
            "type": "object",
            "properties": {
                "sma_window": {"type": "integer", "minimum": 1,
                               "description": "SMA window for volume/spread normalization"},
                "trend_sma": {"type": "integer", "minimum": 1,
                              "description": "SMA window for local trend filter on close"},
                "vol_high": {"type": "number", "exclusiveMinimum": 0,
                             "description": "High relative volume threshold"},
                "vol_low": {"type": "number", "exclusiveMinimum": 0,
                            "description": "Low relative volume threshold"},
                "spread_big": {"type": "number", "exclusiveMinimum": 0,
                               "description": "Large relative spread threshold"},
                "spread_small": {"type": "number", "exclusiveMinimum": 0,
                                 "description": "Small relative spread threshold"},
                "stopping_vol_rel": {"type": "number", "exclusiveMinimum": 0,
                                     "description": "Relative volume threshold for stopping-volume pattern"},
                "buy_close_pos_min": {"type": "number", "minimum": 0, "maximum": 1,
                                      "description": "Min close position for BUY pattern"},
                "sell_close_pos_max": {"type": "number", "minimum": 0, "maximum": 1,
                                       "description": "Max close position for SELL pattern"},
                "enable_buy": {"type": "boolean", "description": "Enable BUY stopping-volume signals"},
                "enable_sell": {"type": "boolean", "description": "Enable SELL symmetric signals"},
            },
            "required": list(_DEFAULTS.keys()),
        }

    @classmethod
    def get_default_parameters(cls) -> dict[str, Any]:
        """Return a copy of the default parameter dictionary.

        :returns: Mapping of every parameter name to its default value as
            defined in the module-level ``_DEFAULTS`` constant.
        :rtype: dict[str, Any]
        """
        return dict(_DEFAULTS)
