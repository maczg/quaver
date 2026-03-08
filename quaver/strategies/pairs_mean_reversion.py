"""Pairs mean-reversion strategy (statistical arbitrage example).

This module provides :class:`PairsMeanReversionStrategy`, a classical
two-leg statistical arbitrage strategy that trades the normalised price
spread between two instruments using z-score entry and exit thresholds.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quaver.types import SignalDirection
from quaver.strategies.base import (
    MultiAssetStrategy,
    MultiAssetStrategyOutput,
    SignalOutput,
)
from quaver.strategies.registry import StrategyRegistry

log = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "spread_window": 60,
    "entry_z": 2.0,
    "exit_z": 0.5,
}


@StrategyRegistry.register("pairs_mean_reversion")
class PairsMeanReversionStrategy(MultiAssetStrategy):
    """Classic two-leg spread (pairs) mean-reversion strategy.

    Computes the raw price spread between two instruments (``close_A - close_B``),
    normalises it to a z-score using a rolling window, and emits paired
    BUY/SELL or CLOSE signals based on entry and exit z-score thresholds.

    **Spread definition**::

        spread[t]  = close_A[t] - close_B[t]
        z_score[t] = (spread[t] - rolling_mean(spread, window))
                     / rolling_std(spread, window)

    **Signal logic**

    * ``z_score > +entry_z`` — SELL A, BUY B (spread expected to fall).
    * ``z_score < -entry_z`` — BUY A, SELL B (spread expected to rise).
    * ``|z_score| < exit_z``  — CLOSE A, CLOSE B (convergence achieved).

    .. important::

       Both legs' signals are always emitted together in one
       :class:`~quaver.strategies.base.MultiAssetStrategyOutput` so that
       the engine applies them atomically.

    :param instrument_a: Identifier of the first leg.  Must match a key
        in the *candles_map* supplied to ``compute()``.
    :type instrument_a: str
    :param instrument_b: Identifier of the second leg.  Must match a key
        in the *candles_map* supplied to ``compute()``.  Must differ from
        *instrument_a*.
    :type instrument_b: str
    :param spread_window: Rolling window for computing the z-score mean and
        standard deviation.  Must be an integer ``>= 2``.  Defaults to
        ``60``.
    :type spread_window: int
    :param entry_z: Z-score magnitude threshold to open a position.  Must
        be a positive number greater than *exit_z*.  Defaults to ``2.0``.
    :type entry_z: float
    :param exit_z: Z-score magnitude threshold to close an open position.
        Must be a positive number less than *entry_z*.  Defaults to ``0.5``.
    :type exit_z: float
    """

    display_name = "Pairs Mean Reversion"
    description = (
        "Classical two-leg statistical arbitrage. Trades the normalised spread "
        "between two instruments using z-score entry/exit thresholds."
    )

    def validate_parameters(self) -> None:
        """Validate all strategy parameters.

        Checks that *instrument_a* and *instrument_b* are non-empty strings
        and are different from each other.  Validates that *spread_window* is
        an integer ``>= 2``, and that both *entry_z* and *exit_z* are positive
        numbers with ``exit_z < entry_z``.

        :raises ValueError: If *instrument_a* or *instrument_b* are missing,
            empty, or identical; if *spread_window* is not an integer ``>= 2``;
            if *entry_z* or *exit_z* are not positive numbers; or if
            ``exit_z >= entry_z``.
        """
        a = self.parameters.get("instrument_a")
        b = self.parameters.get("instrument_b")
        if not isinstance(a, str) or not a:
            raise ValueError(f"instrument_a must be a non-empty string, got {a!r}")
        if not isinstance(b, str) or not b:
            raise ValueError(f"instrument_b must be a non-empty string, got {b!r}")
        if a == b:
            raise ValueError("instrument_a and instrument_b must be different")

        window = self.parameters.get("spread_window", _DEFAULTS["spread_window"])
        if not isinstance(window, int) or window < 2:
            raise ValueError(f"spread_window must be an integer >= 2, got {window!r}")

        for key in ("entry_z", "exit_z"):
            val = self.parameters.get(key, _DEFAULTS[key])
            if not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(f"{key} must be a positive number, got {val!r}")

        entry_z = self.parameters.get("entry_z", _DEFAULTS["entry_z"])
        exit_z = self.parameters.get("exit_z", _DEFAULTS["exit_z"])
        if exit_z >= entry_z:
            raise ValueError(f"exit_z ({exit_z}) must be less than entry_z ({entry_z})")

    def get_required_instrument_ids(self) -> list[str]:
        """Return the two instrument identifiers required by this strategy.

        :returns: A two-element list ``[instrument_a, instrument_b]`` as
            configured in ``self.parameters``.
        :rtype: list[str]
        """
        return [
            self.parameters["instrument_a"],
            self.parameters["instrument_b"],
        ]

    def get_required_candle_count(self) -> int:
        """Return the minimum number of historical candles required per instrument.

        The value is ``spread_window + 10`` to allow the rolling statistics
        to be computed with a small safety buffer.

        :returns: Minimum candle count needed per instrument before ``compute()``
            will produce a signal.
        :rtype: int
        """
        return self.parameters.get("spread_window", _DEFAULTS["spread_window"]) + 10

    def compute(
        self,
        candles_map: dict[str, pd.DataFrame],
        as_of: datetime,
    ) -> MultiAssetStrategyOutput | None:
        """Run pairs mean-reversion logic for both instruments.

        Aligns the two closing-price series on their shorter tail, computes
        the rolling z-score of the price spread, and emits atomically paired
        signals when an entry or exit condition is triggered.  Returns
        ``None`` when either DataFrame is missing, when either series has
        fewer than *spread_window* bars, or when the rolling standard
        deviation is zero.

        :param candles_map: Mapping of instrument identifier to its OHLCV
            DataFrame (ordered by timestamp ascending).  Must contain entries
            for both *instrument_a* and *instrument_b*.  Each DataFrame
            excludes the current bar (no look-ahead).
        :type candles_map: dict[str, pandas.DataFrame]
        :param as_of: Point-in-time timestamp of the current bar being
            evaluated.
        :type as_of: datetime.datetime
        :returns: A :class:`~quaver.strategies.base.MultiAssetStrategyOutput`
            with signals for both legs when an entry or exit condition is met;
            ``None`` otherwise.  Possible signal directions per leg:

            * ``CLOSE`` / ``CLOSE`` — when ``|z_score| < exit_z``.
            * ``SELL`` / ``BUY``    — when ``z_score > +entry_z``.
            * ``BUY``  / ``SELL``   — when ``z_score < -entry_z``.
        :rtype: MultiAssetStrategyOutput or None
        """
        id_a: str = self.parameters["instrument_a"]
        id_b: str = self.parameters["instrument_b"]
        window: int = self.parameters.get("spread_window", _DEFAULTS["spread_window"])
        entry_z: float = self.parameters.get("entry_z", _DEFAULTS["entry_z"])
        exit_z: float = self.parameters.get("exit_z", _DEFAULTS["exit_z"])

        df_a = candles_map.get(id_a)
        df_b = candles_map.get(id_b)
        if df_a is None or df_b is None or len(df_a) < window or len(df_b) < window:
            return None

        closes_a = df_a["close"].astype(float).values
        closes_b = df_b["close"].astype(float).values

        # Align on the shorter series tail
        n = min(len(closes_a), len(closes_b))
        spread = closes_a[-n:] - closes_b[-n:]

        if len(spread) < window:
            return None

        roll_mean = float(np.mean(spread[-window:]))
        roll_std = float(np.std(spread[-window:], ddof=1))

        if roll_std == 0:
            return None

        current_spread = spread[-1]
        z_score = (current_spread - roll_mean) / roll_std

        confidence = min(abs(z_score) / entry_z, 1.0)
        meta = {
            "z_score": round(z_score, 4),
            "spread": round(float(current_spread), 6),
            "roll_mean": round(roll_mean, 6),
            "roll_std": round(roll_std, 6),
        }

        # Exit condition (close both legs)
        if abs(z_score) < exit_z:
            signals = {
                id_a: SignalOutput(
                    direction=SignalDirection.CLOSE,
                    confidence=confidence,
                    notes=f"z={z_score:.3f} < exit_z={exit_z}",
                    metadata=meta,
                ),
                id_b: SignalOutput(
                    direction=SignalDirection.CLOSE,
                    confidence=confidence,
                    notes=f"z={z_score:.3f} < exit_z={exit_z}",
                    metadata=meta,
                ),
            }
            return MultiAssetStrategyOutput(signals=signals, metadata=meta)

        # Entry conditions
        if z_score > entry_z:
            # Spread too high: SELL A, BUY B
            signals = {
                id_a: SignalOutput(
                    direction=SignalDirection.SELL,
                    confidence=confidence,
                    notes=f"z={z_score:.3f} > entry_z={entry_z}",
                    metadata=meta,
                ),
                id_b: SignalOutput(
                    direction=SignalDirection.BUY,
                    confidence=confidence,
                    notes=f"z={z_score:.3f} > entry_z={entry_z}",
                    metadata=meta,
                ),
            }
            return MultiAssetStrategyOutput(signals=signals, metadata=meta)

        if z_score < -entry_z:
            # Spread too low: BUY A, SELL B
            signals = {
                id_a: SignalOutput(
                    direction=SignalDirection.BUY,
                    confidence=confidence,
                    notes=f"z={z_score:.3f} < -entry_z={-entry_z}",
                    metadata=meta,
                ),
                id_b: SignalOutput(
                    direction=SignalDirection.SELL,
                    confidence=confidence,
                    notes=f"z={z_score:.3f} < -entry_z={-entry_z}",
                    metadata=meta,
                ),
            }
            return MultiAssetStrategyOutput(signals=signals, metadata=meta)

        return None

    @classmethod
    def get_default_parameters(cls) -> dict[str, Any]:
        """Return a copy of the default parameter dictionary.

        :returns: Mapping of parameter names to their default values:
            ``spread_window=60``, ``entry_z=2.0``, ``exit_z=0.5``.
            Note that *instrument_a* and *instrument_b* have no defaults and
            must be provided explicitly.
        :rtype: dict[str, Any]
        """
        return dict(_DEFAULTS)
