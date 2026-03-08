"""Pairs mean-reversion strategy (statistical arbitrage example)."""

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
    """
    Classic two-leg spread (pairs) mean-reversion strategy.

    Parameters:
        instrument_a (str): First leg identifier (must match a key in candles_map).
        instrument_b (str): Second leg identifier (must match a key in candles_map).
        spread_window (int): Rolling window for z-score mean/std (default 60).
        entry_z (float): Open when |z_score| > entry_z (default 2.0).
        exit_z (float): Close when |z_score| < exit_z (default 0.5).

    Spread definition:
        spread[t] = close_A[t] - close_B[t]
        z_score[t] = (spread[t] - rolling_mean(spread, window)) / rolling_std(spread, window)

    Signal logic:
        z_score > +entry_z  ->  SELL A, BUY  B  (spread expected to fall)
        z_score < -entry_z  ->  BUY  A, SELL B  (spread expected to rise)
        |z_score| < exit_z  ->  CLOSE A, CLOSE B (convergence achieved)

    IMPORTANT: Both legs' signals are always emitted together in one
    MultiAssetStrategyOutput so the engine applies them atomically.
    """

    display_name = "Pairs Mean Reversion"
    description = (
        "Classical two-leg statistical arbitrage. Trades the normalised spread "
        "between two instruments using z-score entry/exit thresholds."
    )

    def validate_parameters(self) -> None:
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
            raise ValueError(
                f"exit_z ({exit_z}) must be less than entry_z ({entry_z})"
            )

    def get_required_instrument_ids(self) -> list[str]:
        return [
            self.parameters["instrument_a"],
            self.parameters["instrument_b"],
        ]

    def get_required_candle_count(self) -> int:
        return self.parameters.get("spread_window", _DEFAULTS["spread_window"]) + 10

    def compute(
        self,
        candles_map: dict[str, pd.DataFrame],
        as_of: datetime,
    ) -> MultiAssetStrategyOutput | None:
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
        return dict(_DEFAULTS)
