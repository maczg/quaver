"""Mean reversion strategy engine — reference implementation."""

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
    """
    Dual moving-average mean reversion strategy.

    Parameters:
        fast_period (int): Short MA window (default 20).
        slow_period (int): Long MA window (default 50).
        threshold (float): Divergence threshold to trigger a signal (default 0.02 = 2%).

    Signal logic:
        BUY  when fast MA < slow MA by more than threshold (oversold).
        SELL when fast MA > slow MA by more than threshold (overbought).
        Confidence scales with divergence magnitude, capped at 1.0.

    NOTE: SELL signals mean "overbought — expect mean reversion downward".
    Whether the backtest engine opens a short or simply ignores SELL-from-flat
    is controlled by the engine's `allow_shorting` flag (default False).
    """

    display_name = "Mean Reversion"
    description = (
        "Dual moving-average mean reversion. BUY when fast MA is below slow MA "
        "by more than threshold (oversold). SELL when above (overbought)."
    )

    def validate_parameters(self) -> None:
        fast = self.parameters.get("fast_period")
        slow = self.parameters.get("slow_period")
        threshold = self.parameters.get("threshold")

        if not isinstance(fast, int) or fast < 1:
            raise ValueError(f"fast_period must be a positive integer, got {fast!r}")
        if not isinstance(slow, int) or slow < 1:
            raise ValueError(f"slow_period must be a positive integer, got {slow!r}")
        if fast >= slow:
            raise ValueError(
                f"fast_period ({fast}) must be less than slow_period ({slow})"
            )
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError(f"threshold must be a positive number, got {threshold!r}")

    def get_required_candle_count(self) -> int:
        return self.parameters.get("slow_period", _DEFAULTS["slow_period"]) + 10

    def compute(self, candles: pd.DataFrame, as_of: datetime) -> SignalOutput | None:
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
        direction = (
            SignalDirection.BUY if divergence < -threshold else SignalDirection.SELL
        )

        return SignalOutput(
            direction=direction,
            confidence=round(raw_confidence, 4),
            notes=(
                f"fast_ma={fast_ma:.4f} slow_ma={slow_ma:.4f} "
                f"divergence={divergence:.4f}"
            ),
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
        return dict(_DEFAULTS)
