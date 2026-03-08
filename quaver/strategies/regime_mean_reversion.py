"""Regime-based probabilistic mean-reversion strategy engine.

Classifies markets into regimes using ADX + Bollinger Band Width + Volume,
then generates BUY/SELL signals only when expanding-window conditional
probabilities confirm a high likelihood of reversal after a dip/pop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quaver.types import SignalDirection
from quaver.strategies.base import BaseStrategy, SignalOutput
from quaver.strategies.indicators import (
    adx as compute_adx,
    bollinger_band_width,
    bollinger_bands,
    daily_returns,
    rolling_percentile,
    sma,
    volume_relative,
)
from quaver.strategies.registry import StrategyRegistry

log = logging.getLogger(__name__)

# -- Regime labels --
TREND_STRONG_UP = "TREND_STRONG_UP"
TREND_STRONG_DOWN = "TREND_STRONG_DOWN"
TREND_STRONG_UNDEFINED = "TREND_STRONG_UNDEFINED"
TREND_WEAK_UP = "TREND_WEAK_UP"
TREND_WEAK_DOWN = "TREND_WEAK_DOWN"
TREND_WEAK_UNDEFINED = "TREND_WEAK_UNDEFINED"
TRANSITION_CONFIRMED = "TRANSITION_CONFIRMED"
TRANSITION_WEAK = "TRANSITION_WEAK"
COMPRESSION = "COMPRESSION"
RANGE = "RANGE"

_DEFAULTS: dict[str, Any] = {
    "adx_period": 14,
    "bb_period": 20,
    "bb_std": 2.0,
    "bbw_percentile_window": 250,
    "bbw_sma_period": 5,
    "bbw_lookback": 3,
    "sma_fast": 20,
    "sma_slow": 50,
    "volume_sma_period": 20,
    "adx_trend_threshold": 21.0,
    "adx_transition_low": 20.0,
    "volume_strong_threshold": 1.2,
    "volume_normal_threshold": 1.0,
    "return_threshold": 0.02,
    "success_threshold": 0.005,
    "prob_threshold_base": 0.50,
    "prob_threshold_weak": 0.50,
    "prob_threshold_strong": 0.50,
    "winloss_threshold_weak": 1.3,
    "winloss_threshold_strong": 1.3,
    "safemargin": 0.0,
    "min_events": 12,
    "candle_count": 500,
}


@dataclass
class ProbabilityResult:
    """Expanding-window probability result for a direction (long or short)."""

    prob_base: float
    prob_regime: float
    winloss_base: float
    winloss_regime: float
    events_base: int
    events_regime: int


@StrategyRegistry.register("regime_mean_reversion")
class RegimeMeanReversionStrategy(BaseStrategy):
    """Regime-based probabilistic mean-reversion strategy.

    Classifies markets into regimes, computes expanding-window conditional
    probabilities, and generates BUY/SELL signals only when probability
    thresholds are met.
    """

    display_name = "Regime Mean Reversion"
    description = (
        "Regime-based probabilistic mean-reversion strategy. Classifies markets into "
        "regimes using ADX, Bollinger Band Width, and volume, then generates signals "
        "only when expanding-window conditional probabilities confirm reversal likelihood."
    )

    # -- Helpers --

    def _p(self, key: str) -> Any:
        """Get parameter with default fallback."""
        return self.parameters.get(key, _DEFAULTS[key])

    # -- BaseStrategy interface --

    def validate_parameters(self) -> None:
        p = self.parameters

        int_keys = [
            "adx_period", "bb_period", "bbw_percentile_window",
            "bbw_sma_period", "bbw_lookback", "sma_fast", "sma_slow",
            "volume_sma_period", "min_events", "candle_count",
        ]
        for key in int_keys:
            val = p.get(key, _DEFAULTS[key])
            if not isinstance(val, int) or val < 1:
                raise ValueError(f"{key} must be a positive integer, got {val!r}")

        float_keys = [
            "bb_std", "adx_trend_threshold", "adx_transition_low",
            "volume_strong_threshold", "volume_normal_threshold",
            "return_threshold", "success_threshold",
            "prob_threshold_base", "prob_threshold_weak", "prob_threshold_strong",
            "winloss_threshold_weak", "winloss_threshold_strong",
        ]
        for key in float_keys:
            val = p.get(key, _DEFAULTS[key])
            if not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(f"{key} must be a positive number, got {val!r}")

        safemargin = p.get("safemargin", _DEFAULTS["safemargin"])
        if not isinstance(safemargin, (int, float)) or safemargin < 0:
            raise ValueError(f"safemargin must be a non-negative number, got {safemargin!r}")

        sma_fast = p.get("sma_fast", _DEFAULTS["sma_fast"])
        sma_slow = p.get("sma_slow", _DEFAULTS["sma_slow"])
        if sma_fast >= sma_slow:
            raise ValueError(
                f"sma_fast ({sma_fast}) must be less than sma_slow ({sma_slow})"
            )

    def get_required_candle_count(self) -> int:
        return self._p("candle_count")

    def compute(
            self,
            candles: pd.DataFrame,
            as_of: datetime,
    ) -> SignalOutput | None:
        n = len(candles)

        # Minimum data guard
        min_needed = max(
            self._p("bbw_percentile_window") + self._p("bb_period"),
            self._p("sma_slow") + 1,
            2 * self._p("adx_period") + 1,
        )
        if n < min_needed:
            return None

        # -- Extract OHLCV arrays --
        close = candles["close"].astype(float).values
        high = candles["high"].astype(float).values
        low = candles["low"].astype(float).values
        vol = candles["volume"].astype(float).values

        # -- Compute indicators --
        adx_arr, plus_di, minus_di = compute_adx(high, low, close, self._p("adx_period"))
        upper, middle, lower = bollinger_bands(close, self._p("bb_period"), self._p("bb_std"))
        bbw = bollinger_band_width(upper, middle, lower)
        bbw_sma = sma(bbw, self._p("bbw_sma_period"))
        bbw_pct = rolling_percentile(bbw, self._p("bbw_percentile_window"), 20)
        sma_f = sma(close, self._p("sma_fast"))
        sma_s = sma(close, self._p("sma_slow"))
        vol_rel = volume_relative(vol, self._p("volume_sma_period"))
        returns = daily_returns(close)

        # -- BBW classification arrays --
        bbw_expanding = self._compute_bbw_expanding(bbw, bbw_sma)
        bbw_low = self._compute_bbw_low(bbw, bbw_pct)

        # -- Classify all regimes --
        regimes = self._classify_all_regimes(
            adx_arr, bbw_expanding, bbw_low, vol_rel, close, sma_f, sma_s,
        )

        # -- Latest bar index --
        t = n - 1
        current_regime = regimes[t]
        curr_return = returns[t]

        if np.isnan(curr_return) or current_regime is None:
            return None

        # Only generate signals for trending regimes
        if current_regime not in (
                TREND_STRONG_UP, TREND_STRONG_DOWN,
                TREND_WEAK_UP, TREND_WEAK_DOWN,
        ):
            return None

        # -- Compute expanding-window probabilities --
        is_long = current_regime in (TREND_WEAK_UP, TREND_STRONG_UP)
        probs = self._compute_probabilities(returns, regimes, current_regime, t)
        if probs is None:
            return None

        # -- Signal generation --
        return self._generate_signal(
            regime=current_regime,
            curr_return=curr_return,
            probs=probs,
            is_long=is_long,
            adx_val=adx_arr[t] if not np.isnan(adx_arr[t]) else None,
            bbw_val=bbw[t] if not np.isnan(bbw[t]) else None,
            vol_rel_val=vol_rel[t] if not np.isnan(vol_rel[t]) else None,
        )

    # -- Private helpers --

    def _compute_bbw_expanding(
            self,
            bbw: NDArray[np.float64],
            bbw_sma: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """BBW_EXPANDING = BBW > SMA(BBW, bbw_sma_period) AND BBW > BBW[t-bbw_lookback]."""
        n = len(bbw)
        lookback = self._p("bbw_lookback")
        result = np.zeros(n, dtype=bool)
        for i in range(lookback, n):
            if np.isnan(bbw[i]) or np.isnan(bbw_sma[i]) or np.isnan(bbw[i - lookback]):
                continue
            result[i] = (bbw[i] > bbw_sma[i]) and (bbw[i] > bbw[i - lookback])
        return result

    def _compute_bbw_low(
            self,
            bbw: NDArray[np.float64],
            bbw_pct: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """BBW_LOW = BBW <= rolling_percentile(BBW, 250, P20)."""
        n = len(bbw)
        result = np.zeros(n, dtype=bool)
        for i in range(n):
            if np.isnan(bbw[i]) or np.isnan(bbw_pct[i]):
                continue
            result[i] = bbw[i] <= bbw_pct[i]
        return result

    def _classify_regime(
            self,
            adx_val: float,
            bbw_expanding: bool,
            bbw_low: bool,
            vol_rel_val: float,
            close_val: float,
            sma_f_val: float,
            sma_s_val: float,
    ) -> str | None:
        """Classify a single bar into one of 10 regimes."""
        if np.isnan(adx_val) or np.isnan(vol_rel_val):
            return None

        adx_trend = self._p("adx_trend_threshold")
        adx_trans_low = self._p("adx_transition_low")
        vol_strong = self._p("volume_strong_threshold")
        vol_normal = self._p("volume_normal_threshold")

        if adx_val >= adx_trend:
            if bbw_expanding and vol_rel_val >= vol_strong:
                base = "TREND_STRONG"
            else:
                base = "TREND_WEAK"
            # Direction
            if (
                    not np.isnan(sma_f_val)
                    and not np.isnan(sma_s_val)
                    and close_val > sma_f_val > sma_s_val
            ):
                return f"{base}_UP"
            elif (
                    not np.isnan(sma_f_val)
                    and not np.isnan(sma_s_val)
                    and close_val < sma_f_val < sma_s_val
            ):
                return f"{base}_DOWN"
            else:
                return f"{base}_UNDEFINED"

        elif adx_val >= adx_trans_low:
            if bbw_expanding and vol_rel_val >= vol_normal:
                return TRANSITION_CONFIRMED
            else:
                return TRANSITION_WEAK

        else:
            if bbw_low and vol_rel_val >= vol_normal:
                return COMPRESSION
            else:
                return RANGE

    def _classify_all_regimes(
            self,
            adx_arr: NDArray[np.float64],
            bbw_expanding: NDArray[np.bool_],
            bbw_low: NDArray[np.bool_],
            vol_rel: NDArray[np.float64],
            close: NDArray[np.float64],
            sma_f: NDArray[np.float64],
            sma_s: NDArray[np.float64],
    ) -> list[str | None]:
        """Classify regime for every bar."""
        n = len(adx_arr)
        regimes: list[str | None] = [None] * n
        for i in range(n):
            regimes[i] = self._classify_regime(
                adx_arr[i],
                bool(bbw_expanding[i]),
                bool(bbw_low[i]),
                vol_rel[i],
                close[i],
                sma_f[i],
                sma_s[i],
            )
        return regimes

    def _compute_probabilities(
            self,
            returns: NDArray[np.float64],
            regimes: list[str | None],
            current_regime: str,
            t: int,
    ) -> ProbabilityResult | None:
        """Expanding-window conditional probability computation."""
        return_threshold = self._p("return_threshold")
        success_threshold = self._p("success_threshold")
        min_events = self._p("min_events")

        is_long = current_regime in (TREND_WEAK_UP, TREND_STRONG_UP)

        # Accumulators
        base_events = 0
        base_successes = 0
        base_win_sum = 0.0
        base_loss_sum = 0.0
        base_wins = 0
        base_losses = 0

        regime_events = 0
        regime_successes = 0
        regime_win_sum = 0.0
        regime_loss_sum = 0.0
        regime_wins = 0
        regime_losses = 0

        for i in range(t):
            if np.isnan(returns[i]) or i + 1 >= len(returns) or np.isnan(returns[i + 1]):
                continue

            outcome = returns[i + 1]

            if is_long:
                condition = returns[i] <= -return_threshold
                success = outcome >= success_threshold
                win_val = outcome if outcome >= 0 else 0.0
                loss_val = outcome if outcome < 0 else 0.0
            else:
                condition = returns[i] >= return_threshold
                success = -outcome >= success_threshold
                win_val = -outcome if -outcome >= 0 else 0.0
                loss_val = -outcome if -outcome < 0 else 0.0

            if not condition:
                continue

            # Base stats
            base_events += 1
            if success:
                base_successes += 1
            if win_val > 0:
                base_win_sum += win_val
                base_wins += 1
            elif loss_val < 0:
                base_loss_sum += loss_val
                base_losses += 1

            # Regime-specific stats
            if regimes[i] == current_regime:
                regime_events += 1
                if success:
                    regime_successes += 1
                if win_val > 0:
                    regime_win_sum += win_val
                    regime_wins += 1
                elif loss_val < 0:
                    regime_loss_sum += loss_val
                    regime_losses += 1

        if base_events < min_events or regime_events < min_events:
            return None

        prob_base = base_successes / base_events
        prob_regime = regime_successes / regime_events

        avg_win_base = base_win_sum / base_wins if base_wins > 0 else 0.0
        avg_loss_base = abs(base_loss_sum / base_losses) if base_losses > 0 else 0.0
        winloss_base = avg_win_base / avg_loss_base if avg_loss_base > 0 else float("inf")

        avg_win_regime = regime_win_sum / regime_wins if regime_wins > 0 else 0.0
        avg_loss_regime = abs(regime_loss_sum / regime_losses) if regime_losses > 0 else 0.0
        winloss_regime = avg_win_regime / avg_loss_regime if avg_loss_regime > 0 else float("inf")

        return ProbabilityResult(
            prob_base=prob_base,
            prob_regime=prob_regime,
            winloss_base=winloss_base,
            winloss_regime=winloss_regime,
            events_base=base_events,
            events_regime=regime_events,
        )

    def _generate_signal(
            self,
            regime: str,
            curr_return: float,
            probs: ProbabilityResult,
            is_long: bool,
            adx_val: float | None,
            bbw_val: float | None,
            vol_rel_val: float | None,
    ) -> SignalOutput | None:
        """Apply threshold checks and produce BUY/SELL signal."""
        return_threshold = self._p("return_threshold")
        safemargin = self._p("safemargin")

        # Determine which thresholds to use
        is_strong = "STRONG" in regime
        prob_threshold_regime = (
            self._p("prob_threshold_strong") if is_strong
            else self._p("prob_threshold_weak")
        )
        winloss_threshold = (
            self._p("winloss_threshold_strong") if is_strong
            else self._p("winloss_threshold_weak")
        )

        # Check 1: current return triggers dip/pop
        if is_long:
            if curr_return > -return_threshold * (1 + safemargin):
                return None
        else:
            if curr_return < return_threshold * (1 + safemargin):
                return None

        # Check 2: base probability
        if probs.prob_base <= self._p("prob_threshold_base") * (1 + safemargin):
            return None

        # Check 3: regime probability
        if probs.prob_regime <= prob_threshold_regime * (1 + safemargin):
            return None

        # Check 4: win/loss ratio
        if probs.winloss_regime <= winloss_threshold * (1 + safemargin):
            return None

        direction = SignalDirection.BUY if is_long else SignalDirection.SELL
        confidence = round(min(probs.prob_regime, 1.0), 4)

        return SignalOutput(
            direction=direction,
            confidence=confidence,
            notes=(
                f"regime={regime} prob_base={probs.prob_base:.3f} "
                f"prob_regime={probs.prob_regime:.3f} "
                f"winloss_regime={probs.winloss_regime:.3f}"
            ),
            metadata={
                "regime": regime,
                "curr_return": round(curr_return, 6),
                "prob_base": round(probs.prob_base, 4),
                "prob_regime": round(probs.prob_regime, 4),
                "winloss_base": round(probs.winloss_base, 4),
                "winloss_regime": round(probs.winloss_regime, 4),
                "events_base": probs.events_base,
                "events_regime": probs.events_regime,
                "adx": round(adx_val, 4) if adx_val is not None else None,
                "bbw": round(bbw_val, 4) if bbw_val is not None else None,
                "volume_relative": round(vol_rel_val, 4) if vol_rel_val is not None else None,
            },
        )

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "adx_period": {"type": "integer", "minimum": 1, "default": 14},
                "bb_period": {"type": "integer", "minimum": 1, "default": 20},
                "bb_std": {"type": "number", "exclusiveMinimum": 0, "default": 2.0},
                "bbw_percentile_window": {"type": "integer", "minimum": 1, "default": 250},
                "bbw_sma_period": {"type": "integer", "minimum": 1, "default": 5},
                "bbw_lookback": {"type": "integer", "minimum": 1, "default": 3},
                "sma_fast": {"type": "integer", "minimum": 1, "default": 20},
                "sma_slow": {"type": "integer", "minimum": 2, "default": 50},
                "volume_sma_period": {"type": "integer", "minimum": 1, "default": 20},
                "adx_trend_threshold": {"type": "number", "exclusiveMinimum": 0, "default": 21.0},
                "adx_transition_low": {"type": "number", "exclusiveMinimum": 0, "default": 20.0},
                "volume_strong_threshold": {"type": "number", "exclusiveMinimum": 0, "default": 1.2},
                "volume_normal_threshold": {"type": "number", "exclusiveMinimum": 0, "default": 1.0},
                "return_threshold": {"type": "number", "exclusiveMinimum": 0, "default": 0.02},
                "success_threshold": {"type": "number", "exclusiveMinimum": 0, "default": 0.005},
                "prob_threshold_base": {"type": "number", "exclusiveMinimum": 0, "default": 0.50},
                "prob_threshold_weak": {"type": "number", "exclusiveMinimum": 0, "default": 0.50},
                "prob_threshold_strong": {"type": "number", "exclusiveMinimum": 0, "default": 0.50},
                "winloss_threshold_weak": {"type": "number", "exclusiveMinimum": 0, "default": 1.3},
                "winloss_threshold_strong": {"type": "number", "exclusiveMinimum": 0, "default": 1.3},
                "safemargin": {"type": "number", "minimum": 0, "default": 0.0},
                "min_events": {"type": "integer", "minimum": 1, "default": 12},
                "candle_count": {"type": "integer", "minimum": 1, "default": 500},
            },
        }

    @classmethod
    def get_default_parameters(cls) -> dict[str, Any]:
        return dict(_DEFAULTS)
