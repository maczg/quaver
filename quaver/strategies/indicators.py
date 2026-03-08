"""Shared pure-numpy indicator library for strategy engines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sma(values: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Simple Moving Average via np.convolve."""
    n = len(values)
    out = np.full(n, np.nan)
    if period < 1 or n < period:
        return out
    kernel = np.ones(period) / period
    conv = np.convolve(values, kernel, mode="valid")
    out[period - 1:] = conv
    return out


def true_range(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
) -> NDArray[np.float64]:
    """True Range: max(H-L, |H-prevC|, |L-prevC|). Index 0 is NaN."""
    n = len(high)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    hl = high[1:] - low[1:]
    hpc = np.abs(high[1:] - close[:-1])
    lpc = np.abs(low[1:] - close[:-1])
    out[1:] = np.maximum(hl, np.maximum(hpc, lpc))
    return out


def wilder_smooth(values: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Wilder's smoothing (RMA).

    Seed value is the MEAN of the first `period` valid values (index 1..period),
    then applies: out[i] = out[i-1] * (period-1)/period + values[i].

    CORRECTNESS NOTE: The seed must use np.mean, NOT np.sum.
    Using np.sum produces a seed that is `period` times too large and corrupts
    all downstream ATR / ADX values.
    """
    n = len(values)
    out = np.full(n, np.nan)
    if period < 1 or n < period + 1:
        return out
    first_valid = 1          # index 0 is NaN for TR/DM series
    end_seed = first_valid + period
    if end_seed > n:
        return out
    # CORRECT: seed with the MEAN, not the sum
    out[end_seed - 1] = np.mean(values[first_valid:end_seed])
    for i in range(end_seed, n):
        if np.isnan(values[i]):
            break
        out[i] = out[i - 1] * (period - 1) / period + values[i]
    return out


def adx(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Wilder's ADX.

    Returns:
        (adx_arr, plus_di, minus_di) — all length n, NaN where insufficient data.
    """
    n = len(high)
    nan_arr = np.full(n, np.nan)
    if n < 2 * period + 1:
        return nan_arr.copy(), nan_arr.copy(), nan_arr.copy()

    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.full(n, np.nan)
    minus_dm = np.full(n, np.nan)
    plus_dm[1:] = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm[1:] = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr = wilder_smooth(tr, period)
    smooth_plus = wilder_smooth(plus_dm, period)
    smooth_minus = wilder_smooth(minus_dm, period)

    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    valid = ~np.isnan(atr) & (atr != 0)
    plus_di[valid] = 100.0 * smooth_plus[valid] / atr[valid]
    minus_di[valid] = 100.0 * smooth_minus[valid] / atr[valid]

    dx = np.full(n, np.nan)
    di_sum = plus_di + minus_di
    di_valid = ~np.isnan(di_sum) & (di_sum != 0)
    dx[di_valid] = (
        100.0
        * np.abs(plus_di[di_valid] - minus_di[di_valid])
        / di_sum[di_valid]
    )

    adx_arr = np.full(n, np.nan)
    dx_valid_indices = np.where(~np.isnan(dx))[0]
    if len(dx_valid_indices) < period:
        return adx_arr, plus_di, minus_di

    first_dx = dx_valid_indices[0]
    adx_start = first_dx + period - 1
    if adx_start >= n:
        return adx_arr, plus_di, minus_di

    adx_arr[adx_start] = np.mean(dx[first_dx: first_dx + period])
    for i in range(adx_start + 1, n):
        if np.isnan(dx[i]):
            break
        adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period

    return adx_arr, plus_di, minus_di


def bollinger_bands(
    close: NDArray[np.float64],
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Bollinger Bands.

    Returns:
        (upper, middle, lower) — all length n.
    """
    n = len(close)
    nan_arr = np.full(n, np.nan)
    if period < 1 or n < period:
        return nan_arr.copy(), nan_arr.copy(), nan_arr.copy()

    middle = sma(close, period)
    std = np.full(n, np.nan)
    for i in range(period - 1, n):
        std[i] = np.std(close[i - period + 1: i + 1], ddof=0)

    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def bollinger_band_width(
    upper: NDArray[np.float64],
    middle: NDArray[np.float64],
    lower: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Bollinger Band Width = (upper - lower) / middle. NaN where middle == 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        bbw = np.where(middle != 0, (upper - lower) / middle, np.nan)
    bbw = np.where(
        np.isnan(upper) | np.isnan(middle) | np.isnan(lower), np.nan, bbw
    )
    return bbw


def rolling_percentile(
    values: NDArray[np.float64],
    window: int,
    percentile: float,
) -> NDArray[np.float64]:
    """Rolling percentile over a fixed window. NaN where insufficient data."""
    n = len(values)
    out = np.full(n, np.nan)
    if window < 1 or n < window:
        return out
    for i in range(window - 1, n):
        win = values[i - window + 1: i + 1]
        valid = win[~np.isnan(win)]
        if len(valid) > 0:
            out[i] = np.percentile(valid, percentile)
    return out


def daily_returns(close: NDArray[np.float64]) -> NDArray[np.float64]:
    """Percentage returns: (close[t] - close[t-1]) / close[t-1]. Index 0 is NaN."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(
            close[:-1] != 0,
            (close[1:] - close[:-1]) / close[:-1],
            np.nan,
        )
    out[1:] = ret
    return out


def volume_relative(
    volume: NDArray[np.float64],
    period: int = 20,
) -> NDArray[np.float64]:
    """Relative volume = volume / SMA(volume, period). NaN where SMA is 0 or NaN."""
    vol_sma = sma(volume, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(
            (~np.isnan(vol_sma)) & (vol_sma != 0),
            volume / vol_sma,
            np.nan,
        )
    return out
