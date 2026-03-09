"""Shared pure-numpy indicator library for strategy engines.

This module provides a collection of technical analysis indicator functions
implemented exclusively with NumPy. It is intended to be imported by any
strategy that requires price-based signal computation. All functions operate on
:class:`numpy.ndarray` objects and return arrays of the same length as their
primary input, padding leading positions with ``NaN`` wherever insufficient
data exists to produce a meaningful value.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sma(values: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Compute the Simple Moving Average using convolution.

    Calculates a rolling arithmetic mean of *period* consecutive elements via
    :func:`numpy.convolve`. Positions ``0`` through ``period - 2`` are filled
    with ``NaN`` because fewer than *period* samples are available.

    :param values: One-dimensional array of input values.
    :type values: NDArray[np.float64]
    :param period: Number of elements in the rolling window. Must be >= 1.
    :type period: int
    :returns: Array of the same length as *values* containing the SMA values,
        with ``NaN`` for the first ``period - 1`` positions.
    :rtype: NDArray[np.float64]
    """
    n = len(values)
    out = np.full(n, np.nan)
    if period < 1 or n < period:
        return out
    kernel = np.ones(period) / period
    conv = np.convolve(values, kernel, mode="valid")
    out[period - 1 :] = conv
    return out


def true_range(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the True Range for each bar.

    The True Range at index *i* is defined as::

        TR[i] = max(H[i] - L[i], |H[i] - C[i-1]|, |L[i] - C[i-1]|)

    Index 0 is always ``NaN`` because no previous close exists.

    :param high: One-dimensional array of bar high prices.
    :type high: NDArray[np.float64]
    :param low: One-dimensional array of bar low prices.
    :type low: NDArray[np.float64]
    :param close: One-dimensional array of bar closing prices.
    :type close: NDArray[np.float64]
    :returns: Array of the same length as *high* containing True Range values,
        with ``NaN`` at index 0.
    :rtype: NDArray[np.float64]
    """
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
    """Apply Wilder's smoothing (RMA) to an array.

    The seed value is the arithmetic mean of the first *period* valid elements
    (indices ``1`` through ``period``, because index 0 is ``NaN`` for True
    Range and Directional Movement series). Subsequent values are computed as::

        out[i] = out[i-1] * (period - 1) / period + values[i]

    .. note::

        CORRECTNESS NOTE: The seed **must** use :func:`numpy.mean`, **not**
        :func:`numpy.sum`. Using ``np.sum`` produces a seed that is *period*
        times too large and corrupts all downstream ATR and ADX values.

    :param values: One-dimensional array to smooth. Index 0 is expected to be
        ``NaN`` (consistent with True Range and Directional Movement outputs).
    :type values: NDArray[np.float64]
    :param period: Smoothing period (Wilder period). Must be >= 1 and the array
        must contain at least ``period + 1`` elements.
    :type period: int
    :returns: Array of the same length as *values* containing the smoothed
        values, with ``NaN`` for all positions before the first valid seed.
    :rtype: NDArray[np.float64]
    """
    n = len(values)
    out = np.full(n, np.nan)
    if period < 1 or n < period + 1:
        return out
    first_valid = 1  # index 0 is NaN for TR/DM series
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
    """Compute Wilder's Average Directional Index (ADX) and Directional Indicators.

    Implements the full ADX calculation pipeline:

    1. Computes raw Directional Movement (+DM and -DM).
    2. Smooths True Range and Directional Movement with :func:`wilder_smooth`.
    3. Derives +DI and -DI as percentages of smoothed ATR.
    4. Computes DX from the divergence of +DI and -DI.
    5. Smooths DX with a second Wilder pass to produce the ADX line.

    At least ``2 * period + 1`` bars are required; a tuple of three all-``NaN``
    arrays is returned when this threshold is not met.

    :param high: One-dimensional array of bar high prices.
    :type high: NDArray[np.float64]
    :param low: One-dimensional array of bar low prices.
    :type low: NDArray[np.float64]
    :param close: One-dimensional array of bar closing prices.
    :type close: NDArray[np.float64]
    :param period: Wilder smoothing period. Defaults to ``14``.
    :type period: int
    :returns: A three-element tuple ``(adx_arr, plus_di, minus_di)`` where

        - ``adx_arr`` -- Wilder-smoothed ADX values (0-100 scale).
        - ``plus_di`` -- Positive Directional Indicator (+DI, 0-100 scale).
        - ``minus_di`` -- Negative Directional Indicator (-DI, 0-100 scale).

        All three arrays have the same length as *high* and contain ``NaN``
        wherever insufficient data is available.
    :rtype: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
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
    dx[di_valid] = 100.0 * np.abs(plus_di[di_valid] - minus_di[di_valid]) / di_sum[di_valid]

    adx_arr = np.full(n, np.nan)
    dx_valid_indices = np.where(~np.isnan(dx))[0]
    if len(dx_valid_indices) < period:
        return adx_arr, plus_di, minus_di

    first_dx = dx_valid_indices[0]
    adx_start = first_dx + period - 1
    if adx_start >= n:
        return adx_arr, plus_di, minus_di

    adx_arr[adx_start] = np.mean(dx[first_dx : first_dx + period])
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
    """Compute Bollinger Bands around a Simple Moving Average.

    The middle band is the SMA of *close* over *period* bars. The upper and
    lower bands are offset by *num_std* population standard deviations
    (``ddof=0``) of the same rolling window::

        upper  = SMA + num_std * std
        middle = SMA
        lower  = SMA - num_std * std

    :param close: One-dimensional array of closing prices.
    :type close: NDArray[np.float64]
    :param period: Rolling window length for the SMA and standard deviation.
        Must be >= 1. Defaults to ``20``.
    :type period: int
    :param num_std: Number of standard deviations for the band width.
        Defaults to ``2.0``.
    :type num_std: float
    :returns: A three-element tuple ``(upper, middle, lower)`` where

        - ``upper`` -- Upper Bollinger Band array.
        - ``middle`` -- Middle band (SMA) array.
        - ``lower`` -- Lower Bollinger Band array.

        All three arrays have the same length as *close* and contain ``NaN``
        for the first ``period - 1`` positions.
    :rtype: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    """
    n = len(close)
    nan_arr = np.full(n, np.nan)
    if period < 1 or n < period:
        return nan_arr.copy(), nan_arr.copy(), nan_arr.copy()

    middle = sma(close, period)
    std = np.full(n, np.nan)
    for i in range(period - 1, n):
        std[i] = np.std(close[i - period + 1 : i + 1], ddof=0)

    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def bollinger_band_width(
    upper: NDArray[np.float64],
    middle: NDArray[np.float64],
    lower: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the Bollinger Band Width (BBW) normalised by the middle band.

    Band width is calculated as::

        BBW = (upper - lower) / middle

    The result is ``NaN`` wherever *middle* equals zero or any of the three
    input arrays carry a ``NaN`` value at that position.

    :param upper: One-dimensional array of upper Bollinger Band values.
    :type upper: NDArray[np.float64]
    :param middle: One-dimensional array of middle band (SMA) values.
    :type middle: NDArray[np.float64]
    :param lower: One-dimensional array of lower Bollinger Band values.
    :type lower: NDArray[np.float64]
    :returns: Array of BBW values with the same length as *upper*, containing
        ``NaN`` where *middle* is zero or any input is ``NaN``.
    :rtype: NDArray[np.float64]
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        bbw = np.where(middle != 0, (upper - lower) / middle, np.nan)
    bbw = np.where(np.isnan(upper) | np.isnan(middle) | np.isnan(lower), np.nan, bbw)
    return bbw


def rolling_percentile(
    values: NDArray[np.float64],
    window: int,
    percentile: float,
) -> NDArray[np.float64]:
    """Compute a rolling percentile over a fixed-size sliding window.

    For each position *i* >= ``window - 1``, the percentile is computed over
    ``values[i - window + 1 : i + 1]``, excluding any ``NaN`` elements within
    the window. Positions before ``window - 1`` are filled with ``NaN``.

    :param values: One-dimensional array of input values.
    :type values: NDArray[np.float64]
    :param window: Number of elements in the rolling window. Must be >= 1 and
        <= ``len(values)``.
    :type window: int
    :param percentile: Desired percentile in the range ``[0, 100]``.
    :type percentile: float
    :returns: Array of the same length as *values* containing the rolling
        percentile, with ``NaN`` for the first ``window - 1`` positions or
        whenever the window contains no valid (non-``NaN``) values.
    :rtype: NDArray[np.float64]
    """
    n = len(values)
    out = np.full(n, np.nan)
    if window < 1 or n < window:
        return out
    for i in range(window - 1, n):
        win = values[i - window + 1 : i + 1]
        valid = win[~np.isnan(win)]
        if len(valid) > 0:
            out[i] = np.percentile(valid, percentile)
    return out


def daily_returns(close: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute bar-over-bar percentage returns.

    Returns are defined as::

        ret[t] = (close[t] - close[t-1]) / close[t-1]

    Index 0 is always ``NaN`` because no prior close is available. Positions
    where the previous close is zero are also set to ``NaN`` to avoid
    division-by-zero artefacts.

    :param close: One-dimensional array of closing prices.
    :type close: NDArray[np.float64]
    :returns: Array of the same length as *close* containing percentage
        returns, with ``NaN`` at index 0 and wherever the previous close
        equals zero.
    :rtype: NDArray[np.float64]
    """
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
    """Compute the Relative Volume ratio against a simple moving average.

    Relative Volume is defined as::

        RVOL = volume / SMA(volume, period)

    The result is ``NaN`` wherever the SMA is zero or ``NaN`` (i.e. for the
    first ``period - 1`` positions).

    :param volume: One-dimensional array of bar volume values.
    :type volume: NDArray[np.float64]
    :param period: Rolling window length for the volume SMA. Defaults to ``20``.
    :type period: int
    :returns: Array of the same length as *volume* containing the relative
        volume ratio, with ``NaN`` for the first ``period - 1`` positions or
        wherever the SMA is zero.
    :rtype: NDArray[np.float64]
    """
    vol_sma = sma(volume, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(
            (~np.isnan(vol_sma)) & (vol_sma != 0),
            volume / vol_sma,
            np.nan,
        )
    return out


def atr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14,
) -> NDArray[np.float64]:
    """Compute the Average True Range using a simple moving average.

    Calculates the True Range for each bar and then smooths it with a
    :func:`sma` of length *period*.  The first valid ATR value appears at
    index ``period`` (because True Range index 0 is ``NaN``).

    :param high: One-dimensional array of bar high prices.
    :type high: NDArray[np.float64]
    :param low: One-dimensional array of bar low prices.
    :type low: NDArray[np.float64]
    :param close: One-dimensional array of bar closing prices.
    :type close: NDArray[np.float64]
    :param period: Rolling window length for the SMA smoothing.
        Must be >= 1. Defaults to ``14``.
    :type period: int
    :returns: Array of the same length as *high* containing ATR values,
        with ``NaN`` for leading positions where insufficient data exists.
    :rtype: NDArray[np.float64]
    """
    n = len(high)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    tr = true_range(high, low, close)
    # Compute SMA over TR starting from index 1 (index 0 is NaN)
    for i in range(period, n):
        out[i] = np.mean(tr[i - period + 1 : i + 1])
    return out


def rsi(
    close: NDArray[np.float64],
    period: int = 14,
) -> NDArray[np.float64]:
    """Compute the Relative Strength Index (SMA-based).

    Uses simple moving averages of gains and losses over *period* bars::

        RSI = 100 - 100 / (1 + avg_gain / avg_loss)

    The first valid RSI appears at index *period*.  Returns ``100.0``
    when average loss is zero (all gains).

    :param close: One-dimensional array of closing prices.
    :type close: NDArray[np.float64]
    :param period: Lookback period for gain/loss averages.
        Must be >= 1. Defaults to ``14``.
    :type period: int
    :returns: Array of the same length as *close* containing RSI values
        in ``[0, 100]``, with ``NaN`` for the first *period* positions.
    :rtype: NDArray[np.float64]
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = close[1:] - close[:-1]  # length n-1
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    for i in range(period, n):
        avg_g = float(np.mean(gain[i - period : i]))
        avg_l = float(np.mean(loss[i - period : i]))
        if avg_l == 0.0:
            out[i] = 100.0
        else:
            rs = avg_g / avg_l
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def rolling_max(
    values: NDArray[np.float64],
    period: int,
) -> NDArray[np.float64]:
    """Compute the rolling maximum over a fixed-size sliding window.

    For each position *i* >= ``period - 1``, the result is the maximum of
    ``values[i - period + 1 : i + 1]``.  Leading positions are ``NaN``.

    :param values: One-dimensional array of input values.
    :type values: NDArray[np.float64]
    :param period: Number of elements in the rolling window. Must be >= 1.
    :type period: int
    :returns: Array of the same length as *values* containing rolling
        maximum values, with ``NaN`` for the first ``period - 1`` positions.
    :rtype: NDArray[np.float64]
    """
    n = len(values)
    out = np.full(n, np.nan)
    if period < 1 or n < period:
        return out
    for i in range(period - 1, n):
        out[i] = np.nanmax(values[i - period + 1 : i + 1])
    return out


def rolling_min(
    values: NDArray[np.float64],
    period: int,
) -> NDArray[np.float64]:
    """Compute the rolling minimum over a fixed-size sliding window.

    For each position *i* >= ``period - 1``, the result is the minimum of
    ``values[i - period + 1 : i + 1]``.  Leading positions are ``NaN``.

    :param values: One-dimensional array of input values.
    :type values: NDArray[np.float64]
    :param period: Number of elements in the rolling window. Must be >= 1.
    :type period: int
    :returns: Array of the same length as *values* containing rolling
        minimum values, with ``NaN`` for the first ``period - 1`` positions.
    :rtype: NDArray[np.float64]
    """
    n = len(values)
    out = np.full(n, np.nan)
    if period < 1 or n < period:
        return out
    for i in range(period - 1, n):
        out[i] = np.nanmin(values[i - period + 1 : i + 1])
    return out


# ── New indicators ───────────────────────────────────────────────────────────


def ema(values: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Compute the Exponential Moving Average.

    The seed value is the SMA of the first *period* elements. Subsequent
    values use the standard EMA recursion::

        EMA[i] = close[i] * k + EMA[i-1] * (1 - k)

    where ``k = 2 / (period + 1)``.

    :param values: One-dimensional array of input values.
    :type values: NDArray[np.float64]
    :param period: EMA lookback period. Must be >= 1.
    :type period: int
    :returns: Array of the same length as *values* with ``NaN`` for the
        first ``period - 2`` positions (first valid value at index
        ``period - 1``).
    :rtype: NDArray[np.float64]
    """
    n = len(values)
    out = np.full(n, np.nan)
    if period < 1 or n < period:
        return out
    k = 2.0 / (period + 1)
    out[period - 1] = float(np.mean(values[:period]))
    for i in range(period, n):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def macd(
    close: NDArray[np.float64],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Moving Average Convergence Divergence.

    The MACD line is the difference between the fast and slow EMAs. The
    signal line is an EMA of the MACD line. The histogram is their
    difference::

        macd_line  = EMA(close, fast) - EMA(close, slow)
        signal_line = EMA(macd_line, signal)
        histogram   = macd_line - signal_line

    :param close: One-dimensional array of closing prices.
    :type close: NDArray[np.float64]
    :param fast: Fast EMA period. Defaults to ``12``.
    :type fast: int
    :param slow: Slow EMA period. Defaults to ``26``.
    :type slow: int
    :param signal: Signal EMA period. Defaults to ``9``.
    :type signal: int
    :returns: ``(macd_line, signal_line, histogram)`` tuple. All arrays
        have the same length as *close*.
    :rtype: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    """
    n = len(close)
    nan_arr = np.full(n, np.nan)
    if n < slow:
        return nan_arr.copy(), nan_arr.copy(), nan_arr.copy()

    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)

    macd_line = np.full(n, np.nan)
    valid = ~np.isnan(ema_fast) & ~np.isnan(ema_slow)
    macd_line[valid] = ema_fast[valid] - ema_slow[valid]

    # Build a contiguous array from the first valid MACD value for signal EMA
    first_valid = int(np.argmax(valid))
    macd_valid_segment = macd_line[first_valid:]
    signal_segment = ema(macd_valid_segment, signal)

    signal_line = np.full(n, np.nan)
    signal_line[first_valid:] = signal_segment

    histogram = np.full(n, np.nan)
    both_valid = ~np.isnan(macd_line) & ~np.isnan(signal_line)
    histogram[both_valid] = macd_line[both_valid] - signal_line[both_valid]

    return macd_line, signal_line, histogram


def stochastic(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period_k: int = 14,
    period_d: int = 3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the Stochastic Oscillator (%K and %D).

    %K measures where the close sits relative to the high-low range over
    *period_k* bars::

        %K = (close - lowest_low) / (highest_high - lowest_low) * 100

    %D is the SMA of %K over *period_d* bars.

    :param high: One-dimensional array of bar high prices.
    :type high: NDArray[np.float64]
    :param low: One-dimensional array of bar low prices.
    :type low: NDArray[np.float64]
    :param close: One-dimensional array of bar closing prices.
    :type close: NDArray[np.float64]
    :param period_k: Lookback period for %K. Defaults to ``14``.
    :type period_k: int
    :param period_d: SMA period for %D smoothing. Defaults to ``3``.
    :type period_d: int
    :returns: ``(%K, %D)`` tuple. Both arrays have the same length as
        *close*.
    :rtype: tuple[NDArray[np.float64], NDArray[np.float64]]
    """
    n = len(close)
    nan2 = np.full(n, np.nan)
    if period_k < 1 or n < period_k:
        return nan2.copy(), nan2.copy()

    highest = rolling_max(high, period_k)
    lowest = rolling_min(low, period_k)

    pct_k = np.full(n, np.nan)
    denom = highest - lowest
    valid = ~np.isnan(denom) & (denom != 0)
    pct_k[valid] = (close[valid] - lowest[valid]) / denom[valid] * 100.0
    # When denom == 0 (flat range) and we have enough data, %K = 50
    flat = ~np.isnan(highest) & (denom == 0)
    pct_k[flat] = 50.0

    pct_d = sma(pct_k, period_d)
    return pct_k, pct_d


def obv(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute On-Balance Volume.

    OBV is a cumulative total of volume, where volume is added on up-close
    bars and subtracted on down-close bars::

        OBV[0] = NaN
        OBV[i] = OBV[i-1] + sign(close[i] - close[i-1]) * volume[i]

    :param close: One-dimensional array of closing prices.
    :type close: NDArray[np.float64]
    :param volume: One-dimensional array of bar volumes.
    :type volume: NDArray[np.float64]
    :returns: Array of the same length as *close* with ``NaN`` at index 0.
    :rtype: NDArray[np.float64]
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    direction = np.sign(close[1:] - close[:-1])
    signed_vol = direction * volume[1:]
    out[1:] = np.cumsum(signed_vol)
    return out


def vwap(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute cumulative Volume-Weighted Average Price.

    VWAP is the ratio of cumulative typical-price-weighted volume to
    cumulative volume::

        TP = (high + low + close) / 3
        VWAP = cumsum(TP * volume) / cumsum(volume)

    :param high: One-dimensional array of bar high prices.
    :type high: NDArray[np.float64]
    :param low: One-dimensional array of bar low prices.
    :type low: NDArray[np.float64]
    :param close: One-dimensional array of bar closing prices.
    :type close: NDArray[np.float64]
    :param volume: One-dimensional array of bar volumes.
    :type volume: NDArray[np.float64]
    :returns: Array of the same length as *close*. ``NaN`` wherever
        cumulative volume is zero.
    :rtype: NDArray[np.float64]
    """
    tp = (high + low + close) / 3.0
    cum_tp_vol = np.cumsum(tp * volume)
    cum_vol = np.cumsum(volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(cum_vol != 0, cum_tp_vol / cum_vol, np.nan)
    return out.astype(np.float64)


def cci(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 20,
    scalar: float = 0.015,
) -> NDArray[np.float64]:
    """Compute the Commodity Channel Index.

    CCI measures how far the typical price deviates from its SMA,
    normalised by mean absolute deviation::

        TP  = (high + low + close) / 3
        CCI = (TP - SMA(TP, period)) / (scalar * MAD(TP, period))

    :param high: One-dimensional array of bar high prices.
    :type high: NDArray[np.float64]
    :param low: One-dimensional array of bar low prices.
    :type low: NDArray[np.float64]
    :param close: One-dimensional array of bar closing prices.
    :type close: NDArray[np.float64]
    :param period: Lookback period. Defaults to ``20``.
    :type period: int
    :param scalar: Scaling constant. Defaults to ``0.015``.
    :type scalar: float
    :returns: Array of the same length as *close* with ``NaN`` for the
        first ``period - 1`` positions.
    :rtype: NDArray[np.float64]
    """
    n = len(close)
    out = np.full(n, np.nan)
    if period < 1 or n < period:
        return out

    tp = (high + low + close) / 3.0
    tp_sma = sma(tp, period)

    for i in range(period - 1, n):
        window = tp[i - period + 1 : i + 1]
        mad = float(np.mean(np.abs(window - tp_sma[i])))
        if mad == 0.0:
            out[i] = 0.0
        else:
            out[i] = (tp[i] - tp_sma[i]) / (scalar * mad)
    return out


def donchian(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    period: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Donchian Channels.

    The upper channel is the rolling max of *high* and the lower channel
    is the rolling min of *low*. The middle is their average::

        upper  = rolling_max(high, period)
        lower  = rolling_min(low, period)
        middle = (upper + lower) / 2

    :param high: One-dimensional array of bar high prices.
    :type high: NDArray[np.float64]
    :param low: One-dimensional array of bar low prices.
    :type low: NDArray[np.float64]
    :param period: Lookback period. Defaults to ``20``.
    :type period: int
    :returns: ``(upper, middle, lower)`` tuple. All arrays have the same
        length as *high*.
    :rtype: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    """
    upper = rolling_max(high, period)
    lower = rolling_min(low, period)
    middle = (upper + lower) / 2.0
    return upper, middle, lower


def keltner(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 20,
    multiplier: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Keltner Channels.

    The middle band is an EMA of *close*. Upper and lower bands are offset
    by a multiple of the ATR::

        middle = EMA(close, period)
        upper  = middle + multiplier * ATR(high, low, close, period)
        lower  = middle - multiplier * ATR(high, low, close, period)

    :param high: One-dimensional array of bar high prices.
    :type high: NDArray[np.float64]
    :param low: One-dimensional array of bar low prices.
    :type low: NDArray[np.float64]
    :param close: One-dimensional array of bar closing prices.
    :type close: NDArray[np.float64]
    :param period: EMA and ATR lookback period. Defaults to ``20``.
    :type period: int
    :param multiplier: ATR multiplier for channel width. Defaults to ``2.0``.
    :type multiplier: float
    :returns: ``(upper, middle, lower)`` tuple. All arrays have the same
        length as *close*.
    :rtype: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    """
    middle = ema(close, period)
    atr_arr = atr(high, low, close, period)
    upper = middle + multiplier * atr_arr
    lower = middle - multiplier * atr_arr
    return upper, middle, lower
