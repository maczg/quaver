"""Data normalisation and validation utilities."""

from __future__ import annotations

import pandas as pd


_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def normalise_candles(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """Validate, cast, sort, and deduplicate a candles DataFrame.

    Processing steps applied in order:

    1. Raise :exc:`ValueError` if any required column is missing.
       Required columns: ``ts_col``, ``open``, ``high``, ``low``, ``close``,
       ``volume``.
    2. Cast all OHLCV columns to ``float64``.
    3. Parse the timestamp column to ``datetime64[ns, UTC]`` -- strips any
       existing timezone information and then re-localises as UTC to ensure
       consistent merging across instruments.
    4. Sort ascending by the timestamp column, drop duplicate timestamp rows
       (keeping the last occurrence), and reset the index.
    5. Return a copy; the input DataFrame is **never** mutated.

    :param df: Raw OHLCV DataFrame to be normalised.
    :type df: pandas.DataFrame
    :param ts_col: Name of the timestamp column.  Defaults to ``"ts"``.
    :type ts_col: str
    :returns: Cleaned DataFrame with a monotonic, tz-naive UTC timestamp
        column and all OHLCV columns cast to ``float64``.
    :rtype: pandas.DataFrame
    :raises ValueError: If one or more required columns are absent from
        ``df``.
    """
    df = df.copy()

    # Check required columns
    required = _REQUIRED_COLUMNS | {ts_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}"
        )

    # Cast OHLCV
    for col in _REQUIRED_COLUMNS:
        df[col] = df[col].astype("float64")

    # Normalise timestamp — strip tz, convert to datetime64[ns]
    ts = pd.to_datetime(df[ts_col], utc=True)
    df[ts_col] = ts.dt.tz_localize(None)  # store as tz-naive UTC

    # Sort, deduplicate, reset
    df = (
        df.sort_values(ts_col)
        .drop_duplicates(subset=[ts_col], keep="last")
        .reset_index(drop=True)
    )

    return df


def validate_candles(df: pd.DataFrame, required: int, label: str = "") -> None:
    """Raise :exc:`ValueError` if ``df`` has fewer rows than ``required``.

    :param df: Normalised candles DataFrame (output of
        :func:`normalise_candles`).
    :type df: pandas.DataFrame
    :param required: Minimum number of rows needed by the strategy.
    :type required: int
    :param label: Optional instrument label included in the error message for
        easier diagnosis.  Defaults to an empty string.
    :type label: str
    :returns: None
    :rtype: None
    :raises ValueError: If ``len(df) < required``.
    """
    tag = f" for '{label}'" if label else ""
    if len(df) < required:
        raise ValueError(
            f"Insufficient candles{tag}: need {required}, got {len(df)}."
        )
