"""Data normalisation and validation utilities."""

from __future__ import annotations

import pandas as pd


_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def normalise_candles(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """
    Validate, cast, sort, and deduplicate a candles DataFrame.

    Steps:
        1. Raise ValueError if any required column is missing
           (required: ts_col, open, high, low, close, volume).
        2. Cast OHLCV columns to float64.
        3. Parse ts column to datetime64[ns, UTC] — strips timezone if present,
           then re-localises as UTC to ensure consistent merging across instruments.
        4. Sort ascending by ts, drop duplicate ts rows (keep last), reset index.
        5. Return a copy (input is not mutated).

    Args:
        df: Raw OHLCV DataFrame.
        ts_col: Name of the timestamp column (default "ts").

    Returns:
        Cleaned DataFrame with a monotonic, tz-naive UTC timestamp column.

    Raises:
        ValueError: if required columns are missing.
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
    """
    Raise ValueError if df has fewer rows than required.

    Args:
        df: Normalised candles DataFrame.
        required: Minimum number of rows needed.
        label: Optional instrument label for error messages.

    Raises:
        ValueError: if len(df) < required.
    """
    tag = f" for '{label}'" if label else ""
    if len(df) < required:
        raise ValueError(
            f"Insufficient candles{tag}: need {required}, got {len(df)}."
        )
