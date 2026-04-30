"""Synthetic OHLCV builders for behavioural strategy tests.

Each builder returns a deterministic ``pandas.DataFrame`` shaped to either
satisfy or violate a specific strategy's trigger conditions. Tests use these
to assert that strategies fire (or don't fire) as documented.

The builders are intentionally simple and explicit: no randomness unless a
test asks for it, and every shape is described by the parameters of the
function.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def _row(ts: datetime, o: float, h: float, low: float, c: float, v: float) -> dict:
    return {"ts": ts, "open": o, "high": h, "low": low, "close": c, "volume": v}


def flat_market(
    n: int = 250,
    price: float = 100.0,
    spread: float = 0.4,
    volume: float = 1_000_000.0,
    start: datetime = datetime(2023, 1, 1),
) -> pd.DataFrame:
    """Strict flat market — open=close=price each bar, tiny intrabar spread."""
    rows = []
    ts = start
    for _ in range(n):
        rows.append(_row(ts, price, price + spread, price - spread, price, volume))
        ts += timedelta(days=1)
    return pd.DataFrame(rows)


def linear_trend(
    n: int = 250,
    start_price: float = 100.0,
    drift: float = 0.5,
    spread: float = 0.5,
    volume: float = 1_000_000.0,
    start: datetime = datetime(2023, 1, 1),
) -> pd.DataFrame:
    """Linear ramp — close = open + drift each bar.

    drift > 0 → uptrend, drift < 0 → downtrend.
    """
    rows = []
    ts = start
    price = start_price
    for _ in range(n):
        rows.append(
            _row(ts, price, price + spread, price - spread, price + drift, volume)
        )
        price += drift
        ts += timedelta(days=1)
    return pd.DataFrame(rows)


def uptrend_with_dip(
    n_pre: int = 220,
    drift: float = 0.4,
    dip_bars: int = 8,
    dip_drop_per_bar: float = 0.6,
    rebound_jump: float = 1.0,
    start_price: float = 100.0,
    volume: float = 1_000_000.0,
    start: datetime = datetime(2023, 1, 1),
) -> pd.DataFrame:
    """Long uptrend, a small pullback, then one bullish recovery bar.

    Structure:
      - bars [0, n_pre)            — uptrend at +drift / bar
      - bars [n_pre, n_pre+dip_bars) — pullback at -dip_drop_per_bar / bar
      - last bar                    — bullish recovery: close > prior high

    Returns the full DataFrame including the recovery bar at the end.
    """
    rows = []
    ts = start
    price = start_price

    for _ in range(n_pre):
        c = price + drift
        rows.append(_row(ts, price, max(price, c) + 0.3, min(price, c) - 0.3, c, volume))
        price = c
        ts += timedelta(days=1)

    for _ in range(dip_bars):
        c = price - dip_drop_per_bar
        rows.append(_row(ts, price, max(price, c) + 0.3, min(price, c) - 0.3, c, volume))
        price = c
        ts += timedelta(days=1)

    prior_high = rows[-1]["high"]
    o = price
    c = price + rebound_jump
    rows.append(_row(ts, o, max(c, prior_high) + 0.5, o - 0.3, c, volume))
    return pd.DataFrame(rows)


def downtrend_into_support(
    plateau_bars: int = 220,
    plateau_price: float = 100.0,
    drop_bars: int = 14,
    drop_per_bar: float = 0.8,
    rebound_jump: float = 1.5,
    volume: float = 1_000_000.0,
    start: datetime = datetime(2023, 1, 1),
) -> pd.DataFrame:
    """Plateau (so MA200 sits near the plateau) followed by a sharp drop into a
    rolling-low support level, ending with a bullish reversal candle.

    Designed for the "reversal_support" pattern:
      - long flat history → MA200 ~ plateau_price
      - drop_bars steep decline → drives RSI < 30 and creates a fresh
        rolling-min(low, 20)
      - last bar closes above the prior bar's high (bullish trigger)
    """
    rows = []
    ts = start
    for _ in range(plateau_bars):
        rows.append(
            _row(
                ts,
                plateau_price,
                plateau_price + 0.3,
                plateau_price - 0.3,
                plateau_price,
                volume,
            )
        )
        ts += timedelta(days=1)

    price = plateau_price
    for _ in range(drop_bars):
        c = price - drop_per_bar
        rows.append(_row(ts, price, price + 0.2, c - 0.2, c, volume))
        price = c
        ts += timedelta(days=1)

    prior_high = rows[-1]["high"]
    o = price
    c = price + rebound_jump
    rows.append(_row(ts, o, max(c, prior_high) + 0.3, o - 0.2, c, volume))
    return pd.DataFrame(rows)


def trend_then_consolidation_then_breakout(
    trend_bars: int = 60,
    trend_drift: float = 0.6,
    consol_bars: int = 25,
    consol_spread: float = 0.2,
    breakout_jump: float = 4.0,
    start_price: float = 100.0,
    base_volume: float = 1_000_000.0,
    breakout_volume_mult: float = 3.0,
    start: datetime = datetime(2023, 1, 1),
) -> pd.DataFrame:
    """Uptrend → tight consolidation → high-volume breakout above range."""
    rows = []
    ts = start
    price = start_price

    for _ in range(trend_bars):
        c = price + trend_drift
        rows.append(_row(ts, price, max(price, c) + 0.5, min(price, c) - 0.5, c, base_volume))
        price = c
        ts += timedelta(days=1)

    consol_price = price
    for _ in range(consol_bars):
        rows.append(
            _row(
                ts,
                consol_price,
                consol_price + consol_spread,
                consol_price - consol_spread,
                consol_price,
                base_volume * 0.8,
            )
        )
        ts += timedelta(days=1)

    o = consol_price
    c = consol_price + breakout_jump
    rows.append(
        _row(
            ts,
            o,
            c + 0.5,
            o - 0.5,
            c,
            base_volume * breakout_volume_mult,
        )
    )
    return pd.DataFrame(rows)


def vsa_stopping_volume_pattern(
    pre_bars: int = 60,
    drift: float = -0.3,
    base_volume: float = 1_000_000.0,
    final_volume_mult: float = 4.0,
    start_price: float = 100.0,
    start: datetime = datetime(2023, 1, 1),
) -> pd.DataFrame:
    """Local downtrend ending in a narrow-spread bear bar with massive volume
    and close near the top of the range (VSA stopping-volume).

    The final bar is engineered to satisfy *all* BUY conditions of
    :class:`VSAStoppingVolumeStrategy` with default parameters:

      - close < open                       (bear bar)
      - close < trend_sma(close, 20)       (downtrend)
      - vol_rel > 2.0                      (big volume)
      - spread_rel < 0.7                   (narrow spread)
      - close_position > 0.4               (close not on lows)
    """
    rows = []
    ts = start
    price = start_price

    # Build a downtrend with average spread ~ 1.0
    for _ in range(pre_bars):
        o = price
        c = price + drift
        h = max(o, c) + 0.5
        low = min(o, c) - 0.5
        rows.append(_row(ts, o, h, low, c, base_volume))
        price = c
        ts += timedelta(days=1)

    # Final bar — narrow spread, big volume, close near the *top*, still bear.
    # close_position = (close - low) / (high - low) must exceed 0.4.
    o = price
    c = price - 0.05  # bear by a hair
    high = c + 0.05   # close very near the top of the bar
    low = c - 0.35    # total spread = 0.4 vs ~1.0 average → spread_rel ≈ 0.4
    rows.append(_row(ts, o, high, low, c, base_volume * final_volume_mult))
    return pd.DataFrame(rows)


def correlated_pair_with_divergence(
    n: int = 120,
    spread_window: int = 30,
    final_divergence: float = 8.0,
    seed: int = 0,
    start: datetime = datetime(2023, 1, 1),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two correlated price series whose spread is stable until the final
    bars, where instrument B drops sharply (positive z-score on the spread).

    Returns ``(df_a, df_b)``.
    """
    rng = np.random.default_rng(seed)
    ts = start
    rows_a, rows_b = [], []
    price_a, price_b = 100.0, 100.0
    for i in range(n):
        change = rng.normal(0, 0.4)
        noise_b = rng.normal(0, 0.1)
        price_a += change
        price_b += change + noise_b
        # Inject one-sided divergence in the last few bars
        if i >= n - 3:
            price_b -= final_divergence / 3.0
        for rows, p in ((rows_a, price_a), (rows_b, price_b)):
            rows.append(_row(ts, p, p + 0.2, p - 0.2, p, 1000.0))
        ts += timedelta(days=1)
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


def random_walk(
    n: int = 250,
    start_price: float = 100.0,
    sigma: float = 1.0,
    seed: int = 42,
    volume: float = 1_000_000.0,
    start: datetime = datetime(2023, 1, 1),
) -> pd.DataFrame:
    """Geometric-like random walk with stable mean — used to verify that
    strategies do not fire on noise."""
    rng = np.random.default_rng(seed)
    rows = []
    ts = start
    price = start_price
    for _ in range(n):
        change = rng.normal(0, sigma)
        c = price + change
        h = max(price, c) + abs(rng.normal(0, 0.2))
        low = min(price, c) - abs(rng.normal(0, 0.2))
        rows.append(_row(ts, price, h, low, c, volume))
        price = c
        ts += timedelta(days=1)
    return pd.DataFrame(rows)
