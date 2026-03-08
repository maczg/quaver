import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta


def make_candles(
    n: int = 300,
    start_price: float = 100.0,
    volatility: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic OHLCV candles for testing."""
    rng = np.random.default_rng(seed)
    base = start_price
    rows = []
    ts = datetime(2020, 1, 1)
    for _ in range(n):
        change = rng.normal(0, volatility)
        o = base
        c = base + change
        h = max(o, c) + abs(rng.normal(0, 0.2))
        l = min(o, c) - abs(rng.normal(0, 0.2))
        v = abs(rng.normal(1_000_000, 100_000))
        rows.append({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
        base = c
        ts += timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture
def candles_300():
    return make_candles(300)


@pytest.fixture
def candles_500():
    return make_candles(500)
