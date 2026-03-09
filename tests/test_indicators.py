"""Unit tests for indicator functions."""

import numpy as np
import pytest
from numpy.typing import NDArray

from quaver.strategies.indicators import (
    cci,
    donchian,
    ema,
    keltner,
    macd,
    obv,
    stochastic,
    vwap,
)


def _arr(*values: float) -> NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


# ── EMA ──────────────────────────────────────────────────────────────────────


class TestEma:
    def test_known_values(self) -> None:
        vals = _arr(1, 2, 3, 4, 5)
        out = ema(vals, 3)
        # Seed = SMA of first 3 = 2.0
        assert out[2] == pytest.approx(2.0)
        k = 2.0 / 4.0
        expected_3 = 4 * k + 2.0 * (1 - k)
        assert out[3] == pytest.approx(expected_3)

    def test_short_array_all_nan(self) -> None:
        out = ema(_arr(1, 2), 5)
        assert np.all(np.isnan(out))

    def test_shape(self) -> None:
        vals = _arr(1, 2, 3, 4, 5, 6)
        assert ema(vals, 3).shape == vals.shape

    def test_leading_nans(self) -> None:
        out = ema(_arr(1, 2, 3, 4, 5), 3)
        assert np.isnan(out[0])
        assert np.isnan(out[1])
        assert not np.isnan(out[2])

    def test_period_one(self) -> None:
        vals = _arr(5, 10, 15)
        out = ema(vals, 1)
        np.testing.assert_allclose(out, vals)


# ── MACD ─────────────────────────────────────────────────────────────────────


class TestMacd:
    def test_short_array_all_nan(self) -> None:
        ml, sl, hist = macd(_arr(1, 2, 3), fast=12, slow=26, signal=9)
        assert np.all(np.isnan(ml))
        assert np.all(np.isnan(sl))
        assert np.all(np.isnan(hist))

    def test_shape(self) -> None:
        vals = np.arange(50, dtype=np.float64)
        ml, sl, hist = macd(vals)
        assert ml.shape == vals.shape
        assert sl.shape == vals.shape
        assert hist.shape == vals.shape

    def test_macd_line_equals_fast_minus_slow(self) -> None:
        vals = np.arange(50, dtype=np.float64)
        ml, _, _ = macd(vals, fast=12, slow=26, signal=9)
        ema_fast = ema(vals, 12)
        ema_slow = ema(vals, 26)
        valid = ~np.isnan(ml)
        np.testing.assert_allclose(ml[valid], (ema_fast - ema_slow)[valid])

    def test_histogram_equals_macd_minus_signal(self) -> None:
        vals = np.arange(50, dtype=np.float64)
        ml, sl, hist = macd(vals)
        valid = ~np.isnan(hist)
        np.testing.assert_allclose(hist[valid], (ml - sl)[valid])

    def test_leading_nans_macd_line(self) -> None:
        vals = np.arange(50, dtype=np.float64)
        ml, _, _ = macd(vals, fast=12, slow=26)
        # First valid MACD at index slow-1 = 25
        assert np.all(np.isnan(ml[:25]))
        assert not np.isnan(ml[25])


# ── Stochastic ───────────────────────────────────────────────────────────────


class TestStochastic:
    def test_known_values(self) -> None:
        h = _arr(10, 12, 11, 13, 14)
        low = _arr(8, 9, 8, 10, 11)
        c = _arr(9, 11, 10, 12, 13)
        pct_k, pct_d = stochastic(h, low, c, period_k=3, period_d=2)
        # At index 2: highest(h[0:3])=12, lowest(l[0:3])=8, close=10
        # %K = (10-8)/(12-8)*100 = 50
        assert pct_k[2] == pytest.approx(50.0)

    def test_short_array_all_nan(self) -> None:
        pct_k, pct_d = stochastic(_arr(1), _arr(1), _arr(1), period_k=5)
        assert np.all(np.isnan(pct_k))
        assert np.all(np.isnan(pct_d))

    def test_shape(self) -> None:
        n = 20
        h = np.arange(n, dtype=np.float64) + 1
        low = np.arange(n, dtype=np.float64)
        c = np.arange(n, dtype=np.float64) + 0.5
        pct_k, pct_d = stochastic(h, low, c, period_k=5)
        assert pct_k.shape == (n,)
        assert pct_d.shape == (n,)

    def test_flat_data_gives_50(self) -> None:
        h = np.full(10, 100.0)
        low = np.full(10, 100.0)
        c = np.full(10, 100.0)
        pct_k, _ = stochastic(h, low, c, period_k=5)
        assert pct_k[4] == pytest.approx(50.0)

    def test_leading_nans(self) -> None:
        n = 20
        h = np.arange(n, dtype=np.float64) + 1
        low = np.arange(n, dtype=np.float64)
        c = np.arange(n, dtype=np.float64) + 0.5
        pct_k, _ = stochastic(h, low, c, period_k=14)
        assert np.all(np.isnan(pct_k[:13]))
        assert not np.isnan(pct_k[13])


# ── OBV ──────────────────────────────────────────────────────────────────────


class TestObv:
    def test_known_values(self) -> None:
        c = _arr(10, 11, 10, 12)
        v = _arr(100, 200, 150, 300)
        out = obv(c, v)
        assert np.isnan(out[0])
        # Bar 1: up → +200; Bar 2: down → -150; Bar 3: up → +300
        assert out[1] == pytest.approx(200.0)
        assert out[2] == pytest.approx(200.0 - 150.0)
        assert out[3] == pytest.approx(200.0 - 150.0 + 300.0)

    def test_short_array(self) -> None:
        out = obv(_arr(10), _arr(100))
        assert len(out) == 1
        assert np.isnan(out[0])

    def test_shape(self) -> None:
        c = np.arange(10, dtype=np.float64)
        v = np.ones(10)
        assert obv(c, v).shape == (10,)

    def test_flat_close_zero_volume_added(self) -> None:
        c = np.full(5, 100.0)
        v = np.full(5, 1000.0)
        out = obv(c, v)
        # All changes are 0 → sign(0) = 0 → no volume added
        assert out[4] == pytest.approx(0.0)


# ── VWAP ─────────────────────────────────────────────────────────────────────


class TestVwap:
    def test_known_values(self) -> None:
        h = _arr(12, 14)
        low = _arr(8, 10)
        c = _arr(10, 12)
        v = _arr(100, 200)
        out = vwap(h, low, c, v)
        tp0 = (12 + 8 + 10) / 3.0
        tp1 = (14 + 10 + 12) / 3.0
        expected_0 = tp0
        expected_1 = (tp0 * 100 + tp1 * 200) / 300.0
        assert out[0] == pytest.approx(expected_0)
        assert out[1] == pytest.approx(expected_1)

    def test_shape(self) -> None:
        n = 10
        h = np.arange(n, dtype=np.float64) + 1
        low = np.arange(n, dtype=np.float64)
        c = np.arange(n, dtype=np.float64) + 0.5
        v = np.ones(n)
        assert vwap(h, low, c, v).shape == (n,)

    def test_zero_volume_gives_nan(self) -> None:
        h = _arr(10)
        low = _arr(8)
        c = _arr(9)
        v = _arr(0)
        out = vwap(h, low, c, v)
        assert np.isnan(out[0])


# ── CCI ──────────────────────────────────────────────────────────────────────


class TestCci:
    def test_short_array_all_nan(self) -> None:
        out = cci(_arr(1), _arr(1), _arr(1), period=5)
        assert np.all(np.isnan(out))

    def test_shape(self) -> None:
        n = 30
        h = np.arange(n, dtype=np.float64) + 1
        low = np.arange(n, dtype=np.float64)
        c = np.arange(n, dtype=np.float64) + 0.5
        assert cci(h, low, c, period=20).shape == (n,)

    def test_leading_nans(self) -> None:
        n = 30
        h = np.arange(n, dtype=np.float64) + 1
        low = np.arange(n, dtype=np.float64)
        c = np.arange(n, dtype=np.float64) + 0.5
        out = cci(h, low, c, period=20)
        assert np.all(np.isnan(out[:19]))
        assert not np.isnan(out[19])

    def test_flat_data_gives_zero(self) -> None:
        n = 25
        h = np.full(n, 100.0)
        low = np.full(n, 100.0)
        c = np.full(n, 100.0)
        out = cci(h, low, c, period=20)
        assert out[19] == pytest.approx(0.0)


# ── Donchian ─────────────────────────────────────────────────────────────────


class TestDonchian:
    def test_known_values(self) -> None:
        h = _arr(10, 12, 11, 13, 14)
        low = _arr(8, 9, 7, 10, 11)
        upper, middle, lower = donchian(h, low, period=3)
        # At index 2: max(h[0:3])=12, min(l[0:3])=7
        assert upper[2] == pytest.approx(12.0)
        assert lower[2] == pytest.approx(7.0)
        assert middle[2] == pytest.approx(9.5)

    def test_shape(self) -> None:
        n = 20
        h = np.arange(n, dtype=np.float64)
        low = np.arange(n, dtype=np.float64)
        u, m, lo = donchian(h, low, period=5)
        assert u.shape == (n,)
        assert m.shape == (n,)
        assert lo.shape == (n,)

    def test_leading_nans(self) -> None:
        h = np.arange(10, dtype=np.float64)
        low = np.arange(10, dtype=np.float64)
        u, m, lo = donchian(h, low, period=5)
        for arr in (u, m, lo):
            assert np.all(np.isnan(arr[:4]))
            assert not np.isnan(arr[4])

    def test_short_array_all_nan(self) -> None:
        h = _arr(1, 2)
        low = _arr(0, 1)
        u, m, lo = donchian(h, low, period=5)
        assert np.all(np.isnan(u))


# ── Keltner ──────────────────────────────────────────────────────────────────


class TestKeltner:
    def test_shape(self) -> None:
        n = 30
        h = np.arange(n, dtype=np.float64) + 1
        low = np.arange(n, dtype=np.float64)
        c = np.arange(n, dtype=np.float64) + 0.5
        u, m, lo = keltner(h, low, c, period=10)
        assert u.shape == (n,)
        assert m.shape == (n,)
        assert lo.shape == (n,)

    def test_middle_equals_ema(self) -> None:
        n = 30
        h = np.arange(n, dtype=np.float64) + 1
        low = np.arange(n, dtype=np.float64)
        c = np.arange(n, dtype=np.float64) + 0.5
        _, m, _ = keltner(h, low, c, period=10)
        expected = ema(c, 10)
        valid = ~np.isnan(m)
        np.testing.assert_allclose(m[valid], expected[valid])

    def test_upper_gt_lower(self) -> None:
        n = 50
        rng = np.random.default_rng(42)
        base = 100 + np.cumsum(rng.normal(0, 1, n))
        h = base + rng.uniform(0.5, 1.5, n)
        low = base - rng.uniform(0.5, 1.5, n)
        c = base
        u, _, lo = keltner(h, low, c, period=10, multiplier=2.0)
        valid = ~np.isnan(u) & ~np.isnan(lo)
        assert np.all(u[valid] >= lo[valid])

    def test_short_array_all_nan(self) -> None:
        u, m, lo = keltner(_arr(1), _arr(0), _arr(0.5), period=20)
        assert np.all(np.isnan(u))
        assert np.all(np.isnan(m))
        assert np.all(np.isnan(lo))
