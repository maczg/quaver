"""Generate matplotlib figures for strategy documentation.

Run from the repository root:
    python docs/strategies/generate_figures.py

Saves PNGs to docs/strategies/images/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# ── Shared style ────────────────────────────────────────────────────────────

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def _save(fig: plt.Figure, name: str) -> None:
    path = IMAGES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean with NaN for incomplete windows (no edge artifacts)."""
    import pandas as pd

    return pd.Series(data).rolling(window).mean().to_numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Breakout from Consolidation — 3 figures
# ═══════════════════════════════════════════════════════════════════════════


def breakout_consolidation_range() -> None:
    """Tight price range with resistance/support and breakout arrow."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    np.random.seed(10)
    n = 60

    # Pre-consolidation: uptrend
    pre = 100 + np.cumsum(np.random.normal(0.3, 0.8, 20))
    # Consolidation: tight range
    consol = pre[-1] + np.random.normal(0, 0.3, 30)
    # Breakout: sharp move up
    post = consol[-1] + np.cumsum(np.random.normal(0.6, 0.5, 10))

    price = np.concatenate([pre, consol, post])
    days = np.arange(n)

    ax.plot(days, price, color="#2563eb", linewidth=1.5, label="Price")

    # Resistance / support bands
    consol_high = consol.max()
    consol_low = consol.min()
    ax.axhspan(consol_low, consol_high, xmin=20 / n, xmax=50 / n,
               alpha=0.15, color="#f59e0b", label="Consolidation range")
    ax.axhline(consol_high, xmin=20 / n, xmax=50 / n,
               color="#ef4444", linestyle="--", linewidth=1, label="Resistance")
    ax.axhline(consol_low, xmin=20 / n, xmax=50 / n,
               color="#22c55e", linestyle="--", linewidth=1, label="Support")

    # Breakout annotation
    ax.annotate("BREAKOUT", xy=(50, post[0]), xytext=(42, post[0] + 4),
                fontsize=11, fontweight="bold", color="#dc2626",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5))

    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price")
    ax.set_title("Consolidation Range with Breakout")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "consolidation_range")


def breakout_atr_declining() -> None:
    """ATR declining during consolidation period."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.5))

    np.random.seed(11)
    days = np.arange(60)

    # Simulate ATR: high early, declining through consolidation, spike at breakout
    atr = np.concatenate([
        2.5 + np.random.normal(0, 0.15, 15),                      # pre
        np.linspace(2.5, 1.2, 35) + np.random.normal(0, 0.08, 35),  # declining
        np.linspace(1.3, 3.0, 10) + np.random.normal(0, 0.1, 10),   # breakout spike
    ])

    ax.plot(days, atr, color="#8b5cf6", linewidth=2)
    ax.fill_between(days, atr, alpha=0.15, color="#8b5cf6")

    # Highlight declining region
    ax.axvspan(15, 50, alpha=0.08, color="#f59e0b")
    ax.annotate("ATR declining\n(compression building)",
                xy=(32, 1.6), fontsize=10, ha="center",
                color="#92400e", fontstyle="italic")

    ax.annotate("Breakout\nspike", xy=(55, 2.8), xytext=(48, 3.3),
                fontsize=10, color="#dc2626", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.3))

    ax.set_xlabel("Trading Days")
    ax.set_ylabel("ATR (14)")
    ax.set_title("ATR Declining During Consolidation")
    _save(fig, "atr_declining")


def breakout_full_example() -> None:
    """Full breakout example: price + volume subplot."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[3, 1],
                                    sharex=True)
    fig.subplots_adjust(hspace=0.08)

    np.random.seed(12)
    n = 70

    # Price series
    pre = 100 + np.cumsum(np.random.normal(0.25, 0.7, 20))
    consol = pre[-1] + np.random.normal(0, 0.25, 35)
    post = consol[-1] + np.cumsum(np.random.normal(0.5, 0.4, 15))
    price = np.concatenate([pre, consol, post])
    days = np.arange(n)

    # MA50 (simplified as 15-day rolling for visual clarity)
    ma50 = _rolling_mean(price, 15)

    ax1.plot(days, price, color="#2563eb", linewidth=1.5, label="Price")
    ax1.plot(days, ma50, color="#f97316", linewidth=1.2, linestyle="--", label="MA50")

    consol_high = consol.max()
    consol_low = consol.min()
    ax1.axhline(consol_high, xmin=20 / n, xmax=55 / n,
                color="#ef4444", linestyle=":", linewidth=1)
    ax1.axhline(consol_low, xmin=20 / n, xmax=55 / n,
                color="#22c55e", linestyle=":", linewidth=1)
    ax1.axhspan(consol_low, consol_high, xmin=20 / n, xmax=55 / n,
                alpha=0.10, color="#f59e0b")

    # Breakout annotation
    ax1.annotate("Breakout close\nabove 20-day high",
                 xy=(55, post[0]), xytext=(40, post[0] + 5),
                 fontsize=10, fontweight="bold", color="#dc2626",
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.3))

    # Stop annotation
    ax1.annotate("Stop below range",
                 xy=(55, consol_low), xytext=(60, consol_low - 3),
                 fontsize=9, color="#22c55e",
                 arrowprops=dict(arrowstyle="->", color="#22c55e", lw=1))

    ax1.set_ylabel("Price")
    ax1.set_title("Breakout from Consolidation — Full Example")
    ax1.legend(loc="upper left", fontsize=9)

    # Volume subplot
    np.random.seed(13)
    vol = np.random.randint(400, 900, n).astype(float)
    vol[55:60] *= 2.5  # volume spike on breakout
    avg_vol = _rolling_mean(vol, 20)

    colors = ["#ef4444" if v > avg_vol[i] * 1.3 else "#94a3b8"
              for i, v in enumerate(vol)]
    ax2.bar(days, vol, color=colors, width=0.8)
    ax2.plot(days, avg_vol, color="#1e293b", linewidth=1, linestyle="--",
             label="20-day avg volume")
    ax2.annotate("Volume spike", xy=(57, vol[57]), xytext=(45, vol[57] + 200),
                 fontsize=9, color="#dc2626", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1))

    ax2.set_xlabel("Trading Days")
    ax2.set_ylabel("Volume")
    ax2.legend(loc="upper left", fontsize=9)
    _save(fig, "breakout_example")


# ═══════════════════════════════════════════════════════════════════════════
# Pullback in Trend — 3 figures
# ═══════════════════════════════════════════════════════════════════════════


def pullback_ma_layers() -> None:
    """Price with MA20 and MA50 showing layered moving averages."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    np.random.seed(20)
    n = 100
    trend = np.linspace(0, 25, n)
    noise = np.cumsum(np.random.normal(0, 0.6, n))
    price = 100 + trend + noise

    ma20 = _rolling_mean(price, 20)
    ma50 = _rolling_mean(price, 50)

    days = np.arange(n)
    ax.plot(days, price, color="#2563eb", linewidth=1.2, alpha=0.8, label="Price (noisy)")
    ax.plot(days, ma20, color="#f97316", linewidth=2, label="MA20 (smoother)")
    ax.plot(days, ma50, color="#dc2626", linewidth=2, linestyle="--", label="MA50 (smoothest)")

    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price")
    ax.set_title("Moving Average Layers in an Uptrend")
    ax.legend(loc="upper left", fontsize=10)
    _save(fig, "ma_layers")


def pullback_rsi_zones() -> None:
    """RSI with overbought / oversold / neutral zones highlighted."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.5))

    np.random.seed(21)
    n = 100
    # Simulate RSI oscillating between 30 and 70
    rsi = 50 + 20 * np.sin(np.linspace(0, 6 * np.pi, n)) + np.random.normal(0, 3, n)
    rsi = np.clip(rsi, 5, 95)

    ax.plot(rsi, color="#2563eb", linewidth=1.5)

    # Zones
    ax.axhspan(70, 100, alpha=0.12, color="#ef4444", label="Overbought (>70)")
    ax.axhspan(0, 30, alpha=0.12, color="#22c55e", label="Oversold (<30)")
    ax.axhspan(40, 50, alpha=0.12, color="#f59e0b", label="Pullback zone (40-50)")

    ax.axhline(70, color="#ef4444", linestyle="--", linewidth=0.8)
    ax.axhline(50, color="#64748b", linestyle=":", linewidth=0.8)
    ax.axhline(30, color="#22c55e", linestyle="--", linewidth=0.8)

    ax.set_ylim(0, 100)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("RSI (14)")
    ax.set_title("RSI Zones — Pullback in Trend")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "rsi_zones")


def pullback_entry_example() -> None:
    """Pullback entry example: uptrend → dip toward MA20 → resumption."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    np.random.seed(22)
    n = 80

    # Build an uptrend with a clear pullback around day 50-60
    trend = np.linspace(0, 20, n)
    noise = np.cumsum(np.random.normal(0, 0.5, n))
    price = 100 + trend + noise

    # Inject a pullback (dip) at days 48-56
    pullback_depth = np.array([0, -1.5, -3.0, -4.0, -4.5, -3.5, -2.0, -0.5])
    price[48:56] += pullback_depth

    ma20 = _rolling_mean(price, 20)
    ma50 = _rolling_mean(price, 50)

    days = np.arange(n)

    ax.plot(days, price, color="#2563eb", linewidth=1.5, label="Price")
    ax.plot(days, ma20, color="#f97316", linewidth=1.5, linestyle="--", label="MA20")
    ax.plot(days, ma50, color="#dc2626", linewidth=1.5, linestyle=":", label="MA50")

    # Highlight pullback zone
    ax.axvspan(48, 56, alpha=0.10, color="#f59e0b")
    ax.annotate("Pullback\n(2-5 days toward MA20)",
                xy=(52, price[52]), xytext=(35, price[52] - 5),
                fontsize=10, color="#92400e",
                arrowprops=dict(arrowstyle="->", color="#92400e", lw=1.2))

    # Entry point
    entry_day = 56
    ax.plot(entry_day, price[entry_day], "^", color="#22c55e", markersize=14,
            zorder=5, label="Entry")
    ax.annotate("Entry\n(Close > prior High\nor > MA20)",
                xy=(entry_day, price[entry_day]),
                xytext=(entry_day + 5, price[entry_day] + 3),
                fontsize=10, fontweight="bold", color="#16a34a",
                arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.3))

    # Pullback low / stop
    low_day = 52
    ax.plot(low_day, price[low_day], "v", color="#ef4444", markersize=10, zorder=5)
    ax.annotate("Pullback low\n(stop below here)",
                xy=(low_day, price[low_day]),
                xytext=(low_day - 12, price[low_day] - 4),
                fontsize=9, color="#dc2626",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1))

    # RSI note
    ax.text(52, price.min() - 1, "RSI 40-50 here\n(healthy dip)",
            fontsize=9, ha="center", color="#6b7280", fontstyle="italic")

    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price")
    ax.set_title("Pullback in Trend — Entry Example")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "pullback_entry")


# ═══════════════════════════════════════════════════════════════════════════
# Reversal at Support — 3 figures
# ═══════════════════════════════════════════════════════════════════════════


def reversal_ma_levels() -> None:
    """Sharp drop with MA20, MA50, MA200 levels showing target zones."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    np.random.seed(30)
    n = 80

    # Stock that trends then drops sharply
    trend = np.linspace(0, 12, 50)
    noise = np.cumsum(np.random.normal(0, 0.4, 50))
    up = 100 + trend + noise

    # Sharp drop
    drop = up[-1] + np.cumsum(np.random.normal(-0.8, 0.3, 20))
    # Recovery
    recovery = drop[-1] + np.cumsum(np.random.normal(0.4, 0.3, 10))

    price = np.concatenate([up, drop, recovery])
    days = np.arange(n)

    ma20 = _rolling_mean(price, 20)
    ma50 = _rolling_mean(price, 50)
    # Fake MA200 as a lower flat line
    ma200 = np.full(n, price.min() - 3)

    ax.plot(days, price, color="#2563eb", linewidth=1.5, label="Price")
    ax.plot(days, ma20, color="#f97316", linewidth=1.5, linestyle="--", label="MA20 (nearer target)")
    ax.plot(days, ma50, color="#dc2626", linewidth=1.5, linestyle=":", label="MA50 (target zone)")
    ax.axhline(ma200[0], color="#64748b", linewidth=1.5, linestyle="-.",
               label="MA200 (long-term floor)")

    # Annotate the drop
    ax.annotate("Sharp drop", xy=(60, price[60]), xytext=(45, price[60] + 3),
                fontsize=10, color="#dc2626", fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1))

    # Reversal point
    rev_day = 70
    ax.plot(rev_day, price[rev_day], "*", color="#22c55e", markersize=15, zorder=5)
    ax.annotate("Reversal signal\n(near MA200 / support)",
                xy=(rev_day, price[rev_day]),
                xytext=(rev_day - 18, price[rev_day] - 5),
                fontsize=10, fontweight="bold", color="#16a34a",
                arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.3))

    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price")
    ax.set_title("MA Levels — Reversal at Support")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "ma_levels_reversal")


def reversal_rsi_zones() -> None:
    """RSI with oversold threshold highlighted for reversal strategy."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.5))

    np.random.seed(31)
    n = 100
    # RSI that drops below 30 in places
    rsi = 50 + 25 * np.sin(np.linspace(0, 5 * np.pi, n)) + np.random.normal(0, 4, n)
    rsi = np.clip(rsi, 5, 95)

    ax.plot(rsi, color="#2563eb", linewidth=1.5)

    # Zones
    ax.axhspan(70, 100, alpha=0.12, color="#ef4444", label="Overbought (>70) — avoid buying")
    ax.axhspan(0, 30, alpha=0.15, color="#22c55e", label="Oversold (<30) — look for reversal")

    ax.axhline(70, color="#ef4444", linestyle="--", linewidth=0.8)
    ax.axhline(50, color="#64748b", linestyle=":", linewidth=0.8, label="Neutral (50)")
    ax.axhline(30, color="#22c55e", linestyle="--", linewidth=0.8)

    # Mark oversold dips
    oversold_mask = rsi < 30
    ax.fill_between(range(n), rsi, 30, where=oversold_mask,
                    alpha=0.3, color="#22c55e", interpolate=True)

    ax.set_ylim(0, 100)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("RSI (14)")
    ax.set_title("RSI Zones — Reversal at Support")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "rsi_zones_reversal")


def reversal_full_example() -> None:
    """Full reversal example: decline → oversold → bullish trigger → targets."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    np.random.seed(32)
    n = 80

    # Uptrend, then sharp decline, then reversal
    up = 100 + np.cumsum(np.random.normal(0.2, 0.4, 40))
    decline = up[-1] + np.cumsum(np.random.normal(-0.7, 0.3, 20))
    recovery = decline[-1] + np.cumsum(np.random.normal(0.5, 0.35, 20))

    price = np.concatenate([up, decline, recovery])
    days = np.arange(n)

    ma20 = _rolling_mean(price, 20)
    ma50 = _rolling_mean(price, 50)
    ma200 = np.full(n, price.min() - 4)

    ax.plot(days, price, color="#2563eb", linewidth=1.5, label="Price")
    ax.plot(days, ma20, color="#f97316", linewidth=1.3, linestyle="--", label="MA20 (Target 1)")
    ax.plot(days, ma50, color="#dc2626", linewidth=1.3, linestyle=":", label="MA50 (Target 2)")
    ax.axhline(ma200[0], color="#64748b", linewidth=1.2, linestyle="-.",
               label="MA200 (health check)")

    # Decline shading
    ax.axvspan(40, 60, alpha=0.08, color="#ef4444")
    ax.annotate("Sharp decline\n(RSI drops below 30)",
                xy=(50, price[50]), xytext=(30, price[50] - 3),
                fontsize=10, color="#dc2626", fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1))

    # Reversal trigger
    trigger_day = 60
    ax.plot(trigger_day, price[trigger_day], "^", color="#22c55e",
            markersize=14, zorder=5, label="Bullish trigger")
    ax.annotate("Close > prior High\n(entry signal)",
                xy=(trigger_day, price[trigger_day]),
                xytext=(trigger_day + 3, price[trigger_day] - 5),
                fontsize=10, fontweight="bold", color="#16a34a",
                arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.3))

    # Stop loss
    stop_day = 58
    stop_price = price[55:61].min()
    ax.axhline(stop_price, xmin=55 / n, xmax=65 / n,
               color="#ef4444", linestyle="--", linewidth=1)
    ax.annotate("Stop below\nswing low",
                xy=(62, stop_price), xytext=(66, stop_price - 3),
                fontsize=9, color="#dc2626",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1))

    # Target arrows
    target1_y = ma20[65] if not np.isnan(ma20[65]) else price[65] + 3
    target2_y = ma50[65] if not np.isnan(ma50[65]) else price[65] + 6
    ax.annotate("Target 1", xy=(70, target1_y), fontsize=9,
                fontweight="bold", color="#f97316")
    ax.annotate("Target 2", xy=(70, target2_y), fontsize=9,
                fontweight="bold", color="#dc2626")

    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price")
    ax.set_title("Reversal at Support — Full Example")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "reversal_example")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

ALL_GENERATORS = [
    breakout_consolidation_range,
    breakout_atr_declining,
    breakout_full_example,
    pullback_ma_layers,
    pullback_rsi_zones,
    pullback_entry_example,
    reversal_ma_levels,
    reversal_rsi_zones,
    reversal_full_example,
]


def main() -> None:
    print("Generating strategy figures...")
    for gen in ALL_GENERATORS:
        gen()
    print(f"Done — {len(ALL_GENERATORS)} figures saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
