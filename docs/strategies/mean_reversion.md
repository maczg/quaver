# Mean Reversion Strategy

## The idea in one sentence

Prices that move far away from their average tend to snap back — buy the dip, sell the rip.

## Financial background

Mean reversion is one of the oldest observations in financial markets. When a stock's recent price (measured
by a short-term moving average) diverges significantly from its longer-term average, the short-term price
tends to revert toward the mean rather than continue moving further away.

This happens because:

- **Over-reaction to news.** Markets over-shoot on earnings misses, analyst downgrades, or sector rotation.
  Once the initial panic subsides, prices recover toward fair value.
- **Institutional rebalancing.** Funds that track benchmarks periodically rebalance, which pushes stretched
  prices back toward equilibrium.
- **Mean-reverting fundamentals.** Profit margins, multiples, and growth rates tend to revert to long-run
  industry averages, pulling prices along.

Mean reversion works best in **range-bound or mildly trending markets**. In strongly trending markets
(e.g. a tech bubble or a crash), prices can stay far from the mean for extended periods — "the market can
stay irrational longer than you can stay solvent."

---

## How it works

The strategy uses two Simple Moving Averages (SMAs) with different lookback periods:

- **Fast SMA** — a short window (e.g. 20 days) that reacts quickly to recent price changes.
- **Slow SMA** — a longer window (e.g. 50 days) that represents the broader trend.

The **divergence** between the two is calculated as a percentage:

```
divergence = (Fast SMA - Slow SMA) / Slow SMA
```

When the divergence exceeds a configurable **threshold**, the strategy emits a signal.

---

## Signal logic

### BUY signal — "oversold, expect a bounce"

When the fast SMA drops below the slow SMA by more than the threshold, the stock's recent price action is
significantly weaker than its longer-term trend. The strategy interprets this as a temporary dip and
signals BUY.

### SELL signal — "overbought, expect a pullback"

When the fast SMA rises above the slow SMA by more than the threshold, the stock has rallied too far too
fast relative to its broader trend. The strategy signals SELL.

### No signal

When the two averages are close together (divergence within the threshold), the market is "normal" — no
actionable opportunity.

---

## Worked example

Imagine a stock with the following recent closing prices:

```
Day   Close
 1    100.00
 2    101.00
 3     99.50
 4     98.00
 5     97.00   ← recent dip
```

With `fast_period = 5` and `slow_period = 20`:

- **Fast SMA (last 5 days):** (100 + 101 + 99.5 + 98 + 97) / 5 = **99.10**
- **Slow SMA (last 20 days):** assume the previous 15 days averaged around 103, giving a 20-day SMA of
  roughly **101.80**

```
divergence = (99.10 - 101.80) / 101.80 = -0.0265 = -2.65%
```

With `threshold = 0.02` (2%), the divergence of -2.65% exceeds the threshold:

> **Signal: BUY** — the stock is oversold relative to its trend.

The **confidence** scales with how far the divergence exceeds the threshold. At exactly 2% divergence,
confidence is modest (~0.30). At 4% divergence it doubles. At 6%+ it caps at 1.0. This reflects the
intuition that deeper dips carry stronger reversion potential.

---

## When it works well

- **Liquid large-caps** (S&P 500 constituents) — mean reversion is most reliable in deep, liquid markets
  where price dislocations are quickly arbitraged away.
- **Stable macro environments** — sideways or gently trending markets where prices oscillate around a
  well-defined mean.
- **Earnings over-reactions** — a stock drops 5% on a small earnings miss, then recovers over the next
  week as analysts re-assess.

## When it struggles

- **Strong trends** — in a prolonged rally or sell-off, buying dips leads to catching falling knives.
  The fast SMA can stay below the slow SMA for months during a bear market.
- **Regime changes** — when the fundamental picture shifts (e.g. rising interest rates crushing
  growth stocks), the old "mean" is no longer valid and prices establish a new equilibrium.
- **Low-liquidity or speculative names** — small-caps and meme stocks can stay irrational indefinitely.

This is why the more advanced [Regime Mean Reversion](regime_mean_reversion.md) strategy adds market
regime detection before applying mean-reversion logic.

---

## Key concepts

### Simple Moving Average (SMA)

The arithmetic mean of the last *N* closing prices:

```
SMA(N) = (close[t] + close[t-1] + ... + close[t-N+1]) / N
```

| Period | Typical use |
|--------|-------------|
| 5–10 days | Very short-term / noise |
| 20 days | Roughly one trading month — "fast" baseline |
| 50 days | Medium-term trend — used as the "slow" reference |
| 200 days | Long-term trend — institutional benchmark |

A shorter period makes the SMA more responsive to recent price changes; a longer period smooths out noise
and reflects the broader trend.

### Divergence

The percentage difference between the fast and slow SMAs. It measures how far recent prices have strayed
from the longer-term trend:

- **Negative divergence** (fast < slow) — recent prices are below the trend = potential oversold condition.
- **Positive divergence** (fast > slow) — recent prices are above the trend = potential overbought condition.
- **Near zero** — prices are tracking the trend normally.

### Threshold

The minimum divergence required to trigger a signal. Setting it too low produces too many weak signals
(noise). Setting it too high misses genuine opportunities. Typical values range from 1% to 5%, depending
on the volatility of the instrument.

---

## Parameters

| Parameter | Example | What it controls |
|-----------|---------|------------------|
| `fast_period` | 20 | Days in the short SMA — how quickly it reacts to price changes |
| `slow_period` | 50 | Days in the long SMA — the "trend anchor" |
| `threshold` | 0.02 | Minimum divergence (2%) to trigger a signal |

### Tuning guidelines

- **More signals, lower conviction:** decrease the threshold (e.g. 0.01) or bring the periods closer
  together (e.g. 10/30).
- **Fewer signals, higher conviction:** increase the threshold (e.g. 0.04) or widen the period gap
  (e.g. 20/200).
- **Faster reaction:** shorter fast_period (e.g. 5–10 days).
- **Smoother trend reference:** longer slow_period (e.g. 100–200 days).
