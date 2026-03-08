# Regime Mean Reversion Strategy

## The idea in one sentence

Only buy dips (or sell rips) when the market is actually trending and the historical odds of a reversal
are in your favour.

## Financial background

Plain [mean reversion](mean_reversion.md) has a well-known weakness: it buys falling knives in bear
markets and sells into runaway rallies. The reason is that it treats every market day the same — it does
not ask *"what kind of market are we in?"* before placing the trade.

This strategy solves that problem by adding two layers of intelligence on top of basic mean reversion:

1. **Regime classification** — it uses three technical indicators (ADX, Bollinger Band Width, Volume) to
   determine whether the market is trending, transitioning, ranging, or compressed. Signals are only
   generated in trending regimes, where mean-reversion dips and pops are most likely to be temporary.

2. **Probabilistic confirmation** — before acting on a dip or pop, the strategy asks: *"Historically,
   when this stock dipped by this much in this specific regime, how often did it bounce the next day?"*
   A signal is only generated if the historical probability and win/loss ratio both exceed configurable
   thresholds.

The strategy is ported from the QIMPOD5 research framework, where it was originally developed with an
additional VIX momentum filter. VIX filtering will be added in Phase 2 when a VIX data feed is available.

---

## How it works — overview

```
Price data (OHLCV)
    │
    ├─ Compute indicators (ADX, Bollinger Bands, Volume, SMAs, Returns)
    │
    ├─ Classify today's market regime (one of 10 categories)
    │
    ├─ Is the regime a trending one (UP or DOWN)?
    │   └─ No → no signal
    │
    ├─ Did today have a significant dip (for BUY) or pop (for SELL)?
    │   └─ No → no signal
    │
    ├─ Compute expanding-window probabilities:
    │   "In the entire history, when a similar dip happened in this same regime,
    │    how often did the stock recover the next day?"
    │
    ├─ Do the probabilities and win/loss ratio exceed thresholds?
    │   └─ No → no signal
    │
    └─ Emit BUY or SELL with confidence = regime-specific probability
```

---

## Step 1 — Technical indicators

The strategy computes several indicators from daily OHLCV bars. Each indicator captures a different
dimension of market behaviour.

### Average Directional Index (ADX)

**What it measures:** the *strength* of a trend, regardless of direction.

ADX was developed by J. Welles Wilder in the 1970s and is one of the most widely used trend-strength
indicators. It is derived from the Directional Movement system:

1. **+DM (Plus Directional Movement)** — how much today's high exceeds yesterday's high (upward pressure).
2. **-DM (Minus Directional Movement)** — how much today's low is below yesterday's low (downward pressure).
3. These are smoothed using **Wilder's smoothing** (a form of exponential moving average with factor 1/N)
   over 14 periods and normalized by the **Average True Range (ATR)** to produce **+DI** and **-DI**
   (directional indicators, 0–100 each).
4. The **DX** (Directional Index) measures the spread between +DI and -DI: `DX = |+DI - -DI| / (+DI + -DI) * 100`.
5. **ADX** is the smoothed average of DX over another 14 periods.

**Interpretation:**

| ADX value | Market state |
|-----------|-------------|
| < 20 | Weak or no trend — range-bound, choppy |
| 20–25 | Borderline — trend may be emerging or fading |
| 25–50 | Strong trend |
| 50–75 | Very strong trend |
| > 75 | Extreme trend (rare, often near exhaustion) |

ADX alone does not tell you the *direction* — only the *strength*. Direction is determined separately by
comparing the close price to the fast and slow SMAs.

#### Wilder's Smoothing

A specific exponential smoothing method where the decay factor is `(N-1)/N` instead of the usual `2/(N+1)`.
This makes it respond more slowly than a standard EMA of the same period, which is appropriate for
trend-strength indicators because it filters out short-term noise.

```
wilder_smooth[0..N-1] = sum of first N values
wilder_smooth[t] = wilder_smooth[t-1] * (N-1)/N + value[t]
```

#### True Range (TR)

The True Range accounts for overnight gaps, unlike a simple high-minus-low calculation:

```
TR = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
```

For example, if a stock closes at 100, gaps up to open at 105, and trades between 104 and 107, the
simple range is 3 (107-104) but the True Range is 7 (107-100), which better reflects the actual volatility
experienced by a trader holding overnight.

### Bollinger Bands and Bollinger Band Width (BBW)

**Bollinger Bands** are volatility envelopes placed 2 standard deviations above and below a 20-period SMA:

```
Middle band = SMA(20)
Upper band  = SMA(20) + 2 * StdDev(20)
Lower band  = SMA(20) - 2 * StdDev(20)
```

**Bollinger Band Width** normalises the distance between the bands:

```
BBW = (Upper - Lower) / Middle
```

BBW is a pure volatility measure. When volatility expands, the bands widen and BBW increases. When
volatility contracts, the bands squeeze and BBW decreases.

**Why BBW matters for regime detection:**

- **Expanding BBW** (BBW > its 5-period SMA *and* BBW > BBW from 3 bars ago) signals that volatility is
  actively increasing — the market is "breaking out" of a range and a trend may be strengthening.
- **Low BBW** (BBW in the bottom 20th percentile of its 250-day history) signals a volatility squeeze —
  the market is compressed and may be building energy for a move, but is not trending yet.

### Relative Volume

```
Relative Volume = Today's Volume / SMA(Volume, 20)
```

A value above 1.0 means today is busier than average. Volume confirms conviction — a breakout on heavy
volume is more meaningful than one on thin volume.

| Relative volume | Interpretation |
|-----------------|----------------|
| < 0.8 | Unusually quiet — market is disengaged |
| 0.8 – 1.2 | Normal activity |
| 1.2 – 2.0 | Above-average interest — confirms moves |
| > 2.0 | Very high activity — earnings, news, institutional activity |

### Simple Moving Averages (SMA 20 / SMA 50)

Two standard SMAs determine the **direction** of a trend:

- **Uptrend:** Close > SMA(20) > SMA(50) — the price is above both averages, and the short-term average
  is above the long-term average. All three are "stacked" upward.
- **Downtrend:** Close < SMA(20) < SMA(50) — the mirror image. All three are stacked downward.
- **Undefined:** any other arrangement — the trend direction is ambiguous.

### Daily Returns

```
return[t] = (close[t] - close[t-1]) / close[t-1]
```

The daily percentage change. A return of -0.03 means the stock dropped 3% in one day. Returns are used
both to detect dip/pop events and to compute historical probabilities.

---

## Step 2 — Regime classification

The strategy combines ADX, BBW, and Relative Volume to classify every bar into one of 10 market regimes.
Think of regimes as the "weather" of the market — you do not want to apply the same trading logic in a
hurricane as on a calm day.

### The 10 regimes

#### Trending regimes (ADX >= 21)

The market is in a directional trend. Mean-reversion dips are more likely to be temporary pullbacks within
the trend, which makes them safer to buy.

| Regime | Conditions | Meaning |
|--------|-----------|---------|
| **TREND_STRONG_UP** | ADX >= 21, BBW expanding, vol >= 1.2x, Close > SMA20 > SMA50 | Strong uptrend confirmed by rising volatility and heavy volume. Dips are high-conviction buying opportunities. |
| **TREND_STRONG_DOWN** | Same, but Close < SMA20 < SMA50 | Strong downtrend. Pops (short-term rallies) are high-conviction selling opportunities. |
| **TREND_STRONG_UNDEFINED** | Same, but direction ambiguous | Strong trend but direction is unclear — no signal generated. |
| **TREND_WEAK_UP** | ADX >= 21, but either BBW not expanding or vol < 1.2x. Upward direction. | The trend is present but not powerfully confirmed. Dips can still be bought, but with more caution. |
| **TREND_WEAK_DOWN** | Mirror of weak up. | Moderate downtrend. Pops can be sold. |
| **TREND_WEAK_UNDEFINED** | Weak trend, ambiguous direction. | No signal generated. |

#### Transition regimes (ADX between 20 and 21)

The market is on the boundary between trending and ranging. The strategy does not generate signals here —
the direction is too uncertain.

| Regime | Conditions | Meaning |
|--------|-----------|---------|
| **TRANSITION_CONFIRMED** | BBW expanding, vol >= 1.0x | A trend may be starting. Watch, but do not act yet. |
| **TRANSITION_WEAK** | BBW not expanding or vol < 1.0x | Ambiguous. Could go either way. |

#### Ranging regimes (ADX < 20)

The market is not trending. Mean reversion dips could become deeper drawdowns with no clear trend to
anchor a recovery.

| Regime | Conditions | Meaning |
|--------|-----------|---------|
| **COMPRESSION** | BBW in bottom 20th percentile, vol >= 1.0x | A "Bollinger Squeeze" — volatility is extremely low and may precede a sharp move. No signal. |
| **RANGE** | Otherwise | Choppy, directionless market. No signal. |

### Example: reading a regime

```
Date:       2025-03-15
Close:      $185.40
SMA(20):    $182.10
SMA(50):    $178.50
ADX:        28.5
BBW:        0.065 (expanding — above its 5-day SMA and above 3-bar-ago level)
Rel Volume: 1.35

→ ADX 28.5 >= 21    → trending
→ BBW expanding + vol 1.35 >= 1.2  → STRONG
→ Close $185.40 > SMA20 $182.10 > SMA50 $178.50  → UP

Regime: TREND_STRONG_UP
```

This stock is in a strong uptrend with expanding volatility and above-average volume. If it dips 2%+
tomorrow, the strategy will evaluate whether to buy.

---

## Step 3 — Expanding-window probability

Before generating a signal, the strategy checks if history supports the trade. It does this by scanning
every bar in the available history (typically ~500 days) and computing two conditional probabilities:

### Base probability (all regimes)

*"Across all historical days, when this stock dropped >= 2% in a single day, what fraction of the time
did it bounce at least 0.5% the next day?"*

This gives a baseline. If the stock almost never bounces after dips (e.g. it is a chronic decliner), then
even a favourable regime is not enough.

### Regime-specific probability

*"Restrict to only those days when the regime was TREND_STRONG_UP. Same question: after a >= 2% dip,
how often did it bounce >= 0.5% the next day?"*

This is the refined probability. If dips in this specific regime recover more reliably than dips in general,
we have a regime edge.

### Win/loss ratio

In addition to the probability, the strategy computes:

```
Win/Loss Ratio = Average winning return / Average losing return
```

A probability of 60% with a 1.5 win/loss ratio means: 6 out of 10 times you win, and when you win your
average gain is 50% larger than your average loss. This is a positive expected value bet.

### Expanding window

The computation uses an **expanding window** — it uses all available history up to (but not including) the
current bar. This avoids lookahead bias (the model does not peek at future data), while using as much
history as possible.

The outcome for each historical event is the return on the *next* day (t+1), not the same day. This
represents the realistic scenario: you observe a dip today, you buy at the close, and you evaluate the
result tomorrow.

### Minimum events

The strategy requires at least 12 qualifying events (by default) in both the base and regime-specific
histories before it trusts the probabilities. With fewer events, the estimated probability is too noisy
to be useful.

### Example

```
Historical scan of 480 days for AAPL in TREND_STRONG_UP:

Base (all regimes):
  Days with return <= -2%:  45 events
  Next-day recovery >= 0.5%: 29 times
  → Base probability: 29/45 = 64.4%
  → Base W/L ratio: 1.42

Regime-specific (TREND_STRONG_UP only):
  Days with return <= -2%:  18 events
  Next-day recovery >= 0.5%: 14 times
  → Regime probability: 14/18 = 77.8%
  → Regime W/L ratio: 1.65

Today: AAPL is in TREND_STRONG_UP and just dropped -2.3%.
All probability thresholds pass.
→ Signal: BUY, confidence = 0.778
```

---

## Step 4 — Signal generation

All of the following conditions must pass simultaneously for a signal to be emitted:

### BUY (uptrend regimes: TREND_WEAK_UP or TREND_STRONG_UP)

1. **Regime is an uptrend.** Dips in uptrends are pullbacks, not collapses.
2. **Today's return is a dip.** `return <= -return_threshold * (1 + safemargin)` (default: -2%).
3. **Base probability exceeds threshold.** The stock has a general tendency to bounce after dips.
4. **Regime probability exceeds threshold.** The bounce tendency is even stronger in the current regime.
5. **Win/loss ratio exceeds threshold.** The average win is sufficiently larger than the average loss.

### SELL (downtrend regimes: TREND_WEAK_DOWN or TREND_STRONG_DOWN)

Mirror of BUY: a pop (positive return) in a downtrend is interpreted as a temporary rally that will
reverse. All the same probability checks apply in the opposite direction.

### No signal cases

- Non-trending regimes (RANGE, COMPRESSION, TRANSITION) — the strategy abstains.
- Trending but undefined direction — the strategy abstains.
- Dip/pop not large enough — below the return threshold.
- Insufficient historical events — not enough data to trust the probabilities.
- Probabilities or W/L ratio below thresholds — the historical edge is not strong enough.

### Confidence

The signal's confidence value is the **regime-specific probability** (0.0 to 1.0). A confidence of 0.78
means *"in this regime, 78% of similar dips were followed by a bounce."*

---

## Worked example — full pipeline

**Stock:** MSFT, daily bars, 500-day history.

**Today's data:**
```
Close:       $415.20 (yesterday: $425.80)
High:        $420.50
Low:         $413.90
Volume:      45,200,000 (20-day avg: 33,000,000)
```

**Step 1 — Indicators:**
```
Daily return:    (415.20 - 425.80) / 425.80 = -2.49%
ADX(14):         31.2
SMA(20):         $420.50
SMA(50):         $408.30
Bollinger BBW:   0.072 (5-day SMA of BBW: 0.058, BBW 3 bars ago: 0.055)
Relative volume: 45.2M / 33M = 1.37
```

**Step 2 — Regime classification:**
```
ADX 31.2 >= 21                        → trending
BBW 0.072 > SMA 0.058 and > 0.055    → BBW expanding
Rel volume 1.37 >= 1.2               → volume confirmed
→ TREND_STRONG

Close $415.20 < SMA20 $420.50        → wait, close is BELOW SMA20...
                                        but SMA20 $420.50 > SMA50 $408.30
→ Close < SMA20 but SMA20 > SMA50    → direction = UNDEFINED

Regime: TREND_STRONG_UNDEFINED → no signal generated
```

The strategy does **not** buy here because even though there is a dip, the trend direction is ambiguous.
The price just crossed below the 20-day SMA, which could signal the start of a deeper correction rather
than a buyable pullback.

**Alternative scenario — if close were $421.50 (return -1.01%):**
```
Close $421.50 > SMA20 $420.50 > SMA50 $408.30 → TREND_STRONG_UP
But return -1.01% > -2% threshold → dip not large enough → no signal
```

**Alternative scenario — if close were $416.50 with SMA20 at $415.00:**
```
Close $416.50 > SMA20 $415.00 > SMA50 $408.30 → TREND_STRONG_UP
Return -2.18% <= -2% threshold → dip qualifies

Historical scan: base prob 61%, regime prob 73%, regime W/L 1.52
All thresholds pass.

→ BUY, confidence = 0.73
```

---

## The safety margin parameter

The `safemargin` parameter (default 0.0) adds a buffer to all thresholds. With `safemargin = 0.10` (10%):

- Return threshold becomes `-2% * 1.10 = -2.2%` — requires a deeper dip.
- Probability threshold becomes `50% * 1.10 = 55%` — requires higher conviction.
- W/L threshold becomes `1.3 * 1.10 = 1.43` — requires a better risk/reward profile.

Use this to make the strategy more conservative without changing each threshold individually.

---

## VIX filtering (Phase 2 — not yet implemented)

In the original QIMPOD5 research, the strategy included a **VIX momentum filter** as an additional gate:

### What is the VIX?

The **CBOE Volatility Index (VIX)** measures the market's expectation of 30-day volatility, derived from
S&P 500 option prices. It is often called the "fear gauge":

| VIX level | Market mood |
|-----------|-------------|
| < 15 | Complacent — low expected volatility |
| 15–20 | Normal |
| 20–30 | Elevated fear — uncertainty rising |
| 30–40 | High fear — significant market stress |
| > 40 | Panic (e.g. COVID crash, 2008 crisis) |

### How VIX filtering works

The idea is to avoid buying dips during periods of rising systemic fear. Even if a stock is in a strong
uptrend with high recovery probability, a spike in VIX suggests the broader market environment is
deteriorating and individual stock pullbacks may deepen.

The filter checks:

1. **VIX level** — is VIX below a maximum threshold (e.g. 30)?
2. **VIX momentum** — is VIX declining or stable (not spiking)?

If VIX is spiking above the threshold, all BUY signals are suppressed regardless of individual stock
conditions. SELL signals in downtrends may be enhanced (falling markets + rising VIX = strong sell).

This filter will be added in Phase 2 once a VIX data feed is integrated into the platform.

---

## Parameters

### Indicator parameters

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `adx_period` | 14 | Lookback for ADX calculation. Standard is 14. Shorter = more responsive, noisier. |
| `bb_period` | 20 | Bollinger Bands SMA period. Standard is 20. |
| `bb_std` | 2.0 | Standard deviations for the bands. 2.0 captures ~95% of price action. |
| `bbw_percentile_window` | 250 | Rolling window (~1 year) for determining "low" BBW via 20th percentile. |
| `bbw_sma_period` | 5 | SMA period for detecting BBW expansion (is BBW rising short-term?). |
| `bbw_lookback` | 3 | Bars to look back for BBW expansion confirmation. |
| `sma_fast` | 20 | Fast SMA for trend direction (~1 month). |
| `sma_slow` | 50 | Slow SMA for trend direction (~2.5 months). |
| `volume_sma_period` | 20 | Lookback for average volume (relative volume denominator). |

### Regime thresholds

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `adx_trend_threshold` | 21.0 | ADX above this = market is trending. |
| `adx_transition_low` | 20.0 | ADX in the [20, 21) zone = transition. |
| `volume_strong_threshold` | 1.2 | Relative volume for TREND_STRONG classification. |
| `volume_normal_threshold` | 1.0 | Relative volume for TRANSITION/COMPRESSION classification. |

### Signal thresholds

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `return_threshold` | 0.02 | Minimum daily drop (2%) to qualify as a "dip" for BUY, or daily rise for SELL. |
| `success_threshold` | 0.005 | Minimum next-day recovery (0.5%) to count as a "success" in probability computation. |
| `prob_threshold_base` | 0.50 | Minimum base probability (across all regimes) to proceed. |
| `prob_threshold_weak` | 0.50 | Minimum regime probability in TREND_WEAK regimes. |
| `prob_threshold_strong` | 0.50 | Minimum regime probability in TREND_STRONG regimes. |
| `winloss_threshold_weak` | 1.3 | Minimum win/loss ratio in TREND_WEAK regimes. |
| `winloss_threshold_strong` | 1.3 | Minimum win/loss ratio in TREND_STRONG regimes. |
| `safemargin` | 0.0 | Safety buffer applied multiplicatively to all thresholds. |
| `min_events` | 12 | Minimum historical qualifying events to trust the probability estimate. |

### Data parameters

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `candle_count` | 500 | Number of daily bars to request (~2 years). More data = more events for probability, but older data may be less relevant. |

### Tuning guidelines

- **More aggressive (more signals):** lower `return_threshold` to 0.015, lower probability thresholds to
  0.45, lower `min_events` to 8.
- **More conservative (fewer, higher-quality signals):** raise `return_threshold` to 0.03, raise
  probability thresholds to 0.60, set `safemargin` to 0.10.
- **For volatile stocks:** increase `return_threshold` (a 2% dip in a volatile stock is routine, not a
  signal-worthy event).
- **For stable blue-chips:** decrease `return_threshold` (a 1.5% dip in a stable stock may already be
  significant).
