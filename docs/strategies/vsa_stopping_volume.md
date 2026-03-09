# VSA Stopping Volume Strategy

## The idea in one sentence

When a falling stock meets a wall of volume but the candle stays narrow, the selling pressure is being absorbed — a reversal may be near.

## Financial background

Volume Spread Analysis (VSA) is a method developed from the work of Richard Wyckoff (1920s) and later refined
by Tom Williams. The core idea is that professional operators leave footprints in the relationship between
volume and price spread (high minus low). By reading these footprints, a trader can detect accumulation
(smart money buying) and distribution (smart money selling) before the price visibly turns.

**Stopping Volume** is a specific VSA pattern that appears near the end of a downtrend:

- A stock has been falling (downtrend).
- A candle appears with **very high volume** but a **narrow spread** (the range between high and low is small
  relative to recent bars).
- The candle closes **above its midpoint** (not on its lows), despite being a down candle.

This combination tells a story: massive selling hit the market, but equally massive buying absorbed it. The
narrow spread means the sellers could not push the price down despite the volume. The close above the midpoint
confirms that buyers stepped in. The selling pressure has been "stopped."

This is similar to what institutional trading desks call **absorption** — large limit buy orders sitting below
the market that soak up sell orders without letting the price fall further.

The strategy also detects the **symmetric SELL pattern**: in an uptrend, a narrow-spread bull candle with
high volume and a close below the midpoint suggests distribution — smart money selling into retail buying
enthusiasm.

---

## How it works

The strategy computes four quantitative features from OHLCV data, then checks for the stopping-volume
pattern on the latest candle.

### Computed features

**Spread:** The range of each candle: `High - Low`.

**Close Position:** Where the close sits within the candle's range:

```
close_position = (Close - Low) / (High - Low)
```

A value of 0.0 means closing on the low; 1.0 means closing on the high; 0.5 is the midpoint.

**Relative Volume:** Today's volume divided by the 20-period SMA of volume:

```
vol_rel = Volume / SMA(Volume, 20)
```

A value of 2.0 means today's volume is twice the recent average — unusually heavy activity.

**Relative Spread:** Today's spread divided by the 20-period SMA of spread:

```
spread_rel = Spread / SMA(Spread, 20)
```

A value below 0.7 means today's candle is significantly narrower than recent bars.

### Trend filter

A simple moving average of the close price determines the local trend:

- **Downtrend:** Close < SMA(Close, 20)
- **Uptrend:** Close > SMA(Close, 20)

### Volume-Spread Matrix

The strategy classifies bars into states based on relative volume and relative spread:

| State | Condition | Meaning |
|-------|-----------|---------|
| **healthy_move** | vol_rel > 1.5 and spread_rel > 1.3 | High volume + wide spread = genuine directional move |
| **absorption_trap** | vol_rel > 1.5 and spread_rel < 0.7 | High volume + narrow spread = absorption / stopping volume |
| **no_supply_or_demand** | vol_rel < 0.7 and spread_rel > 0.8 | Low volume + normal/wide spread = no conviction |

The `absorption_trap` state is the key pattern — it provides a bonus to signal confidence.

---

## Signal logic

### BUY signal — "stopping volume at the bottom"

All of these must be true simultaneously:

1. **Downtrend.** Close is below the trend SMA — the stock is in a local decline.
2. **Bear candle.** Close < Open — the candle is red (down).
3. **High volume.** Relative volume exceeds `stopping_vol_rel` (default 2.0x) — heavy activity.
4. **Narrow spread.** Relative spread is below `spread_small` (default 0.7x) — the candle is compressed.
5. **Not closing on lows.** Close position is above `buy_close_pos_min` (default 0.4) — buyers prevented
   a close near the bottom.

**Reading:** Sellers dumped into the market but buyers absorbed everything. The narrow spread despite huge
volume means the price could not be pushed lower. A reversal is building.

### SELL signal — "distribution at the top"

The mirror image of the BUY pattern:

1. **Uptrend.** Close is above the trend SMA.
2. **Bull candle.** Close > Open — the candle is green (up).
3. **High volume.** Relative volume exceeds `stopping_vol_rel`.
4. **Narrow spread.** Relative spread is below `spread_small`.
5. **Not closing on highs.** Close position is below `sell_close_pos_max` (default 0.6) — sellers
   prevented a close near the top.

**Reading:** Buyers pushed prices up but smart money distributed (sold) into the rally. The narrow spread
despite heavy volume means the buying pressure is being met with selling. A pullback may follow.

### No signal

- No downtrend or uptrend (close equals the trend SMA).
- Volume is not high enough (below the stopping volume threshold).
- Spread is not narrow enough (a wide spread means it was a genuine directional move, not absorption).
- Close position does not match the pattern (closing on lows in a BUY setup means sellers won).

### Confidence

Confidence is composed of two parts:

1. **Volume score:** Scales from 0.5 (at the threshold) to 1.0 as relative volume increases. More volume
   means more conviction that absorption is occurring.
2. **Absorption bonus:** +0.1 if the bar is classified as `absorption_trap` in the volume-spread matrix.

```
vol_score = min((vol_rel - threshold) / threshold / 2 + 0.5, 1.0)
confidence = min(vol_score + absorption_bonus, 1.0)
```

---

## Worked example

**Stock:** AAPL, daily bars, 25 recent candles.

**Today's data:**
```
Open:    $182.50
High:    $183.10
Low:     $181.80
Close:   $182.00
Volume:  125,000,000 (20-day avg: 55,000,000)
```

**Step 1 — Features:**
```
Spread:          183.10 - 181.80 = 1.30
Close position:  (182.00 - 181.80) / (183.10 - 181.80) = 0.15 / 1.30 = 0.154
Relative volume: 125M / 55M = 2.27
Relative spread: 1.30 / SMA(spread, 20) = 1.30 / 2.10 = 0.619
Trend SMA(20):   $184.50
```

**Step 2 — Trend:**
```
Close $182.00 < Trend SMA $184.50 → downtrend ✓
```

**Step 3 — Pattern check:**
```
Bear candle:    Close $182.00 < Open $182.50 → ✓
High volume:    vol_rel 2.27 > stopping_vol_rel 2.0 → ✓
Narrow spread:  spread_rel 0.619 < spread_small 0.7 → ✓
Close position: 0.154 < buy_close_pos_min 0.4 → ✗ (closing too close to the low)

→ No signal — close position is too low.
```

The close position check failed. The candle closed near its low, meaning sellers still had control despite
the volume. This is not absorption — it is a capitulation-style bar where supply overwhelmed demand.

**Alternative — if close were $182.40:**
```
Close position: (182.40 - 181.80) / (183.10 - 181.80) = 0.462

All checks pass:
  Downtrend ✓, Bear candle ✓, vol_rel 2.27 > 2.0 ✓,
  spread_rel 0.619 < 0.7 ✓, close_pos 0.462 > 0.4 ✓

Matrix state: vol_rel 2.27 > 1.5 and spread_rel 0.619 < 0.7 → absorption_trap
Absorption bonus: +0.1

vol_score = min((2.27 - 2.0) / 2.0 / 2.0 + 0.5, 1.0) = min(0.5675, 1.0) = 0.5675
confidence = min(0.5675 + 0.1, 1.0) = 0.6675

→ BUY, confidence = 0.67
```

---

## When it works well

- **End of pullbacks in uptrends** — a stock in a longer-term uptrend dips temporarily, and stopping volume
  marks the bottom of the dip.
- **Capitulation bottoms in liquid stocks** — large-cap stocks that sell off sharply and find institutional
  support.
- **Stocks with consistent volume patterns** — the relative volume indicator is most meaningful when the
  stock has regular trading activity.

## When it struggles

- **Low-liquidity stocks** — erratic volume makes relative volume readings unreliable.
- **News-driven gaps** — a stock that gaps down on bad news with high volume may not be showing absorption.
  The high volume could be genuine panic selling, and the gap removes spread context.
- **Extended bear markets** — stopping volume signals in a prolonged decline can be premature. The "absorption"
  may be temporary buying that gets overwhelmed by the next wave of selling.

---

## Key concepts

### Volume Spread Analysis (VSA)

A methodology for reading the supply-demand balance by analysing the relationship between volume
(how much) and spread (how far price moved). The premise: price moves on the back of professional activity,
and professionals leave their footprints in the volume-spread relationship.

### Stopping Volume

High volume with a narrow spread and a close not on the lows. Indicates that selling is being absorbed by
large buy orders. Named "stopping" because the selling pressure is being "stopped" — it cannot drive prices
lower despite the heavy turnover.

### Absorption

When large limit orders (typically institutional) absorb incoming market orders without allowing the price
to move. On a chart, this appears as high volume but minimal price movement (narrow spread). Absorption
at lows suggests accumulation (buying); at highs, it suggests distribution (selling).

### Close Position

Where the closing price falls within the candle's range. A close near the high (close_position > 0.7)
shows buyers dominated the session. A close near the low (close_position < 0.3) shows sellers dominated.
For stopping volume, a mid-range close (0.4–0.7) is critical — it shows neither side fully won, which
is the hallmark of absorption.

---

## Parameters

### Lookback parameters

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `sma_window` | 20 | SMA window for normalising volume and spread |
| `trend_sma` | 20 | SMA window for the local trend filter on close |

### Volume-Spread Matrix thresholds

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `vol_high` | 1.5 | Relative volume threshold for "high volume" (matrix classification) |
| `vol_low` | 0.7 | Relative volume threshold for "low volume" (matrix classification) |
| `spread_big` | 1.3 | Relative spread threshold for "wide spread" |
| `spread_small` | 0.7 | Relative spread threshold for "narrow spread" (used in pattern checks) |

### Stopping-volume pattern thresholds

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `stopping_vol_rel` | 2.0 | Minimum relative volume required for the stopping-volume pattern |
| `buy_close_pos_min` | 0.4 | Minimum close position for BUY (rejects candles closing on their lows) |
| `sell_close_pos_max` | 0.6 | Maximum close position for SELL (rejects candles closing on their highs) |

### Pattern toggles

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `enable_buy` | true | Enable BUY stopping-volume signals |
| `enable_sell` | true | Enable SELL distribution signals |

### Tuning guidelines

- **More signals, lower conviction:** lower `stopping_vol_rel` to 1.5, widen close position range
  (`buy_close_pos_min` to 0.3, `sell_close_pos_max` to 0.7).
- **Fewer signals, higher conviction:** raise `stopping_vol_rel` to 2.5, tighten `spread_small` to 0.5
  (requires even narrower spreads).
- **BUY only:** set `enable_sell` to false — useful if you only want to detect bottoming patterns.
- **Shorter trend context:** reduce `trend_sma` to 10 for faster trend detection.
- **Longer normalisation window:** increase `sma_window` to 30 for smoother volume/spread baselines (reduces
  noise but may miss short-lived volume spikes).