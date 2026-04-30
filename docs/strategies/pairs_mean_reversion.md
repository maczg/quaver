# Pairs Mean Reversion Strategy

## The idea in one sentence

Two related stocks usually move together — when their price gap stretches abnormally far, bet that it snaps back.

## Financial background

Pairs trading is the original **statistical arbitrage** strategy, popularised by Nunzio
Tartaglia's quant team at Morgan Stanley in the mid-1980s. The premise: if two assets share
a common economic driver — same sector, same supply chain, same macro exposure — their
prices should track each other within a stable, mean-reverting band. When the gap (the
**spread**) widens beyond its historical norm, one leg is temporarily mispriced relative to
the other. Selling the rich leg and buying the cheap leg captures the convergence.

What makes it attractive:

- **Market-neutral by construction.** Long one leg, short the other → broad market moves
  cancel out. The PnL depends on the spread converging, not on equity-market direction.
- **Self-financing.** The short proceeds fund the long, so capital usage is low (modulo
  margin requirements).
- **Sector-bet hedged.** Trading two oil majors against each other is not a bet on oil; it
  is a bet on relative pricing.

The classic textbook pair is two integrated oil companies (Exxon vs. Chevron), two large
US banks (JPMorgan vs. Bank of America), or two index ETFs tracking the same underlying
(SPY vs. IVV). Cointegration tests (Engle–Granger, Johansen) are typically used to *select*
pairs; the strategy below assumes you have already chosen a pair worth trading.

Why it can stop working:

- **Structural break.** A merger, regulatory shock, or business-model divergence permanently
  re-prices one leg. The spread is no longer mean-reverting — you are now holding a directional
  bet.
- **Liquidity vacuum.** If one leg becomes hard to borrow, the short side is forced to cover
  at the worst possible time.
- **Crowding.** Famous pairs (Pepsi/Coca-Cola, Ford/GM) are arbitraged so aggressively that
  the edge is razor-thin and erodes during quiet markets.

---

## How it works

The strategy maintains a rolling z-score of the **price spread** between two instruments
``A`` and ``B``:

```
spread[t]  = close_A[t] - close_B[t]
z_score[t] = (spread[t] - rolling_mean(spread, window))
             / rolling_std(spread, window)
```

The z-score measures how unusual today's gap is relative to the recent past. A z-score of
``+2`` means the spread is two standard deviations above its rolling mean — i.e. extreme
divergence in one direction.

Two thresholds drive the trade:

- **`entry_z`** — magnitude required to *open* a paired position (default `2.0`).
- **`exit_z`** — magnitude tight enough to *close* the position (default `0.5`).

`exit_z` must be strictly less than `entry_z`, so positions only close after meaningful
convergence.

---

## Signal logic

The two legs are always traded **atomically**: every signal emission contains both legs in
one `MultiAssetStrategyOutput` so the engine applies them as a single transaction.

### Entry — spread too high (`z_score > +entry_z`)

A is rich, B is cheap. The strategy expects the spread to fall.

> SELL A, BUY B

### Entry — spread too low (`z_score < -entry_z`)

A is cheap, B is rich. The strategy expects the spread to rise.

> BUY A, SELL B

### Exit — convergence (`|z_score| < exit_z`)

The spread has tightened back inside the inner band. The strategy closes both legs
simultaneously regardless of direction.

> CLOSE A, CLOSE B

### No signal — middle band (`exit_z ≤ |z_score| ≤ entry_z`)

The spread is wider than the exit threshold but not yet at the entry threshold. The
strategy holds existing positions and does not open new ones — a hysteresis band that
prevents whipsaw.

---

## Worked example

Pair: AAPL and MSFT. Parameters: `spread_window = 60`, `entry_z = 2.0`, `exit_z = 0.5`.

Today's snapshot:

```
close_AAPL = 180.00
close_MSFT = 420.00
spread     = 180.00 - 420.00 = -240.00

rolling mean(spread, 60) = -250.00
rolling std (spread, 60) =    4.00

z_score = (-240.00 - (-250.00)) / 4.00 = 10.00 / 4.00 = +2.50
```

Today's spread (`-240`) is unusually high relative to the typical `-250 ± 4` band. AAPL is
rich vs. MSFT.

Because `2.50 > entry_z = 2.0`:

> **Signal: SELL AAPL, BUY MSFT** — bet the spread drops back toward `-250`.

Two weeks later, the spread has compressed back to `-249` and the z-score is now `+0.25`.
Because `|0.25| < exit_z = 0.5`:

> **Signal: CLOSE AAPL, CLOSE MSFT** — convergence achieved, lock in the gain.

The **confidence** scales linearly with `|z_score| / entry_z`, capped at `1.0`. A z-score
of exactly `entry_z` triggers confidence `1.0`; weaker entries from the schema are not
possible because `|z_score| ≤ entry_z` would not have triggered an entry in the first
place.

---

## When it works well

- **Cointegrated pairs in stable regimes.** Two firms in the same industry, with similar
  business mix, traded on the same exchange. The spread oscillates around a stable mean
  for years.
- **Liquid large-caps.** Tight bid-ask, easy borrow, low slippage. Pairs trading is
  fundamentally a high-turnover, low-edge game; transaction costs eat returns fast.
- **Mid-volatility environments.** Enough movement to generate `2σ` events, not so much
  that the rolling mean shifts faster than positions can converge.

## When it struggles

- **Structural breaks.** A 2013-style spin-off (eBay/PayPal) or 2023-style banking shock
  (SVB) permanently re-prices one leg. The "mean" the strategy assumes no longer exists.
- **High-volatility crashes.** Correlations break down — *every* spread widens at the same
  time, and the rolling-window estimates lag the new regime.
- **Asymmetric borrow costs.** If shorting one leg is expensive (high HTB fee), the
  realised PnL is much worse than the model PnL.
- **Single-pair concentration.** A diversified pairs book of 20+ uncorrelated pairs has a
  far smoother equity curve than betting on one pair, even a "perfect" one.

---

## Key concepts

### Spread

The **price difference** between the two legs (not a ratio, not log prices). This works
when the two instruments have similar nominal price levels. For pairs with very different
price scales (e.g. BRK.A vs. SPY), a log-spread or hedge-ratio formulation is more
appropriate — outside the scope of this engine.

### Rolling z-score

Standardises the spread against its own recent history rather than a fixed mean. This
adapts gradually to slow drifts in the relationship while still flagging sharp dislocations.

### Entry / exit hysteresis

Keeping `exit_z < entry_z` creates a **dead band** between `±exit_z` and `±entry_z` where
the strategy neither opens nor closes. Without this, every wiggle around the threshold
would trigger a re-entry. Typical ratios: `entry_z = 2.0` with `exit_z = 0.0` (close at
the mean) or `exit_z = 0.5` (close just inside the mean).

### Atomic two-leg signals

Both legs always flip together, packaged in a single `MultiAssetStrategyOutput`. This
ensures the engine never holds a half-built pair (long one leg without the offsetting
short), which would convert the trade into an unintended directional bet.

### Pair selection (out of band)

This engine assumes the user has already identified a pair worth trading. In production,
pair selection typically uses:

- **Pearson / Kendall correlation** — quick first filter (`> 0.7`).
- **Cointegration tests** — Engle–Granger or Johansen — to confirm a stable long-run
  relationship.
- **Half-life of mean reversion** — Ornstein-Uhlenbeck fit; reject pairs with half-life
  much longer than the holding-period budget.

---

## Parameters

| Parameter       | Default  | What it controls |
|-----------------|----------|------------------|
| `instrument_a`  | required | First leg's identifier (must match a key in `candles_map`) |
| `instrument_b`  | required | Second leg's identifier (must differ from `instrument_a`) |
| `spread_window` | 60       | Rolling window for the z-score mean and stdev |
| `entry_z`       | 2.0      | Z-score magnitude required to open a paired position |
| `exit_z`        | 0.5      | Z-score magnitude required to close an open position (must be `< entry_z`) |

### Tuning guidelines

- **More trades, smaller edge:** lower `entry_z` (e.g. `1.5`). Expect more whipsaws and
  more sensitivity to transaction costs.
- **Fewer trades, higher conviction:** raise `entry_z` (e.g. `2.5–3.0`). Holding periods
  lengthen; idle capital between signals is the cost.
- **Faster adaptation:** shorten `spread_window` (e.g. `20–40`). The mean tracks regime
  shifts more responsively but is noisier.
- **Slower, smoother:** lengthen `spread_window` (e.g. `120–252`). The mean is stable but
  takes longer to recognise a structural break.
- **Tighter exits:** drop `exit_z` toward `0.0` to ride convergence all the way to the
  mean. Trades off larger per-trade gain against more time exposed.

---

## References

- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). *Pairs Trading: Performance of a
  Relative-Value Arbitrage Rule.* Review of Financial Studies, 19(3), 797–827.
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis.* Wiley.
- Engle, R. F., & Granger, C. W. J. (1987). *Co-integration and Error Correction:
  Representation, Estimation, and Testing.* Econometrica, 55(2), 251–276.
