# Time-Series Momentum (TSMOM) Strategy

## The idea in one sentence

Whatever an asset has been doing for the past year, expect it to keep doing — buy if its 12-month return is positive, sell if it is negative.

## Financial background

Time-series momentum is the empirical anomaly documented by Moskowitz, Ooi & Pedersen (2012,
*Journal of Financial Economics*) across dozens of equity-index, currency, commodity, and
bond futures: the **sign** of an asset's trailing 12-month return predicts the sign of its
next-month return, and the effect is remarkably stable across asset classes and decades.

Unlike *cross-sectional* momentum (which ranks assets against each other and goes long the
winners / short the losers), time-series momentum looks at each asset in isolation. The
trade is purely a bet that recent direction persists.

Why it works:

- **Slow diffusion of information.** News and macro shifts take weeks to fully impound into
  prices, so trends emerge gradually rather than all at once.
- **Behavioural under-reaction followed by over-reaction.** Investors anchor on stale priors
  and only revise gradually, then chase performance once the move is undeniable — both
  effects extend the trend.
- **Risk-premium harvesting.** Persistent demand for hedging (e.g. equity index puts, FX
  carry trades) creates structural one-way pressure that trend-followers are paid to absorb.

TSMOM works best in **persistent directional regimes** — sustained bull markets, secular
bear markets, currency revaluations, commodity super-cycles. It bleeds during **range-bound
chop**, where mean-reversion strategies thrive.

This makes it structurally complementary to the mean-reversion engines in quaver: the two
profit in opposite environments and tend to be uncorrelated.

---

## How it works

The rule is intentionally minimal — there are no moving averages, no oscillators, no
pattern recognition. Just one number: the cumulative return over a trailing window.

For a closing-price series and a `lookback_period` (default `252` ≈ one trading year):

```
trailing_return = close[-1 - skip_period] / close[-lookback_period - skip_period] - 1
```

The optional `skip_period` (default `21` ≈ one trading month) drops the most recent month
from the window. This implements the classic **"12-1" momentum** convention popularised by
Asness, Moskowitz & Pedersen (2013) — it neutralises short-horizon reversal noise that
contaminates the most recent bars.

The **sign** of `trailing_return` becomes the trade direction.

---

## Signal logic

### BUY signal — "uptrend, expect persistence"

When the trailing return is positive (and exceeds `threshold` in magnitude), the asset has
been trending up. TSMOM bets the trend continues.

### SELL signal — "downtrend, expect persistence"

When the trailing return is negative (and exceeds `threshold` in magnitude), the asset has
been trending down. TSMOM emits SELL — meaning *short* if the engine has `allow_shorting`
enabled, otherwise *stay flat*.

### No signal

When the absolute trailing return is at or below `threshold`, the trend is too weak to
trade. The default `threshold = 0.0` means any non-zero return triggers a signal.

---

## Worked example

Assume `lookback_period = 252` (1 year), `skip_period = 21` (1 month), and a stock with the
following salient prices:

```
Index in series         Bar offset   Close
─────────────────────────────────────────────
close[-1]               today        108.00
close[-22]              ~1 mo ago    105.00   ← end of window (skip=21)
close[-274]             ~13 mo ago    90.00   ← start of window
```

The trailing window is `[-274, -22]` — i.e. the year that ended one month ago.

```
trailing_return = 105.00 / 90.00 - 1 = 0.1667 = +16.67%
```

With `threshold = 0.0`, `|0.1667| > 0`:

> **Signal: BUY** — the asset gained ~17% over the trailing window.

The **confidence** scales linearly with the magnitude of the return, capped at `1.0`. A 10%
return maps to confidence `1.00`; a 5% return maps to `0.50`; a 1% return maps to `0.10`.
This reflects the intuition that stronger trends provide stronger evidence of persistence.

---

## When it works well

- **Sustained directional regimes** — secular bull markets, prolonged bear markets,
  currency or commodity super-cycles. The 2009–2020 equity bull run and the 1980s bond bull
  run are textbook TSMOM environments.
- **Asset classes with strong serial correlation** — futures on equity indices, government
  bonds, major currencies, and broad commodities historically show 12-month autocorrelation
  of 0.05–0.10, which is small but exploitable across many contracts.
- **Diversified portfolios** — TSMOM is designed to be applied across many uncorrelated
  assets simultaneously; the per-asset edge is tiny but compounds via diversification.

## When it struggles

- **Range-bound or whipsawing markets** — when prices oscillate without committing to a
  direction, TSMOM repeatedly buys tops and sells bottoms. 2015–2016 in equities and most
  of 2023 in bonds are recent examples.
- **Sharp regime reversals** — turning points (e.g. the 2009 equity bottom, the 2022 bond
  collapse) catch trend-followers on the wrong side until the new trend is long enough to
  flip the trailing-return sign.
- **Single-asset deployment** — the per-asset hit rate is ~52–55%; the strategy relies on
  diversification across many uncorrelated trends to deliver smooth equity curves.

This is why TSMOM is best paired with a mean-reversion engine such as
[Mean Reversion](mean_reversion.md) or [Regime Mean Reversion](regime_mean_reversion.md):
the two have negatively correlated PnL profiles and together produce smoother returns than
either in isolation.

---

## Key concepts

### Trailing return

The cumulative percentage change of `close` over the lookback window. This is the **only**
input the strategy looks at — there are no moving averages, indicators, or pattern matches.

### Lookback period

How far back to measure the trend. The 12-month / 252-day window is the academic standard
because it balances enough history to dampen noise against enough recency to remain
relevant. Shorter windows (e.g. 60 days) make the strategy more reactive but noisier;
longer windows (e.g. 504 days) make it slower but more stable.

### Skip period (12-1 convention)

The most recent ~1 month of returns shows weak short-horizon **reversal** rather than
momentum. Skipping that final month — measuring the return from 13 months ago to 1 month
ago, hence "12-1" — historically improves Sharpe by ~0.1–0.2 across asset classes. Set
`skip_period = 0` to disable and use a flush-to-today window.

### Threshold

The minimum absolute trailing return required to trade. The default `0.0` means trade on
any signed return. Raising it (e.g. `0.05` = 5%) filters out weak trends at the cost of
fewer trading opportunities. Useful when transaction costs are high.

### Confidence scaling

The strategy emits `confidence = min(|trailing_return| / 0.10, 1.0)` — a 10% trailing
return is the saturation point. Downstream position-sizing logic (e.g. `size_by_risk()`)
can scale exposure by this confidence so that stronger trends earn larger positions.

---

## Parameters

| Parameter         | Default | What it controls |
|-------------------|---------|------------------|
| `lookback_period` | 252     | Bars over which to measure the trailing return (≈ 1 trading year) |
| `skip_period`     | 21      | Most-recent bars to drop from the window — 0 disables (≈ 1 trading month) |
| `threshold`       | 0.0     | Minimum absolute trailing return to trigger a signal |

### Tuning guidelines

- **More signals, faster reaction:** shorten `lookback_period` (e.g. 60–120) and/or set
  `threshold = 0.0`. Expect more whipsaws.
- **Fewer signals, higher conviction:** lengthen `lookback_period` (e.g. 504) and/or raise
  `threshold` (e.g. 0.05). Expect longer holding periods.
- **Disable the skip:** set `skip_period = 0` if you trust the most recent month or if your
  bars are higher-frequency (intraday) where the 12-1 convention is less applicable.
- **Risk-equalise across assets:** TSMOM signals are direction-only; pair the engine with
  volatility-targeted position sizing so each asset contributes equal risk to the
  portfolio.

---

## References

- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). *Time Series Momentum.*
  Journal of Financial Economics, 104(2), 228–250.
- Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). *Value and Momentum
  Everywhere.* Journal of Finance, 68(3), 929–985.