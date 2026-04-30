Indicators
==========

The :mod:`quaver.strategies.indicators` module provides a pure-NumPy library of
technical-analysis primitives shared by the bundled strategy engines.  All
functions accept one-dimensional :class:`numpy.ndarray` inputs and return arrays
of the same length, padding leading positions with ``NaN`` when insufficient
history is available.

.. note::

   The indicator API has no dependency on :mod:`pandas`, :mod:`talib`, or any
   other charting library.  This keeps the strategy layer light and
   deterministic across platforms.

Overview
--------

The 21 indicators fall into five families.

Trend / smoothing
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :func:`~quaver.strategies.indicators.sma`
     - Simple Moving Average — arithmetic mean over a rolling window.
   * - :func:`~quaver.strategies.indicators.ema`
     - Exponential Moving Average — weighted toward recent values.
   * - :func:`~quaver.strategies.indicators.wilder_smooth`
     - Wilder smoothing — slower EMA variant used in ADX / RSI.
   * - :func:`~quaver.strategies.indicators.macd`
     - Moving Average Convergence Divergence (MACD line, signal, histogram).

Volatility / range
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :func:`~quaver.strategies.indicators.true_range`
     - Per-bar True Range (max of three high / low / prev-close ranges).
   * - :func:`~quaver.strategies.indicators.atr`
     - Average True Range — Wilder-smoothed True Range.
   * - :func:`~quaver.strategies.indicators.bollinger_bands`
     - Upper, middle, and lower Bollinger Bands.
   * - :func:`~quaver.strategies.indicators.bollinger_band_width`
     - Normalised band width — proxy for volatility regime.
   * - :func:`~quaver.strategies.indicators.donchian`
     - Donchian channel — rolling max / min of high / low.
   * - :func:`~quaver.strategies.indicators.keltner`
     - Keltner channel — EMA ± ATR multiple.

Momentum / oscillators
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :func:`~quaver.strategies.indicators.rsi`
     - Relative Strength Index — 0–100 momentum oscillator.
   * - :func:`~quaver.strategies.indicators.adx`
     - Average Directional Index — trend-strength magnitude.
   * - :func:`~quaver.strategies.indicators.stochastic`
     - Stochastic %K and %D oscillators.
   * - :func:`~quaver.strategies.indicators.cci`
     - Commodity Channel Index — deviation from typical price.

Volume / flow
~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :func:`~quaver.strategies.indicators.obv`
     - On-Balance Volume — cumulative signed volume.
   * - :func:`~quaver.strategies.indicators.vwap`
     - Volume-Weighted Average Price (rolling).
   * - :func:`~quaver.strategies.indicators.volume_relative`
     - Current volume as a multiple of recent average.

Statistical helpers
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :func:`~quaver.strategies.indicators.daily_returns`
     - Bar-over-bar percentage returns.
   * - :func:`~quaver.strategies.indicators.rolling_max`
     - Rolling maximum over a window.
   * - :func:`~quaver.strategies.indicators.rolling_min`
     - Rolling minimum over a window.
   * - :func:`~quaver.strategies.indicators.rolling_percentile`
     - Rolling quantile of a series.

Conventions
-----------

- **NaN padding** — every indicator pads leading positions with ``NaN`` until
  enough history exists.  Strategy code should filter or guard against ``NaN``
  before making decisions.
- **No look-ahead** — indicators only consume values up to and including the
  current index; the engine layer is responsible for excluding the current bar
  when point-in-time evaluation is required.
- **Float64 throughout** — inputs are expected to be ``np.float64`` arrays.
  Pass ``df["close"].to_numpy(dtype=float)`` from a pandas DataFrame.

Reference
---------

.. automodule:: quaver.strategies.indicators
   :members:
   :undoc-members:
   :show-inheritance:
