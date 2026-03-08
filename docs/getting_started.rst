Getting Started
===============

Installation
------------

Install from source using pip:

.. code-block:: bash

   pip install -e ".[dev,docs]"

Quick Example
-------------

Run a single-asset mean-reversion backtest:

.. code-block:: python

   import pandas as pd
   from quaver.backtest import run_backtest

   # Load your OHLCV data
   candles = pd.read_csv("data.csv")

   result = run_backtest(
       engine_name="mean_reversion",
       parameters={
           "fast_period": 20,
           "slow_period": 50,
           "threshold": 0.02,
       },
       candles=candles,
       instrument_id="AAPL",
       initial_capital=10_000.0,
   )

   print(result.summary())

Run a multi-asset pairs backtest:

.. code-block:: python

   from quaver.backtest import run_multi_asset_backtest

   results = run_multi_asset_backtest(
       engine_name="pairs_mean_reversion",
       parameters={
           "instrument_a": "AAPL",
           "instrument_b": "MSFT",
           "spread_window": 60,
           "entry_z": 2.0,
           "exit_z": 0.5,
       },
       candles_map={
           "AAPL": candles_aapl,
           "MSFT": candles_msft,
       },
       initial_capital=10_000.0,
   )

   for iid, r in results.items():
       print(f"{iid}: {r.summary()}")

Available Strategies
--------------------

- ``mean_reversion`` — Dual moving-average mean reversion
- ``regime_mean_reversion`` — Regime-based probabilistic mean reversion
- ``vsa_stopping_volume`` — VSA stopping-volume reversal pattern
- ``pairs_mean_reversion`` — Statistical arbitrage pairs trading
