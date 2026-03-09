Changelog
=========

v0.1.2 — 2026-03-09
--------------------

Features
^^^^^^^^

- Add exit rules (stop-loss, take-profit, trailing stop) to backtest engine
  (``4ad641f``)

  - ``ExitReason`` enum: SIGNAL, STOP_LOSS, TAKE_PROFIT, TRAILING_STOP, END_OF_DATA
  - ``ExitRules`` frozen dataclass with percentage-based global rules
  - Per-trade metadata overrides for absolute price levels
  - Exit checks run each bar before ``strategy.compute()`` using high/low prices
  - Pessimistic priority: stop-loss > trailing stop > take-profit
  - Fill at trigger price level, not bar close
  - Fully backward-compatible (all defaults disabled)

- Add risk-based position sizing utility ``size_by_risk()`` (``4ad641f``)

- Add ``get_parameter_schema()`` to all strategies missing it (``a8748bb``)

  - MeanReversionStrategy, BreakoutConsolidationStrategy,
    PullbackTrendStrategy, ReversalSupportStrategy,
    PairsMeanReversionStrategy

- Add breakout, pullback, and reversal strategies with indicators (``aa72385``)

  - ``breakout_consolidation`` — breakout from low-volatility consolidation
  - ``pullback_trend`` — trend continuation on pullback to short-term MA
  - ``reversal_support`` — counter-trend reversal at support with RSI confirmation
  - 20+ pure-NumPy technical indicators in ``quaver.strategies.indicators``

Documentation
^^^^^^^^^^^^^

- Create 6 feature showcase Jupyter notebooks (``4ad641f``)

  - ``01_indicators_showcase`` — all 20+ indicators with visualizations
  - ``02_trading_costs`` — commission and slippage impact comparison
  - ``03_exit_rules`` — stop-loss, take-profit, trailing stop demos
  - ``04_position_sizing`` — fixed vs risk-based sizing
  - ``05_all_strategies`` — all 7 strategy engines + pairs trading
  - ``06_backtest_metrics`` — all 19 BacktestResult metrics

- Integrate strategy guides into Sphinx documentation (``0de129d``)

Bug Fixes
^^^^^^^^^

- Fix close-and-reverse in backtest engines and numpy scalar leak (``b8b7bc3``)
- Fix version 0.1.1 in pyproject.toml (``54493c7``)

Chore
^^^^^

- Add Makefile and fix lint/format issues to pass all checks (``ce50e6b``)
- Update IntelliJ project config for Python 3.12 SDK and Black (``57f34db``)

v0.1.0 — 2026-03-08
--------------------

Initial release of **pyquaver** (``import quaver``).

Features
^^^^^^^^

- Implement quaver library with 4 strategy engines, backtest engine, and test
  suite (``ae334e4``)

  - ``mean_reversion`` — dual moving-average mean reversion
  - ``regime_mean_reversion`` — regime-based probabilistic mean reversion
  - ``vsa_stopping_volume`` — VSA stopping-volume reversal pattern
  - ``pairs_mean_reversion`` — statistical arbitrage pairs trading
  - Single-asset and multi-asset walk-forward backtest engines
  - Portfolio tracker with long/short support
  - ``BacktestResult`` with equity curve, Sharpe, drawdown, profit factor

- PyPI release setup: LICENSE, README, ``project.urls``, optional dependency
  groups (``289077b``)

- Add example Jupyter notebook with yfinance-backed end-to-end backtests

Documentation
^^^^^^^^^^^^^

- Set up Sphinx project with autodoc, rST docstrings, and Read the Docs theme
  (``130650b``)
- Getting started guide with PyPI install instructions

CI/CD
^^^^^

- GitHub Actions workflows for CI (lint + test), Sphinx docs deploy to GitHub
  Pages, and PyPI publishing via trusted publisher (``b3ac757``)
- Dev builds auto-published to TestPyPI on push to ``develop``

Bug Fixes
^^^^^^^^^

- Rename PyPI distribution to ``pyquaver`` to avoid name conflict (``8cc98de``)
- Resolve all ruff lint errors across codebase (``14b5eef``)
- Resolve all mypy strict mode errors with full type annotations (``23570b5``)

Style
^^^^^

- Apply ``ruff format`` to entire codebase (``1180af6``)
