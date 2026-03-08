Changelog
=========

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
