"""quaver.strategies — strategy engines and infrastructure.

This package exposes the base classes, registry, and concrete engine
implementations that make up the quaver strategy layer.

**Auto-registration**

Importing this package is the single step required to populate
:class:`~quaver.strategies.registry.StrategyRegistry` with all built-in
engines.  Each sub-module listed below is imported here solely to execute its
:func:`@StrategyRegistry.register <quaver.strategies.registry.StrategyRegistry.register>`
decorator calls as a side-effect; no public names are re-exported from those
modules at this level.

Built-in engines registered on import:

- ``quaver.strategies.mean_reversion``
- ``quaver.strategies.regime_mean_reversion``
- ``quaver.strategies.vsa_stopping_volume``
- ``quaver.strategies.pairs_mean_reversion``
- ``quaver.strategies.breakout_consolidation``
- ``quaver.strategies.pullback_trend``
- ``quaver.strategies.reversal_support``

After importing this package the full list of registered engine names is
available via
:meth:`~quaver.strategies.registry.StrategyRegistry.list_engines`.
"""

# Import strategy modules to trigger @StrategyRegistry.register decorators
import quaver.strategies.mean_reversion  # noqa: F401
import quaver.strategies.regime_mean_reversion  # noqa: F401
import quaver.strategies.vsa_stopping_volume  # noqa: F401
import quaver.strategies.pairs_mean_reversion  # noqa: F401
import quaver.strategies.breakout_consolidation  # noqa: F401
import quaver.strategies.pullback_trend  # noqa: F401
import quaver.strategies.reversal_support  # noqa: F401
