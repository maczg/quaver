"""quaver.strategies — strategy engines and infrastructure."""

# Import strategy modules to trigger @StrategyRegistry.register decorators
import quaver.strategies.mean_reversion  # noqa: F401
import quaver.strategies.regime_mean_reversion  # noqa: F401
import quaver.strategies.vsa_stopping_volume  # noqa: F401
import quaver.strategies.pairs_mean_reversion  # noqa: F401
