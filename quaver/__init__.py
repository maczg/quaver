"""quaver — quantitative signal generation and walk-forward backtesting.

``quaver`` provides the core abstractions for building, registering, and
executing trading strategy engines that produce point-in-time signals from
OHLCV candle data.

**Key concepts**

- **Strategy engines** — subclasses of :class:`BaseStrategy` (single-asset) or
  :class:`MultiAssetStrategy` (multi-asset / pairs) that implement
  :meth:`~BaseStrategy.validate_parameters` and :meth:`~BaseStrategy.compute`.
- **Signal output** — :class:`SignalOutput` (single-asset) and
  :class:`MultiAssetStrategyOutput` (multi-asset) carry the direction,
  confidence, and optional metadata produced by a ``compute()`` call.
- **Universe filtering** — :class:`UniverseFilter` declares which instrument
  sub-set a strategy engine should be applied to.
- **Shared types** — :class:`SignalDirection`, :class:`SignalStrength`,
  :class:`TimeFrame`, :class:`InstrumentType`, and :class:`EngineInfo` are
  pure-Python enums / dataclasses with no ORM dependency, making them safe to
  import from both the engine layer and the backend model layer.

**Typical usage**

.. code-block:: python

    from quaver import BaseStrategy, SignalOutput, SignalDirection

    @StrategyRegistry.register("my_engine")
    class MyEngine(BaseStrategy):
        def validate_parameters(self) -> None:
            ...

        def compute(self, candles, as_of):
            return SignalOutput(direction=SignalDirection.BUY, confidence=0.8)

.. rubric:: Public API

.. autosummary::

    BaseStrategy
    MultiAssetStrategy
    MultiAssetStrategyOutput
    SignalOutput
    UniverseFilter
    SignalDirection
    SignalStrength
    TimeFrame
    InstrumentType
    EngineInfo
"""

from quaver.types import (
    SignalDirection,
    SignalStrength,
    TimeFrame,
    InstrumentType,
    EngineInfo,
)
from quaver.strategies.base import (
    BaseStrategy,
    SignalOutput,
    UniverseFilter,
    MultiAssetStrategy,
    MultiAssetStrategyOutput,
)

__all__ = [
    "BaseStrategy",
    "MultiAssetStrategy",
    "MultiAssetStrategyOutput",
    "SignalOutput",
    "UniverseFilter",
    "SignalDirection",
    "SignalStrength",
    "TimeFrame",
    "InstrumentType",
    "EngineInfo",
]
