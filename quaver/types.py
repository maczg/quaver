"""Shared pure-Python types for quaver.

These enums and dataclasses have NO SQLAlchemy dependency.
They are the single source of truth for enum values shared between
quaver engines and any backend models.

.. rubric:: Exported types

- :class:`SignalDirection`
- :class:`SignalStrength`
- :class:`TimeFrame`
- :class:`InstrumentType`
- :class:`EngineInfo`
"""

import enum
from dataclasses import dataclass
from typing import Any


class SignalDirection(str, enum.Enum):
    """Direction of a trading signal emitted by a strategy engine.

    Members:

    - ``BUY`` — enter or increase a long position.
    - ``SELL`` — enter or increase a short position.
    - ``HOLD`` — maintain the current position without change.
    - ``CLOSE`` — exit an open position entirely.
    """

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class SignalStrength(str, enum.Enum):
    """Qualitative strength of a trading signal.

    Members:

    - ``STRONG`` — high-conviction signal; engine is highly confident.
    - ``MODERATE`` — medium-conviction signal; conditions partially met.
    - ``WEAK`` — low-conviction signal; marginal or early setup.
    """

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class TimeFrame(str, enum.Enum):
    """Canonical candle time-frame identifiers.

    Members:

    - ``M1``  — 1-minute bars.
    - ``M5``  — 5-minute bars.
    - ``M15`` — 15-minute bars.
    - ``H1``  — 1-hour bars.
    - ``H4``  — 4-hour bars.
    - ``D1``  — daily bars.
    - ``W1``  — weekly bars.
    """

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class ExitReason(str, enum.Enum):
    """Reason a position was closed during a backtest.

    Members:

    - ``SIGNAL`` — closed by a strategy signal (SELL, CLOSE, or reversal).
    - ``STOP_LOSS`` — closed by a stop-loss trigger.
    - ``TAKE_PROFIT`` — closed by a take-profit trigger.
    - ``TRAILING_STOP`` — closed by a trailing-stop trigger.
    - ``END_OF_DATA`` — force-closed at the end of the data series.
    """

    SIGNAL = "signal"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    END_OF_DATA = "end_of_data"


class InstrumentType(str, enum.Enum):
    """Classification of a financial instrument.

    Members:

    - ``STOCK``     — publicly traded equity share.
    - ``ETF``       — exchange-traded fund.
    - ``BOND``      — fixed-income debt instrument.
    - ``COMMODITY`` — raw material or primary agricultural product.
    - ``CRYPTO``    — cryptocurrency or digital asset.
    - ``INDEX``     — market index (non-tradeable reference).
    - ``UNKNOWN``   — instrument type cannot be determined.
    """

    STOCK = "stock"
    ETF = "etf"
    BOND = "bond"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    INDEX = "index"
    UNKNOWN = "unknown"


@dataclass
class EngineInfo:
    """Engine metadata for registry sync.

    Carries the human-readable and schema-level information that describes a
    registered strategy engine.  Instances are constructed by strategy modules
    and consumed by the backend to keep the database in sync with the registry.

    :param slug: Machine-readable unique identifier for the engine (e.g.
        ``"mean_reversion"``).  Must be a valid Python identifier and match the
        key used in :class:`~quaver.strategies.registry.StrategyRegistry`.
    :type slug: str
    :param name: Human-readable display name shown in UIs (e.g.
        ``"Mean Reversion"``).
    :type name: str
    :param description: Optional prose description of the engine's logic and
        intended use-case.
    :type description: str or None
    :param parameter_schema: Optional JSON Schema ``dict`` describing the
        ``parameters`` accepted by this engine.  Used by the backend for
        validation and UI form generation.
    :type parameter_schema: dict[str, Any] or None
    :param universe_constraints: Optional ``dict`` expressing hard constraints
        on the instrument universe this engine can operate on (e.g. allowed
        instrument types, minimum market-cap).
    :type universe_constraints: dict[str, Any] or None
    :param default_parameters: Optional ``dict`` of default parameter values
        applied when a user creates a new engine instance without specifying
        parameters explicitly.
    :type default_parameters: dict[str, Any] or None
    """

    slug: str
    name: str
    description: str | None = None
    parameter_schema: dict[str, Any] | None = None
    universe_constraints: dict[str, Any] | None = None
    default_parameters: dict[str, Any] | None = None
