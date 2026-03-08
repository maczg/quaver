"""Shared pure-Python types for quaver.

These enums and dataclasses have NO SQLAlchemy dependency.
They are the single source of truth for enum values shared between
quaver engines and any backend models.
"""

import enum
from dataclasses import dataclass, field
from typing import Any


class SignalDirection(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class SignalStrength(str, enum.Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class TimeFrame(str, enum.Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class InstrumentType(str, enum.Enum):
    STOCK = "stock"
    ETF = "etf"
    BOND = "bond"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    INDEX = "index"
    UNKNOWN = "unknown"


@dataclass
class EngineInfo:
    """Engine metadata for registry sync."""

    slug: str
    name: str
    description: str | None = None
    parameter_schema: dict[str, Any] | None = None
    universe_constraints: dict[str, Any] | None = None
    default_parameters: dict[str, Any] | None = None