"""Base strategy classes, signal output, and universe filter."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from quaver.types import SignalDirection

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalOutput:
    """Immutable signal produced by a strategy's compute() method."""

    direction: SignalDirection
    confidence: float  # 0.0–1.0
    notes: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


@dataclass
class UniverseFilter:
    """Filters the set of listings a strategy operates on.

    Pure data container. The ORM-coupled .apply() logic lives in the backend.
    """

    exchange_codes: list[str] | None = None
    instrument_types: list[str] | None = None
    countries: list[str] | None = None
    sectors: list[str] | None = None
    listing_ids: list[int] | None = None

    @classmethod
    def from_params(cls, parameters: dict[str, Any] | None) -> UniverseFilter:
        if not parameters:
            return cls()
        universe = parameters.get("universe", {})
        if not isinstance(universe, dict):
            return cls()
        return cls(
            exchange_codes=universe.get("exchange_codes"),
            instrument_types=universe.get("instrument_types"),
            countries=universe.get("countries"),
            sectors=universe.get("sectors"),
            listing_ids=universe.get("listing_ids"),
        )


class BaseStrategy(ABC):
    """
    Abstract base class for all single-asset trading strategies.

    Engines receive candle data as a pd.DataFrame with columns:
    ts, open, high, low, close, volume (minimum required).
    Rows are ordered by ts ASC.

    Subclasses must implement:
        validate_parameters(): check parameters has required keys/types.
        compute(): produce a signal (or None) for a single listing's candles.

    Subclasses should define class attributes:
        display_name: str
        description: str
    """

    display_name: str = ""
    description: str = ""

    def __init__(self, parameters: dict[str, Any]) -> None:
        self.parameters: dict[str, Any] = parameters or {}

    @abstractmethod
    def validate_parameters(self) -> None:
        """Validate parameters. Raise ValueError on invalid config."""
        ...

    @abstractmethod
    def compute(
        self,
        candles: pd.DataFrame,
        as_of: datetime,
    ) -> SignalOutput | None:
        """
        Run strategy logic on a single listing's candles.

        Args:
            candles: OHLCV DataFrame ordered by ts ASC.
                     Contains columns: open, high, low, close, volume (minimum).
                     IMPORTANT: does NOT include the current bar being evaluated.
            as_of: The point-in-time for evaluation (timestamp of current bar).

        Returns:
            A SignalOutput if a signal is generated, None otherwise.
        """
        ...

    def get_required_candle_count(self) -> int:
        """Return how many historical candles this strategy needs. Default 200."""
        return 200

    def get_universe_filter(self) -> UniverseFilter:
        """Build a UniverseFilter from strategy parameters."""
        return UniverseFilter.from_params(self.parameters)

    def validate_candles(self, candles: pd.DataFrame) -> bool:
        """Return True if we have enough candles. Override for custom logic."""
        return len(candles) >= self.get_required_candle_count()

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        """Return a JSON Schema dict describing this engine's parameters."""
        return {}

    @classmethod
    def get_universe_constraints(cls) -> dict[str, Any]:
        """Return constraints/defaults for universe filtering."""
        return {}

    @classmethod
    def get_default_parameters(cls) -> dict[str, Any]:
        """Return default parameter values for this engine."""
        return {}


# ── Multi-asset extension ────────────────────────────────────────────────────

@dataclass(frozen=True)
class MultiAssetStrategyOutput:
    """
    Signals produced by a multi-asset compute() call.

    Keys are instrument identifiers (e.g. listing_id or ticker string).
    All signals for a paired/grouped trade should be emitted together in a
    single MultiAssetStrategyOutput so the engine can apply them atomically.
    """

    signals: dict[str, SignalOutput]       # instrument_id -> signal
    metadata: dict[str, Any] | None = None  # strategy-level metadata


class MultiAssetStrategy(ABC):
    """
    Base class for strategies that need candles from multiple instruments
    simultaneously (e.g. pairs trading, basket strategies).

    compute() receives a dict mapping instrument_id -> DataFrame of candles,
    and returns signals for zero or more instruments.

    IMPORTANT: all signals in a single MultiAssetStrategyOutput are applied
    atomically by the engine. Emit signals for all legs together so that
    partial fills cannot occur.
    """

    display_name: str = ""
    description: str = ""

    def __init__(self, parameters: dict[str, Any]) -> None:
        self.parameters: dict[str, Any] = parameters or {}

    @abstractmethod
    def validate_parameters(self) -> None:
        """Validate parameters. Raise ValueError on invalid config."""
        ...

    @abstractmethod
    def compute(
        self,
        candles_map: dict[str, pd.DataFrame],
        as_of: datetime,
    ) -> MultiAssetStrategyOutput | None:
        """
        Args:
            candles_map: mapping of instrument_id -> OHLCV DataFrame (ASC).
                         Each DataFrame excludes the current bar (no look-ahead).
            as_of: point-in-time timestamp of the current bar.

        Returns:
            MultiAssetStrategyOutput with signals keyed by instrument_id, or None.
        """
        ...

    def get_required_instrument_ids(self) -> list[str]:
        """Declare which instrument_ids this strategy requires."""
        return []

    def get_required_candle_count(self) -> int:
        """Minimum candles needed per instrument. Default 200."""
        return 200

    def get_universe_filter(self) -> UniverseFilter:
        return UniverseFilter.from_params(self.parameters)

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def get_default_parameters(cls) -> dict[str, Any]:
        return {}
