"""Base strategy classes, signal output, and universe filter.

This module defines the public contracts that every quaver strategy engine
must satisfy:

- :class:`SignalOutput` — immutable result of a single-asset ``compute()`` call.
- :class:`UniverseFilter` — instrument-set filter derived from engine parameters.
- :class:`BaseStrategy` — abstract base for single-asset engines.
- :class:`MultiAssetStrategyOutput` — immutable result of a multi-asset
  ``compute()`` call.
- :class:`MultiAssetStrategy` — abstract base for multi-asset / pairs engines.
"""

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
    """Immutable signal produced by a strategy's ``compute()`` method.

    :param direction: The trading direction recommended by the engine.
    :type direction: SignalDirection
    :param confidence: Conviction score in the closed interval ``[0.0, 1.0]``.
        ``0.0`` indicates no conviction; ``1.0`` indicates maximum conviction.
    :type confidence: float
    :param notes: Optional free-text explanation or debug information attached
        to the signal.
    :type notes: str or None
    :param metadata: Optional arbitrary key/value pairs for strategy-specific
        data that the backend may persist or forward downstream.
    :type metadata: dict[str, Any] or None
    :raises ValueError: If ``confidence`` is outside the range ``[0.0, 1.0]``.
    """

    direction: SignalDirection
    confidence: float  # 0.0–1.0
    notes: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class UniverseFilter:
    """Filters the set of listings a strategy operates on.

    Pure data container. The ORM-coupled ``.apply()`` logic lives in the backend.

    :param exchange_codes: Restrict the universe to listings on these exchange
        codes (e.g. ``["XNYS", "XNAS"]``).  ``None`` means no restriction.
    :type exchange_codes: list[str] or None
    :param instrument_types: Restrict the universe to these instrument type
        strings (values of :class:`~quaver.types.InstrumentType`).
        ``None`` means no restriction.
    :type instrument_types: list[str] or None
    :param countries: Restrict the universe to listings domiciled in these ISO
        country codes (e.g. ``["US", "GB"]``).  ``None`` means no restriction.
    :type countries: list[str] or None
    :param sectors: Restrict the universe to listings in these sector names.
        ``None`` means no restriction.
    :type sectors: list[str] or None
    :param listing_ids: Pin the universe to an explicit set of listing primary
        keys.  ``None`` means no restriction.
    :type listing_ids: list[int] or None
    """

    exchange_codes: list[str] | None = None
    instrument_types: list[str] | None = None
    countries: list[str] | None = None
    sectors: list[str] | None = None
    listing_ids: list[int] | None = None

    @classmethod
    def from_params(cls, parameters: dict[str, Any] | None) -> UniverseFilter:
        """Construct a :class:`UniverseFilter` from a raw parameters dictionary.

        Reads the nested ``"universe"`` key from *parameters*, mapping its
        sub-keys to the corresponding filter fields.  Missing or malformed
        values fall back to ``None`` (no restriction).

        :param parameters: Raw engine parameters dict, typically stored in the
            database alongside the engine configuration.  May be ``None`` or
            omit the ``"universe"`` key entirely.
        :type parameters: dict[str, Any] or None
        :returns: A populated :class:`UniverseFilter`; all fields default to
            ``None`` when *parameters* is absent or ``"universe"`` is missing.
        :rtype: UniverseFilter
        """
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
    """Abstract base class for all single-asset trading strategies.

    Engines receive candle data as a :class:`pandas.DataFrame` with columns
    ``ts``, ``open``, ``high``, ``low``, ``close``, ``volume`` (minimum
    required).  Rows are ordered by ``ts`` ASC.

    Subclasses **must** implement:

    - :meth:`validate_parameters` — check that ``parameters`` contains the
      required keys and valid value types.
    - :meth:`compute` — produce a :class:`SignalOutput` (or ``None``) for a
      single listing's candle history.

    Subclasses **should** define class attributes:

    - ``display_name: str`` — human-readable name shown in UIs.
    - ``description: str`` — prose explanation of the strategy logic.

    :param parameters: Engine configuration key/value pairs supplied by the
        user.  Validated by :meth:`validate_parameters` before ``compute()``
        is called.
    :type parameters: dict[str, Any]
    """

    display_name: str = ""
    description: str = ""

    def __init__(self, parameters: dict[str, Any]) -> None:
        self.parameters: dict[str, Any] = parameters or {}

    @abstractmethod
    def validate_parameters(self) -> None:
        """Validate engine parameters.

        Inspect ``self.parameters`` and raise :exc:`ValueError` for any
        missing required keys or out-of-range values.

        :raises ValueError: If ``self.parameters`` is invalid or incomplete.
        """
        ...

    @abstractmethod
    def compute(
        self,
        candles: pd.DataFrame,
        as_of: datetime,
    ) -> SignalOutput | None:
        """Run strategy logic on a single listing's candle history.

        :param candles: OHLCV :class:`~pandas.DataFrame` ordered by ``ts`` ASC.
            Contains at minimum the columns ``open``, ``high``, ``low``,
            ``close``, and ``volume``.  **Does not include the current bar
            being evaluated** — look-ahead into the current bar is forbidden.
        :type candles: pandas.DataFrame
        :param as_of: Point-in-time timestamp of the current bar being
            evaluated.  Use this value (not ``datetime.utcnow()``) for all
            time-dependent logic to ensure reproducible back-tests.
        :type as_of: datetime.datetime
        :returns: A :class:`SignalOutput` when the strategy generates a signal,
            or ``None`` when conditions are not met.
        :rtype: SignalOutput or None
        """
        ...

    def get_required_candle_count(self) -> int:
        """Return the minimum number of historical candles this strategy needs.

        The engine runner will not call :meth:`compute` when fewer candles are
        available.  Override this method to raise or lower the default.

        :returns: Minimum required candle count.  Defaults to ``200``.
        :rtype: int
        """
        return 200

    def get_universe_filter(self) -> UniverseFilter:
        """Build a :class:`UniverseFilter` from the engine's parameters.

        :returns: A :class:`UniverseFilter` derived from
            ``self.parameters["universe"]``, or an unrestricted filter when the
            key is absent.
        :rtype: UniverseFilter
        """
        return UniverseFilter.from_params(self.parameters)

    def validate_candles(self, candles: pd.DataFrame) -> bool:
        """Return ``True`` if *candles* meets the minimum row requirement.

        Override this method to apply custom validation logic (e.g. checking
        for required columns or data quality constraints).

        :param candles: The candle :class:`~pandas.DataFrame` to validate.
        :type candles: pandas.DataFrame
        :returns: ``True`` when the DataFrame has at least
            :meth:`get_required_candle_count` rows, ``False`` otherwise.
        :rtype: bool
        """
        return len(candles) >= self.get_required_candle_count()

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        """Return a JSON Schema ``dict`` describing this engine's parameters.

        The returned schema is stored in
        :attr:`~quaver.types.EngineInfo.parameter_schema` and used by the
        backend for validation and UI form generation.  Override to provide a
        concrete schema.

        :returns: A JSON Schema-compatible dictionary, or an empty ``dict``
            when no schema is defined.
        :rtype: dict[str, Any]
        """
        return {}

    @classmethod
    def get_universe_constraints(cls) -> dict[str, Any]:
        """Return constraints or defaults for universe filtering.

        The returned dict is stored in
        :attr:`~quaver.types.EngineInfo.universe_constraints`.  Override to
        express hard limits on the instrument universe (e.g. allowed instrument
        types, minimum market-cap tier).

        :returns: A dictionary of universe constraint definitions, or an empty
            ``dict`` when no constraints are defined.
        :rtype: dict[str, Any]
        """
        return {}

    @classmethod
    def get_default_parameters(cls) -> dict[str, Any]:
        """Return default parameter values for this engine.

        The returned dict is stored in
        :attr:`~quaver.types.EngineInfo.default_parameters` and applied when a
        user creates a new engine instance without specifying parameters.

        :returns: A dictionary of default parameter values, or an empty
            ``dict`` when no defaults are defined.
        :rtype: dict[str, Any]
        """
        return {}


# ── Multi-asset extension ────────────────────────────────────────────────────


@dataclass(frozen=True)
class MultiAssetStrategyOutput:
    """Signals produced by a multi-asset ``compute()`` call.

    Keys of :attr:`signals` are instrument identifiers (e.g. ``listing_id``
    cast to ``str``, or a ticker string).  All signals for a paired or grouped
    trade should be emitted together in a single
    :class:`MultiAssetStrategyOutput` so the engine can apply them atomically.

    :param signals: Mapping of ``instrument_id`` to the corresponding
        :class:`SignalOutput` for that instrument.
    :type signals: dict[str, SignalOutput]
    :param metadata: Optional strategy-level metadata shared across all signals
        in this output (e.g. spread value, regime label).
    :type metadata: dict[str, Any] or None
    """

    signals: dict[str, SignalOutput]  # instrument_id -> signal
    metadata: dict[str, Any] | None = None  # strategy-level metadata


class MultiAssetStrategy(ABC):
    """Base class for strategies that need candles from multiple instruments simultaneously.

    Intended for use cases such as pairs trading and basket strategies where
    the signal for one instrument depends on the price history of one or more
    other instruments.

    :meth:`compute` receives a mapping of ``instrument_id`` to a
    :class:`~pandas.DataFrame` of candles and returns signals for zero or more
    instruments.

    .. important::

        All signals in a single :class:`MultiAssetStrategyOutput` are applied
        **atomically** by the engine.  Emit signals for all legs of a trade
        together so that partial fills cannot occur.

    :param parameters: Engine configuration key/value pairs supplied by the
        user.  Validated by :meth:`validate_parameters` before ``compute()``
        is called.
    :type parameters: dict[str, Any]
    """

    display_name: str = ""
    description: str = ""

    def __init__(self, parameters: dict[str, Any]) -> None:
        self.parameters: dict[str, Any] = parameters or {}

    @abstractmethod
    def validate_parameters(self) -> None:
        """Validate engine parameters.

        Inspect ``self.parameters`` and raise :exc:`ValueError` for any
        missing required keys or out-of-range values.

        :raises ValueError: If ``self.parameters`` is invalid or incomplete.
        """
        ...

    @abstractmethod
    def compute(
        self,
        candles_map: dict[str, pd.DataFrame],
        as_of: datetime,
    ) -> MultiAssetStrategyOutput | None:
        """Run strategy logic across multiple instruments simultaneously.

        :param candles_map: Mapping of ``instrument_id`` to an OHLCV
            :class:`~pandas.DataFrame` ordered by ``ts`` ASC.  Each DataFrame
            **excludes the current bar** — no look-ahead into the bar being
            evaluated.
        :type candles_map: dict[str, pandas.DataFrame]
        :param as_of: Point-in-time timestamp of the current bar being
            evaluated.  Use this value for all time-dependent calculations to
            ensure reproducible back-tests.
        :type as_of: datetime.datetime
        :returns: A :class:`MultiAssetStrategyOutput` with signals keyed by
            ``instrument_id``, or ``None`` when no signal conditions are met.
        :rtype: MultiAssetStrategyOutput or None
        """
        ...

    def get_required_instrument_ids(self) -> list[str]:
        """Declare which instrument IDs this strategy requires.

        The engine runner uses this list to fetch candle data for all required
        instruments before calling :meth:`compute`.  Override to return the
        specific identifiers the strategy depends on.

        :returns: List of required instrument identifier strings.  Defaults to
            an empty list.
        :rtype: list[str]
        """
        return []

    def get_required_candle_count(self) -> int:
        """Return the minimum number of candles needed per instrument.

        The engine runner will not call :meth:`compute` when any instrument has
        fewer candles available than this threshold.

        :returns: Minimum required candle count per instrument.  Defaults to
            ``200``.
        :rtype: int
        """
        return 200

    def get_universe_filter(self) -> UniverseFilter:
        """Build a :class:`UniverseFilter` from the engine's parameters.

        :returns: A :class:`UniverseFilter` derived from
            ``self.parameters["universe"]``, or an unrestricted filter when the
            key is absent.
        :rtype: UniverseFilter
        """
        return UniverseFilter.from_params(self.parameters)

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        """Return a JSON Schema ``dict`` describing this engine's parameters.

        :returns: A JSON Schema-compatible dictionary, or an empty ``dict``
            when no schema is defined.
        :rtype: dict[str, Any]
        """
        return {}

    @classmethod
    def get_default_parameters(cls) -> dict[str, Any]:
        """Return default parameter values for this engine.

        :returns: A dictionary of default parameter values, or an empty
            ``dict`` when no defaults are defined.
        :rtype: dict[str, Any]
        """
        return {}
