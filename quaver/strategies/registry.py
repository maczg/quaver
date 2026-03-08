"""Strategy engine registry — maps engine names to strategy classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from quaver.strategies.base import BaseStrategy, MultiAssetStrategy


class DuplicateEngineError(Exception):
    """Raised when registering an engine name that already exists."""


class EngineNotFoundError(Exception):
    """Raised when looking up an engine name that is not registered."""


class StrategyRegistry:
    """Central registry mapping engine names -> BaseStrategy subclasses."""

    _engines: dict[str, type[BaseStrategy] | type[MultiAssetStrategy]] = {}

    @classmethod
    def register(cls, engine_name: str):
        """Decorator to register a strategy engine class."""

        def decorator(strategy_cls):
            if engine_name in cls._engines:
                raise DuplicateEngineError(
                    f"Engine '{engine_name}' is already registered "
                    f"to {cls._engines[engine_name].__name__}"
                )
            cls._engines[engine_name] = strategy_cls
            return strategy_cls

        return decorator

    @classmethod
    def get(cls, engine_name: str):
        """Look up a registered engine by name. Raises EngineNotFoundError."""
        try:
            return cls._engines[engine_name]
        except KeyError:
            available = ", ".join(sorted(cls._engines)) or "(none)"
            raise EngineNotFoundError(
                f"Engine '{engine_name}' not found. Available: {available}"
            )

    @classmethod
    def list_engines(cls) -> list[str]:
        """Return a sorted list of all registered engine names."""
        return sorted(cls._engines)

    @classmethod
    def all(cls) -> dict:
        """Return the full engine registry as a dict (copy)."""
        return dict(cls._engines)

    @classmethod
    def get_strategy_kind(cls, engine_name: str) -> Literal["single", "multi"]:
        """Return 'single' for BaseStrategy subclasses, 'multi' for MultiAssetStrategy."""
        from quaver.strategies.base import MultiAssetStrategy
        strategy_cls = cls.get(engine_name)
        if issubclass(strategy_cls, MultiAssetStrategy):
            return "multi"
        return "single"

    @classmethod
    def clear(cls) -> None:
        """Remove all registered engines. USE IN TESTS ONLY (teardown).

        Prevents cross-test pollution when stub engines are registered.
        """
        cls._engines.clear()
