"""Strategy engine registry — maps engine names to strategy classes.

:class:`StrategyRegistry` is a class-level singleton (backed by a plain
``dict`` class variable) that maps engine name strings to their corresponding
:class:`~quaver.strategies.base.BaseStrategy` or
:class:`~quaver.strategies.base.MultiAssetStrategy` subclasses.

Engines are registered via the :meth:`StrategyRegistry.register` decorator at
module import time.  Importing :mod:`quaver.strategies` triggers all built-in
registrations automatically.

.. rubric:: Exceptions

- :class:`DuplicateEngineError` — raised when a name collision occurs during
  registration.
- :class:`EngineNotFoundError` — raised when a look-up finds no matching
  engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from quaver.strategies.base import BaseStrategy, MultiAssetStrategy


class DuplicateEngineError(Exception):
    """Raised when registering an engine name that already exists.

    Prevents silent overwrites of previously registered engines.  The error
    message includes both the conflicting name and the class that owns the
    original registration.

    :param message: Explanation of which engine name is duplicated and which
        class holds the original registration.
    :type message: str
    """


class EngineNotFoundError(Exception):
    """Raised when looking up an engine name that is not registered.

    The error message includes the requested name and the full list of
    currently registered engine names to assist with debugging typos.

    :param message: Explanation of which engine name was not found and what
        names are currently available.
    :type message: str
    """


class StrategyRegistry:
    """Central registry mapping engine names to :class:`~quaver.strategies.base.BaseStrategy` subclasses.

    All state is stored in the ``_engines`` class variable so that a single
    shared registry is available for the lifetime of the process without
    requiring explicit instantiation.

    Engines are added via the :meth:`register` class-method decorator, which is
    typically applied at module level in each engine's source file.  Importing
    :mod:`quaver.strategies` causes all built-in engine modules to be imported,
    populating this registry as a side-effect.
    """

    _engines: dict[str, type[BaseStrategy] | type[MultiAssetStrategy]] = {}

    @classmethod
    def register(cls, engine_name: str):
        """Decorator factory that registers a strategy engine class.

        Apply to a :class:`~quaver.strategies.base.BaseStrategy` or
        :class:`~quaver.strategies.base.MultiAssetStrategy` subclass to add it
        to the registry under *engine_name*.

        .. code-block:: python

            @StrategyRegistry.register("my_engine")
            class MyEngine(BaseStrategy):
                ...

        :param engine_name: Unique string key for this engine.  Must not
            already exist in the registry.
        :type engine_name: str
        :returns: A class decorator that registers the decorated class and
            returns it unchanged.
        :rtype: Callable[[type], type]
        :raises DuplicateEngineError: If *engine_name* is already present in
            the registry.
        """

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
        """Look up a registered engine class by name.

        :param engine_name: The engine name used during registration.
        :type engine_name: str
        :returns: The :class:`~quaver.strategies.base.BaseStrategy` or
            :class:`~quaver.strategies.base.MultiAssetStrategy` subclass
            registered under *engine_name*.
        :rtype: type[BaseStrategy] | type[MultiAssetStrategy]
        :raises EngineNotFoundError: If *engine_name* is not present in the
            registry.  The error message lists all available engine names.
        """
        try:
            return cls._engines[engine_name]
        except KeyError:
            available = ", ".join(sorted(cls._engines)) or "(none)"
            raise EngineNotFoundError(
                f"Engine '{engine_name}' not found. Available: {available}"
            )

    @classmethod
    def list_engines(cls) -> list[str]:
        """Return a sorted list of all registered engine names.

        :returns: Alphabetically sorted list of engine name strings.
        :rtype: list[str]
        """
        return sorted(cls._engines)

    @classmethod
    def all(cls) -> dict:
        """Return the full engine registry as a shallow copy.

        The returned ``dict`` maps engine name strings to their strategy
        classes.  Modifications to the returned dict do not affect the registry.

        :returns: Shallow copy of the internal ``_engines`` mapping.
        :rtype: dict[str, type[BaseStrategy] | type[MultiAssetStrategy]]
        """
        return dict(cls._engines)

    @classmethod
    def get_strategy_kind(cls, engine_name: str) -> Literal["single", "multi"]:
        """Return the kind of strategy registered under *engine_name*.

        :param engine_name: The engine name to inspect.
        :type engine_name: str
        :returns: ``"multi"`` if the engine is a
            :class:`~quaver.strategies.base.MultiAssetStrategy` subclass,
            ``"single"`` if it is a
            :class:`~quaver.strategies.base.BaseStrategy` subclass.
        :rtype: Literal["single", "multi"]
        :raises EngineNotFoundError: If *engine_name* is not present in the
            registry.
        """
        from quaver.strategies.base import MultiAssetStrategy
        strategy_cls = cls.get(engine_name)
        if issubclass(strategy_cls, MultiAssetStrategy):
            return "multi"
        return "single"

    @classmethod
    def clear(cls) -> None:
        """Remove all registered engines from the registry.

        .. warning::

            This method is intended **for use in tests only** (teardown
            fixtures).  Calling it in production code will leave the registry
            empty and cause :exc:`EngineNotFoundError` on any subsequent
            look-up.  It exists to prevent cross-test pollution when stub
            engines are registered inside individual test cases.

        :returns: None
        """
        cls._engines.clear()
