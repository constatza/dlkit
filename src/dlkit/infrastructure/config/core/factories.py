"""Factories for settings-driven component construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from inspect import isclass
from typing import Any, cast

from dlkit.infrastructure.config.core.base_settings import ComponentSettings
from dlkit.infrastructure.utils.general import import_object, kwargs_compatible_with

from .context import BuildContext


class ComponentFactory[T](ABC):
    """Abstract factory for creating components from settings.

    This replaces the old build() method pattern with a proper factory pattern
    that separates configuration from construction logic.
    """

    @abstractmethod
    def create(self, settings: ComponentSettings, context: BuildContext) -> T:
        """Create a component instance from settings.

        Args:
            settings: The component settings
            context: Build context with dependencies and overrides

        Returns:
            T: The created component instance
        """


class DefaultComponentFactory[T](ComponentFactory[T]):
    """Default factory implementation for most components.

    This handles the common case of resolving a class and calling its constructor
    with compatible keyword arguments.
    """

    def create(self, settings: ComponentSettings, context: BuildContext) -> T:
        """Create component using default pattern: resolve class and construct.

        Args:
            settings: Component settings containing class reference
            context: Build context with overrides

        Returns:
            T: The constructed component

        Raises:
            TypeError: If the resolved target is not a class or callable
            ValueError: If construction fails
        """
        target = self._resolve_class(settings, context)

        # If an already-constructed instance is provided, return it directly (bypass construction)
        if not isclass(target) and not callable(target):
            return cast(T, target)

        if isclass(target):
            init_kwargs = self._prepare_init_kwargs(settings, context, target)
            return cast(T, target(**init_kwargs))
        if callable(target):
            # Handle function/callable targets: return the callable itself
            # rather than invoking it here. Many callables (e.g., loss functions)
            # require runtime tensors as positional arguments.
            return cast(T, target)
        raise TypeError(
            f"Resolved target must be a class, callable, or instance, got {type(target)}"
        )

    def _resolve_class(
        self, settings: ComponentSettings, context: BuildContext
    ) -> type[T] | Callable:
        """Resolve the target class from settings.

        Args:
            settings: Component settings with class reference

        Returns:
            type[T] | Callable: The resolved class or callable
        """
        target = settings.name
        if isinstance(target, str):
            # Ensure config/working directory is importable to resolve local modules
            try:
                import sys as _sys
                from pathlib import Path as _Path

                wd = getattr(context, "working_directory", None)
                if wd is not None:
                    rp = str(_Path(wd).resolve())
                    if rp not in _sys.path:
                        _sys.path.insert(0, rp)
            except Exception:
                pass
            # Try to resolve via user registries first (with forced selection),
            # then fall back to import for built-in/third-party objects.
            kind = self._infer_kind_from_settings(settings)
            if kind is not None:
                from dlkit.infrastructure.registry.resolve import resolve_component

                resolved = resolve_component(kind, target, module_path=settings.module_path)
            else:
                resolved = import_object(target, fallback_module=settings.module_path or "")
            # Only allow class/callable via string resolution; reject arbitrary instances
            from inspect import isclass as _isclass

            if _isclass(resolved) or callable(resolved):
                return resolved
            raise TypeError(f"Resolved target must be a class or callable, got {type(resolved)}")
        if isinstance(target, type) or callable(target):
            return target
        # Allow passing already constructed instances (object) as name
        return cast("type[T] | Callable[..., object]", target)

    @staticmethod
    def _infer_kind_from_settings(settings: ComponentSettings) -> str | None:
        """Infer component kind from settings type.

        Returns one of: "model", "dataset", "datamodule", "loss", "metric" or None if unknown.
        """
        try:
            # Import locally to avoid circular imports at module import time
            from dlkit.infrastructure.config.datamodule_settings import DataModuleSettings
            from dlkit.infrastructure.config.dataset_settings import DatasetSettings
            from dlkit.infrastructure.config.model_components import (
                LossComponentSettings,
                MetricComponentSettings,
                ModelComponentSettings,
            )

            if isinstance(settings, ModelComponentSettings):
                return "model"
            if isinstance(settings, DatasetSettings):
                return "dataset"
            if isinstance(settings, DataModuleSettings):
                return "datamodule"
            if isinstance(settings, LossComponentSettings):
                return "loss"
            if isinstance(settings, MetricComponentSettings):
                return "metric"
        except Exception:
            return None
        return None

    def _prepare_init_kwargs(
        self, settings: ComponentSettings, context: BuildContext, cls: type
    ) -> dict[str, Any]:
        """Prepare initialization kwargs compatible with the target class.

        Args:
            settings: Component settings
            context: Build context with overrides
            cls: Target class

        Returns:
            dict[str, Any]: Compatible initialization kwargs
        """
        # Get base kwargs from settings and filter them for compatibility
        base_kwargs = settings.get_init_kwargs()

        # If a nested "params" dict is present (common in configs),
        # merge its contents into the top-level kwargs for constructor compatibility.
        try:
            nested_params = (
                base_kwargs.pop("params") if isinstance(base_kwargs.get("params"), dict) else None
            )
            if nested_params:
                # Do not overwrite explicitly set top-level keys
                for k, v in nested_params.items():
                    base_kwargs.setdefault(k, v)
        except Exception:
            # Be resilient to any unexpected structure
            pass
        # For classes that accept **kwargs, allow all kwargs through without checking.
        from inspect import Parameter, signature

        try:
            target_fn = cls.__init__ if isclass(cls) else cls
            accepts_var_keyword = any(
                p.kind == Parameter.VAR_KEYWORD for p in signature(target_fn).parameters.values()
            )
        except ValueError, TypeError:
            accepts_var_keyword = True  # can't introspect → assume permissive

        if accepts_var_keyword:
            return {**base_kwargs, **context.overrides}

        # Strict constructor: any kwarg from settings that isn't a recognised
        # parameter is a configuration error — raise instead of silently dropping.
        compatible_base = kwargs_compatible_with(cls, **base_kwargs)
        unrecognized = set(base_kwargs) - set(compatible_base)
        if unrecognized:
            raise TypeError(
                f"{cls.__name__}.__init__() got unexpected keyword arguments: {sorted(unrecognized)}"
            )

        # Merge with context overrides and filter overrides for compatibility.
        final_kwargs = {**compatible_base, **context.overrides}
        return kwargs_compatible_with(cls, **final_kwargs)


class ComponentRegistry:
    """Registry for managing component factories.

    This allows registration of custom factories for specific component types
    and provides a centralized way to manage component creation.
    """

    def __init__(self):
        self._factories: dict[type, ComponentFactory] = {}
        self._default_factory = DefaultComponentFactory()

    def register_factory(self, settings_type: type, factory: ComponentFactory) -> None:
        """Register a custom factory for a settings type.

        Args:
            settings_type: The settings class type
            factory: The factory to handle this settings type
        """
        self._factories[settings_type] = factory

    def get_factory(self, settings_type: type) -> ComponentFactory[Any]:
        """Get the appropriate factory for a settings type.

        Args:
            settings_type: The settings class type

        Returns:
            ComponentFactory: The factory to use
        """
        return self._factories.get(settings_type, self._default_factory)

    def create_component(self, settings: ComponentSettings, context: BuildContext) -> Any:
        """Create a component using the appropriate factory.

        Args:
            settings: Component settings
            context: Build context

        Returns:
            T: The created component
        """
        factory = self.get_factory(type(settings))
        return factory.create(settings, context)


class FactoryProvider:
    """Global factory provider singleton.

    This provides a global access point to the component registry
    and maintains backward compatibility during the transition.
    """

    _instance: ComponentRegistry | None = None

    @classmethod
    def get_registry(cls) -> ComponentRegistry:
        """Get the global component registry.

        Returns:
            ComponentRegistry: The global registry instance
        """
        if cls._instance is None:
            cls._instance = ComponentRegistry()
        return cls._instance

    @classmethod
    def create_component(cls, settings: ComponentSettings, context: BuildContext) -> Any:
        """Create a component using the global registry.

        Args:
            settings: Component settings
            context: Build context

        Returns:
            T: The created component
        """
        registry = cls.get_registry()
        return registry.create_component(settings, context)

    @classmethod
    def register_factory(cls, settings_type: type, factory: ComponentFactory) -> None:
        """Register a factory with the global registry.

        Args:
            settings_type: Settings type to register for
            factory: Factory to register
        """
        registry = cls.get_registry()
        registry.register_factory(settings_type, factory)

    @classmethod
    def reset_for_testing(cls) -> ComponentRegistry | None:
        """Save and reset singleton state for test isolation.

        Returns:
            ComponentRegistry | None: Previous registry state to pass to restore_for_testing.
        """
        saved = cls._instance
        cls._instance = None
        return saved

    @classmethod
    def restore_for_testing(cls, saved: ComponentRegistry | None) -> None:
        """Restore singleton state saved by reset_for_testing.

        Args:
            saved: Registry state returned by reset_for_testing.
        """
        cls._instance = saved


# Import here to avoid circular imports
