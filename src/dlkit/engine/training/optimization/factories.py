"""Optimizer and scheduler factory implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib import import_module
from typing import cast

import torch
import torch.optim

from dlkit.infrastructure.config.optimizer_component import (
    OptimizerComponentSettings,
    SchedulerComponentSettings,
)

# torch param_groups: each entry is a dict with "params" key and arbitrary optimizer kwargs
type ParamGroup = dict[str, object]


class IOptimizerFactory(ABC):
    """Abstract interface for creating optimizer instances.

    Implementations are responsible for resolving the optimizer class
    (by name, module path, or callable) and instantiating it with
    the provided parameter groups and configuration.
    """

    @abstractmethod
    def create(self, param_groups: list[ParamGroup]) -> torch.optim.Optimizer:
        """Create an optimizer instance.

        Args:
            param_groups: Parameter groups for the optimizer, as defined by
                torch.optim.Optimizer. Each group is a dict with keys like
                "params", "lr", "weight_decay", etc.

        Returns:
            A torch.optim.Optimizer instance configured and ready to use.
        """
        ...


class TorchOptimizerFactory(IOptimizerFactory):
    """Factory for creating torch.optim.Optimizer instances from settings.

    Resolves the optimizer class by name and optional module path,
    then instantiates it with param_groups and extra keyword arguments
    from the settings.

    Attributes:
        _settings: The optimizer component settings.
    """

    def __init__(self, settings: OptimizerComponentSettings) -> None:
        """Initialize the factory with optimizer settings.

        Args:
            settings: Optimizer component configuration including name,
                module_path, and additional hyperparameters.
        """
        self._settings = settings

    def create(self, param_groups: list[ParamGroup]) -> torch.optim.Optimizer:
        """Create a torch optimizer instance.

        Args:
            param_groups: Parameter groups for the optimizer.

        Returns:
            Instantiated optimizer.

        Raises:
            ImportError: If the optimizer class cannot be imported.
            TypeError: If the optimizer class cannot be instantiated
                with the provided parameters. Passing kwargs that are
                incompatible with the chosen optimizer raises here rather
                than being silently dropped.
        """
        optimizer_cls = self._resolve_optimizer_class()

        # get_init_kwargs() excludes identity fields (name, module_path), None values,
        # and unset fields. Only kwargs the user explicitly set are forwarded.
        # Each optimizer uses its own PyTorch defaults for the rest.
        # Passing an incompatible kwarg raises TypeError here — intentional, not silent.
        kwargs = self._settings.get_init_kwargs()

        return optimizer_cls(param_groups, **kwargs)

    def _resolve_optimizer_class(self) -> Callable[..., torch.optim.Optimizer]:
        """Resolve the optimizer class from settings.

        Returns:
            The optimizer class or callable that creates optimizers.

        Raises:
            ImportError: If the optimizer cannot be imported.
            TypeError: If the resolved name is not a class.
        """
        name = self._settings.name

        # If name is already a class, use it directly
        if isinstance(name, type):
            return cast(Callable[..., torch.optim.Optimizer], name)

        # If name is a callable, use it (assume it's the class or factory)
        if callable(name) and not isinstance(name, str):
            return cast(Callable[..., torch.optim.Optimizer], name)

        # Otherwise, it should be a string name; import it
        if not isinstance(name, str):
            raise TypeError(f"Optimizer name must be str, type, or callable; got {type(name)}")

        module_path = self._settings.module_path or "torch.optim"
        full_path = f"{module_path}:{name}"

        return self._import_class(full_path, module_path)

    @staticmethod
    def _import_class(full_path: str, fallback_module: str) -> Callable[..., torch.optim.Optimizer]:
        """Import a class by module:name or name notation.

        Args:
            full_path: Import path in format "module:class" or just "class".
            fallback_module: Module to use if no module is specified.

        Returns:
            The imported class or callable that creates optimizers.

        Raises:
            ImportError: If the class cannot be imported.
        """
        if ":" in full_path:
            module_name, class_name = full_path.split(":", 1)
        else:
            module_name = fallback_module
            class_name = full_path

        module = import_module(module_name)
        cls = getattr(module, class_name)

        if cls is None:
            raise ImportError(f"Could not find {class_name} in {module_name}")

        return cls


class IMuonOptimizerFactory(IOptimizerFactory):
    """Marker abstract class for Muon-specific optimizer factories.

    Muon has a different construction contract than standard torch optimizers
    (e.g., different hyperparameter handling, manual updates). Implementations
    of this interface follow Muon's API.
    """

    pass


class ISchedulerFactory(ABC):
    """Abstract interface for creating learning rate scheduler instances.

    Implementations resolve the scheduler class and instantiate it with
    the provided optimizer and configuration.
    """

    @abstractmethod
    def create(self, optimizer: torch.optim.Optimizer) -> object:
        """Create a scheduler instance.

        Args:
            optimizer: The optimizer to schedule learning rates for.

        Returns:
            A scheduler instance (typically torch.optim.lr_scheduler._LRScheduler
            or compatible). The return type is object to allow custom scheduler types.
        """
        ...


class TorchSchedulerFactory(ISchedulerFactory):
    """Factory for creating torch.optim.lr_scheduler instances from settings.

    Resolves the scheduler class by name and optional module path,
    then instantiates it with the optimizer and extra keyword arguments
    from the settings.

    Attributes:
        _settings: The scheduler component settings.
    """

    def __init__(self, settings: SchedulerComponentSettings) -> None:
        """Initialize the factory with scheduler settings.

        Args:
            settings: Scheduler component configuration including name,
                module_path, monitor, and additional hyperparameters.
        """
        self._settings = settings

    def create(self, optimizer: torch.optim.Optimizer) -> object:
        """Create a torch scheduler instance.

        Args:
            optimizer: The optimizer to schedule.

        Returns:
            Instantiated scheduler.

        Raises:
            ImportError: If the scheduler class cannot be imported.
            TypeError: If the scheduler class cannot be instantiated.
        """
        # Resolve the scheduler class
        scheduler_cls = self._resolve_scheduler_class()

        # get_init_kwargs() excludes identity fields (name, module_path), None values, and
        # unset fields. "monitor" and "frequency" are also excluded because they are
        # Lightning metadata forwarded via configure_optimizers(), not scheduler constructor args.
        kwargs = self._settings.get_init_kwargs(exclude={"monitor", "frequency"})

        # Create and return scheduler
        return scheduler_cls(optimizer, **kwargs)

    def _resolve_scheduler_class(self) -> Callable[..., object]:
        """Resolve the scheduler class from settings.

        Returns:
            The scheduler class or callable that creates schedulers.

        Raises:
            ImportError: If the scheduler cannot be imported.
            TypeError: If the resolved name is not a class.
        """
        name = self._settings.name

        # If name is already a class, use it directly
        if isinstance(name, type):
            return cast(Callable[..., object], name)

        # If name is a callable, use it
        if callable(name) and not isinstance(name, str):
            return cast(Callable[..., object], name)

        # Otherwise, it should be a string name; import it
        if not isinstance(name, str):
            raise TypeError(f"Scheduler name must be str, type, or callable; got {type(name)}")

        module_path = self._settings.module_path or "torch.optim.lr_scheduler"
        full_path = f"{module_path}:{name}"

        return self._import_class(full_path, module_path)

    @staticmethod
    def _import_class(full_path: str, fallback_module: str) -> Callable[..., object]:
        """Import a class by module:name or name notation.

        Args:
            full_path: Import path in format "module:class" or just "class".
            fallback_module: Module to use if no module is specified.

        Returns:
            The imported class or callable that creates schedulers.

        Raises:
            ImportError: If the class cannot be imported.
        """
        if ":" in full_path:
            module_name, class_name = full_path.split(":", 1)
        else:
            module_name = fallback_module
            class_name = full_path

        module = import_module(module_name)
        cls = getattr(module, class_name)

        if cls is None:
            raise ImportError(f"Could not find {class_name} in {module_name}")

        return cls
