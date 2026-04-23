"""Factory for creating Lightning wrappers.

Provides convenient factory methods for creating the appropriate wrapper type
based on model characteristics and explicit type requests.
"""

from typing import Any

from torch import nn

from dlkit.infrastructure.config import (
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.infrastructure.config.data_entries import DataEntry
from dlkit.infrastructure.utils.logging_config import get_logger

from .base import CoreLightningWrapper
from .graph import GraphLightningWrapper
from .standard import StandardLightningWrapper
from .timeseries import TimeSeriesLightningWrapper
from .wrapper_types import WrapperComponents

logger = get_logger(__name__)


class WrapperFactory:
    """Factory for creating appropriate Lightning wrappers.

    Uses heuristics to determine the best wrapper type for a given model and
    provides convenient methods for explicit wrapper creation.

    Example:
        ```python
        wrapper = WrapperFactory.create_standard_wrapper(
            model_settings=model_settings,
            settings=wrapper_settings,
            entry_configs=data_configs,
        )
        ```
    """

    @staticmethod
    def create_wrapper(
        model_settings: ModelComponentSettings,
        settings: WrapperComponentSettings,
        wrapper_type: str = "auto",
        entry_configs: tuple[DataEntry, ...] | None = None,
        components: WrapperComponents | None = None,
        **kwargs: Any,
    ) -> CoreLightningWrapper:
        """Create an appropriate Lightning wrapper for the given configuration.

        Args:
            model_settings: Model configuration settings.
            settings: Wrapper configuration settings.
            wrapper_type: Type of wrapper ('auto', 'standard', 'graph', 'timeseries').
            entry_configs: Data entry configurations for pipeline setup.
            components: Pre-built WrapperComponents (required).
            **kwargs: Additional arguments passed to wrapper.

        Returns:
            Appropriate ProcessingLightningWrapper instance.
        """
        if wrapper_type == "auto":
            wrapper_type = WrapperFactory._detect_wrapper_type(model_settings)

        match wrapper_type:
            case "graph":
                return WrapperFactory.create_graph_wrapper(
                    model_settings=model_settings,
                    settings=settings,
                    entry_configs=entry_configs,
                    components=components,
                    **kwargs,
                )
            case "timeseries":
                return WrapperFactory.create_timeseries_wrapper(
                    model_settings=model_settings,
                    settings=settings,
                    entry_configs=entry_configs,
                    components=components,
                    **kwargs,
                )
            case _:
                return WrapperFactory.create_standard_wrapper(
                    model_settings=model_settings,
                    settings=settings,
                    entry_configs=entry_configs,
                    components=components,
                    **kwargs,
                )

    @staticmethod
    def create_standard_wrapper(
        model_settings: ModelComponentSettings,
        settings: WrapperComponentSettings,
        entry_configs: tuple[DataEntry, ...] | None = None,
        components: WrapperComponents | None = None,
        **kwargs: Any,
    ) -> StandardLightningWrapper:
        """Create a standard Lightning wrapper for tensor/TensorDict-based models.

        Args:
            model_settings: Model configuration settings.
            settings: Wrapper configuration settings.
            entry_configs: Data entry configurations.
            components: Pre-built WrapperComponents (required).
            **kwargs: Additional arguments passed to wrapper.

        Returns:
            StandardLightningWrapper instance.

        Raises:
            TypeError: If components is None (must be provided by caller).
        """
        if components is None:
            raise TypeError(
                "WrapperFactory.create_standard_wrapper() requires components. "
                "Call build_wrapper_components() from dlkit.engine.workflows.factories.component_builders first."
            )
        return StandardLightningWrapper(
            model_settings=model_settings,
            settings=settings,
            components=components,
            entry_configs=entry_configs or (),
            **kwargs,
        )

    @staticmethod
    def create_graph_wrapper(
        model_settings: ModelComponentSettings,
        settings: WrapperComponentSettings,
        entry_configs: tuple[DataEntry, ...] | None = None,
        components: WrapperComponents | None = None,
        **kwargs: Any,
    ) -> GraphLightningWrapper:
        """Create a graph Lightning wrapper for PyTorch Geometric models.

        Args:
            model_settings: Model configuration settings.
            settings: Wrapper configuration settings.
            entry_configs: Data entry configurations.
            components: Pre-built WrapperComponents (required).
            **kwargs: Additional arguments passed to wrapper.

        Returns:
            GraphLightningWrapper instance.

        Raises:
            TypeError: If components is None (must be provided by caller).
        """
        if components is None:
            raise TypeError(
                "WrapperFactory.create_graph_wrapper() requires components. "
                "Call build_wrapper_components() from dlkit.engine.workflows.factories.component_builders first."
            )
        return GraphLightningWrapper(
            model_settings=model_settings,
            settings=settings,
            entry_configs=entry_configs,
            components=components,
            **kwargs,
        )

    @staticmethod
    def create_timeseries_wrapper(
        model_settings: ModelComponentSettings,
        settings: WrapperComponentSettings,
        entry_configs: tuple[DataEntry, ...] | None = None,
        components: WrapperComponents | None = None,
        **kwargs: Any,
    ) -> TimeSeriesLightningWrapper:
        """Create a timeseries Lightning wrapper.

        Args:
            model_settings: Model configuration settings.
            settings: Wrapper configuration settings.
            entry_configs: Data entry configurations.
            components: Pre-built WrapperComponents (required).
            **kwargs: Additional arguments passed to wrapper.

        Returns:
            TimeSeriesLightningWrapper instance.

        Raises:
            TypeError: If components is None (must be provided by caller).
        """
        if components is None:
            raise TypeError(
                "WrapperFactory.create_timeseries_wrapper() requires components. "
                "Call build_wrapper_components() from dlkit.engine.workflows.factories.component_builders first."
            )
        return TimeSeriesLightningWrapper(
            model_settings=model_settings,
            settings=settings,
            entry_configs=entry_configs,
            components=components,
            **kwargs,
        )

    @staticmethod
    def _detect_wrapper_type(model_settings: ModelComponentSettings) -> str:
        """Detect the appropriate wrapper type based on model characteristics.

        Uses string inspection (class name, module path) to infer model family
        without instantiating the model (avoids FactoryProvider in core).

        Args:
            model_settings: Model configuration settings to analyze.

        Returns:
            Detected wrapper type string ('standard', 'graph').
        """
        # Extract class name and module path from settings
        name = model_settings.name
        if isinstance(name, type):
            class_name = name.__name__.lower()
            module_name = getattr(name, "__module__", "").lower()
        else:
            class_name = str(name).lower()
            module_path = getattr(model_settings, "module_path", "") or ""
            module_name = module_path.lower()

        # Check for graph indicators
        graph_indicators = ["graph", "gnn", "gcn", "gat", "sage", "gin", "pgnn"]
        if any(indicator in class_name for indicator in graph_indicators):
            return "graph"
        if "torch_geometric" in module_name or "pyg" in module_name:
            return "graph"

        return "standard"

    @staticmethod
    def create_wrapper_from_checkpoint(
        checkpoint_path: str,
        wrapper_type: str = "auto",
        **kwargs: Any,
    ) -> CoreLightningWrapper:
        """Create a wrapper and load it from a checkpoint.

        Args:
            checkpoint_path: Path to the Lightning checkpoint.
            wrapper_type: Type of wrapper to create.
            **kwargs: Additional arguments for wrapper creation.

        Returns:
            Loaded ProcessingLightningWrapper instance.
        """
        match wrapper_type:
            case "standard":
                wrapper_class = StandardLightningWrapper
            case "graph":
                wrapper_class = GraphLightningWrapper
            case _:
                wrapper_class = StandardLightningWrapper

        return wrapper_class.load_from_checkpoint(checkpoint_path, **kwargs)

    @staticmethod
    def get_available_wrapper_types() -> dict[str, type]:
        """Get a mapping of available wrapper types to their classes.

        Returns:
            Dictionary mapping type names to wrapper classes.
        """
        return {
            "standard": StandardLightningWrapper,
            "graph": GraphLightningWrapper,
            "timeseries": TimeSeriesLightningWrapper,
        }

    @staticmethod
    def create_wrapper_with_defaults(
        model: nn.Module,
        wrapper_type: str = "auto",
        **kwargs: Any,
    ) -> CoreLightningWrapper:
        """Create a wrapper with sensible defaults for quick experimentation.

        Args:
            model: PyTorch model to wrap.
            wrapper_type: Type of wrapper to create.
            **kwargs: Additional arguments to override defaults.

        Returns:
            ProcessingLightningWrapper instance with defaults.
        """
        if "settings" not in kwargs:
            kwargs["settings"] = WrapperComponentSettings()

        if "model_settings" not in kwargs:
            kwargs["model_settings"] = ModelComponentSettings(
                name=model.__class__, module_path=model.__class__.__module__
            )

        extra_kwargs = {k: v for k, v in kwargs.items() if k not in {"model_settings", "settings"}}
        return WrapperFactory.create_wrapper(
            model_settings=kwargs["model_settings"],
            settings=kwargs["settings"],
            wrapper_type=wrapper_type,
            **extra_kwargs,
        )
