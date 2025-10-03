"""Factory classes for creating Lightning wrappers.

This module provides factory classes that automatically detect the appropriate
wrapper type based on model characteristics and provide convenient creation methods.
"""

import warnings

from torch import nn

from dlkit.tools.config import (
    BuildContext,
    FactoryProvider,
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.tools.config.data_entries import DataEntry
from dlkit.core.shape_specs import IShapeSpec
from .base import ProcessingLightningWrapper
from .standard import StandardLightningWrapper, BareWrapper
from .graph import GraphLightningWrapper
from .timeseries import TimeSeriesLightningWrapper


class WrapperFactory:
    """Factory for creating appropriate Lightning wrappers.
    
    This factory uses heuristics to determine the best wrapper type for a given
    model and provides convenient methods for wrapper creation.
    
    Example:
        ```python
        # Automatic wrapper detection
        wrapper = WrapperFactory.create_wrapper(
            model_settings=model_settings,
            settings=wrapper_settings,
            shape_spec=shape_spec
        )
        
        # Explicit wrapper type
        wrapper = WrapperFactory.create_standard_wrapper(
            model_settings=model_settings,
            settings=wrapper_settings,
            shape_spec=shape_spec
        )
        ```
    """
    
    @staticmethod
    def create_wrapper(
        model_settings: ModelComponentSettings,
        settings: WrapperComponentSettings,
        wrapper_type: str = "auto",
        entry_configs: dict[str, DataEntry] | None = None,
        **kwargs
    ):
        """Create an appropriate Lightning wrapper for the given configuration.
        
        Args:
            model_settings: Model configuration settings
            settings: Wrapper configuration settings
            wrapper_type: Type of wrapper ("auto", "standard", "graph", "bare")
            entry_configs: Data entry configurations for pipeline setup
            **kwargs: Additional arguments passed to wrapper
            
        Returns:
            Appropriate ProcessingLightningWrapper instance
        """
        if wrapper_type == "auto":
            wrapper_type = WrapperFactory._detect_wrapper_type(model_settings)
        
        # Create wrapper based on detected/specified type
        if wrapper_type == "graph":
            return WrapperFactory.create_graph_wrapper(
                model_settings=model_settings,
                settings=settings,
                entry_configs=entry_configs,
                **kwargs
            )
        elif wrapper_type == "timeseries":
            return WrapperFactory.create_timeseries_wrapper(
                model_settings=model_settings,
                settings=settings,
                entry_configs=entry_configs,
                **kwargs
            )
        elif wrapper_type == "bare":
            return WrapperFactory.create_bare_wrapper(
                model_settings=model_settings,
                **kwargs
            )
        else:  # "standard"
            return WrapperFactory.create_standard_wrapper(
                model_settings=model_settings,
                settings=settings,
                entry_configs=entry_configs,
                **kwargs
            )
    
    @staticmethod
    def create_standard_wrapper(
        model_settings: ModelComponentSettings,
        settings: WrapperComponentSettings,
        entry_configs: dict[str, DataEntry] | None = None,
        shape_spec: IShapeSpec | None = None,
        unified_shape: IShapeSpec | None = None,
        **kwargs
    ) -> StandardLightningWrapper:
        """Create a standard Lightning wrapper for tensor-based models.

        Args:
            model_settings: Model configuration settings
            settings: Wrapper configuration settings
            entry_configs: Data entry configurations for pipeline setup
            shape_spec: Shape specification inferred upstream
            unified_shape: Explicit shape specification for ABC models
            **kwargs: Additional arguments passed to wrapper

        Returns:
            StandardLightningWrapper instance
        """
        # Prefer unified_shape over legacy shape_spec
        final_shape_spec = unified_shape or shape_spec

        return StandardLightningWrapper(
            model_settings=model_settings,
            settings=settings,
            entry_configs=entry_configs,
            shape_spec=final_shape_spec,
            **kwargs
        )
    
    @staticmethod
    def create_graph_wrapper(
        model_settings: ModelComponentSettings,
        settings: WrapperComponentSettings,
        entry_configs: dict[str, DataEntry] | None = None,
        shape_spec: IShapeSpec | None = None,
        **kwargs
    ) -> GraphLightningWrapper:
        """Create a graph Lightning wrapper for PyTorch Geometric models.

        Args:
            model_settings: Model configuration settings
            settings: Wrapper configuration settings
            entry_configs: Data entry configurations for pipeline setup
            shape_spec: Shape specification for graph models
            **kwargs: Additional arguments passed to wrapper

        Returns:
            GraphLightningWrapper instance
        """
        return GraphLightningWrapper(
            model_settings=model_settings,
            settings=settings,
            entry_configs=entry_configs,
            shape_spec=shape_spec,
            **kwargs
        )
    
    @staticmethod
    def create_bare_wrapper(
        model_settings: ModelComponentSettings,
        **kwargs
    ) -> BareWrapper:
        """Create a bare Lightning wrapper with minimal functionality.
        
        Args:
            model_settings: Model configuration settings
            **kwargs: Additional arguments passed to wrapper
            
        Returns:
            BareWrapper instance
        """
        return BareWrapper(
            model_settings=model_settings,
            **kwargs
        )

    @staticmethod
    def create_timeseries_wrapper(
        model_settings: ModelComponentSettings,
        settings: WrapperComponentSettings,
        entry_configs: dict[str, DataEntry] | None = None,
        shape_spec: IShapeSpec | None = None,
        **kwargs
    ) -> TimeSeriesLightningWrapper:
        """Create a timeseries Lightning wrapper.

        Args:
            model_settings: Model configuration settings
            settings: Wrapper configuration settings
            entry_configs: Data entry configurations for pipeline setup
            shape_spec: Shape specification for timeseries models
            **kwargs: Additional arguments passed to wrapper
        """
        return TimeSeriesLightningWrapper(
            model_settings=model_settings,
            settings=settings,
            entry_configs=entry_configs,
            shape_spec=shape_spec,
            **kwargs
        )
    
    @staticmethod
    def _detect_wrapper_type(model_settings: ModelComponentSettings) -> str:
        """Detect the appropriate wrapper type based on model characteristics.
        
        Args:
            model_settings: Model configuration settings to analyze
            
        Returns:
            Detected wrapper type string ("standard", "graph")
        """
        # Build model to inspect its characteristics
        try:
            model = FactoryProvider.create_component(model_settings, BuildContext(mode="inspection"))
            model_name = model.__class__.__name__.lower()
            model_module = model.__class__.__module__.lower()
            
            # Check for graph models
            graph_indicators = [
                'graph', 'gnn', 'gcn', 'gat', 'sage', 'gin', 'pgnn'
            ]
            
            if any(indicator in model_name for indicator in graph_indicators):
                return "graph"
            
            if 'torch_geometric' in model_module or 'pyg' in model_module:
                return "graph"
            
            # Check for graph-related methods
            if hasattr(model, 'forward') and hasattr(model.forward, '__annotations__'):
                annotations = model.forward.__annotations__
                for param_type in annotations.values():
                    if hasattr(param_type, '__name__'):
                        type_name = param_type.__name__.lower()
                        if 'dataflow' in type_name and 'graph' in type_name:
                            return "graph"
            
            # Default to standard for regular neural networks
            return "standard"
            
        except Exception:
            # If model building fails, default to standard
            warnings.warn(
                "Could not build model for wrapper type detection, defaulting to 'standard'",
                UserWarning
            )
            return "standard"
    
    @staticmethod
    def create_wrapper_from_checkpoint(
        checkpoint_path: str,
        wrapper_type: str = "auto",
        **kwargs
    ):
        """Create a wrapper and load it from a checkpoint.
        
        Args:
            checkpoint_path: Path to the Lightning checkpoint
            wrapper_type: Type of wrapper to create
            **kwargs: Additional arguments for wrapper creation
            
        Returns:
            Loaded ProcessingLightningWrapper instance
        """
        # Determine wrapper class
        if wrapper_type == "standard":
            wrapper_class = StandardLightningWrapper
        elif wrapper_type == "graph":
            wrapper_class = GraphLightningWrapper
        elif wrapper_type == "bare":
            wrapper_class = BareWrapper
        else:
            # Auto-detect from checkpoint if possible
            wrapper_class = StandardLightningWrapper
        
        # Load from checkpoint
        return wrapper_class.load_from_checkpoint(checkpoint_path, **kwargs)
    
    @staticmethod
    def get_available_wrapper_types():
        """Get a mapping of available wrapper types to their classes.
        
        Returns:
            Dictionary mapping type names to wrapper classes
        """
        return {
            "standard": StandardLightningWrapper,
            "graph": GraphLightningWrapper,
            "timeseries": TimeSeriesLightningWrapper,
            "bare": BareWrapper,
        }
    
    @staticmethod
    def create_wrapper_with_defaults(
        model: nn.Module,
        wrapper_type: str = "auto",
        **kwargs
    ):
        """Create a wrapper with sensible defaults for quick experimentation.
        
        Args:
            model: PyTorch model to wrap
            wrapper_type: Type of wrapper to create
            **kwargs: Additional arguments to override defaults
            
        Returns:
            ProcessingLightningWrapper instance with defaults
        """
        from dlkit.tools.config import WrapperComponentSettings
        
        # Create default settings
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
