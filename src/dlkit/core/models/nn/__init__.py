from .base import DLKitModel

# Import commonly used models for convenience
from .ffnn.linear import LinearNetwork
from .ffnn.norm_scaled import (
    NormScaledFFNN,
    NormScaledLinearFFNN,
    NormScaledConstantWidthFFNN,
)

# Import new wrappers directly
from dlkit.core.models.wrappers import (
    StandardLightningWrapper as BaseWrapper,
    GraphLightningWrapper,
    WrapperFactory,
)

# Expose all submodules for convenient access
from . import attention
from . import cae
from . import encoder
from . import ffnn
from . import graph
from . import primitives

__all__ = [
    "DLKitModel",
    "LinearNetwork",
    "NormScaledFFNN",
    "NormScaledLinearFFNN",
    "NormScaledConstantWidthFFNN",
    "BaseWrapper",
    "GraphLightningWrapper",
    "WrapperFactory",
    "attention",
    "cae",
    "encoder",
    "ffnn",
    "graph",
    "primitives",
]
