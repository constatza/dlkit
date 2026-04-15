from .linear import LinearNetwork
from .parametric import (
    ConstantWidthParametricFFNN,
    EmbeddedParametricFFNN,
    ParametricDenseBlock,
)
from .parametric_variants import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthSPDFactorizedFFNN,
    ConstantWidthSPDFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
)
from .plain import ConstantWidthSimpleFFNN, SimpleFeedForwardNN
from .scale_equivariant import (
    ScaleEquivariantConstantWidthFFNN,
    ScaleEquivariantFFNN,
)
from .scale_equivariant_deep import (
    ScaleEquivariantConstantWidthFactorizedFFNN,
    ScaleEquivariantConstantWidthSPDFactorizedFFNN,
    ScaleEquivariantConstantWidthSPDFFNN,
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSPDFFNN,
)
from .simple import ConstantWidthFFNN, FeedForwardNN

__all__ = [
    # Residual networks
    "FeedForwardNN",
    "ConstantWidthFFNN",
    # Plain (no residual) networks
    "SimpleFeedForwardNN",
    "ConstantWidthSimpleFFNN",
    # Linear baseline
    "LinearNetwork",
    # ScaleEquivariant
    "ScaleEquivariantFFNN",
    "ScaleEquivariantConstantWidthFFNN",
    # Parametric base classes
    "ParametricDenseBlock",
    "ConstantWidthParametricFFNN",
    "EmbeddedParametricFFNN",
    # Constant-width parametric variants
    "ConstantWidthSPDFFNN",
    "ConstantWidthSPDFactorizedFFNN",
    "ConstantWidthFactorizedFFNN",
    # Embedded parametric variants
    "EmbeddedSPDFFNN",
    "EmbeddedSPDFactorizedFFNN",
    "EmbeddedFactorizedFFNN",
    # ScaleEquivariant deep variants
    "ScaleEquivariantConstantWidthSPDFFNN",
    "ScaleEquivariantConstantWidthSPDFactorizedFFNN",
    "ScaleEquivariantConstantWidthFactorizedFFNN",
    "ScaleEquivariantEmbeddedSPDFFNN",
    "ScaleEquivariantEmbeddedSPDFactorizedFFNN",
    "ScaleEquivariantEmbeddedFactorizedFFNN",
]
