from .constrained import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthParametricFFNN,
    ConstantWidthSimpleFactorizedFFNN,
    ConstantWidthSimpleSPDFactorizedFFNN,
    ConstantWidthSimpleSPDFFNN,
    ConstantWidthSPDFactorizedFFNN,
    ConstantWidthSPDFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedParametricFFNN,
    EmbeddedSimpleFactorizedFFNN,
    EmbeddedSimpleSPDFactorizedFFNN,
    EmbeddedSimpleSPDFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
    ParametricDenseBlock,
)
from .linear import LinearNetwork
from .residual import ConstantWidthFFNN, FeedForwardNN
from .scale_equivariant import (
    ScaleEquivariantConstantWidthFactorizedFFNN,
    ScaleEquivariantConstantWidthFFNN,
    ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
    ScaleEquivariantConstantWidthSimpleFFNN,
    ScaleEquivariantConstantWidthSimpleSPDFactorizedFFNN,
    ScaleEquivariantConstantWidthSimpleSPDFFNN,
    ScaleEquivariantConstantWidthSPDFactorizedFFNN,
    ScaleEquivariantConstantWidthSPDFFNN,
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSPDFFNN,
    ScaleEquivariantEmbeddedSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSPDFFNN,
    ScaleEquivariantFFNN,
)
from .simple import ConstantWidthSimpleFFNN, SimpleFeedForwardNN

__all__ = [
    # Residual dense networks
    "FeedForwardNN",
    "ConstantWidthFFNN",
    # Plain dense networks
    "SimpleFeedForwardNN",
    "ConstantWidthSimpleFFNN",
    # Linear baseline
    "LinearNetwork",
    # Scale-equivariant dense variants
    "ScaleEquivariantFFNN",
    "ScaleEquivariantConstantWidthFFNN",
    "ScaleEquivariantConstantWidthSimpleFFNN",
    # Constrained low-level builders
    "ParametricDenseBlock",
    "ConstantWidthParametricFFNN",
    "EmbeddedParametricFFNN",
    # Constant-width constrained variants
    "ConstantWidthSPDFFNN",
    "ConstantWidthSimpleSPDFFNN",
    "ConstantWidthSPDFactorizedFFNN",
    "ConstantWidthSimpleSPDFactorizedFFNN",
    "ConstantWidthFactorizedFFNN",
    "ConstantWidthSimpleFactorizedFFNN",
    # Embedded constrained variants
    "EmbeddedSPDFFNN",
    "EmbeddedSimpleSPDFFNN",
    "EmbeddedSPDFactorizedFFNN",
    "EmbeddedSimpleSPDFactorizedFFNN",
    "EmbeddedFactorizedFFNN",
    "EmbeddedSimpleFactorizedFFNN",
    # Scale-equivariant constrained variants
    "ScaleEquivariantConstantWidthSPDFFNN",
    "ScaleEquivariantConstantWidthSimpleSPDFFNN",
    "ScaleEquivariantConstantWidthSPDFactorizedFFNN",
    "ScaleEquivariantConstantWidthSimpleSPDFactorizedFFNN",
    "ScaleEquivariantConstantWidthFactorizedFFNN",
    "ScaleEquivariantConstantWidthSimpleFactorizedFFNN",
    "ScaleEquivariantEmbeddedSPDFFNN",
    "ScaleEquivariantEmbeddedSimpleSPDFFNN",
    "ScaleEquivariantEmbeddedSPDFactorizedFFNN",
    "ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN",
    "ScaleEquivariantEmbeddedFactorizedFFNN",
    "ScaleEquivariantEmbeddedSimpleFactorizedFFNN",
]
