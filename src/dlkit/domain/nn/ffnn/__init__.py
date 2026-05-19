from .constrained import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthParametricFFNN,
    ConstantWidthSimpleFactorizedFFNN,
    ConstantWidthSimpleParametricFFNN,
    ConstantWidthSimpleSPDFactorizedFFNN,
    ConstantWidthSimpleSPDFFNN,
    ConstantWidthSPDFactorizedFFNN,
    ConstantWidthSPDFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedParametricFFNN,
    EmbeddedSimpleFactorizedFFNN,
    EmbeddedSimpleParametricFFNN,
    EmbeddedSimpleSPDFactorizedFFNN,
    EmbeddedSimpleSPDFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
    ParametricDenseBlock,
)
from .gated import GatedMLP
from .linear import (
    FactorizedLinearNetwork,
    LinearNetwork,
    SPDFactorizedLinearNetwork,
    SPDLinearNetwork,
    SymmetricFactorizedLinearNetwork,
    SymmetricLinearNetwork,
)
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
    ScaleEquivariantFeedForwardNN,
    ScaleEquivariantSimpleFeedForwardNN,
)
from .simple import ConstantWidthSimpleFFNN, SimpleFeedForwardNN

__all__ = [
    # Residual dense networks
    "FeedForwardNN",
    "ConstantWidthFFNN",
    # Gated networks
    "GatedMLP",
    # Plain dense networks
    "SimpleFeedForwardNN",
    "ConstantWidthSimpleFFNN",
    # Linear baseline
    "LinearNetwork",
    "FactorizedLinearNetwork",
    "SymmetricLinearNetwork",
    "SPDLinearNetwork",
    "SymmetricFactorizedLinearNetwork",
    "SPDFactorizedLinearNetwork",
    # Scale-equivariant dense variants (variable-width)
    "ScaleEquivariantFeedForwardNN",
    "ScaleEquivariantSimpleFeedForwardNN",
    # Scale-equivariant dense variants (constant-width)
    "ScaleEquivariantConstantWidthFFNN",
    "ScaleEquivariantConstantWidthSimpleFFNN",
    # Constrained low-level builders
    "ParametricDenseBlock",
    "ConstantWidthParametricFFNN",
    "ConstantWidthSimpleParametricFFNN",
    "EmbeddedParametricFFNN",
    "EmbeddedSimpleParametricFFNN",
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
