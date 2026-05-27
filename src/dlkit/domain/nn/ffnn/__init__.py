from .constrained import (
    SPDFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedParametricFFNN,
    EmbeddedSimpleFactorizedFFNN,
    EmbeddedSimpleParametricFFNN,
    EmbeddedSimpleSPDFactorizedFFNN,
    EmbeddedSimpleSPDFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
    FactorizedFFNN,
    ParametricDenseBlock,
    SimpleFactorizedFFNN,
    SimpleSPDFactorizedFFNN,
    SimpleSPDFFNN,
    SPDFactorizedFFNN,
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
    ScaleEquivariantConstantWidthFFNN,
    ScaleEquivariantConstantWidthSimpleFFNN,
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSPDFFNN,
    ScaleEquivariantEmbeddedSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSPDFFNN,
    ScaleEquivariantFactorizedFFNN,
    ScaleEquivariantFeedForwardNN,
    ScaleEquivariantSimpleFactorizedFFNN,
    ScaleEquivariantSimpleFeedForwardNN,
    ScaleEquivariantSimpleSPDFactorizedFFNN,
    ScaleEquivariantSimpleSPDFFNN,
    ScaleEquivariantSPDFactorizedFFNN,
    ScaleEquivariantSPDFFNN,
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
    "EmbeddedParametricFFNN",
    "EmbeddedSimpleParametricFFNN",
    # Embedded SPD variants (all-SPD, square)
    "EmbeddedSPDFFNN",
    "EmbeddedSimpleSPDFFNN",
    "EmbeddedSPDFactorizedFFNN",
    "EmbeddedSimpleSPDFactorizedFFNN",
    # Non-embedded SPD variants (all-SPD, square)
    "SPDFFNN",
    "SimpleSPDFFNN",
    "SPDFactorizedFFNN",
    "SimpleSPDFactorizedFFNN",
    # Embedded Factorized variants
    "EmbeddedFactorizedFFNN",
    "EmbeddedSimpleFactorizedFFNN",
    # Non-embedded Factorized variants
    "FactorizedFFNN",
    "SimpleFactorizedFFNN",
    # Scale-equivariant embedded SPD variants
    "ScaleEquivariantEmbeddedSPDFFNN",
    "ScaleEquivariantEmbeddedSimpleSPDFFNN",
    "ScaleEquivariantEmbeddedSPDFactorizedFFNN",
    "ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN",
    # Scale-equivariant non-embedded SPD variants
    "ScaleEquivariantSPDFFNN",
    "ScaleEquivariantSimpleSPDFFNN",
    "ScaleEquivariantSPDFactorizedFFNN",
    "ScaleEquivariantSimpleSPDFactorizedFFNN",
    # Scale-equivariant Factorized variants
    "ScaleEquivariantEmbeddedFactorizedFFNN",
    "ScaleEquivariantEmbeddedSimpleFactorizedFFNN",
    "ScaleEquivariantFactorizedFFNN",
    "ScaleEquivariantSimpleFactorizedFFNN",
]
