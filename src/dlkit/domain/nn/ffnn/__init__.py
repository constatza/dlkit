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
from .residual import FFNN, EmbeddedFFNN, VarWidthFFNN
from .scale_equivariant import (
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSPDFFNN,
    ScaleEquivariantEmbeddedSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSPDFFNN,
    ScaleEquivariantFactorizedFFNN,
    ScaleEquivariantFFNN,
    ScaleEquivariantSimpleFactorizedFFNN,
    ScaleEquivariantSimpleSPDFactorizedFFNN,
    ScaleEquivariantSimpleSPDFFNN,
    ScaleEquivariantSPDFactorizedFFNN,
    ScaleEquivariantSPDFFNN,
)

__all__ = [
    # VarWidth (explicit per-layer widths)
    "VarWidthFFNN",
    # Constant-width
    "FFNN",
    "EmbeddedFFNN",
    # Gated
    "GatedMLP",
    # Linear baseline
    "LinearNetwork",
    "FactorizedLinearNetwork",
    "SymmetricLinearNetwork",
    "SPDLinearNetwork",
    "SymmetricFactorizedLinearNetwork",
    "SPDFactorizedLinearNetwork",
    # Scale-equivariant constant-width
    "ScaleEquivariantFFNN",
    # Constrained low-level builders
    "ParametricDenseBlock",
    "EmbeddedParametricFFNN",
    "EmbeddedSimpleParametricFFNN",
    # Embedded SPD variants
    "EmbeddedSPDFFNN",
    "EmbeddedSimpleSPDFFNN",
    "EmbeddedSPDFactorizedFFNN",
    "EmbeddedSimpleSPDFactorizedFFNN",
    # Non-embedded SPD variants
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
