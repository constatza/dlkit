from .constrained import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthSimpleFactorizedFFNN,
    ConstantWidthSoftplusFactorizedFFNN,
    EmbeddedFactorizedEndFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedFullyFactorizedFFNN,
    EmbeddedFullySoftplusFactorizedFFNN,
    EmbeddedParametricFFNN,
    EmbeddedSimpleFactorizedEndFFNN,
    EmbeddedSimpleFactorizedFFNN,
    EmbeddedSimpleFullyFactorizedFFNN,
    EmbeddedSimpleFullySoftplusFactorizedFFNN,
    EmbeddedSimpleParametricFFNN,
    EmbeddedSimpleSoftplusFactorizedEndFFNN,
    EmbeddedSimpleSoftplusFactorizedFFNN,
    EmbeddedSoftplusFactorizedEndFFNN,
    EmbeddedSoftplusFactorizedFFNN,
    FactorizedFFNN,
    ParametricDenseBlock,
    SimpleFactorizedFFNN,
)
from .film import FiLMBlock, FiLMEmbeddedFFNN, FiLMFFNN, FiLMResidualBlock, VarWidthFiLMFFNN
from .gated import GatedMLP
from .linear import (
    FactorizedLinearNetwork,
    LinearNetwork,
)
from .residual import FFNN, EmbeddedFFNN, VarWidthFFNN
from .scale_equivariant import (
    ScaleEquivariantConstantWidthFactorizedFFNN,
    ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
    ScaleEquivariantConstantWidthSoftplusFactorizedFFNN,
    ScaleEquivariantEmbeddedFactorizedEndFFNN,
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedFullyFactorizedFFNN,
    ScaleEquivariantEmbeddedFullySoftplusFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFactorizedEndFFNN,
    ScaleEquivariantEmbeddedSimpleFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFullyFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFullySoftplusFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSoftplusFactorizedEndFFNN,
    ScaleEquivariantEmbeddedSimpleSoftplusFactorizedFFNN,
    ScaleEquivariantEmbeddedSoftplusFactorizedEndFFNN,
    ScaleEquivariantEmbeddedSoftplusFactorizedFFNN,
    ScaleEquivariantFactorizedFFNN,
    ScaleEquivariantFFNN,
    ScaleEquivariantFiLMEmbeddedFFNN,
    ScaleEquivariantFiLMFFNN,
    ScaleEquivariantSimpleFactorizedFFNN,
    ScaleEquivariantVarWidthFiLMFFNN,
)

__all__ = [
    # VarWidth (explicit per-layer widths)
    "VarWidthFFNN",
    # Constant-width
    "FFNN",
    "EmbeddedFFNN",
    # FiLM-conditioned
    "FiLMBlock",
    "FiLMEmbeddedFFNN",
    "FiLMFFNN",
    "FiLMResidualBlock",
    "VarWidthFiLMFFNN",
    # Gated
    "GatedMLP",
    # Linear baseline
    "LinearNetwork",
    "FactorizedLinearNetwork",
    # Scale-equivariant constant-width
    "ScaleEquivariantFFNN",
    # Scale-equivariant FiLM-conditioned
    "ScaleEquivariantFiLMFFNN",
    "ScaleEquivariantFiLMEmbeddedFFNN",
    "ScaleEquivariantVarWidthFiLMFFNN",
    # Constrained low-level builders
    "ParametricDenseBlock",
    "EmbeddedParametricFFNN",
    "EmbeddedSimpleParametricFFNN",
    "ConstantWidthFactorizedFFNN",
    "ConstantWidthSimpleFactorizedFFNN",
    "ConstantWidthSoftplusFactorizedFFNN",
    # Embedded Factorized variants
    "EmbeddedFactorizedFFNN",
    "EmbeddedSimpleFactorizedFFNN",
    # Embedded Softplus-Factorized variants
    "EmbeddedSoftplusFactorizedFFNN",
    "EmbeddedSimpleSoftplusFactorizedFFNN",
    # Embedded FactorizedEnd variants (plain Linear embedding, FactorizedLinear regression)
    "EmbeddedFactorizedEndFFNN",
    "EmbeddedSimpleFactorizedEndFFNN",
    "EmbeddedSoftplusFactorizedEndFFNN",
    "EmbeddedSimpleSoftplusFactorizedEndFFNN",
    # Embedded FullyFactorized variants (FactorizedLinear embedding, body, and regression)
    "EmbeddedFullyFactorizedFFNN",
    "EmbeddedSimpleFullyFactorizedFFNN",
    "EmbeddedFullySoftplusFactorizedFFNN",
    "EmbeddedSimpleFullySoftplusFactorizedFFNN",
    # Non-embedded Factorized variants
    "FactorizedFFNN",
    "SimpleFactorizedFFNN",
    # Scale-equivariant Factorized variants
    "ScaleEquivariantEmbeddedFactorizedFFNN",
    "ScaleEquivariantEmbeddedSimpleFactorizedFFNN",
    # Scale-equivariant Embedded Softplus-Factorized variants
    "ScaleEquivariantEmbeddedSoftplusFactorizedFFNN",
    "ScaleEquivariantEmbeddedSimpleSoftplusFactorizedFFNN",
    # Scale-equivariant FactorizedEnd variants
    "ScaleEquivariantEmbeddedFactorizedEndFFNN",
    "ScaleEquivariantEmbeddedSimpleFactorizedEndFFNN",
    "ScaleEquivariantEmbeddedSoftplusFactorizedEndFFNN",
    "ScaleEquivariantEmbeddedSimpleSoftplusFactorizedEndFFNN",
    # Scale-equivariant FullyFactorized variants
    "ScaleEquivariantEmbeddedFullyFactorizedFFNN",
    "ScaleEquivariantEmbeddedSimpleFullyFactorizedFFNN",
    "ScaleEquivariantEmbeddedFullySoftplusFactorizedFFNN",
    "ScaleEquivariantEmbeddedSimpleFullySoftplusFactorizedFFNN",
    "ScaleEquivariantFactorizedFFNN",
    "ScaleEquivariantSimpleFactorizedFFNN",
    "ScaleEquivariantConstantWidthFactorizedFFNN",
    "ScaleEquivariantConstantWidthSimpleFactorizedFFNN",
    "ScaleEquivariantConstantWidthSoftplusFactorizedFFNN",
]
