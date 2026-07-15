from .constrained import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthSimpleFactorizedFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedParametricFFNN,
    EmbeddedSimpleFactorizedFFNN,
    EmbeddedSimpleParametricFFNN,
    ParametricDenseBlock,
)
from .factorized_composites import (
    ConstantWidthHyper,
    ConstantWidthHyperFactorized,
    ConstantWidthMoE,
    ConstantWidthMoEFactorized,
    EmbeddedHyper,
    EmbeddedHyperFactorized,
    EmbeddedMoE,
    EmbeddedMoEFactorized,
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
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFactorizedFFNN,
    ScaleEquivariantFFNN,
    ScaleEquivariantFiLMEmbeddedFFNN,
    ScaleEquivariantFiLMFFNN,
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
    # Embedded Factorized variants (FactorizedLinear embedding, body, and regression)
    "EmbeddedFactorizedFFNN",
    "EmbeddedSimpleFactorizedFFNN",
    # Hyper/MoE variants
    "ConstantWidthHyper",
    "ConstantWidthHyperFactorized",
    "EmbeddedHyper",
    "EmbeddedHyperFactorized",
    "ConstantWidthMoE",
    "ConstantWidthMoEFactorized",
    "EmbeddedMoE",
    "EmbeddedMoEFactorized",
    # Scale-equivariant Factorized variants
    "ScaleEquivariantEmbeddedFactorizedFFNN",
    "ScaleEquivariantEmbeddedSimpleFactorizedFFNN",
    "ScaleEquivariantConstantWidthFactorizedFFNN",
    "ScaleEquivariantConstantWidthSimpleFactorizedFFNN",
]
