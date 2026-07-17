from .constrained import (
    EmbeddedFactorizedFFNN,
    FactorizedFFNN,
    ParametricDenseBlock,
)
from .film import FiLMBlock, FiLMEmbeddedFFNN, FiLMFFNN, FiLMResidualBlock, VarWidthFiLMFFNN
from .gated import GatedMLP
from .hyper_moe import EmbeddedHyperFFNN, EmbeddedMoEFFNN
from .linear import (
    FactorizedLinearNetwork,
    LinearNetwork,
)
from .residual import FFNN, VarWidthFFNN
from .scale_equivariant import (
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedHyperFFNN,
    ScaleEquivariantEmbeddedMoEFFNN,
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
    # Factorized variants
    "FactorizedFFNN",
    "EmbeddedFactorizedFFNN",
    # Hyper/MoE variants
    "EmbeddedHyperFFNN",
    "EmbeddedMoEFFNN",
    # Scale-equivariant Factorized variants
    "ScaleEquivariantEmbeddedFactorizedFFNN",
    # Scale-equivariant Hyper/MoE variants
    "ScaleEquivariantEmbeddedHyperFFNN",
    "ScaleEquivariantEmbeddedMoEFFNN",
]
