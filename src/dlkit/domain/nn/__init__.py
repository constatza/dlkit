from . import attention, cae, contracts, encoder, ffnn, operators, primitives, spectral
from .contracts import (
    EntryConsumer,
    InputSpec,
)
from .ffnn.constrained import (
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
from .ffnn.film import FiLMBlock, FiLMEmbeddedFFNN, FiLMFFNN, FiLMResidualBlock, VarWidthFiLMFFNN
from .ffnn.gated import GatedMLP
from .ffnn.linear import (
    FactorizedLinearNetwork,
    LinearNetwork,
)
from .ffnn.residual import FFNN, EmbeddedFFNN, VarWidthFFNN
from .ffnn.scale_equivariant import (
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
from .operators import (
    DeepONet,
    EmbeddedDeepONet,
    FFNNDeepONet,
    FourierNeuralOperator1d,
    GridOperatorBase,
    IGridOperator,
    IOperatorNetwork,
    IQueryOperator,
    VarWidthDeepONet,
)
from .parameter_roles import ParameterRole
from .primitives import (
    GatedConvolutionBlock1d,
    GatedDeconvolutionBlock1d,
    GLUGate,
    GRNGate,
    IGatingMechanism,
    SwiGLUGate,
    UVGate,
)
from .spectral import (
    DualPathFFNN,
    FactorizedFourierFeatureNetwork,
    FourierAugmented,
    FourierEnhancedFFNN,
    FourierFeatureNetwork,
    HashEncodingNetwork,
    ISpectralLayer,
    ModifiedMLP,
    ScaleEquivariantFactorizedFourierFeatureNetwork,
    ScaleEquivariantFourierFeatureNetwork,
    ScaleEquivariantModifiedMLP,
    ScaleEquivariantSiren,
    Siren,
    SpectralDualPath,
)

__all__ = [
    # Contracts
    "EntryConsumer",
    "InputSpec",
    "contracts",
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
    # Parameter roles
    "ParameterRole",
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
    # Spectral / frequency-domain networks
    "ISpectralLayer",
    "FourierAugmented",
    "SpectralDualPath",
    "FourierEnhancedFFNN",
    "DualPathFFNN",
    # Coordinate spectral-bias networks
    "FourierFeatureNetwork",
    "FactorizedFourierFeatureNetwork",
    "HashEncodingNetwork",
    "Siren",
    "ModifiedMLP",
    "ScaleEquivariantFactorizedFourierFeatureNetwork",
    "ScaleEquivariantFourierFeatureNetwork",
    "ScaleEquivariantSiren",
    "ScaleEquivariantModifiedMLP",
    # Neural operators
    "IOperatorNetwork",
    "IGridOperator",
    "IQueryOperator",
    "GridOperatorBase",
    "FourierNeuralOperator1d",
    "DeepONet",
    "VarWidthDeepONet",
    "FFNNDeepONet",
    "EmbeddedDeepONet",
    # Gating mechanisms and gated building blocks
    "IGatingMechanism",
    "GLUGate",
    "SwiGLUGate",
    "GRNGate",
    "UVGate",
    "GatedConvolutionBlock1d",
    "GatedDeconvolutionBlock1d",
    # Submodules
    "attention",
    "cae",
    "encoder",
    "ffnn",
    "operators",
    "primitives",
    "spectral",
]
