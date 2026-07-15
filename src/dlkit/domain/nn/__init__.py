from . import attention, cae, contracts, encoder, ffnn, operators, primitives, spectral
from .contracts import (
    EntryConsumer,
    InputSpec,
)
from .ffnn import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthHyper,
    ConstantWidthHyperFactorized,
    ConstantWidthMoE,
    ConstantWidthMoEFactorized,
    ConstantWidthSimpleFactorizedFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedHyper,
    EmbeddedHyperFactorized,
    EmbeddedMoE,
    EmbeddedMoEFactorized,
    EmbeddedParametricFFNN,
    EmbeddedSimpleFactorizedFFNN,
    EmbeddedSimpleParametricFFNN,
    ParametricDenseBlock,
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
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFactorizedFFNN,
    ScaleEquivariantFFNN,
    ScaleEquivariantFiLMEmbeddedFFNN,
    ScaleEquivariantFiLMFFNN,
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
    GraphHyperConnection,
    GraphHyperSequential,
    GRNGate,
    HyperConnection,
    HyperSequential,
    IGatingMechanism,
    LaneExpand,
    LaneMixingStats,
    LaneReduce,
    MoESequential,
    RoutingDecision,
    RoutingStats,
    SparseMoE,
    SwiGLUGate,
    TopKRouter,
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
    "TopKRouter",
    "SparseMoE",
    "RoutingDecision",
    "RoutingStats",
    "LaneExpand",
    "LaneReduce",
    "GraphHyperSequential",
    "HyperSequential",
    "HyperConnection",
    "GraphHyperConnection",
    "LaneMixingStats",
    "MoESequential",
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
