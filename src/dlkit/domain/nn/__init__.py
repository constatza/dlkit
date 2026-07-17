from . import attention, cae, contracts, encoder, ffnn, operators, primitives, spectral
from .contracts import (
    EntryConsumer,
    InputSpec,
)
from .ffnn import (
    EmbeddedFactorizedFFNN,
    EmbeddedHyperFFNN,
    EmbeddedMoEFFNN,
    FactorizedFFNN,
)
from .ffnn.film import FiLMBlock, FiLMEmbeddedFFNN, FiLMFFNN, FiLMResidualBlock, VarWidthFiLMFFNN
from .ffnn.gated import GatedMLP
from .ffnn.linear import (
    FactorizedLinearNetwork,
    LinearNetwork,
)
from .ffnn.residual import FFNN, VarWidthFFNN
from .ffnn.scale_equivariant import (
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedHyperFFNN,
    ScaleEquivariantEmbeddedMoEFFNN,
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
    HyperDeepONet,
    IGridOperator,
    IOperatorNetwork,
    IQueryOperator,
    MoEDeepONet,
    VarWidthDeepONet,
)
from .parameter_roles import ParameterRole
from .primitives import (
    DenseBlock,
    DenseBlockKind,
    DenseLinearKind,
    DenseMLPBlock,
    GateActivationName,
    GatedConvolutionBlock1d,
    GatedDeconvolutionBlock1d,
    GatedDenseMLPBlock,
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
    ParametricDenseBlock,
    RoutingDecision,
    RoutingStats,
    SparseMoE,
    SwiGLUGate,
    TopKRouter,
    UVGate,
    make_dense_block,
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
    "DenseBlock",
    "DenseBlockKind",
    "DenseLinearKind",
    "DenseMLPBlock",
    "GatedDenseMLPBlock",
    "GateActivationName",
    "make_dense_block",
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
    "HyperDeepONet",
    "MoEDeepONet",
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
