from . import attention, cae, contracts, encoder, ffnn, graph, operators, primitives, spectral
from .contracts import (
    BranchTrunkSpec,
    ContractConsumer,
    GraphContractSpec,
    GridOperatorSpec,
    ModelContractSpec,
    SequenceSpec,
    TabulaRSpec,
)
from .ffnn.constrained import (
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
from .ffnn.gated import GatedMLP
from .ffnn.linear import (
    FactorizedLinearNetwork,
    LinearNetwork,
    SPDFactorizedLinearNetwork,
    SPDLinearNetwork,
    SymmetricFactorizedLinearNetwork,
    SymmetricLinearNetwork,
)
from .ffnn.residual import FFNN, EmbeddedFFNN, VarWidthFFNN
from .ffnn.scale_equivariant import (
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
from .graph import (
    GATv2Message,
    GATv2Projection,
    ScaledGATv2Projection,
    ScaledSimpleGATv2Projection,
    SimpleGATv2Message,
    SimpleGATv2Projection,
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
    FourierAugmented,
    FourierEnhancedFFNN,
    FourierFeatureNetwork,
    HashEncodingNetwork,
    ISpectralLayer,
    ModifiedMLP,
    ScaleEquivariantFourierFeatureNetwork,
    ScaleEquivariantModifiedMLP,
    ScaleEquivariantSiren,
    Siren,
    SpectralDualPath,
)

__all__ = [
    # Contracts
    "BranchTrunkSpec",
    "ContractConsumer",
    "GraphContractSpec",
    "GridOperatorSpec",
    "ModelContractSpec",
    "SequenceSpec",
    "TabulaRSpec",
    "contracts",
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
    # Parameter roles
    "ParameterRole",
    # Scale-equivariant constant-width
    "ScaleEquivariantFFNN",
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
    # Graph neural networks
    "GATv2Message",
    "SimpleGATv2Message",
    "GATv2Projection",
    "SimpleGATv2Projection",
    "ScaledGATv2Projection",
    "ScaledSimpleGATv2Projection",
    # Spectral / frequency-domain networks
    "ISpectralLayer",
    "FourierAugmented",
    "SpectralDualPath",
    "FourierEnhancedFFNN",
    "DualPathFFNN",
    # Coordinate spectral-bias networks
    "FourierFeatureNetwork",
    "HashEncodingNetwork",
    "Siren",
    "ModifiedMLP",
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
    "graph",
    "operators",
    "primitives",
    "spectral",
]
