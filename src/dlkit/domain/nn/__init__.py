from . import attention, cae, encoder, ffnn, graph, operators, primitives, spectral
from .ffnn.constrained import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthParametricFFNN,
    ConstantWidthSimpleFactorizedFFNN,
    ConstantWidthSimpleSPDFactorizedFFNN,
    ConstantWidthSimpleSPDFFNN,
    ConstantWidthSPDFactorizedFFNN,
    ConstantWidthSPDFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedParametricFFNN,
    EmbeddedSimpleFactorizedFFNN,
    EmbeddedSimpleSPDFactorizedFFNN,
    EmbeddedSimpleSPDFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
    ParametricDenseBlock,
)
from .ffnn.linear import LinearNetwork
from .ffnn.residual import ConstantWidthFFNN, FeedForwardNN
from .ffnn.scale_equivariant import (
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
    ScaleEquivariantFFNN,
)
from .ffnn.simple import ConstantWidthSimpleFFNN, SimpleFeedForwardNN
from .operators import (
    DeepONet,
    FourierNeuralOperator1d,
    GridOperatorBase,
    IGridOperator,
    IOperatorNetwork,
    IQueryOperator,
    MLPDeepONet,
)
from .parameter_roles import ParameterRole
from .role_provider import IParameterRoleProvider
from .spectral import (
    DualPathFFNN,
    FourierAugmented,
    FourierEnhancedFFNN,
    ISpectralLayer,
    SpectralDualPath,
)

__all__ = [
    "LinearNetwork",
    "FeedForwardNN",
    "ConstantWidthFFNN",
    # Parameter roles and protocols
    "ParameterRole",
    "IParameterRoleProvider",
    # ScaleEquivariant dense variants
    "ScaleEquivariantFFNN",
    "ScaleEquivariantConstantWidthFFNN",
    "ScaleEquivariantConstantWidthSimpleFFNN",
    # Plain dense variants
    "SimpleFeedForwardNN",
    "ConstantWidthSimpleFFNN",
    # Constrained low-level builders
    "ParametricDenseBlock",
    "ConstantWidthParametricFFNN",
    "EmbeddedParametricFFNN",
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
    # ScaleEquivariant constrained variants
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
    # Spectral / frequency-domain networks
    "ISpectralLayer",
    "FourierAugmented",
    "SpectralDualPath",
    "FourierEnhancedFFNN",
    "DualPathFFNN",
    # Neural operators
    "IOperatorNetwork",
    "IGridOperator",
    "IQueryOperator",
    "GridOperatorBase",
    "FourierNeuralOperator1d",
    "DeepONet",
    "MLPDeepONet",
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
