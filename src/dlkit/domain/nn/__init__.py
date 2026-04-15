from . import attention, cae, encoder, ffnn, graph, operators, primitives, spectral
from .ffnn.linear import LinearNetwork
from .ffnn.parametric import (
    ConstantWidthParametricFFNN,
    EmbeddedParametricFFNN,
    ParametricDenseBlock,
)
from .ffnn.parametric_variants import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthSPDFactorizedFFNN,
    ConstantWidthSPDFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
)
from .ffnn.plain import ConstantWidthSimpleFFNN, SimpleFeedForwardNN
from .ffnn.scale_equivariant import (
    ScaleEquivariantConstantWidthFFNN,
    ScaleEquivariantFFNN,
)
from .ffnn.scale_equivariant_deep import (
    ScaleEquivariantConstantWidthFactorizedFFNN,
    ScaleEquivariantConstantWidthSPDFactorizedFFNN,
    ScaleEquivariantConstantWidthSPDFFNN,
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSPDFFNN,
)
from .operators import (
    DeepONet,
    FourierNeuralOperator1d,
    GridOperatorBase,
    IGridOperator,
    IOperatorNetwork,
    IQueryOperator,
    MLPDeepONet,
)
from .spectral import (
    DualPathFFNN,
    FourierAugmented,
    FourierEnhancedFFNN,
    ISpectralLayer,
    SpectralDualPath,
)

__all__ = [
    "LinearNetwork",
    # ScaleEquivariant
    "ScaleEquivariantFFNN",
    "ScaleEquivariantConstantWidthFFNN",
    # Plain (no residual)
    "SimpleFeedForwardNN",
    "ConstantWidthSimpleFFNN",
    # Parametric base classes
    "ParametricDenseBlock",
    "ConstantWidthParametricFFNN",
    "EmbeddedParametricFFNN",
    # Constant-width parametric variants
    "ConstantWidthSPDFFNN",
    "ConstantWidthSPDFactorizedFFNN",
    "ConstantWidthFactorizedFFNN",
    # Embedded parametric variants
    "EmbeddedSPDFFNN",
    "EmbeddedSPDFactorizedFFNN",
    "EmbeddedFactorizedFFNN",
    # ScaleEquivariant deep variants
    "ScaleEquivariantConstantWidthSPDFFNN",
    "ScaleEquivariantConstantWidthSPDFactorizedFFNN",
    "ScaleEquivariantConstantWidthFactorizedFFNN",
    "ScaleEquivariantEmbeddedSPDFFNN",
    "ScaleEquivariantEmbeddedSPDFactorizedFFNN",
    "ScaleEquivariantEmbeddedFactorizedFFNN",
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
