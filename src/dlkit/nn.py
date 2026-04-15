"""User-facing neural network namespace.

Thin re-export of ``dlkit.domain.nn`` so users can write::

    from dlkit.nn import ConstantWidthFFNN, ScaleEquivariantFFNN
    import dlkit.nn as nn

instead of::

    from dlkit.domain.nn import ConstantWidthFFNN
"""

from dlkit.domain.nn import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthParametricFFNN,
    ConstantWidthSimpleFFNN,
    ConstantWidthSPDFactorizedFFNN,
    ConstantWidthSPDFFNN,
    DeepONet,
    DualPathFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedParametricFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
    FourierAugmented,
    FourierEnhancedFFNN,
    FourierNeuralOperator1d,
    GridOperatorBase,
    IGridOperator,
    IOperatorNetwork,
    IQueryOperator,
    ISpectralLayer,
    LinearNetwork,
    MLPDeepONet,
    ParametricDenseBlock,
    ScaleEquivariantConstantWidthFactorizedFFNN,
    ScaleEquivariantConstantWidthFFNN,
    ScaleEquivariantConstantWidthSPDFactorizedFFNN,
    ScaleEquivariantConstantWidthSPDFFNN,
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSPDFFNN,
    ScaleEquivariantFFNN,
    SimpleFeedForwardNN,
    SpectralDualPath,
    attention,
    cae,
    encoder,
    ffnn,
    graph,
    operators,
    primitives,
    spectral,
)

__all__ = [
    "LinearNetwork",
    # ScaleEquivariant
    "ScaleEquivariantFFNN",
    "ScaleEquivariantConstantWidthFFNN",
    # Plain
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
