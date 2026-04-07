"""User-facing neural network namespace.

Thin re-export of ``dlkit.domain.nn`` so users can write::

    from dlkit.nn import ConstantWidthFFNN, NormScaledFFNN
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
    EmbeddedFactorizedFFNN,
    EmbeddedParametricFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
    LinearNetwork,
    NormScaledConstantWidthFactorizedFFNN,
    NormScaledConstantWidthFFNN,
    NormScaledConstantWidthSPDFactorizedFFNN,
    NormScaledConstantWidthSPDFFNN,
    NormScaledEmbeddedFactorizedFFNN,
    NormScaledEmbeddedSPDFactorizedFFNN,
    NormScaledEmbeddedSPDFFNN,
    NormScaledFactorizedLinear,
    NormScaledFFNN,
    NormScaledLinearFFNN,
    NormScaledSPDFactorizedLinear,
    NormScaledSPDLinear,
    NormScaledSymmetricFactorizedLinear,
    NormScaledSymmetricLinear,
    ParametricDenseBlock,
    SimpleFeedForwardNN,
    attention,
    cae,
    encoder,
    ffnn,
    graph,
    primitives,
)

__all__ = [
    "LinearNetwork",
    # NormScaled single-layer
    "NormScaledFFNN",
    "NormScaledLinearFFNN",
    "NormScaledConstantWidthFFNN",
    "NormScaledSymmetricLinear",
    "NormScaledSPDLinear",
    "NormScaledFactorizedLinear",
    "NormScaledSymmetricFactorizedLinear",
    "NormScaledSPDFactorizedLinear",
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
    # NormScaled deep variants
    "NormScaledConstantWidthSPDFFNN",
    "NormScaledConstantWidthSPDFactorizedFFNN",
    "NormScaledConstantWidthFactorizedFFNN",
    "NormScaledEmbeddedSPDFFNN",
    "NormScaledEmbeddedSPDFactorizedFFNN",
    "NormScaledEmbeddedFactorizedFFNN",
    # Submodules
    "attention",
    "cae",
    "encoder",
    "ffnn",
    "graph",
    "primitives",
]
