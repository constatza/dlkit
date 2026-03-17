from .simple import FeedForwardNN, ConstantWidthFFNN
from .plain import SimpleFeedForwardNN, ConstantWidthSimpleFFNN
from .linear import LinearNetwork
from .norm_scaled import (
    NormScaledFFNN,
    NormScaledLinearFFNN,
    NormScaledConstantWidthFFNN,
    NormScaledSymmetricLinear,
    NormScaledSPDLinear,
    NormScaledFactorizedLinear,
    NormScaledSymmetricFactorizedLinear,
    NormScaledSPDFactorizedLinear,
)
from .parametric import (
    ParametricDenseBlock,
    ConstantWidthParametricFFNN,
    EmbeddedParametricFFNN,
)
from .parametric_variants import (
    ConstantWidthSPDFFNN,
    ConstantWidthSPDFactorizedFFNN,
    ConstantWidthFactorizedFFNN,
    EmbeddedSPDFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedFactorizedFFNN,
)
from .norm_scaled_deep import (
    NormScaledConstantWidthSPDFFNN,
    NormScaledConstantWidthSPDFactorizedFFNN,
    NormScaledConstantWidthFactorizedFFNN,
    NormScaledEmbeddedSPDFFNN,
    NormScaledEmbeddedSPDFactorizedFFNN,
    NormScaledEmbeddedFactorizedFFNN,
)

__all__ = [
    # Residual networks
    "FeedForwardNN",
    "ConstantWidthFFNN",
    # Plain (no residual) networks
    "SimpleFeedForwardNN",
    "ConstantWidthSimpleFFNN",
    # Linear baseline
    "LinearNetwork",
    # NormScaled (single-layer)
    "NormScaledFFNN",
    "NormScaledLinearFFNN",
    "NormScaledConstantWidthFFNN",
    "NormScaledSymmetricLinear",
    "NormScaledSPDLinear",
    "NormScaledFactorizedLinear",
    "NormScaledSymmetricFactorizedLinear",
    "NormScaledSPDFactorizedLinear",
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
]
