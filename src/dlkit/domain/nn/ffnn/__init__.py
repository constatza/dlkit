from .linear import LinearNetwork
from .norm_scaled import (
    NormScaledConstantWidthFFNN,
    NormScaledFactorizedLinear,
    NormScaledFFNN,
    NormScaledLinearFFNN,
    NormScaledSPDFactorizedLinear,
    NormScaledSPDLinear,
    NormScaledSymmetricFactorizedLinear,
    NormScaledSymmetricLinear,
)
from .norm_scaled_deep import (
    NormScaledConstantWidthFactorizedFFNN,
    NormScaledConstantWidthSPDFactorizedFFNN,
    NormScaledConstantWidthSPDFFNN,
    NormScaledEmbeddedFactorizedFFNN,
    NormScaledEmbeddedSPDFactorizedFFNN,
    NormScaledEmbeddedSPDFFNN,
)
from .parametric import (
    ConstantWidthParametricFFNN,
    EmbeddedParametricFFNN,
    ParametricDenseBlock,
)
from .parametric_variants import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthSPDFactorizedFFNN,
    ConstantWidthSPDFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
)
from .plain import ConstantWidthSimpleFFNN, SimpleFeedForwardNN
from .simple import ConstantWidthFFNN, FeedForwardNN

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
