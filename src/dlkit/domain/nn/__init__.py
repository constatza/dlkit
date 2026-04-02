from . import attention, cae, encoder, ffnn, graph, primitives
from .base import DLKitModel
from .ffnn.linear import LinearNetwork
from .ffnn.norm_scaled import (
    NormScaledConstantWidthFFNN,
    NormScaledFactorizedLinear,
    NormScaledFFNN,
    NormScaledLinearFFNN,
    NormScaledSPDFactorizedLinear,
    NormScaledSPDLinear,
    NormScaledSymmetricFactorizedLinear,
    NormScaledSymmetricLinear,
)
from .ffnn.norm_scaled_deep import (
    NormScaledConstantWidthFactorizedFFNN,
    NormScaledConstantWidthSPDFactorizedFFNN,
    NormScaledConstantWidthSPDFFNN,
    NormScaledEmbeddedFactorizedFFNN,
    NormScaledEmbeddedSPDFactorizedFFNN,
    NormScaledEmbeddedSPDFFNN,
)
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

__all__ = [
    "DLKitModel",
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
