from .base import DLKitModel

from .ffnn.linear import LinearNetwork
from .ffnn.norm_scaled import (
    NormScaledFFNN,
    NormScaledLinearFFNN,
    NormScaledConstantWidthFFNN,
    NormScaledSymmetricLinear,
    NormScaledSPDLinear,
    NormScaledFactorizedLinear,
    NormScaledSymmetricFactorizedLinear,
    NormScaledSPDFactorizedLinear,
)
from .ffnn.plain import SimpleFeedForwardNN, ConstantWidthSimpleFFNN
from .ffnn.parametric import (
    ParametricDenseBlock,
    ConstantWidthParametricFFNN,
    EmbeddedParametricFFNN,
)
from .ffnn.parametric_variants import (
    ConstantWidthSPDFFNN,
    ConstantWidthSPDFactorizedFFNN,
    ConstantWidthFactorizedFFNN,
    EmbeddedSPDFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedFactorizedFFNN,
)
from .ffnn.norm_scaled_deep import (
    NormScaledConstantWidthSPDFFNN,
    NormScaledConstantWidthSPDFactorizedFFNN,
    NormScaledConstantWidthFactorizedFFNN,
    NormScaledEmbeddedSPDFFNN,
    NormScaledEmbeddedSPDFactorizedFFNN,
    NormScaledEmbeddedFactorizedFFNN,
)

from . import attention
from . import cae
from . import encoder
from . import ffnn
from . import graph
from . import primitives

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
