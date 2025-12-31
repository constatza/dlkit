from .base import (
    BaseTransform,
    Transform,
    FittableTransform,
    InvertibleTransform,
    ShapeAwareTransform,
)
from .chain import TransformChain
from .permute import Permutation
from .minmax import MinMaxScaler
from .pca import PCA
from .standard import StandardScaler
from .subset import TensorSubset
from .spectral import SpectralRadiusNorm
from .sample_norm import SampleNormL2
from .manager import TransformManager
from .errors import (
    TransformError,
    TransformNotFittedError,
    ShapeMismatchError,
    TransformChainError,
    TransformApplicationError,
    InvalidTransformConfigurationError,
)

# Backward compatibility: keep old interface names as aliases
IFittableTransform = FittableTransform  # Deprecated: use FittableTransform Protocol
IInvertibleTransform = InvertibleTransform  # Deprecated: use InvertibleTransform Protocol
IShapeAwareTransform = ShapeAwareTransform  # Deprecated: use ShapeAwareTransform Protocol

__all__ = [
    "Transform",
    "TransformChain",
    "MinMaxScaler",
    "PCA",
    "StandardScaler",
    "Permutation",
    "TensorSubset",
    "BaseTransform",
    "SpectralRadiusNorm",
    "SampleNormL2",
    "TransformManager",
    # Protocols
    "FittableTransform",
    "InvertibleTransform",
    "ShapeAwareTransform",
    # Deprecated: backward compatibility
    "IFittableTransform",
    "IInvertibleTransform",
    "IShapeAwareTransform",
    # Errors
    "TransformError",
    "TransformNotFittedError",
    "ShapeMismatchError",
    "TransformChainError",
    "TransformApplicationError",
    "InvalidTransformConfigurationError",
]
