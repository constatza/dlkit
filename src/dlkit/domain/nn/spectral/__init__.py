from .base import ISpectralLayer
from .ffnn import DualPathFFNN, FourierAugmented, FourierEnhancedFFNN, SpectralDualPath
from .layers import FourierLayer, SpectralConv1d

__all__ = [
    # Protocol
    "ISpectralLayer",
    # Primitives
    "SpectralConv1d",
    "FourierLayer",
    # Composable base classes
    "FourierAugmented",
    "SpectralDualPath",
    # Convenience constructors
    "FourierEnhancedFFNN",
    "DualPathFFNN",
]
