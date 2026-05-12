from .base import ISpectralLayer
from .ffnn import DualPathFFNN, FourierAugmented, FourierEnhancedFFNN, SpectralDualPath
from .layers import FourierLayer, SpectralConv1d
from .pinn import FourierFeatureNetwork, ModifiedMLP, SirenFFNN

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
    # PINN networks
    "FourierFeatureNetwork",
    "SirenFFNN",
    "ModifiedMLP",
]
