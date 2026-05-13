from .base import ISpectralLayer
from .coordinate import (
    FourierFeatureNetwork,
    ModifiedMLP,
    ScaleEquivariantFourierFeatureNetwork,
    ScaleEquivariantModifiedMLP,
    ScaleEquivariantSiren,
    Siren,
)
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
    # Coordinate spectral-bias networks
    "FourierFeatureNetwork",
    "Siren",
    "ModifiedMLP",
    "ScaleEquivariantFourierFeatureNetwork",
    "ScaleEquivariantSiren",
    "ScaleEquivariantModifiedMLP",
]
