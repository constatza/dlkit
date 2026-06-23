from .base import ISpectralLayer
from .coordinate import (
    FactorizedFourierFeatureNetwork,
    FourierFeatureNetwork,
    HashEncodingNetwork,
    ModifiedMLP,
    ScaleEquivariantFactorizedFourierFeatureNetwork,
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
    "FactorizedFourierFeatureNetwork",
    "HashEncodingNetwork",
    "Siren",
    "ModifiedMLP",
    "ScaleEquivariantFactorizedFourierFeatureNetwork",
    "ScaleEquivariantFourierFeatureNetwork",
    "ScaleEquivariantSiren",
    "ScaleEquivariantModifiedMLP",
]
