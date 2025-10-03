from pydantic import ConfigDict, validate_call

# from dlkit.core.datatypes.dataset import Shape  # Removed - using IShapeSpec
from dlkit.core.shape_specs import IShapeSpec
from typing import Any
from dlkit.core.models.nn.cae.base import CAE
from dlkit.core.models.nn.cae import SkipCAE1d


class LinearCAE1d(CAE):
    latent_channels: int
    latent_width: int
    latent_size: int
    num_layers: int
    kernel_size: int
    shape_spec: Any  # IShapeSpec | None

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        shape_spec: Any = None,  # IShapeSpec | None
        latent_channels: int = 5,
        latent_width: int = 10,
        latent_size: int = 10,
        num_layers: int = 3,
        kernel_size: int = 3,
        **kwargs
    ):
        super().__init__(shape_spec=shape_spec, **kwargs)

        # Store hyperparameters for this model
        self.latent_channels = latent_channels
        self.latent_width = latent_width
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        # Create the actual implementation using SkipCAE1d
        # Note: This creates a composition rather than proper inheritance
        # This is a temporary implementation to maintain backward compatibility
        self._impl = SkipCAE1d(
            shape_spec=shape_spec,
            latent_channels=latent_channels,
            latent_width=latent_width,
            latent_size=latent_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            activation=lambda x: x,
        )

    def encode(self, x):
        """Delegate to the implementation."""
        return self._impl.encode(x)

    def decode(self, x):
        """Delegate to the implementation."""
        return self._impl.decode(x)
