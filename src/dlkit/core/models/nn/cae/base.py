import abc
from typing import Any, Dict

from torch import nn, Tensor
from dlkit.core.models.nn.base import ShapeAwareModel
from dlkit.core.shape_specs import IShapeSpec


class CAE(ShapeAwareModel):
    def __init__(self, *, unified_shape: IShapeSpec, **kwargs):
        """Initialize CAE with shape specification.

        Args:
            unified_shape: Shape specification for input/output data
            **kwargs: Additional parameters passed to ShapeAwareModel
        """
        super().__init__(unified_shape=unified_shape, **kwargs)

        # CAE models require shape specs for proper initialization
        # Shape validation happens automatically in ShapeAwareModel

    @abc.abstractmethod
    def encode(self, *args, **kwargs) -> Any:
        """Encode input to latent space."""
        ...

    @abc.abstractmethod
    def decode(self, *args, **kwargs) -> Any:
        """Decode latent representation back to original space."""
        ...

    def forward(self, x: Any) -> Any:
        """Forward pass through autoencoder (encode -> decode)."""
        encoding = self.encode(x)
        return self.decode(encoding)

    def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        """Prediction step that returns both reconstruction and latent representation."""
        x = batch[0]
        latent = self.encode(x)
        y = self.decode(latent)

        # Ensure we return tensors
        if isinstance(y, Tensor):
            y_tensor = y
        else:
            # Handle case where decode returns tuple (like VAE)
            y_tensor = y[0] if isinstance(y, tuple) else y

        if isinstance(latent, Tensor):
            latent_tensor = latent
        else:
            # Handle case where encode returns tuple (like VAE)
            latent_tensor = latent[0] if isinstance(latent, tuple) else latent

        return {"predictions": y_tensor.detach(), "latent": latent_tensor.detach()}

    def accepts_shape(self, shape_spec: IShapeSpec) -> bool:
        """Check if this CAE can accept the given shape specification."""
        # CAE-specific validation: we typically need input shape
        input_shape = shape_spec.get_input_shape()
        if input_shape is None:
            return False

        # Input shape should have positive dimensions
        if len(input_shape) == 0 or any(d <= 0 for d in input_shape):
            return False

        return True
