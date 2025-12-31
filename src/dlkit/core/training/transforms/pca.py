from functools import wraps
from typing import TYPE_CHECKING

import torch
from loguru import logger

from .base import Transform
from .errors import TransformNotFittedError, InvalidTransformConfigurationError, ShapeMismatchError
from .shape_inference import register_shape_inference

if TYPE_CHECKING:
    from dlkit.core.shape_specs import IShapeSpec


def reshaper2d(func):
    """Decorator to handle multi-dimensional inputs by flattening to 2D."""

    @wraps(func)
    def wrapper(*args):
        data_position = len(args) - 1
        data = args[data_position]
        shape = data.shape
        if len(shape) < 2:
            return func(*args)
        else:
            data = data.reshape(-1, data.shape[-1])
            processed = func(*args[:-1], data)
            return processed.reshape((-1,) + shape[1:-1] + processed.shape[1:])

    return wrapper


class PCA(Transform):
    """Principal Component Analysis (PCA) transformer using torch.pca_lowrank.

    This transformer computes the principal components efficiently by leveraging
    torch.pca_lowrank. It projects data onto the computed components and provides
    an inverse transformation for approximate reconstruction.

    The transformer treats all dimensions prior to the last as sample dimensions.
    For example, for data with shape (N, T, D), it will be reshaped to (N*T, D)
    where the last dimension is considered as the feature dimension.

    The number of components can be validated against input shape via configure_shape(),
    or determined dynamically during fit().
    """

    mean: torch.Tensor | None
    components: torch.Tensor | None
    explained_variance: torch.Tensor | None
    explained_variance_ratio: torch.Tensor | None
    total_explained_variance: float | None
    n_components: int
    n_power_iterations: int
    _shape_configured: bool

    def __init__(self, *, n_components: int, n_power_iterations: int = 2) -> None:
        """Initialize the PCA transformer.

        Args:
            n_components: Number of principal components to compute.
            n_power_iterations: Number of power iterations for pca_lowrank. Defaults to 2.

        Raises:
            InvalidTransformConfigurationError: If n_components <= 0.

        Example:
            >>> pca = PCA(n_components=10)
            >>> pca.fit(train_data)
            >>> reduced = pca(train_data)
        """
        if n_components <= 0:
            raise InvalidTransformConfigurationError(
                f"n_components must be positive, got {n_components}"
            )

        super().__init__()
        self.n_components = n_components
        self.n_power_iterations = n_power_iterations
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.total_explained_variance = None
        self._shape_configured = False

    def configure_shape(self, shape_spec: "IShapeSpec", entry_name: str) -> None:
        """Configure PCA with shape information for validation.

        Args:
            shape_spec: Shape specification containing entry shapes.
            entry_name: Name of the entry to get shape for.

        Raises:
            ShapeMismatchError: If n_components > input features.
        """
        shape = shape_spec.get_shape(entry_name)
        if shape is None:
            return

        n_features = shape[-1]  # Last dimension is features
        if self.n_components > n_features:
            raise ShapeMismatchError(
                expected=(n_features,),
                actual=(self.n_components,),
                context=f"n_components ({self.n_components}) must be <= n_features ({n_features})",
            )

        self._shape_configured = True

    def fit(self, data: torch.Tensor, dim: int = -1) -> None:
        """Fit the PCA transformer on the input

        This method computes the mean, principal components, explained variance,
        and explained variance ratio using torch.pca_lowrank. The dataflow is manually
        centered before applying pca_lowrank. If the input dataflow has more than 2 dimensions,
        all dimensions except the last are flattened into the sample dimension.

        Args:
            data (torch.Tensor): Input dataflow of shape (..., n_features). For example,
                (N, T, D) where D is the feature dimension.
            dim (int, optional): Dimension to be used as the feature dimension (to be reduced). Defaults to -1.
        """
        # swqp dim to the last dimension for easier handling.
        data = data.transpose(dim, -1)

        if len(data.shape) > 2:
            data = data.reshape(-1, data.shape[-1])

        n_samples, _ = data.shape

        # Compute the mean manually for later use in forward and inverse transformations.
        self.mean = torch.mean(data, dim=0, keepdim=True)
        data_centered = data - self.mean

        # Compute total variance: sum of squared deviations divided by (n_samples - 1).
        total_variance = torch.sum(data_centered.pow(2)) / (n_samples - 1)

        # Use torch.pca_lowrank on already centered dataflow by setting center=False.
        U, S, V = torch.pca_lowrank(
            data_centered,
            q=self.n_components,
            center=False,
            niter=self.n_power_iterations,
        )
        # Principal components are the right singular vectors.
        # V has shape (n_features, n_components); we transpose to (n_components, n_features)
        self.components = V.T

        # Compute the explained variance from singular values using Bessel's correction.
        self.explained_variance = S**2 / (n_samples - 1)

        # Compute the explained variance ratio with respect to the total variance.
        self.explained_variance_ratio = self.explained_variance / total_variance
        self.total_explained_variance = torch.sum(self.explained_variance_ratio).item()

        self.fitted = True
        logger.info(f"PCA total explained variance ratio: {self.total_explained_variance:.4e}")

    @reshaper2d
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Project the input data onto the principal components.

        Args:
            data: Input data of shape (..., n_features).

        Returns:
            Projected data of shape (..., n_components).

        Raises:
            TransformNotFittedError: If fit() hasn't been called yet.
        """
        if not self.fitted or self.mean is None or self.components is None:
            raise TransformNotFittedError("PCA")

        # Center the data using the stored mean
        data_centered = data - self.mean
        # Project onto principal components
        projected = torch.matmul(data_centered, self.components.T)

        return projected

    @reshaper2d
    def inverse_transform(self, projected: torch.Tensor) -> torch.Tensor:
        """Perform inverse transformation (approximate reconstruction).

        Reconstructs the original data from projected data. If n_components
        is less than the original feature dimensions, reconstruction is approximate.

        Args:
            projected: Projected data of shape (..., n_components).

        Returns:
            Reconstructed data of shape (..., n_features).

        Raises:
            TransformNotFittedError: If fit() hasn't been called yet.
        """
        if not self.fitted or self.mean is None or self.components is None:
            raise TransformNotFittedError("PCA")

        device = projected.device
        mean = self.mean.to(device)
        components = self.components.to(device)

        # Reconstruct: approximate inverse of PCA projection
        reconstructed = torch.matmul(projected, components) + mean

        return reconstructed


# Register shape inference function (PCA reduces last dimension)
@register_shape_inference(PCA)
def _infer_pca_output_shape(
    input_shape: tuple[int, ...], n_components: int, **kwargs
) -> tuple[int, ...]:
    """PCA reduces the last dimension to n_components."""
    return input_shape[:-1] + (n_components,)
