from functools import wraps

import torch
from loguru import logger

from .base import Transform
from .errors import TransformNotFittedError, InvalidTransformConfigurationError, ShapeMismatchError


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
        # Register empty placeholder buffers for checkpoint support
        self.register_buffer("mean", torch.tensor([]))
        self.register_buffer("components", torch.tensor([]))
        self.register_buffer("explained_variance", torch.tensor([]))
        self.register_buffer("explained_variance_ratio", torch.tensor([]))
        self.register_buffer("total_explained_variance", torch.tensor(0.0))

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
        mean = torch.mean(data, dim=0, keepdim=True)
        data_centered = data - mean

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
        components = V.T

        # Compute the explained variance from singular values using Bessel's correction.
        explained_variance = S**2 / (n_samples - 1)

        # Compute the explained variance ratio with respect to the total variance.
        explained_variance_ratio = explained_variance / total_variance
        total_explained_variance = torch.sum(explained_variance_ratio).item()

        # Register buffers for checkpoint support
        self.register_buffer("mean", mean)
        self.register_buffer("components", components)
        self.register_buffer("explained_variance", explained_variance)
        self.register_buffer("explained_variance_ratio", explained_variance_ratio)
        self.register_buffer("total_explained_variance", torch.tensor(total_explained_variance))

        self.fitted = True
        logger.info(
            f"PCA total explained variance ratio: {self.total_explained_variance.item():.4e}"
        )

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Pre-allocate buffers with correct shape from checkpoint before loading.

        Args:
            state_dict: Full state dictionary.
            prefix: Module prefix for this module's keys.
            local_metadata: Local metadata dict.
            strict: Whether to enforce strict key matching.
            missing_keys: List to accumulate missing key names.
            unexpected_keys: List to accumulate unexpected key names.
            error_msgs: List to accumulate error messages.
        """
        for name in (
            "mean",
            "components",
            "explained_variance",
            "explained_variance_ratio",
            "total_explained_variance",
        ):
            key = f"{prefix}{name}"
            if key in state_dict:
                self.register_buffer(name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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
        if not self.fitted:
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
        if not self.fitted:
            raise TransformNotFittedError("PCA")

        device = projected.device
        mean = self.mean.to(device)
        components = self.components.to(device)

        # Reconstruct: approximate inverse of PCA projection
        reconstructed = torch.matmul(projected, components) + mean

        return reconstructed

    def infer_output_shape(self, in_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Infer output shape. PCA reduces the last dimension to n_components.

        Args:
            in_shape: Input tensor shape.

        Returns:
            Output shape with last dimension = n_components.
        """
        return in_shape[:-1] + (self.n_components,)
