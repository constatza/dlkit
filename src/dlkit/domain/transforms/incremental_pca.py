from typing import Any, cast

import torch
from loguru import logger

from .base import Transform, reshaper2d
from .errors import InvalidTransformConfigurationError, TransformNotFittedError


class IncrementalPCA(Transform):
    """PCA that supports streaming fit via the IncrementalFittableTransform protocol.

    Calls sklearn.decomposition.IncrementalPCA.partial_fit() during update_fit()
    so the full dataset need not fit in memory. forward() and inverse_transform()
    are pure torch — no sklearn at inference time.

    Forward:  (x - mean) @ components.T
    Inverse:  x @ components + mean   (approximate if n_components < n_features)

    Suitable for large datasets. When data fits in memory, prefer PCA (faster,
    one-shot SVD via torch.pca_lowrank).

    Args:
        n_components: Number of principal components.
        batch_size: Mini-batch size passed to sklearn IncrementalPCA.
    """

    def __init__(self, *, n_components: int, batch_size: int = 256) -> None:
        """Initialize the IncrementalPCA transformer.

        Args:
            n_components: Number of principal components to retain.
            batch_size: Mini-batch size for the sklearn IncrementalPCA estimator.

        Raises:
            InvalidTransformConfigurationError: If n_components <= 0.
        """
        if n_components <= 0:
            raise InvalidTransformConfigurationError(
                f"n_components must be positive, got {n_components}"
            )
        super().__init__()
        self.n_components = n_components
        self.batch_size = batch_size
        self._estimator: Any = None
        self.register_buffer("mean", torch.tensor([]))
        self.register_buffer("components", torch.tensor([]))
        self.register_buffer("explained_variance_ratio", torch.tensor([]))

    # --- IncrementalFittableTransform protocol ---

    def reset_fit_state(self) -> None:
        """Initialise a fresh IncrementalPCA estimator before the streaming pass.

        Must be called before the first update_fit() call.
        """
        from sklearn.decomposition import IncrementalPCA as _IPCA

        self._estimator = _IPCA(n_components=self.n_components, batch_size=self.batch_size)
        self.fitted = False

    def update_fit(self, batch: torch.Tensor) -> None:
        """Accumulate one mini-batch into the incremental fit.

        Args:
            batch: Tensor of shape (..., n_features).
        """
        np_batch = batch.detach().cpu().numpy()
        if np_batch.ndim > 2:
            np_batch = np_batch.reshape(-1, np_batch.shape[-1])
        self._estimator.partial_fit(np_batch)

    def finalize_fit(self) -> None:
        """Extract fitted parameters into torch buffers and discard the estimator."""
        self.register_buffer("mean", torch.from_numpy(self._estimator.mean_.copy()).float())
        self.register_buffer(
            "components", torch.from_numpy(self._estimator.components_.copy()).float()
        )
        explained_variance_ratio = torch.from_numpy(
            self._estimator.explained_variance_ratio_.copy()
        ).float()
        self.register_buffer("explained_variance_ratio", explained_variance_ratio)
        self._estimator = None
        self.fitted = True
        logger.info(
            "IncrementalPCA total explained variance ratio: {:.4e}",
            explained_variance_ratio.sum().item(),
        )

    # --- FittableTransform protocol (full-data path via TransformChain.fit) ---

    def fit(self, data: torch.Tensor) -> None:
        """Fit on a full in-memory dataset.

        Delegates to the streaming protocol (reset → update → finalize) so the
        behaviour is identical whether fit() or the incremental path is used.

        Args:
            data: Input tensor of shape (..., n_features).
        """
        self.reset_fit_state()
        self.update_fit(data)
        self.finalize_fit()

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
        for name in ("mean", "components", "explained_variance_ratio"):
            key = f"{prefix}{name}"
            if key in state_dict:
                self.register_buffer(name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @reshaper2d
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input onto the principal components.

        Args:
            x: Input tensor of shape (..., n_features).

        Returns:
            Projected tensor of shape (..., n_components).

        Raises:
            TransformNotFittedError: If fit() has not been called.
        """
        if not self.fitted:
            raise TransformNotFittedError("IncrementalPCA")
        mean = cast(torch.Tensor, self.mean)
        components = cast(torch.Tensor, self.components)
        return torch.matmul(x - mean, components.T)

    @reshaper2d
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate reconstruction from projected representation.

        Args:
            x: Projected tensor of shape (..., n_components).

        Returns:
            Reconstructed tensor of shape (..., n_features).

        Raises:
            TransformNotFittedError: If fit() has not been called.
        """
        if not self.fitted:
            raise TransformNotFittedError("IncrementalPCA")
        device = x.device
        mean = cast(torch.Tensor, self.mean).to(device)
        components = cast(torch.Tensor, self.components).to(device)
        return torch.matmul(x, components) + mean

    def infer_output_shape(self, in_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return output shape: last dim replaced by n_components.

        Args:
            in_shape: Input shape.

        Returns:
            Output shape with last dimension = n_components.
        """
        return in_shape[:-1] + (self.n_components,)
