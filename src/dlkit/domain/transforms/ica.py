from typing import cast

import torch
from loguru import logger

from .base import Transform, reshaper2d
from .errors import InvalidTransformConfigurationError, TransformNotFittedError


class ICA(Transform):
    """Independent Component Analysis via sklearn FastICA.

    Uses sklearn.decomposition.FastICA during fit() to compute the whitened
    unmixing matrix, then stores it as torch buffers. forward() and
    inverse_transform() are pure torch — no sklearn at inference time.

    Forward:  (x - mean) @ components.T
    Inverse:  x @ mixing.T + mean  (approximate for n_components < n_features)

    Args:
        n_components: Number of independent components.
        fun: ICA contrast function — "logcosh" (default), "exp", or "cube".
        max_iter: Maximum iterations for the FastICA algorithm.
        random_state: Random seed for reproducibility.
    """

    requires_materialized_fit = True

    def __init__(
        self,
        *,
        n_components: int,
        fun: str = "logcosh",
        max_iter: int = 200,
        random_state: int = 0,
    ) -> None:
        """Initialize the ICA transformer.

        Args:
            n_components: Number of independent components to extract.
            fun: ICA contrast function. One of "logcosh", "exp", or "cube".
            max_iter: Maximum number of iterations for FastICA.
            random_state: Random seed for reproducibility.

        Raises:
            InvalidTransformConfigurationError: If n_components <= 0.
        """
        if n_components <= 0:
            raise InvalidTransformConfigurationError(
                f"n_components must be positive, got {n_components}"
            )
        super().__init__()
        self.n_components = n_components
        self.fun = fun
        self.max_iter = max_iter
        self.random_state = random_state
        self.register_buffer("mean", torch.tensor([]))
        self.register_buffer("components", torch.tensor([]))
        self.register_buffer("mixing", torch.tensor([]))

    def fit(self, data: torch.Tensor) -> None:
        """Fit ICA using sklearn FastICA on the full dataset.

        Args:
            data: Input tensor of shape (..., n_features).
        """
        from sklearn.decomposition import FastICA

        np_data = data.detach().cpu().numpy()
        if np_data.ndim > 2:
            np_data = np_data.reshape(-1, np_data.shape[-1])

        estimator = FastICA(
            n_components=self.n_components,
            fun=self.fun,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        estimator.fit(np_data)
        logger.info("ICA FastICA converged in {} iterations", estimator.n_iter_)

        self.register_buffer("mean", torch.from_numpy(estimator.mean_.copy()).float())
        self.register_buffer("components", torch.from_numpy(estimator.components_.copy()).float())
        self.register_buffer("mixing", torch.from_numpy(estimator.mixing_.copy()).float())
        self.fitted = True

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor | bool],
        prefix: str,
        local_metadata: dict[str, int],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
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
        for name in ("mean", "components", "mixing"):
            val = state_dict.get(f"{prefix}{name}")
            if isinstance(val, torch.Tensor):
                self.register_buffer(name, torch.empty_like(val))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @reshaper2d
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ICA projection: (x - mean) @ components.T

        Args:
            x: Input tensor of shape (..., n_features).

        Returns:
            Independent components of shape (..., n_components).

        Raises:
            TransformNotFittedError: If fit() has not been called.
        """
        if not self.fitted:
            raise TransformNotFittedError("ICA")
        device = x.device
        mean = cast(torch.Tensor, self.mean).to(device)
        components = cast(torch.Tensor, self.components).to(device)
        return torch.matmul(x - mean, components.T)

    @reshaper2d
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate reconstruction: x @ mixing.T + mean

        Args:
            x: Independent components of shape (..., n_components).

        Returns:
            Reconstructed tensor of shape (..., n_features).

        Raises:
            TransformNotFittedError: If fit() has not been called.
        """
        if not self.fitted:
            raise TransformNotFittedError("ICA")
        device = x.device
        mean = cast(torch.Tensor, self.mean).to(device)
        mixing = cast(torch.Tensor, self.mixing).to(device)
        return torch.matmul(x, mixing.T) + mean

    def infer_output_shape(self, in_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return output shape: last dim replaced by n_components.

        Args:
            in_shape: Input shape.

        Returns:
            Output shape with last dimension = n_components.
        """
        return in_shape[:-1] + (self.n_components,)
