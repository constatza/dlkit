from typing import cast

import torch
from loguru import logger

from .base import Transform, reshaper2d
from .errors import InvalidTransformConfigurationError, TransformNotFittedError


class TruncatedSVD(Transform):
    """Dimensionality reduction via truncated SVD (no mean-centering).

    Unlike PCA, does not subtract the data mean before decomposition.
    Implemented in pure torch via torch.svd_lowrank. Prefer PCA when data
    is centred; prefer TruncatedSVD for sparse or non-negative data where
    the mean offset is meaningful.

    Args:
        n_components: Number of singular components to retain.
        n_iter: Number of power iterations for randomised SVD.
    """

    def __init__(self, *, n_components: int, n_iter: int = 4) -> None:
        """Initialize the TruncatedSVD transformer.

        Args:
            n_components: Number of singular components to retain.
            n_iter: Number of power iterations for randomised SVD. Defaults to 4.

        Raises:
            InvalidTransformConfigurationError: If n_components <= 0.
        """
        if n_components <= 0:
            raise InvalidTransformConfigurationError(
                f"n_components must be positive, got {n_components}"
            )
        super().__init__()
        self.n_components = n_components
        self.n_iter = n_iter
        self.register_buffer("components", torch.tensor([]))
        self.register_buffer("singular_values", torch.tensor([]))
        self.register_buffer("explained_energy_ratio", torch.tensor(0.0))

    def fit(self, data: torch.Tensor) -> None:
        """Compute truncated SVD on data.

        Args:
            data: Input tensor of shape (..., n_features).
        """
        if data.ndim > 2:
            data = data.reshape(-1, data.shape[-1])
        _, S, V = torch.svd_lowrank(data, q=self.n_components, niter=self.n_iter)
        total_energy = torch.linalg.norm(data.float(), ord="fro") ** 2
        explained_energy_ratio = S.float().pow(2).sum() / total_energy
        # V: (n_features, n_components) → store as (n_components, n_features)
        self.register_buffer("components", V.T.clone())
        self.register_buffer("singular_values", S.clone())
        self.register_buffer("explained_energy_ratio", explained_energy_ratio.clone())
        self.fitted = True
        logger.info(
            "TruncatedSVD explained energy ratio: {:.4e}",
            explained_energy_ratio.item(),
        )

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
        for name in ("components", "singular_values", "explained_energy_ratio"):
            val = state_dict.get(f"{prefix}{name}")
            if isinstance(val, torch.Tensor):
                self.register_buffer(name, torch.empty_like(val))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @reshaper2d
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input onto the truncated singular components.

        Args:
            x: Input tensor of shape (..., n_features).

        Returns:
            Projected tensor of shape (..., n_components).

        Raises:
            TransformNotFittedError: If fit() has not been called.
        """
        if not self.fitted:
            raise TransformNotFittedError("TruncatedSVD")
        components = cast(torch.Tensor, self.components)
        return torch.matmul(x, components.T)

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
            raise TransformNotFittedError("TruncatedSVD")
        device = x.device
        components = cast(torch.Tensor, self.components).to(device)
        return torch.matmul(x, components)

    def infer_output_shape(self, in_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return output shape: last dim replaced by n_components.

        Args:
            in_shape: Input shape.

        Returns:
            Output shape with last dimension = n_components.
        """
        return in_shape[:-1] + (self.n_components,)
