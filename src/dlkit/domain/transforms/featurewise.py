from collections.abc import Sequence

import torch
from torch import Tensor

from dlkit.domain.transforms.base import (
    IncrementalFittableTransform,
    InvertibleTransform,
    Transform,
)


class FeatureWise(Transform):
    """Wraps any transform and applies it to a selected subset of feature indices.

    The inner transform operates only on the selected slice; the remaining
    features pass through unchanged. All fitting and inverse protocols are
    delegated to the inner transform when they are satisfied.

    This is primarily a programmatic tool for composing existing (fittable)
    transforms with index selection. For config-driven use, prefer transforms
    that inherit PartialTransform directly.

    Args:
        transform: Any Transform instance to wrap.
        indices: Feature positions along index_dim to apply the transform to.
        index_dim: Axis that holds the feature dimension. Defaults to -1.

    Example:
        >>> from dlkit.domain.transforms import StandardScaler
        >>> t = FeatureWise(StandardScaler(dim=0), indices=[0, 2, 5])
        >>> t.fit(train_data)
        >>> y = t(data)
        >>> x = t.inverse_transform(y)
    """

    def __init__(
        self,
        transform: Transform,
        *,
        indices: Sequence[int],
        index_dim: int = -1,
    ) -> None:
        """Initialize FeatureWise.

        Args:
            transform: Inner transform instance to apply to selected features.
            indices: Feature indices to apply the transform to. Required.
            index_dim: Axis along which to select features.
        """
        super().__init__()
        self._transform = transform
        self.indices: tuple[int, ...] = tuple(indices)
        self.index_dim = index_dim

    def _idx_tensor(self, device: torch.device) -> Tensor:
        return torch.tensor(self.indices, device=device, dtype=torch.long)

    def _gather(self, x: Tensor) -> Tensor:
        dim = self.index_dim % x.ndim
        return torch.index_select(x, dim, self._idx_tensor(x.device))

    def _scatter(self, x: Tensor, transformed: Tensor) -> Tensor:
        dim = self.index_dim % x.ndim
        out = x.clone()
        out.index_copy_(dim, self._idx_tensor(x.device), transformed)
        return out

    def _is_truly_fittable(self) -> bool:
        return type(self._transform).fit is not Transform.fit

    # --- fitting delegation ---

    @property
    def fitted(self) -> bool:
        """Returns True if inner transform is ready: either non-fittable or already fitted."""
        if self._is_truly_fittable():
            return self._transform.fitted
        return True

    @fitted.setter
    def fitted(self, value: bool) -> None:
        if self._is_truly_fittable():
            self._transform.fitted = value

    def fit(self, data: Tensor) -> None:
        """Fit the inner transform on the selected feature slice.

        Args:
            data: Full input tensor; only the selected slice is passed to the inner transform.
        """
        if self._is_truly_fittable():
            self._transform.fit(self._gather(data))

    def reset_fit_state(self) -> None:
        """Delegate to inner transform if it supports incremental fitting."""
        if isinstance(self._transform, IncrementalFittableTransform) and self._is_truly_fittable():
            self._transform.reset_fit_state()

    def update_fit(self, batch: Tensor) -> None:
        """Delegate incremental fit update on the selected feature slice.

        Args:
            batch: Full batch tensor; only the selected slice is forwarded.
        """
        if isinstance(self._transform, IncrementalFittableTransform) and self._is_truly_fittable():
            self._transform.update_fit(self._gather(batch))

    def finalize_fit(self) -> None:
        """Delegate incremental fit finalisation to inner transform."""
        if isinstance(self._transform, IncrementalFittableTransform) and self._is_truly_fittable():
            self._transform.finalize_fit()

    # --- forward / inverse ---

    def forward(self, x: Tensor) -> Tensor:
        """Apply inner transform to selected features, pass others through.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with the same shape as x, selected features transformed.
        """
        return self._scatter(x, self._transform(self._gather(x)))

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Invert the inner transform on selected features.

        Args:
            x: Transformed tensor of any shape.

        Returns:
            Tensor with selected features reconstructed.

        Raises:
            TypeError: If the inner transform is not invertible.
        """
        if not isinstance(self._transform, InvertibleTransform):
            raise TypeError(f"{type(self._transform).__name__} does not support inverse_transform.")
        return self._scatter(x, self._transform.inverse_transform(self._gather(x)))
