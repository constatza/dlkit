from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

# Minimum argument passed to _inverse_softplus to avoid log(0) / NaN.
# Chosen small enough to not interfere with valid near-zero inputs.
_SOFTPLUS_INV_CLAMP: float = 1e-14


def _validate_square_matrix(x: torch.Tensor, name: str) -> None:
    """Validate that a tensor is a square 2-D matrix.

    Args:
        x: Input tensor.
        name: Tensor name used in error messages.

    Raises:
        ValueError: If the tensor is not 2-D and square.
    """
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(x.shape)}")
    if x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be square, got shape={tuple(x.shape)}")


def _validate_matrix_rows(x: torch.Tensor, rows: int, name: str) -> None:
    """Validate that a tensor is a 2-D matrix with the expected row count.

    Args:
        x: Input tensor.
        rows: Expected number of rows.
        name: Tensor name used in error messages.

    Raises:
        ValueError: If the tensor is not 2-D or has the wrong number of rows.
    """
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(x.shape)}")
    if x.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows, got shape={tuple(x.shape)}")


def _validate_matrix_cols(x: torch.Tensor, cols: int, name: str) -> None:
    """Validate that a tensor is a 2-D matrix with the expected column count.

    Args:
        x: Input tensor.
        cols: Expected number of columns.
        name: Tensor name used in error messages.

    Raises:
        ValueError: If the tensor is not 2-D or has the wrong number of columns.
    """
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(x.shape)}")
    if x.shape[1] != cols:
        raise ValueError(f"{name} must have {cols} columns, got shape={tuple(x.shape)}")


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    """Compute a numerically stable inverse softplus.

    Args:
        y: Positive tensor (must be > 0).

    Returns:
        Tensor x such that softplus(x) ≈ y.
    """
    return y + torch.log(-torch.expm1(-y))


class Symmetric(nn.Module):
    """Parametrize a square matrix as symmetric.

    Applies the upper-triangular symmetrisation: ``(X.triu() + X.triu(1)^T)``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a symmetric matrix.

        Args:
            x: Unconstrained square matrix of shape ``(n, n)``.

        Returns:
            Symmetric matrix of the same shape.

        Raises:
            ValueError: If *x* is not a square 2-D tensor.
        """
        _validate_square_matrix(x, "x")
        return x.triu() + x.triu(1).transpose(-1, -2)

    def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
        """Return one valid preimage of a symmetric matrix.

        Args:
            w: Symmetric square matrix of shape ``(n, n)``.

        Returns:
            Upper-triangular representative of *w*.

        Raises:
            ValueError: If *w* is not a square 2-D tensor.
        """
        _validate_square_matrix(w, "w")
        return w.triu()


class SPD(nn.Module):
    """Parametrize a square matrix as symmetric positive definite.

    Uses a lower-Cholesky factor with softplus-activated diagonal to ensure
    positive definiteness.

    Attributes:
        min_diag (float): Positive floor added to each Cholesky diagonal element.
    """

    def __init__(self, min_diag: float = 1e-4) -> None:
        """Initialize the SPD parametrization.

        Args:
            min_diag: Positive diagonal floor for numerical stability.
        """
        super().__init__()
        self.min_diag = float(min_diag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an SPD matrix.

        Args:
            x: Unconstrained square matrix of shape ``(n, n)``.

        Returns:
            Symmetric positive-definite matrix of the same shape.

        Raises:
            ValueError: If *x* is not a square 2-D tensor.
        """
        _validate_square_matrix(x, "x")
        l = torch.tril(x)
        diag_raw = torch.diagonal(l, dim1=-2, dim2=-1)
        diag_pos = F.softplus(diag_raw) + self.min_diag
        l = l - torch.diag_embed(diag_raw) + torch.diag_embed(diag_pos)
        return l @ l.transpose(-1, -2)

    def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
        """Return one valid preimage of an SPD matrix.

        Raises ``NotImplementedError`` when *w* is not positive-definite so
        that PyTorch's parametrize machinery falls back to treating the current
        raw parameter as the identity (the effective weight is ``forward(raw)``
        regardless).  This is the intended signal — see PyTorch docs for
        ``register_parametrization``.

        Args:
            w: Symmetric positive-definite matrix of shape ``(n, n)``.

        Returns:
            Lower-triangular representative in unconstrained space.

        Raises:
            ValueError: If *w* is not a square 2-D tensor.
            NotImplementedError: If *w* is not positive-definite (Cholesky fails).
        """
        _validate_square_matrix(w, "w")
        try:
            l = torch.linalg.cholesky(w)
        except RuntimeError as exc:
            raise NotImplementedError(
                "SPD.right_inverse requires a positive-definite matrix; "
                f"Cholesky failed: {exc}"
            ) from exc
        diag_pos = torch.diagonal(l, dim1=-2, dim2=-1)
        safe = (diag_pos - self.min_diag).clamp(min=_SOFTPLUS_INV_CLAMP)
        diag_raw = _inverse_softplus(safe)
        return torch.tril(l, diagonal=-1) + torch.diag_embed(diag_raw)


class PositiveRowScale(nn.Module):
    """Apply positive row-wise scaling: ``W = diag(exp(s)) @ A``.

    Owns an independent ``log_scale`` parameter (one scalar per row).

    Attributes:
        log_scale (nn.Parameter): Learnable log-scale of shape ``(rows,)``.
    """

    def __init__(self, rows: int, mean: float = 0.0, std: float = 0.1) -> None:
        """Initialize the row scaling parametrization.

        Args:
            rows: Number of rows in the target weight matrix.
            mean: Mean for log-scale initialisation. Default ``0.0`` gives
                unit scale (``exp(0) = 1``).
            std: Standard deviation for log-scale initialisation.
        """
        super().__init__()
        self.log_scale = nn.Parameter(torch.empty(rows))
        nn.init.normal_(self.log_scale, mean=mean, std=std)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """Apply row scaling.

        Args:
            a: Base weight matrix of shape ``(rows, cols)``.

        Returns:
            Row-scaled matrix of the same shape.

        Raises:
            ValueError: If *a* does not have the expected number of rows.
        """
        _validate_matrix_rows(a, self.log_scale.numel(), "a")
        return torch.exp(self.log_scale).unsqueeze(1) * a

    def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
        """Identity right-inverse — raw weight is returned unchanged.

        Because ``W = diag(exp(s)) @ A`` has two free variables (``s`` and
        ``A``) for one equation, the inverse is under-determined. Returning
        the input unchanged satisfies PyTorch's parametrize contract while
        leaving the scale parameter free.

        Args:
            w: Effective weight tensor.

        Returns:
            *w* unchanged.
        """
        return w


class PositiveColumnScale(nn.Module):
    """Apply positive column-wise scaling: ``W = A @ diag(exp(s))``.

    Owns an independent ``log_scale`` parameter (one scalar per column).

    Attributes:
        log_scale (nn.Parameter): Learnable log-scale of shape ``(cols,)``.
    """

    def __init__(self, cols: int, mean: float = 0.0, std: float = 0.1) -> None:
        """Initialize the column scaling parametrization.

        Args:
            cols: Number of columns in the target weight matrix.
            mean: Mean for log-scale initialisation. Default ``0.0`` gives
                unit scale (``exp(0) = 1``).
            std: Standard deviation for log-scale initialisation.
        """
        super().__init__()
        self.log_scale = nn.Parameter(torch.empty(cols))
        nn.init.normal_(self.log_scale, mean=mean, std=std)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """Apply column scaling.

        Args:
            a: Base weight matrix of shape ``(rows, cols)``.

        Returns:
            Column-scaled matrix of the same shape.

        Raises:
            ValueError: If *a* does not have the expected number of columns.
        """
        _validate_matrix_cols(a, self.log_scale.numel(), "a")
        return a * torch.exp(self.log_scale).unsqueeze(0)

    def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
        """Identity right-inverse — raw weight is returned unchanged.

        Because ``W = A @ diag(exp(s))`` has two free variables (``s`` and
        ``A``) for one equation, the inverse is under-determined. Returning
        the input unchanged satisfies PyTorch's parametrize contract while
        leaving the scale parameter free.

        Args:
            w: Effective weight tensor.

        Returns:
            *w* unchanged.
        """
        return w


class PositiveSandwichScale(nn.Module):
    """Apply positive sandwich scaling: ``W = D @ A @ D`` where ``D = diag(exp(s))``.

    Preserves symmetry and positive definiteness of the base matrix.

    Attributes:
        log_scale (nn.Parameter): Learnable log-scale of shape ``(size,)``.
    """

    def __init__(self, size: int, mean: float = 0.0, std: float = 0.1) -> None:
        """Initialize the sandwich scaling parametrization.

        Args:
            size: Dimension of the square weight matrix.
            mean: Mean for log-scale initialisation. Default ``0.0`` gives
                unit scale (``exp(0) = 1``).
            std: Standard deviation for log-scale initialisation.
        """
        super().__init__()
        self.log_scale = nn.Parameter(torch.empty(size))
        nn.init.normal_(self.log_scale, mean=mean, std=std)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """Apply sandwich scaling.

        Args:
            a: Base square weight matrix of shape ``(size, size)``.

        Returns:
            Sandwich-scaled matrix of the same shape.

        Raises:
            ValueError: If *a* is not square or does not match *size*.
        """
        _validate_square_matrix(a, "a")
        _validate_matrix_rows(a, self.log_scale.numel(), "a")
        scale = torch.exp(self.log_scale)
        return scale.unsqueeze(1) * a * scale.unsqueeze(0)

    def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
        """Identity right-inverse — raw weight is returned unchanged.

        Because ``W = D @ A @ D`` has two free variables (``D`` and ``A``)
        for one equation, the inverse is under-determined. Returning the
        input unchanged satisfies PyTorch's parametrize contract.

        Args:
            w: Effective weight tensor.

        Returns:
            *w* unchanged.
        """
        return w


class PositiveScalarScale(nn.Module):
    """Apply a positive scalar scaling: ``W = exp(s) * A``.

    Owns an independent ``log_scale`` scalar parameter.

    Attributes:
        log_scale (nn.Parameter): Learnable scalar log-scale.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.1) -> None:
        """Initialize the scalar scaling parametrization.

        Args:
            mean: Mean for log-scale initialisation. Default ``0.0`` gives
                unit scale (``exp(0) = 1``).
            std: Standard deviation for log-scale initialisation.
        """
        super().__init__()
        self.log_scale = nn.Parameter(torch.empty(()))
        nn.init.normal_(self.log_scale, mean=mean, std=std)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """Apply scalar scaling.

        Args:
            a: Base tensor of any shape.

        Returns:
            Positively scaled tensor of the same shape.
        """
        return torch.exp(self.log_scale) * a

    def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
        """Identity right-inverse — raw weight is returned unchanged.

        Because ``W = exp(s) * A`` has two free variables (``s`` and ``A``)
        for one equation, the inverse is under-determined. Returning the
        input unchanged satisfies PyTorch's parametrize contract.

        Args:
            w: Effective weight tensor.

        Returns:
            *w* unchanged.
        """
        return w
