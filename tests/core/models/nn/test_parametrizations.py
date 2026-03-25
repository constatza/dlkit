from __future__ import annotations

import pytest
import torch

from dlkit.core.models.nn.primitives.parametrizations import (
    SPD,
    PositiveColumnScale,
    PositiveRowScale,
    PositiveSandwichScale,
    PositiveScalarScale,
    Symmetric,
)

# ---------------------------------------------------------------------------
# Fixtures — shared matrix data
# ---------------------------------------------------------------------------


@pytest.fixture
def square_base() -> torch.Tensor:
    """Fixed 3×3 square matrix."""
    return torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=torch.float32,
    )


@pytest.fixture
def symmetric_matrix(square_base: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix derived from *square_base*."""
    return 0.5 * (square_base + square_base.T)


@pytest.fixture
def spd_matrix() -> torch.Tensor:
    """Fixed 3×3 symmetric positive-definite matrix outside the new SPD image."""
    return torch.tensor(
        [[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]],
        dtype=torch.float32,
    )


@pytest.fixture
def representable_spd_matrix(
    spd_parametrization: SPD,
    symmetric_matrix: torch.Tensor,
) -> torch.Tensor:
    """Matrix produced by the SPD parametrization itself."""
    return spd_parametrization(symmetric_matrix)


@pytest.fixture
def rectangular_matrix() -> torch.Tensor:
    """Fixed 3×2 rectangular matrix."""
    return torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
    )


@pytest.fixture
def row_scale_values() -> torch.Tensor:
    """Row scale values (one per row of the rectangular matrix)."""
    return torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)


@pytest.fixture
def column_scale_values() -> torch.Tensor:
    """Column scale values (one per column of the rectangular matrix)."""
    return torch.tensor([2.0, 3.0], dtype=torch.float32)


@pytest.fixture
def sandwich_scale_values() -> torch.Tensor:
    """Sandwich scale values for a 3×3 matrix."""
    return torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)


@pytest.fixture
def scalar_scale_value() -> torch.Tensor:
    """Scalar scale value."""
    return torch.tensor(3.0, dtype=torch.float32)


@pytest.fixture
def expected_row_scaled(
    rectangular_matrix: torch.Tensor,
    row_scale_values: torch.Tensor,
) -> torch.Tensor:
    """Expected output of row scaling, derived from scale fixture."""
    return row_scale_values.unsqueeze(1) * rectangular_matrix


@pytest.fixture
def expected_column_scaled(
    rectangular_matrix: torch.Tensor,
    column_scale_values: torch.Tensor,
) -> torch.Tensor:
    """Expected output of column scaling, derived from scale fixture."""
    return rectangular_matrix * column_scale_values.unsqueeze(0)


# ---------------------------------------------------------------------------
# Fixtures — parametrization instances
# ---------------------------------------------------------------------------


@pytest.fixture
def symmetric_parametrization() -> Symmetric:
    """Symmetric parametrization instance."""
    return Symmetric()


@pytest.fixture
def spd_parametrization() -> SPD:
    """SPD parametrization instance with default min_diag."""
    return SPD(min_diag=1e-4)


@pytest.fixture
def row_scale_parametrization(row_scale_values: torch.Tensor) -> PositiveRowScale:
    """PositiveRowScale configured with exact *row_scale_values*."""
    p = PositiveRowScale(rows=row_scale_values.numel())
    with torch.no_grad():
        p.log_scale.copy_(torch.log(row_scale_values))
    return p


@pytest.fixture
def column_scale_parametrization(
    column_scale_values: torch.Tensor,
) -> PositiveColumnScale:
    """PositiveColumnScale configured with exact *column_scale_values*."""
    p = PositiveColumnScale(cols=column_scale_values.numel())
    with torch.no_grad():
        p.log_scale.copy_(torch.log(column_scale_values))
    return p


@pytest.fixture
def sandwich_scale_parametrization(
    sandwich_scale_values: torch.Tensor,
) -> PositiveSandwichScale:
    """PositiveSandwichScale configured with exact *sandwich_scale_values*."""
    p = PositiveSandwichScale(size=sandwich_scale_values.numel())
    with torch.no_grad():
        p.log_scale.copy_(torch.log(sandwich_scale_values))
    return p


@pytest.fixture
def scalar_scale_parametrization(
    scalar_scale_value: torch.Tensor,
) -> PositiveScalarScale:
    """PositiveScalarScale configured with exact *scalar_scale_value*."""
    p = PositiveScalarScale()
    with torch.no_grad():
        p.log_scale.copy_(torch.log(scalar_scale_value))
    return p


# ---------------------------------------------------------------------------
# Tests — Symmetric
# ---------------------------------------------------------------------------


class TestSymmetric:
    """Tests for the Symmetric parametrization."""

    def test_forward_produces_symmetric_matrix(
        self,
        symmetric_parametrization: Symmetric,
        square_base: torch.Tensor,
    ) -> None:
        """forward() must produce a symmetric matrix."""
        result = symmetric_parametrization(square_base)
        assert torch.allclose(result, result.T)

    def test_right_inverse_round_trips(
        self,
        symmetric_parametrization: Symmetric,
        symmetric_matrix: torch.Tensor,
    ) -> None:
        """forward(right_inverse(W)) must equal W for any symmetric W."""
        preimage = symmetric_parametrization.right_inverse(symmetric_matrix)
        result = symmetric_parametrization(preimage)
        assert torch.allclose(result, symmetric_matrix)

    def test_forward_raises_on_non_square(
        self,
        symmetric_parametrization: Symmetric,
        rectangular_matrix: torch.Tensor,
    ) -> None:
        """forward() must raise ValueError for a non-square matrix."""
        with pytest.raises(ValueError, match="square"):
            symmetric_parametrization(rectangular_matrix)

    def test_right_inverse_raises_on_non_square(
        self,
        symmetric_parametrization: Symmetric,
        rectangular_matrix: torch.Tensor,
    ) -> None:
        """right_inverse() must raise ValueError for a non-square matrix."""
        with pytest.raises(ValueError, match="square"):
            symmetric_parametrization.right_inverse(rectangular_matrix)

    def test_forward_raises_on_1d_input(
        self,
        symmetric_parametrization: Symmetric,
    ) -> None:
        """forward() must raise ValueError for a 1-D tensor."""
        with pytest.raises(ValueError, match="2D"):
            symmetric_parametrization(torch.randn(4))


# ---------------------------------------------------------------------------
# Tests — SPD
# ---------------------------------------------------------------------------


class TestSPD:
    """Tests for the SPD parametrization."""

    def test_forward_produces_symmetric_matrix(
        self,
        spd_parametrization: SPD,
        symmetric_matrix: torch.Tensor,
    ) -> None:
        """forward() must produce a symmetric matrix."""
        result = spd_parametrization(symmetric_matrix)
        assert torch.allclose(result, result.T, atol=1e-6)

    def test_forward_produces_positive_definite_matrix(
        self,
        spd_parametrization: SPD,
        symmetric_matrix: torch.Tensor,
    ) -> None:
        """forward() must produce a matrix with all positive eigenvalues."""
        result = spd_parametrization(symmetric_matrix)
        eigvals = torch.linalg.eigvalsh(result)
        assert torch.all(eigvals > 0.0)

    def test_right_inverse_round_trips(
        self,
        spd_parametrization: SPD,
        representable_spd_matrix: torch.Tensor,
    ) -> None:
        """forward(right_inverse(W)) must equal W for any representable SPD matrix W."""
        preimage = spd_parametrization.right_inverse(representable_spd_matrix)
        result = spd_parametrization(preimage)
        assert torch.allclose(result, representable_spd_matrix, atol=1e-5)

    def test_right_inverse_no_nan_near_min_diag(
        self,
        spd_parametrization: SPD,
    ) -> None:
        """right_inverse() must not produce NaN when diagonal slack ≈ min_diag."""
        w = torch.eye(3) * (spd_parametrization.min_diag + 1e-6)
        result = spd_parametrization.right_inverse(w)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_right_inverse_rejects_non_representable_spd_matrix(
        self,
        spd_parametrization: SPD,
        spd_matrix: torch.Tensor,
    ) -> None:
        """right_inverse() must reject SPD matrices outside the diagonal-dominant image."""
        with pytest.raises(NotImplementedError, match="diagonally-dominant SPD image"):
            spd_parametrization.right_inverse(spd_matrix)

    def test_forward_raises_on_non_square(
        self,
        spd_parametrization: SPD,
        rectangular_matrix: torch.Tensor,
    ) -> None:
        """forward() must raise ValueError for a non-square matrix."""
        with pytest.raises(ValueError, match="square"):
            spd_parametrization(rectangular_matrix)

    def test_forward_raises_on_non_symmetric(
        self,
        spd_parametrization: SPD,
        square_base: torch.Tensor,
    ) -> None:
        """forward() must reject non-symmetric inputs."""
        with pytest.raises(ValueError, match="symmetric"):
            spd_parametrization(square_base)


# ---------------------------------------------------------------------------
# Tests — PositiveRowScale
# ---------------------------------------------------------------------------


class TestPositiveRowScale:
    """Tests for the PositiveRowScale parametrization."""

    def test_forward_scales_rows_correctly(
        self,
        row_scale_parametrization: PositiveRowScale,
        rectangular_matrix: torch.Tensor,
        expected_row_scaled: torch.Tensor,
    ) -> None:
        """forward() must independently scale each row."""
        result = row_scale_parametrization(rectangular_matrix)
        assert torch.allclose(result, expected_row_scaled)

    def test_right_inverse_is_identity(
        self,
        row_scale_parametrization: PositiveRowScale,
        rectangular_matrix: torch.Tensor,
    ) -> None:
        """right_inverse() must return the input unchanged."""
        result = row_scale_parametrization.right_inverse(rectangular_matrix)
        assert torch.equal(result, rectangular_matrix)

    def test_default_mean_gives_near_unit_scale(self) -> None:
        """Default mean=0.0 should give log_scale ≈ 0, so exp(s) ≈ 1."""
        p = PositiveRowScale(rows=8)
        # With std=0.1 most values should be near 1 after exp.
        assert p.log_scale.abs().mean().item() < 0.5

    def test_forward_raises_on_row_mismatch(
        self,
        row_scale_parametrization: PositiveRowScale,
    ) -> None:
        """forward() must raise ValueError when row count mismatches."""
        wrong = torch.randn(2, 3)  # expects 3 rows, giving 2
        with pytest.raises(ValueError, match="rows"):
            row_scale_parametrization(wrong)


# ---------------------------------------------------------------------------
# Tests — PositiveColumnScale
# ---------------------------------------------------------------------------


class TestPositiveColumnScale:
    """Tests for the PositiveColumnScale parametrization."""

    def test_forward_scales_columns_correctly(
        self,
        column_scale_parametrization: PositiveColumnScale,
        rectangular_matrix: torch.Tensor,
        expected_column_scaled: torch.Tensor,
    ) -> None:
        """forward() must independently scale each column."""
        result = column_scale_parametrization(rectangular_matrix)
        assert torch.allclose(result, expected_column_scaled)

    def test_right_inverse_is_identity(
        self,
        column_scale_parametrization: PositiveColumnScale,
        rectangular_matrix: torch.Tensor,
    ) -> None:
        """right_inverse() must return the input unchanged."""
        result = column_scale_parametrization.right_inverse(rectangular_matrix)
        assert torch.equal(result, rectangular_matrix)

    def test_forward_raises_on_col_mismatch(
        self,
        column_scale_parametrization: PositiveColumnScale,
    ) -> None:
        """forward() must raise ValueError when column count mismatches."""
        wrong = torch.randn(3, 5)
        with pytest.raises(ValueError, match="columns"):
            column_scale_parametrization(wrong)


# ---------------------------------------------------------------------------
# Tests — PositiveSandwichScale
# ---------------------------------------------------------------------------


class TestPositiveSandwichScale:
    """Tests for the PositiveSandwichScale parametrization."""

    def test_forward_preserves_symmetry(
        self,
        sandwich_scale_parametrization: PositiveSandwichScale,
        symmetric_matrix: torch.Tensor,
    ) -> None:
        """forward() must preserve symmetry of the base matrix."""
        result = sandwich_scale_parametrization(symmetric_matrix)
        assert torch.allclose(result, result.T)

    def test_forward_preserves_positive_definiteness(
        self,
        sandwich_scale_parametrization: PositiveSandwichScale,
        spd_matrix: torch.Tensor,
    ) -> None:
        """forward() must preserve positive definiteness."""
        result = sandwich_scale_parametrization(spd_matrix)
        eigvals = torch.linalg.eigvalsh(result)
        assert torch.all(eigvals > 0.0)

    def test_right_inverse_is_identity(
        self,
        sandwich_scale_parametrization: PositiveSandwichScale,
        symmetric_matrix: torch.Tensor,
    ) -> None:
        """right_inverse() must return the input unchanged."""
        result = sandwich_scale_parametrization.right_inverse(symmetric_matrix)
        assert torch.equal(result, symmetric_matrix)

    def test_forward_raises_on_non_square(
        self,
        sandwich_scale_parametrization: PositiveSandwichScale,
        rectangular_matrix: torch.Tensor,
    ) -> None:
        """forward() must raise ValueError for a non-square matrix."""
        with pytest.raises(ValueError, match="square"):
            sandwich_scale_parametrization(rectangular_matrix)

    def test_forward_raises_on_size_mismatch(
        self,
        sandwich_scale_parametrization: PositiveSandwichScale,
    ) -> None:
        """forward() must raise ValueError when matrix size mismatches log_scale."""
        wrong = torch.randn(2, 2)
        with pytest.raises(ValueError, match="rows"):
            sandwich_scale_parametrization(wrong)


# ---------------------------------------------------------------------------
# Tests — PositiveScalarScale
# ---------------------------------------------------------------------------


class TestPositiveScalarScale:
    """Tests for the PositiveScalarScale parametrization."""

    def test_forward_scales_tensor_uniformly(
        self,
        scalar_scale_parametrization: PositiveScalarScale,
        rectangular_matrix: torch.Tensor,
        scalar_scale_value: torch.Tensor,
    ) -> None:
        """forward() must uniformly scale the entire tensor."""
        result = scalar_scale_parametrization(rectangular_matrix)
        expected = scalar_scale_value * rectangular_matrix
        assert torch.allclose(result, expected)

    def test_right_inverse_is_identity(
        self,
        scalar_scale_parametrization: PositiveScalarScale,
        rectangular_matrix: torch.Tensor,
    ) -> None:
        """right_inverse() must return the input unchanged."""
        result = scalar_scale_parametrization.right_inverse(rectangular_matrix)
        assert torch.equal(result, rectangular_matrix)

    def test_default_mean_gives_near_unit_scale(self) -> None:
        """Default mean=0.0 should give log_scale ≈ 0, so exp(s) ≈ 1."""
        p = PositiveScalarScale()
        assert abs(p.log_scale.item()) < 0.5
