"""Comprehensive tests for PCA transformation following SOLID principles."""

from __future__ import annotations

import pytest
import torch

from dlkit.domain.transforms.pca import PCA


def _expect_tensor(value: object) -> torch.Tensor:
    assert isinstance(value, torch.Tensor)
    return value


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_2d_data() -> torch.Tensor:
    """Simple 2D dataset for basic PCA testing.

    Returns:
        Tensor of shape (100, 5) with known statistical properties.
    """
    torch.manual_seed(42)
    return torch.randn(100, 5)


@pytest.fixture
def correlated_data() -> torch.Tensor:
    """Correlated data where PCA should capture significant variance.

    Returns:
        Tensor of shape (200, 10) with strong correlations.
    """
    torch.manual_seed(123)
    n_samples = 200
    # Create correlated features
    base = torch.randn(n_samples, 3)
    # Add correlated features
    expanded = torch.cat(
        [
            base,
            base[:, :2] * 2 + torch.randn(n_samples, 2) * 0.1,
            base[:, :3] * 1.5 + torch.randn(n_samples, 3) * 0.1,
            torch.randn(n_samples, 2) * 0.5,
        ],
        dim=1,
    )
    return expanded


@pytest.fixture
def multidim_data() -> torch.Tensor:
    """Multi-dimensional data (N, T, D) for testing reshaping logic.

    Returns:
        Tensor of shape (20, 8, 6) representing sequences.
    """
    torch.manual_seed(456)
    return torch.randn(20, 8, 6)


@pytest.fixture
def pca_2_components() -> PCA:
    """PCA transformer with 2 components (unfitted).

    Returns:
        Unfitted PCA instance.
    """
    return PCA(n_components=2)


@pytest.fixture
def pca_5_components() -> PCA:
    """PCA transformer with 5 components (unfitted).

    Returns:
        Unfitted PCA instance.
    """
    return PCA(n_components=5)


# ============================================================================
# Good Path Tests
# ============================================================================


class TestPCAGoodPath:
    """Test expected behavior of PCA transformation."""

    def test_fit_sets_fitted_flag(
        self, pca_2_components: PCA, simple_2d_data: torch.Tensor
    ) -> None:
        """Fitting PCA should set the fitted flag to True."""
        assert not pca_2_components.fitted
        pca_2_components.fit(simple_2d_data)
        assert pca_2_components.fitted

    def test_fit_computes_components(
        self, pca_2_components: PCA, simple_2d_data: torch.Tensor
    ) -> None:
        """Fitting should compute principal components with correct shape."""
        pca_2_components.fit(simple_2d_data)

        assert pca_2_components.components is not None
        assert pca_2_components.components.shape == (2, 5)  # (n_components, n_features)

    def test_fit_computes_mean(self, pca_2_components: PCA, simple_2d_data: torch.Tensor) -> None:
        """Fitting should compute and store the data mean."""
        pca_2_components.fit(simple_2d_data)

        assert pca_2_components.mean is not None
        assert pca_2_components.mean.shape == (1, 5)

        # Mean should be close to actual data mean
        expected_mean = simple_2d_data.mean(dim=0, keepdim=True)
        assert torch.allclose(_expect_tensor(pca_2_components.mean), expected_mean, atol=1e-5)

    def test_fit_computes_explained_variance(
        self, pca_2_components: PCA, simple_2d_data: torch.Tensor
    ) -> None:
        """Fitting should compute explained variance and ratio."""
        pca_2_components.fit(simple_2d_data)

        assert pca_2_components.explained_variance is not None
        assert pca_2_components.explained_variance_ratio is not None
        assert pca_2_components.total_explained_variance is not None

        # Variance should be positive
        explained_variance = _expect_tensor(pca_2_components.explained_variance)
        explained_variance_ratio = _expect_tensor(pca_2_components.explained_variance_ratio)
        assert torch.all(explained_variance > 0)
        assert torch.all(explained_variance_ratio > 0)

        # Ratio should be between 0 and 1
        assert torch.all(explained_variance_ratio <= 1.0)

    def test_forward_projects_to_lower_dimension(
        self, pca_2_components: PCA, simple_2d_data: torch.Tensor
    ) -> None:
        """Forward transform should reduce dimensionality."""
        pca_2_components.fit(simple_2d_data)
        projected = pca_2_components.forward(simple_2d_data)

        assert projected.shape == (100, 2)  # Reduced from 5 to 2 dimensions

    def test_inverse_transform_reconstructs_data(
        self, pca_2_components: PCA, simple_2d_data: torch.Tensor
    ) -> None:
        """Inverse transform should reconstruct approximately."""
        pca_2_components.fit(simple_2d_data)
        projected = pca_2_components.forward(simple_2d_data)
        reconstructed = pca_2_components.inverse_transform(projected)

        assert reconstructed.shape == simple_2d_data.shape
        # With only 2 components, reconstruction won't be perfect
        # but should be reasonably close
        assert not torch.allclose(reconstructed, simple_2d_data, atol=1e-5)

    def test_full_rank_pca_perfect_reconstruction(
        self, pca_5_components: PCA, simple_2d_data: torch.Tensor
    ) -> None:
        """PCA with all components should perfectly reconstruct."""
        pca_5_components.fit(simple_2d_data)
        projected = pca_5_components.forward(simple_2d_data)
        reconstructed = pca_5_components.inverse_transform(projected)

        # Full-rank PCA should give near-perfect reconstruction
        assert torch.allclose(reconstructed, simple_2d_data, atol=1e-4)

    def test_explained_variance_ratio_sums_correctly(self, correlated_data: torch.Tensor) -> None:
        """Total explained variance should be reasonable for correlated data."""
        pca = PCA(n_components=3)
        pca.fit(correlated_data)

        # With highly correlated data, 3 components should capture significant variance
        assert pca.total_explained_variance is not None
        assert _expect_tensor(pca.total_explained_variance) > 0.5  # At least 50%

    def test_multidimensional_data_handling(self, multidim_data: torch.Tensor) -> None:
        """PCA should handle 3D data by reshaping correctly."""
        pca = PCA(n_components=3)
        pca.fit(multidim_data)

        # Forward should flatten (20, 8, 6) -> (160, 6) internally then project to (160, 3)
        projected = pca.forward(multidim_data)
        # Output should be reshaped back to (20, 8, 3)
        assert projected.shape == (20, 8, 3)

        # Inverse should restore to (20, 8, 6)
        reconstructed = pca.inverse_transform(projected)
        assert reconstructed.shape == multidim_data.shape

    def test_centered_data_has_zero_mean_projection(
        self, pca_2_components: PCA, simple_2d_data: torch.Tensor
    ) -> None:
        """Projected data should be centered (mean near zero)."""
        pca_2_components.fit(simple_2d_data)
        projected = pca_2_components.forward(simple_2d_data)

        # Mean of projected data should be near zero
        mean_proj = projected.mean(dim=0)
        assert torch.allclose(mean_proj, torch.zeros_like(mean_proj), atol=1e-5)

    def test_components_are_orthogonal(
        self, pca_2_components: PCA, simple_2d_data: torch.Tensor
    ) -> None:
        """Principal components should be orthogonal."""
        pca_2_components.fit(simple_2d_data)

        components = pca_2_components.components
        assert components is not None
        components_tensor = _expect_tensor(components)

        # Compute inner product matrix
        inner_products = torch.matmul(components_tensor, components_tensor.T)

        # Should be close to identity (orthonormal)
        identity = torch.eye(2)
        assert torch.allclose(inner_products, identity, atol=1e-4)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestPCAEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_forward_before_fit_raises_error(
        self, pca_2_components: PCA, simple_2d_data: torch.Tensor
    ) -> None:
        """Calling forward before fit should raise RuntimeError."""
        from dlkit.domain.transforms.errors import TransformNotFittedError

        with pytest.raises(TransformNotFittedError):
            pca_2_components.forward(simple_2d_data)

    def test_inverse_before_fit_raises_error(
        self, pca_2_components: PCA, simple_2d_data: torch.Tensor
    ) -> None:
        """Calling inverse_transform before fit should raise RuntimeError."""
        from dlkit.domain.transforms.errors import TransformNotFittedError

        projected = torch.randn(100, 2)
        with pytest.raises(TransformNotFittedError):
            pca_2_components.inverse_transform(projected)

    def test_single_component_pca(self, simple_2d_data: torch.Tensor) -> None:
        """PCA with single component should work."""
        pca = PCA(n_components=1)
        pca.fit(simple_2d_data)
        projected = pca.forward(simple_2d_data)

        assert projected.shape == (100, 1)
        assert pca.explained_variance is not None
        assert pca.explained_variance.shape == (1,)

    def test_1d_data_handling(self) -> None:
        """PCA should handle 1D data (single feature)."""
        torch.manual_seed(789)
        data_1d = torch.randn(50, 1)

        pca = PCA(n_components=1)
        pca.fit(data_1d)
        projected = pca.forward(data_1d)

        # Should preserve the single dimension
        assert projected.shape == (50, 1)

    def test_fit_with_custom_dim_parameter(self, multidim_data: torch.Tensor) -> None:
        """Fit should respect the dim parameter for feature dimension."""
        pca = PCA(n_components=3)

        # Fit along last dimension (default)
        pca.fit(multidim_data, dim=-1)
        assert pca.fitted

        # Components should match the last dimension size
        assert pca.components is not None
        assert _expect_tensor(pca.components).shape[1] == 6  # Last dim of (20, 8, 6)

    def test_device_transfer_in_inverse_transform(
        self, pca_2_components: PCA, simple_2d_data: torch.Tensor
    ) -> None:
        """Inverse transform should handle device transfer correctly."""
        pca_2_components.fit(simple_2d_data)
        projected = pca_2_components.forward(simple_2d_data)

        # Move projected data to same device (CPU in this case)
        projected_same = projected.to("cpu")
        reconstructed = pca_2_components.inverse_transform(projected_same)

        assert reconstructed.device == projected_same.device


# ============================================================================
# Property-Based Tests
# ============================================================================


class TestPCAProperties:
    """Test mathematical properties that should always hold."""

    def test_variance_monotonically_decreasing(self, simple_2d_data: torch.Tensor) -> None:
        """Explained variance should be sorted in descending order."""
        pca = PCA(n_components=4)
        pca.fit(simple_2d_data)

        assert pca.explained_variance is not None
        variances = _expect_tensor(pca.explained_variance)

        # Check monotonically decreasing
        for i in range(len(variances) - 1):
            assert variances[i] >= variances[i + 1]

    def test_projection_preserves_variance_order(self, correlated_data: torch.Tensor) -> None:
        """Projected data variance should match explained variance ordering."""
        pca = PCA(n_components=5)
        pca.fit(correlated_data)
        projected = pca.forward(correlated_data)

        # Compute variance of each projected component
        proj_var = projected.var(dim=0, unbiased=True)

        # Should match explained variance
        assert pca.explained_variance is not None
        assert torch.allclose(proj_var, _expect_tensor(pca.explained_variance), atol=1e-4)

    def test_full_rank_reconstruction_machine_precision(self) -> None:
        """Full-rank PCA should reconstruct with only machine error.

        This tests the mathematical correctness of the algorithm.
        """
        torch.manual_seed(999)
        n_samples, n_features = 50, 8
        data = torch.randn(n_samples, n_features)

        # Use ALL components (full rank)
        pca = PCA(n_components=n_features)
        pca.fit(data)

        projected = pca.forward(data)
        reconstructed = pca.inverse_transform(projected)

        # Reconstruction error should be at machine precision
        mse = torch.mean((data - reconstructed) ** 2).item()
        assert mse < 1e-10, f"Full-rank reconstruction MSE {mse} exceeds machine precision"

        # Max absolute error should also be tiny
        max_abs_error = torch.max(torch.abs(data - reconstructed)).item()
        assert max_abs_error < 1e-5, f"Max error {max_abs_error} too large"

    def test_explained_variance_ratio_mathematical_correctness(self) -> None:
        """Explained variance ratios must sum to ≤1.0 and be monotonic.

        This is a fundamental mathematical property of PCA.
        """
        torch.manual_seed(888)
        data = torch.randn(100, 10)

        pca = PCA(n_components=8)
        pca.fit(data)

        assert pca.explained_variance_ratio is not None
        ratios = _expect_tensor(pca.explained_variance_ratio)

        # Each ratio must be in [0, 1]
        assert torch.all(ratios >= 0.0)
        assert torch.all(ratios <= 1.0)

        # Sum of ratios must be ≤ 1.0 (we're not using all components)
        total = torch.sum(ratios).item()
        assert total <= 1.0, f"Variance ratios sum to {total} > 1.0"

        # Ratios must be monotonically decreasing
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1], f"Variance not decreasing at index {i}"

    def test_known_data_known_components(self) -> None:
        """Test PCA on synthetic data with known principal components."""
        # Create data aligned with coordinate axes (known PCs)
        torch.manual_seed(777)
        n_samples = 200

        # First component: large variance along first axis
        # Second component: medium variance along second axis
        # Third component: small variance along third axis
        comp1 = torch.randn(n_samples) * 10.0  # Large variance
        comp2 = torch.randn(n_samples) * 3.0  # Medium variance
        comp3 = torch.randn(n_samples) * 0.5  # Small variance

        data = torch.stack([comp1, comp2, comp3], dim=1)

        pca = PCA(n_components=3)
        pca.fit(data)

        # First component should explain most variance
        assert pca.explained_variance_ratio is not None
        ratios = _expect_tensor(pca.explained_variance_ratio)

        # First should be much larger than second
        assert ratios[0] > ratios[1]
        assert ratios[1] > ratios[2]

        # First component should explain > 80% of variance (comp1 has 10x variance)
        assert ratios[0] > 0.80

    def test_component_removal_increases_reconstruction_error(self) -> None:
        """Removing components must increase reconstruction error.

        This validates that components capture real variance.
        """
        torch.manual_seed(666)
        data = torch.randn(100, 10)

        # Test with different numbers of components
        errors = []
        for n_comp in [10, 7, 5, 3, 1]:
            pca = PCA(n_components=n_comp)
            pca.fit(data)
            projected = pca.forward(data)
            reconstructed = pca.inverse_transform(projected)
            mse = torch.mean((data - reconstructed) ** 2).item()
            errors.append(mse)

        # Errors must increase monotonically as we remove components
        for i in range(len(errors) - 1):
            assert errors[i] < errors[i + 1], (
                f"Error with more components ({errors[i]}) >= error with fewer ({errors[i + 1]})"
            )


# ============================================================================
# Integration Tests
# ============================================================================


class TestPCAIntegration:
    """Test PCA in realistic usage scenarios."""

    def test_fit_transform_inverse_pipeline(self, correlated_data: torch.Tensor) -> None:
        """Complete pipeline: fit -> transform -> inverse."""
        pca = PCA(n_components=5)

        # Fit
        pca.fit(correlated_data)
        assert pca.fitted

        # Transform
        projected = pca.forward(correlated_data)
        assert projected.shape[1] == 5

        # Inverse
        reconstructed = pca.inverse_transform(projected)
        assert reconstructed.shape == correlated_data.shape

        # Check reconstruction quality
        mse = torch.mean((reconstructed - correlated_data) ** 2)
        assert mse < 1.0  # Should have reasonable reconstruction

    def test_transform_new_data_after_fit(self, simple_2d_data: torch.Tensor) -> None:
        """PCA should work on new data after fitting."""
        # Split data
        train_data = simple_2d_data[:80]
        test_data = simple_2d_data[80:]

        pca = PCA(n_components=3)
        pca.fit(train_data)

        # Transform test data
        test_projected = pca.forward(test_data)
        assert test_projected.shape == (20, 3)

        # Reconstruct test data
        test_reconstructed = pca.inverse_transform(test_projected)
        assert test_reconstructed.shape == test_data.shape

    def test_batched_processing(self, simple_2d_data: torch.Tensor) -> None:
        """PCA should give consistent results with batched processing."""
        pca = PCA(n_components=3)
        pca.fit(simple_2d_data)

        # Process all at once
        all_projected = pca.forward(simple_2d_data)

        # Process in batches
        batch_size = 25
        batched_projected = []
        for i in range(0, len(simple_2d_data), batch_size):
            batch = simple_2d_data[i : i + batch_size]
            batched_projected.append(pca.forward(batch))

        batched_result = torch.cat(batched_projected, dim=0)

        # Should be identical
        assert torch.allclose(all_projected, batched_result, atol=1e-6)
