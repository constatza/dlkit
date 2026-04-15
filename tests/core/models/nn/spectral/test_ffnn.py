"""Tests for spectral FFNN families: composable bases and convenience constructors."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dlkit.domain.nn.spectral import (
    DualPathFFNN,
    FourierAugmented,
    FourierEnhancedFFNN,
    SpectralDualPath,
)


class TestFourierEnhancedFFNN:
    def test_output_shape(self, flat_input: torch.Tensor, n_modes: int) -> None:
        model = FourierEnhancedFFNN(
            in_features=16, out_features=4, hidden_size=32, num_layers=2, n_modes=n_modes
        )
        assert model(flat_input).shape == (flat_input.shape[0], 4)

    def test_scale_equivariance_is_not_guaranteed(
        self, flat_input: torch.Tensor, n_modes: int
    ) -> None:
        """FourierEnhancedFFNN is NOT scale-equivariant by design — verify this."""
        model = FourierEnhancedFFNN(
            in_features=16, out_features=4, hidden_size=32, num_layers=2, n_modes=n_modes
        )
        scale = 3.0
        out_base = model(flat_input)
        out_scaled = model(flat_input * scale)
        assert not torch.allclose(out_scaled, out_base * scale, atol=1e-3)

    def test_is_fourier_augmented_subclass(self, n_modes: int) -> None:
        model = FourierEnhancedFFNN(
            in_features=8, out_features=2, hidden_size=16, num_layers=2, n_modes=n_modes
        )
        assert isinstance(model, FourierAugmented)

    def test_n_modes_larger_than_features_does_not_raise(self, batch_size: int) -> None:
        model = FourierEnhancedFFNN(
            in_features=4, out_features=2, hidden_size=8, num_layers=2, n_modes=100
        )
        x = torch.randn(batch_size, 4)
        assert model(x).shape == (batch_size, 2)

    def test_is_differentiable(self, flat_input: torch.Tensor, n_modes: int) -> None:
        model = FourierEnhancedFFNN(
            in_features=16, out_features=4, hidden_size=32, num_layers=2, n_modes=n_modes
        )
        x = flat_input.requires_grad_(True)
        model(x).sum().backward()
        assert x.grad is not None


class TestDualPathFFNN:
    @pytest.fixture(params=["add", "concat"])
    def merge_mode(self, request: pytest.FixtureRequest) -> str:
        return request.param  # type: ignore[return-value]

    def test_output_shape(self, flat_input: torch.Tensor, n_modes: int, merge_mode: str) -> None:
        model = DualPathFFNN(
            in_features=16,
            out_features=4,
            hidden_size=32,
            num_layers=2,
            n_modes=n_modes,
            merge=merge_mode,
        )
        assert model(flat_input).shape == (flat_input.shape[0], 4)

    def test_is_spectral_dual_path_subclass(self, n_modes: int) -> None:
        model = DualPathFFNN(
            in_features=8, out_features=2, hidden_size=16, num_layers=2, n_modes=n_modes
        )
        assert isinstance(model, SpectralDualPath)

    def test_rejects_invalid_merge(self) -> None:
        with pytest.raises(ValueError, match="merge must be"):
            DualPathFFNN(
                in_features=8,
                out_features=2,
                hidden_size=16,
                num_layers=2,
                n_modes=4,
                merge="invalid",  # type: ignore[arg-type]
            )

    def test_is_differentiable(self, flat_input: torch.Tensor, n_modes: int) -> None:
        model = DualPathFFNN(
            in_features=16, out_features=4, hidden_size=32, num_layers=2, n_modes=n_modes
        )
        x = flat_input.requires_grad_(True)
        model(x).sum().backward()
        assert x.grad is not None


class TestFourierAugmented:
    """Tests for the composable FourierAugmented base."""

    def test_custom_backbone_injected(self, flat_input: torch.Tensor, n_modes: int) -> None:
        """Any backbone sized to (in_features + n_modes*2) can be injected."""
        in_features, out_features = 16, 4
        augmented_in = in_features + n_modes * 2
        backbone = nn.Sequential(
            nn.Linear(augmented_in, 32), nn.ReLU(), nn.Linear(32, out_features)
        )
        model = FourierAugmented(backbone=backbone, n_modes=n_modes)
        assert model(flat_input).shape == (flat_input.shape[0], out_features)

    def test_is_nn_module(self, n_modes: int) -> None:
        model = FourierAugmented(backbone=nn.Linear(16 + n_modes * 2, 4), n_modes=n_modes)
        assert isinstance(model, nn.Module)

    def test_is_differentiable(self, flat_input: torch.Tensor, n_modes: int) -> None:
        in_features = flat_input.shape[-1]
        backbone = nn.Linear(in_features + n_modes * 2, 4)
        model = FourierAugmented(backbone=backbone, n_modes=n_modes)
        x = flat_input.requires_grad_(True)
        model(x).sum().backward()
        assert x.grad is not None

    def test_n_modes_larger_than_input_does_not_raise(self, batch_size: int) -> None:
        in_features = 4
        n_modes = 200
        backbone = nn.Linear(in_features + n_modes * 2, 2)
        model = FourierAugmented(backbone=backbone, n_modes=n_modes)
        x = torch.randn(batch_size, in_features)
        assert model(x).shape == (batch_size, 2)


class TestSpectralDualPath:
    """Tests for the composable SpectralDualPath base."""

    @pytest.fixture
    def custom_model(self, n_modes: int) -> SpectralDualPath:
        hidden = 32
        spatial = nn.Sequential(nn.Linear(16, hidden), nn.ReLU())
        spectral = nn.Sequential(nn.Linear(n_modes * 2, hidden), nn.ReLU())
        projection = nn.Linear(hidden, 4)
        return SpectralDualPath(
            spatial_branch=spatial,
            spectral_branch=spectral,
            projection=projection,
            n_modes=n_modes,
            merge="add",
        )

    def test_output_shape(self, custom_model: SpectralDualPath, flat_input: torch.Tensor) -> None:
        assert custom_model(flat_input).shape == (flat_input.shape[0], 4)

    def test_concat_merge(self, flat_input: torch.Tensor, n_modes: int) -> None:
        hidden = 16
        spatial = nn.Sequential(nn.Linear(16, hidden), nn.ReLU())
        spectral = nn.Sequential(nn.Linear(n_modes * 2, hidden), nn.ReLU())
        projection = nn.Linear(hidden * 2, 4)
        model = SpectralDualPath(
            spatial_branch=spatial,
            spectral_branch=spectral,
            projection=projection,
            n_modes=n_modes,
            merge="concat",
        )
        assert model(flat_input).shape == (flat_input.shape[0], 4)

    def test_rejects_invalid_merge(self, n_modes: int) -> None:
        with pytest.raises(ValueError, match="merge must be"):
            SpectralDualPath(
                spatial_branch=nn.Identity(),
                spectral_branch=nn.Identity(),
                projection=nn.Identity(),
                n_modes=n_modes,
                merge="invalid",  # type: ignore[arg-type]
            )

    def test_is_differentiable(
        self, custom_model: SpectralDualPath, flat_input: torch.Tensor
    ) -> None:
        x = flat_input.requires_grad_(True)
        custom_model(x).sum().backward()
        assert x.grad is not None
