"""Tests for convolutional autoencoders and VAE.

Tests SkipCAE1d, LinearCAE1d, VAE1d, vae_loss, and reparameterize
for shape transformations, inheritance, and loss functions.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dlkit.domain.nn.base import DLKitModel
from dlkit.domain.nn.cae import (
    LinearCAE1d,
    SkipCAE1d,
    VAE1d,
    reparameterize,
    vae_loss,
)
from dlkit.domain.nn.cae.base import CAE


@pytest.fixture
def cae_kwargs() -> dict:
    """Minimal CAE configuration: 2ch in/out, 16-length sequence."""
    return dict(
        in_channels=2,
        in_length=16,
        latent_channels=2,
        latent_size=4,
        latent_width=2,
        num_layers=2,
    )


@pytest.fixture
def skip_cae(cae_kwargs: dict) -> SkipCAE1d:
    """Instantiated SkipCAE1d with minimal config."""
    return SkipCAE1d(**cae_kwargs)


@pytest.fixture
def linear_cae(cae_kwargs: dict) -> LinearCAE1d:
    """Instantiated LinearCAE1d with minimal config."""
    return LinearCAE1d(**cae_kwargs)


@pytest.fixture
def vae(cae_kwargs: dict) -> VAE1d:
    """Instantiated VAE1d with minimal config."""
    return VAE1d(**cae_kwargs)


class TestSkipCAE1d:
    """Tests for skip-connection convolutional autoencoder."""

    def test_forward_preserves_shape(self, skip_cae: SkipCAE1d, conv_input: torch.Tensor) -> None:
        """Forward pass should preserve input shape."""
        out = skip_cae(conv_input)
        assert out.shape == conv_input.shape

    def test_encode_decode_roundtrip(self, skip_cae: SkipCAE1d, conv_input: torch.Tensor) -> None:
        """encode() → decode() should preserve shape."""
        z = skip_cae.encode(conv_input)
        out = skip_cae.decode(z)
        assert out.shape == conv_input.shape

    def test_encode_output_shape(self, skip_cae: SkipCAE1d, conv_input: torch.Tensor) -> None:
        """encode() should return latent vector of size latent_size."""
        z = skip_cae.encode(conv_input)
        assert z.shape == (conv_input.shape[0], 4)  # latent_size=4

    def test_is_cae(self, skip_cae: SkipCAE1d) -> None:
        """SkipCAE1d should be an instance of CAE."""
        assert isinstance(skip_cae, CAE)

    def test_is_dlkit_model(self, skip_cae: SkipCAE1d) -> None:
        """SkipCAE1d should be an instance of DLKitModel."""
        assert isinstance(skip_cae, DLKitModel)

    def test_has_parameters(self, skip_cae: SkipCAE1d) -> None:
        """SkipCAE1d should have trainable parameters."""
        assert len(list(skip_cae.parameters())) > 0

    def test_has_encode_decode_methods(self, skip_cae: SkipCAE1d) -> None:
        """SkipCAE1d should have encode() and decode() methods."""
        assert hasattr(skip_cae, "encode")
        assert hasattr(skip_cae, "decode")
        assert callable(skip_cae.encode)
        assert callable(skip_cae.decode)

    def test_with_batch_normalization(self, cae_kwargs: dict, conv_input: torch.Tensor) -> None:
        """SkipCAE1d with batch norm should work."""
        cae_kwargs["normalize"] = "batch"
        cae = SkipCAE1d(**cae_kwargs)
        cae.eval()
        out = cae(conv_input)
        assert out.shape == conv_input.shape

    def test_with_layer_normalization(self, cae_kwargs: dict, conv_input: torch.Tensor) -> None:
        """SkipCAE1d with layer norm should work."""
        cae_kwargs["normalize"] = "layer"
        cae = SkipCAE1d(**cae_kwargs)
        out = cae(conv_input)
        assert out.shape == conv_input.shape

    def test_with_dropout(self, cae_kwargs: dict, conv_input: torch.Tensor) -> None:
        """SkipCAE1d with dropout should work."""
        cae_kwargs["dropout"] = 0.2
        cae = SkipCAE1d(**cae_kwargs)
        cae.eval()
        out = cae(conv_input)
        assert out.shape == conv_input.shape


class TestLinearCAE1d:
    """Tests for linear convolutional autoencoder."""

    def test_forward_preserves_shape(
        self, linear_cae: LinearCAE1d, conv_input: torch.Tensor
    ) -> None:
        """Forward pass should preserve input shape."""
        out = linear_cae(conv_input)
        assert out.shape == conv_input.shape

    def test_is_skip_cae(self, linear_cae: LinearCAE1d) -> None:
        """LinearCAE1d must be a SkipCAE1d (direct inheritance)."""
        assert isinstance(linear_cae, SkipCAE1d)

    def test_is_cae(self, linear_cae: LinearCAE1d) -> None:
        """LinearCAE1d should be an instance of CAE."""
        assert isinstance(linear_cae, CAE)

    def test_is_dlkit_model(self, linear_cae: LinearCAE1d) -> None:
        """LinearCAE1d should be an instance of DLKitModel."""
        assert isinstance(linear_cae, DLKitModel)

    def test_no_impl_delegation(self, linear_cae: LinearCAE1d) -> None:
        """Old delegation pattern should be gone (no _impl)."""
        assert not hasattr(linear_cae, "_impl")

    def test_encode_decode_roundtrip(
        self, linear_cae: LinearCAE1d, conv_input: torch.Tensor
    ) -> None:
        """encode() → decode() should preserve shape."""
        z = linear_cae.encode(conv_input)
        out = linear_cae.decode(z)
        assert out.shape == conv_input.shape

    def test_has_encoder_decoder_attributes(self, linear_cae: LinearCAE1d) -> None:
        """LinearCAE1d should have encoder/decoder components."""
        assert hasattr(linear_cae, "encoder")
        assert hasattr(linear_cae, "decoder")


class TestVAE1d:
    """Tests for variational autoencoder."""

    def test_forward_returns_triple(self, vae: VAE1d, conv_input: torch.Tensor) -> None:
        """forward() should return (recon, mu, logvar) tuple."""
        result = vae(conv_input)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_reconstruction_shape(self, vae: VAE1d, conv_input: torch.Tensor) -> None:
        """Reconstructed output should match input shape."""
        recon, mu, logvar = vae(conv_input)
        assert recon.shape == conv_input.shape

    def test_latent_shape(self, vae: VAE1d, conv_input: torch.Tensor) -> None:
        """mu and logvar should have shape (batch, latent_size)."""
        _, mu, logvar = vae(conv_input)
        assert mu.shape == logvar.shape
        assert mu.shape == (conv_input.shape[0], 4)  # latent_size=4
        assert mu.ndim == 2

    def test_is_dlkit_model(self, vae: VAE1d) -> None:
        """VAE1d should be an instance of DLKitModel."""
        assert isinstance(vae, DLKitModel)

    def test_not_cae(self, vae: VAE1d) -> None:
        """VAE1d should NOT be an instance of CAE."""
        assert not isinstance(vae, CAE)

    def test_no_lightning_methods(self, vae: VAE1d) -> None:
        """VAE1d should not have Lightning-specific methods."""
        assert not hasattr(vae, "predict_step")
        assert not hasattr(vae, "loss_function")

    def test_encode_returns_mu_logvar(self, vae: VAE1d, conv_input: torch.Tensor) -> None:
        """encode() should return (mu, logvar)."""
        mu, logvar = vae.encode(conv_input)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(logvar, torch.Tensor)
        assert mu.shape == logvar.shape

    def test_decode_works(self, vae: VAE1d, batch_size: int) -> None:
        """decode() should work with latent (mu, logvar) pair."""
        mu = torch.randn(batch_size, 4)  # latent_size=4
        logvar = torch.randn(batch_size, 4)
        recon, mu_out, logvar_out = vae.decode(mu, logvar)
        assert recon.shape == (batch_size, 2, 16)  # in_channels, in_length

    def test_has_parameters(self, vae: VAE1d) -> None:
        """VAE1d should have trainable parameters."""
        assert len(list(vae.parameters())) > 0

    def test_stores_loss_weights(self, cae_kwargs: dict) -> None:
        """VAE1d should store alpha and beta weights."""
        vae_with_weights = VAE1d(**cae_kwargs, alpha=0.5, beta=0.2)
        assert vae_with_weights.alpha == 0.5
        assert vae_with_weights.beta == 0.2


class TestVAELoss:
    """Tests for VAE loss function."""

    def test_returns_scalar(self, vae: VAE1d, conv_input: torch.Tensor) -> None:
        """vae_loss should return a scalar tensor."""
        recon, mu, logvar = vae(conv_input)
        loss = vae_loss(recon, conv_input, mu, logvar)
        assert loss.ndim == 0
        assert isinstance(loss, torch.Tensor)

    def test_alpha_zero_ignores_reconstruction(self, vae: VAE1d, conv_input: torch.Tensor) -> None:
        """With alpha=0, reconstruction loss should not affect total."""
        recon, mu, logvar = vae(conv_input)
        loss_with_recon = vae_loss(recon, conv_input, mu, logvar, alpha=0.0, beta=1.0)
        loss_bad_recon = vae_loss(
            torch.zeros_like(recon), conv_input, mu, logvar, alpha=0.0, beta=1.0
        )
        assert torch.isclose(loss_with_recon, loss_bad_recon, atol=1e-5)

    def test_beta_zero_is_mse_only(self, vae: VAE1d, conv_input: torch.Tensor) -> None:
        """With beta=0, loss should equal MSE only."""
        recon, mu, logvar = vae(conv_input)
        loss = vae_loss(recon, conv_input, mu, logvar, alpha=1.0, beta=0.0)
        mse = nn.functional.mse_loss(recon, conv_input)
        assert torch.isclose(loss, mse, atol=1e-5)

    def test_loss_is_positive(self, vae: VAE1d, conv_input: torch.Tensor) -> None:
        """VAE loss should be positive."""
        recon, mu, logvar = vae(conv_input)
        loss = vae_loss(recon, conv_input, mu, logvar)
        assert loss.item() > 0

    def test_different_weights_change_loss(self, vae: VAE1d, conv_input: torch.Tensor) -> None:
        """Different alpha/beta weights should produce different losses."""
        recon, mu, logvar = vae(conv_input)
        loss1 = vae_loss(recon, conv_input, mu, logvar, alpha=1.0, beta=0.1)
        loss2 = vae_loss(recon, conv_input, mu, logvar, alpha=0.5, beta=0.1)
        assert not torch.isclose(loss1, loss2, atol=1e-5)


class TestReparameterize:
    """Tests for reparameterization trick."""

    def test_output_shape(self, batch_size: int) -> None:
        """Output should have same shape as mu."""
        mu = torch.zeros(batch_size, 4)
        logvar = torch.zeros(batch_size, 4)
        z = reparameterize(mu, logvar)
        assert z.shape == (batch_size, 4)

    def test_zero_variance_returns_mean(self, batch_size: int) -> None:
        """With very small variance, output should be close to mean."""
        mu = torch.ones(batch_size, 4) * 5.0
        logvar = torch.full((batch_size, 4), -30.0)  # near-zero std
        z = reparameterize(mu, logvar)
        assert torch.allclose(z, mu, atol=1e-2)

    def test_nonzero_variance_differs_from_mean(self) -> None:
        """With nonzero variance, some samples should differ from mean."""
        batch_size = 100
        mu = torch.zeros(batch_size, 4)
        logvar = torch.zeros(batch_size, 4)  # std=1
        z = reparameterize(mu, logvar)
        # With std=1, mean should be close to 0 but some samples differ
        assert not torch.allclose(z, mu, atol=1e-2)

    def test_is_differentiable(self) -> None:
        """Reparameterization should be differentiable."""
        mu = torch.randn(3, 4, requires_grad=True)
        logvar = torch.randn(3, 4, requires_grad=True)
        z = reparameterize(mu, logvar)
        loss = z.sum()
        loss.backward()
        assert mu.grad is not None
        assert logvar.grad is not None

    def test_batch_independence(self) -> None:
        """Different samples in batch should have different noise."""
        mu = torch.zeros(10, 4)
        logvar = torch.zeros(10, 4)
        z = reparameterize(mu, logvar)
        # Not all samples should be identical (with high probability)
        assert not torch.allclose(z[0], z[1], atol=1e-5)
