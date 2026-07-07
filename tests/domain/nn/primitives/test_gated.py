from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from dlkit.domain.nn.primitives.gated import (
    GatedConvolutionBlock1d,
    GatedDeconvolutionBlock1d,
    GLUGate,
    GRNGate,
    IGatingMechanism,
    SwiGLUGate,
    UVGate,
)


class TestGLUGate:
    """Tests for GLUGate."""

    def test_output_shape(self, hidden_input: Tensor, original_input: Tensor) -> None:
        """GLUGate output shape must match (batch, hidden_size)."""
        gate = GLUGate(8)
        out = gate(hidden_input, original_input)
        assert out.shape == hidden_input.shape

    def test_x_not_used(self, hidden_input: Tensor, original_input: Tensor) -> None:
        """GLUGate ignores x: same h with two different x tensors yields identical output."""
        gate = GLUGate(8)
        with torch.no_grad():
            out1 = gate(hidden_input, original_input)
            out2 = gate(hidden_input, -original_input)
        assert torch.allclose(out1, out2)

    def test_gradient_flows(self, hidden_input: Tensor, original_input: Tensor) -> None:
        """Gradients must flow back to hidden_input through GLUGate."""
        hidden_input.requires_grad_(True)
        gate = GLUGate(8)
        out = gate(hidden_input, original_input)
        out.sum().backward()
        assert hidden_input.grad is not None

    def test_isinstance_protocol(self, hidden_input: Tensor, original_input: Tensor) -> None:
        """GLUGate must satisfy the IGatingMechanism protocol."""
        gate = GLUGate(8)
        assert isinstance(gate, IGatingMechanism)


class TestSwiGLUGate:
    """Tests for SwiGLUGate."""

    def test_output_shape(self, hidden_input: Tensor, original_input: Tensor) -> None:
        """SwiGLUGate output shape must match (batch, hidden_size)."""
        gate = SwiGLUGate(8)
        out = gate(hidden_input, original_input)
        assert out.shape == hidden_input.shape

    def test_x_not_used(self, hidden_input: Tensor, original_input: Tensor) -> None:
        """SwiGLUGate ignores x: same h with two different x tensors yields identical output."""
        gate = SwiGLUGate(8)
        with torch.no_grad():
            out1 = gate(hidden_input, original_input)
            out2 = gate(hidden_input, -original_input)
        assert torch.allclose(out1, out2)

    def test_gradient_flows(self, hidden_input: Tensor, original_input: Tensor) -> None:
        """Gradients must flow back to hidden_input through SwiGLUGate."""
        hidden_input.requires_grad_(True)
        gate = SwiGLUGate(8)
        out = gate(hidden_input, original_input)
        out.sum().backward()
        assert hidden_input.grad is not None

    def test_isinstance_protocol(self, hidden_input: Tensor, original_input: Tensor) -> None:
        """SwiGLUGate must satisfy the IGatingMechanism protocol."""
        gate = SwiGLUGate(8)
        assert isinstance(gate, IGatingMechanism)

    def test_bias_false_matches_shazeer_convention(self) -> None:
        """bias=False must drop bias on both projections (Shazeer 2020 convention)."""
        gate = SwiGLUGate(8, bias=False)
        assert gate.content_proj.bias is None
        assert gate.gate_proj.bias is None


class TestGRNGate:
    """Tests for GRNGate."""

    def test_output_shape_no_context(self, hidden_input: Tensor) -> None:
        """GRNGate(context_size=None) output shape must match (batch, hidden_size).

        When context_size is None, x must have the same width as h (hidden_size=8).
        """
        gate = GRNGate(8)
        out = gate(hidden_input, hidden_input)
        assert out.shape == hidden_input.shape

    def test_output_shape_with_context(self, hidden_input: Tensor, original_input: Tensor) -> None:
        """GRNGate(context_size=6) output shape must match (batch, hidden_size)."""
        gate = GRNGate(8, context_size=6)
        out = gate(hidden_input, original_input)
        assert out.shape == hidden_input.shape

    def test_layernorm_present(self) -> None:
        """GRNGate must have a LayerNorm attribute named 'norm'."""
        gate = GRNGate(8)
        assert hasattr(gate, "norm")
        assert isinstance(gate.norm, nn.LayerNorm)

    def test_gradient_flows(self, hidden_input: Tensor) -> None:
        """Gradients must flow back to hidden_input through GRNGate."""
        hidden_input.requires_grad_(True)
        gate = GRNGate(8)
        out = gate(hidden_input, hidden_input)
        out.sum().backward()
        assert hidden_input.grad is not None

    def test_isinstance_protocol(self, hidden_input: Tensor) -> None:
        """GRNGate must satisfy the IGatingMechanism protocol."""
        gate = GRNGate(8)
        assert isinstance(gate, IGatingMechanism)

    def test_content_and_gate_read_from_bottleneck_output(
        self, hidden_input: Tensor, original_input: Tensor
    ) -> None:
        """content_proj/gate_proj must read from bottleneck(eta2), not eta2 directly.

        Regression for the missing intermediate linear vs. Lim et al. 2020's
        published GRN (eta2 = ELU(...), eta1 = bottleneck(eta2), then both the
        content and gate branches read eta1, not eta2).
        """
        gate = GRNGate(8, context_size=6)
        with torch.no_grad():
            eta2 = torch.nn.functional.elu(
                gate.w2(hidden_input) + gate.context_proj(original_input)
            )
            eta1 = gate.bottleneck(eta2)
            expected_glu = gate.content_proj(eta1) * torch.sigmoid(gate.gate_proj(eta1))
            expected = gate.norm(hidden_input + expected_glu)
            actual = gate(hidden_input, original_input)
        torch.testing.assert_close(actual, expected)


class TestUVGate:
    """Tests for UVGate."""

    def test_output_shape(self, hidden_input: Tensor, original_input: Tensor) -> None:
        """UVGate output shape must match (batch, hidden_size)."""
        gate = UVGate(in_features=6, hidden_size=8)
        out = gate(hidden_input, original_input)
        assert out.shape == hidden_input.shape

    def test_x_modulates_output(self, hidden_input: Tensor, original_input: Tensor) -> None:
        """UVGate uses x: same h with two DIFFERENT x tensors yields different outputs."""
        gate = UVGate(in_features=6, hidden_size=8)
        with torch.no_grad():
            out1 = gate(hidden_input, original_input)
            out2 = gate(hidden_input, -original_input)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows_through_x(self, hidden_input: Tensor, original_input: Tensor) -> None:
        """Gradients must flow back to original_input (x) through UVGate."""
        original_input.requires_grad_(True)
        gate = UVGate(in_features=6, hidden_size=8)
        out = gate(hidden_input, original_input)
        out.sum().backward()
        assert original_input.grad is not None

    def test_isinstance_protocol(self, hidden_input: Tensor, original_input: Tensor) -> None:
        """UVGate must satisfy the IGatingMechanism protocol."""
        gate = UVGate(6, 8)
        assert isinstance(gate, IGatingMechanism)


class TestGatedConvolutionBlock1d:
    """Tests for GatedConvolutionBlock1d."""

    def test_output_shape(self, conv_gate_input: Tensor) -> None:
        """GatedConvolutionBlock1d output shape must be (batch, out_channels, T)."""
        out_channels = 4
        block = GatedConvolutionBlock1d(in_channels=8, out_channels=out_channels, in_timesteps=16)
        out = block(conv_gate_input)
        assert out.shape == (conv_gate_input.shape[0], out_channels, conv_gate_input.shape[2])

    def test_same_padding_preserves_length(self, conv_gate_input: Tensor) -> None:
        """padding='same' must preserve the temporal dimension length."""
        block = GatedConvolutionBlock1d(
            in_channels=8, out_channels=4, in_timesteps=16, padding="same"
        )
        out = block(conv_gate_input)
        assert out.shape[2] == conv_gate_input.shape[2]

    def test_layernorm_accepted(self, conv_gate_input: Tensor) -> None:
        """normalize='layer' must not raise during construction or forward pass."""
        out_channels = 4
        block = GatedConvolutionBlock1d(
            in_channels=8, out_channels=out_channels, in_timesteps=16, normalize="layer"
        )
        out = block(conv_gate_input)
        assert out.shape == (conv_gate_input.shape[0], out_channels, conv_gate_input.shape[2])

    def test_gradient_flows(self, conv_gate_input: Tensor) -> None:
        """Gradients must flow back to conv_gate_input through GatedConvolutionBlock1d."""
        conv_gate_input.requires_grad_(True)
        block = GatedConvolutionBlock1d(in_channels=8, out_channels=4, in_timesteps=16)
        out = block(conv_gate_input)
        out.sum().backward()
        assert conv_gate_input.grad is not None


class TestGatedDeconvolutionBlock1d:
    """Tests for GatedDeconvolutionBlock1d."""

    def test_output_shape(self, conv_gate_input: Tensor) -> None:
        """GatedDeconvolutionBlock1d output shape must be (batch, out_channels, T)."""
        out_channels = 4
        block = GatedDeconvolutionBlock1d(in_channels=8, out_channels=out_channels, in_timesteps=16)
        out = block(conv_gate_input)
        assert out.shape == (conv_gate_input.shape[0], out_channels, conv_gate_input.shape[2])

    def test_same_padding_stride_gt1_raises(self) -> None:
        """padding='same' with stride!=1 must raise ValueError."""
        with pytest.raises(ValueError):
            GatedDeconvolutionBlock1d(
                in_channels=8, out_channels=4, in_timesteps=16, padding="same", stride=2
            )

    def test_gradient_flows(self, conv_gate_input: Tensor) -> None:
        """Gradients must flow back to conv_gate_input through GatedDeconvolutionBlock1d."""
        conv_gate_input.requires_grad_(True)
        block = GatedDeconvolutionBlock1d(in_channels=8, out_channels=4, in_timesteps=16)
        out = block(conv_gate_input)
        out.sum().backward()
        assert conv_gate_input.grad is not None

    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_same_padding_preserves_length_for_odd_kernels(
        self, conv_gate_input: Tensor, kernel_size: int
    ) -> None:
        """Regression: odd kernel_size + stride=1 + padding='same' must preserve length."""
        block = GatedDeconvolutionBlock1d(
            in_channels=8, out_channels=4, in_timesteps=16, kernel_size=kernel_size
        )
        out = block(conv_gate_input)
        assert out.shape[2] == conv_gate_input.shape[2]

    @pytest.mark.parametrize("kernel_size", [2, 4, 6])
    def test_same_padding_even_kernel_raises(self, kernel_size: int) -> None:
        """Regression: even kernel_size + padding='same' silently broke length preservation."""
        with pytest.raises(ValueError, match='"same" padding'):
            GatedDeconvolutionBlock1d(
                in_channels=8, out_channels=4, in_timesteps=16, kernel_size=kernel_size
            )
