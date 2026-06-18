# Convolutional Autoencoder (CAE) Module

## Overview
The CAE module provides convolutional autoencoder architectures for 1D temporal data, including standard autoencoders, skip-connection variants, and variational autoencoders (VAE). These models compress high-dimensional temporal sequences into compact latent representations and reconstruct them.

## Architecture & Design Patterns
- **Abstract Base Class Pattern**: `CAE` defines the encode/decode interface that all autoencoders implement
- **Template Method Pattern**: Base `CAE.forward()` orchestrates encode→decode flow
- **Composition over Inheritance**: Models compose encoder/decoder modules from the encoder package
- **Entry-Based Construction**: All models implement `from_entries(input_shapes, output_shapes, **kwargs)` via the `EntryConsumer` protocol
- **Reparameterization Trick**: VAE uses standard Gaussian reparameterization for differentiable sampling

Key architectural decisions:
- All CAEs require explicit shape kwargs (`in_channels`, `in_length`) or dataset entry shapes via `from_entries`
- Latent representation always bottlenecks through vector space
- Skip connections in encoder/decoder for better gradient flow
- Flexible activation, normalization, and dropout options
- VAE includes a standalone `vae_loss()` pure function combining reconstruction and KL divergence

## Module Structure

### Public API
| Name | Type | Purpose |
|------|------|---------|
| `CAE` | Abstract Class | Base class for all autoencoders |
| `SkipCAE1d` | Class | 1D convolutional autoencoder with skip connections |
| `LinearCAE1d` | Class | Linear autoencoder (identity activation, subclass of `SkipCAE1d`) |
| `VAE1d` | Class | Variational autoencoder |
| `reparameterize` | Function | Reparameterization trick (μ, logσ² → z) |
| `vae_loss` | Function | VAE loss: α·MSE + β·KL |

### Protocols/Interfaces
| Name | Methods | Purpose |
|------|---------|---------|
| `CAE` (abstract) | `encode()`, `decode()`, `forward()` | Autoencoder interface |

## Dependencies

### Internal Dependencies
- `dlkit.common.sources`: `InputShapes`, `OutputShapes` for entry-based construction
- `dlkit.domain.nn.encoder.skip`: `SkipEncoder1d`, `SkipDecoder1d` for feature extraction
- `dlkit.domain.nn.encoder.latent`: `VectorToTensorBlock`, `TensorToVectorBlock` for latent space conversion
- `dlkit.domain.nn.types`: `NormalizerName` for normalization type hints
- `dlkit.domain.nn.utils`: `build_channel_schedule` for layer width scheduling

### External Dependencies
- `torch`: PyTorch tensor operations and neural network modules
- `torch.distributions.normal`: Normal distribution for VAE reparameterization

## Key Components

### Component 1: `CAE` (Abstract Base Class)

**Purpose**: Abstract base class defining the autoencoder interface. Provides `forward()` as encode→decode composition.

**Abstract Methods**:
- `encode(x: Tensor) -> Tensor` - Encode input to latent space (must implement)
- `decode(x: Tensor) -> Tensor` - Decode latent representation (must implement)

**Concrete Methods**:
- `forward(x: Tensor) -> Tensor` - Full autoencoder pass (encode→decode)

---

### Component 2: `SkipCAE1d`

**Purpose**: 1D convolutional autoencoder with skip connections throughout encoder/decoder for improved gradient flow and feature preservation.

**Constructor Parameters** (keyword-only):
- `in_channels: int` - Number of input channels
- `in_length: int` - Length of input sequence
- `latent_channels: int` - Number of channels in bottleneck feature map
- `latent_size: int` - Dimension of compressed latent vector
- `latent_width: int` - Temporal width of bottleneck feature map (default: 1)
- `num_layers: int` - Number of encoder/decoder layers (default: 3)
- `kernel_size: int` - Convolution kernel size (default: 3)
- `activation: Callable` - Activation function (default: `nn.functional.gelu`)
- `normalize: NormalizerName | None` - Normalization type: "batch", "layer", "instance", or None
- `dropout: float` - Dropout probability (default: 0.0)
- `transpose: bool` - Transpose before latent compression (default: False)
- `dilation: int` - Convolution dilation rate (default: 1)

**Class Method**:
- `from_entries(input_shapes: InputShapes, output_shapes: OutputShapes, **kwargs) -> Self` — extracts `in_channels` from the first input shape's channel dim and `in_length` from its second dim (defaults to 1)

**Example**:
```python
from dlkit.domain.nn.cae import SkipCAE1d
import torch

# Direct construction
cae = SkipCAE1d(in_channels=32, in_length=100, latent_channels=8, latent_size=64, num_layers=3)

# Entry-based construction (used by the build factory)
input_shapes = {"x": (32, 100)}
output_shapes = {"y": (32, 100)}
cae = SkipCAE1d.from_entries(input_shapes, output_shapes, latent_channels=8, latent_size=64, num_layers=3)

x = torch.randn(16, 32, 100)  # (batch, channels, timesteps)
latent = cae.encode(x)        # Shape: (16, 64)
reconstructed = cae.decode(latent)  # Shape: (16, 32, 100)
output = cae(x)               # Equivalent to decode(encode(x))
```

**Implementation Notes**:
- Uses `build_channel_schedule` to compute gradual channel/width reduction across layers
- Encoder: `SkipEncoder1d` + `TensorToVectorBlock` for latent compression
- Decoder: `VectorToTensorBlock` + `SkipDecoder1d` for reconstruction

---

### Component 3: `LinearCAE1d`

**Purpose**: Simplified autoencoder subclass that uses an identity activation (no non-linearity). Equivalent to `SkipCAE1d(activation=lambda x: x)`.

**Constructor Parameters** (keyword-only):
- `in_channels: int` - Number of input channels
- `in_length: int` - Length of input sequence
- `latent_channels: int` - Bottleneck channels (default: 5)
- `latent_width: int` - Bottleneck width (default: 10)
- `latent_size: int` - Latent vector dimension (default: 10)
- `num_layers: int` - Number of layers (default: 3)
- `kernel_size: int` - Kernel size (default: 3)

**Example**:
```python
from dlkit.domain.nn.cae import LinearCAE1d
import torch

cae = LinearCAE1d(in_channels=32, in_length=100, latent_size=64)
x = torch.randn(16, 32, 100)
output = cae(x)
```

---

### Component 4: `VAE1d`

**Purpose**: Variational autoencoder implementing the reparameterization trick. Learns a probabilistic latent space with Gaussian distribution.

**Constructor Parameters** (keyword-only):
- `in_channels: int` - Number of input channels
- `in_length: int` - Input sequence length
- `latent_channels: int` - Bottleneck feature map channels
- `latent_size: int` - Latent vector dimension
- `latent_width: int` - Bottleneck feature map width (default: 1)
- `num_layers: int` - Number of encoder/decoder layers (default: 3)
- `kernel_size: int` - Convolution kernel size (default: 3)
- `activation: Callable` - Activation function (default: `gelu`)
- `normalize: NormalizerName | None` - Normalization type
- `dropout: float` - Dropout probability (default: 0.0)
- `scale_of_latent: int` - Multiplier for intermediate latent dimension (default: 4)

**Class Method**:
- `from_entries(input_shapes: InputShapes, output_shapes: OutputShapes, **kwargs) -> Self`

**Forward Methods**:
- `encode(x: Tensor) -> tuple[Tensor, Tensor]` - Returns (mu, logvar)
- `decode(mu: Tensor, logvar: Tensor) -> tuple[Tensor, Tensor, Tensor]` - Returns (reconstruction, mu, logvar)
- `forward(x: Tensor) -> tuple[Tensor, Tensor, Tensor]` - Full VAE pass

**Example**:
```python
from dlkit.domain.nn.cae import VAE1d
from dlkit.domain.nn.cae.vae import vae_loss
import torch

vae = VAE1d(in_channels=32, in_length=100, latent_channels=8, latent_size=64, num_layers=3)
x = torch.randn(16, 32, 100)

recon, mu, logvar = vae(x)
loss = vae_loss(recon, x, mu, logvar, alpha=1.0, beta=0.1)
```

---

### Component 5: `reparameterize` / `vae_loss`

**`reparameterize(mu, logvar) -> Tensor`**: Samples z = μ + σ·ε where ε ~ N(0,1).

**`vae_loss(predictions, targets, mu, logvar, *, alpha, beta) -> Tensor`**: Returns α·MSE + β·KL(q||N(0,1)).

## Testing

- Unit tests: `tests/domain/nn/cae/` or `tests/core/models/nn/test_cae.py`
- Integration coverage via `tests/integration/`

## Related Modules
- `dlkit.common.sources`: `InputShapes`, `OutputShapes`
- `dlkit.domain.nn.encoder`: Encoder/decoder building blocks
- `dlkit.domain.nn.primitives`: Convolution and dense blocks
- `dlkit.engine.adapters.lightning`: Lightning wrappers for training CAEs
