# Encoder Module

## Overview
The encoder module provides building blocks for progressive downsampling (encoding) and upsampling (decoding) of temporal sequences, along with latent space conversion utilities. These components are the fundamental building blocks used by CAE, VAE, and other sequence-to-latent architectures in DLKit.

## Architecture & Design Patterns
- **Encoder-Decoder Symmetry**: Encoder and decoder use matching architectures with reversed parameters
- **Skip Connections**: Every layer includes residual connections for gradient flow
- **Progressive Transformation**: Gradual channel/timestep changes across layers (no abrupt jumps)
- **Latent Space Bridges**: Conversion modules bridge tensor feature maps and vector latent codes
- **Adaptive Pooling**: Handles variable-length sequences via adaptive average pooling
- **Composable Blocks**: Small, focused modules that compose into larger architectures

Key architectural decisions:
- Encoder reduces spatial/temporal dimensions while increasing channel complexity
- Decoder mirrors encoder structure but in reverse
- Latent conversion uses adaptive pooling + dense layers for flexibility
- All layers support configurable activation, normalization, and dropout
- Skip connections prevent vanishing gradients in deep networks

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `SkipEncoder1d` | Class | Progressive downsampling with skip connections | N/A |
| `SkipDecoder1d` | Class | Progressive upsampling with skip connections | N/A |
| `VectorToTensorBlock` | Class | Converts latent vector to 2D feature map | N/A |
| `TensorToVectorBlock` | Class | Converts 2D feature map to latent vector | N/A |

### Internal Components
None - all components are public API building blocks.

### Protocols/Interfaces
None - standard nn.Module interface.

## Dependencies

### Internal Dependencies
- `dlkit.domain.nn.primitives.dense`: `DenseBlock` for fully connected layers
- `dlkit.domain.nn.primitives.convolutional`: `ConvolutionBlock1d` for conv layers
- `dlkit.domain.nn.primitives.skip`: `SkipConnection` for residual connections
- `dlkit.runtime.data.graph.types`: `NormalizerName` type hints

### External Dependencies
- `torch`: PyTorch tensor operations and neural network modules
- `torch.nn`: Neural network building blocks
- `torch.nn.functional`: Functional API (GELU, interpolation)

## Key Components

### Component 1: `SkipEncoder1d`

**Purpose**: Progressive encoder that reduces temporal resolution while increasing channel depth using skip connections. Each layer combines convolution with skip/residual connection, followed by adaptive interpolation to target timesteps.

**Constructor Parameters**:
- `channels: Sequence[int]` - List of channel sizes for each layer (length = num_layers + 1)
- `timesteps: Sequence[int]` - List of temporal resolutions for each layer (length = num_layers + 1)
- `kernel_size: int = 3` - Convolution kernel size
- `activation: Callable = nn.functional.gelu` - Activation function
- `normalize: NormalizerName | None = None` - Normalization type ("batch", "layer", "instance", None)
- `reduce: Callable = nn.functional.interpolate` - Downsampling function (default: interpolate)
- `dilation: int = 1` - Convolution dilation (increases per layer: i+1)
- `dropout: float = 0.0` - Dropout probability

**Returns**: N/A (constructor)

**Forward Pass**:
- Input: `x: Tensor` - Shape (batch, channels[0], timesteps[0])
- Output: `Tensor` - Shape (batch, channels[-1], timesteps[-1])

**Example**:
```python
from dlkit.domain.nn.encoder import SkipEncoder1d
import torch

# Define progressive channel/timestep reduction
channels = [32, 64, 128, 256]  # 3 layers
timesteps = [100, 50, 25, 10]  # Progressively reduce temporal resolution

encoder = SkipEncoder1d(
    channels=channels,
    timesteps=timesteps,
    kernel_size=5,
    activation=torch.nn.functional.relu,
    normalize="batch",
    dropout=0.1,
)

# Encode temporal sequence
x = torch.randn(16, 32, 100)  # (batch, channels, timesteps)
encoded = encoder(x)  # Shape: (16, 256, 10)
```

**Implementation Notes**:
- Each layer: `SkipConnection(ConvolutionBlock1d(...))` + `reduce()`
- Skip connection handles channel mismatch with 1x1 convolution
- Dilation increases per layer (i+1) for larger receptive fields
- Interpolation reduces timesteps to target resolution
- Supports "same" padding to maintain spatial dimensions within layer
- All layers share same kernel_size, activation, normalize, dropout

---

### Component 2: `SkipDecoder1d`

**Purpose**: Progressive decoder that reconstructs temporal sequences from compressed representations. Mirrors encoder architecture but operates in reverse (upsampling). Inherits from `SkipEncoder1d` and adds a final regression layer.

**Constructor Parameters**:
- `channels: Sequence[int]` - List of channel sizes (reversed from encoder)
- `timesteps: Sequence[int]` - List of temporal resolutions (reversed from encoder)
- `kernel_size: int = 3` - Convolution kernel size
- `activation: Callable = nn.functional.gelu` - Activation function
- `normalize: NormalizerName | None = None` - Normalization type
- `reduce: Callable = nn.functional.interpolate` - Upsampling function (interpolate)
- `dilation: int = 1` - Convolution dilation
- `dropout: float = 0.0` - Dropout probability

**Returns**: N/A (constructor)

**Forward Pass**:
- Input: `x: Tensor` - Shape (batch, channels[0], timesteps[0])
- Output: `Tensor` - Shape (batch, channels[-1], timesteps[-1])

**Example**:
```python
from dlkit.domain.nn.encoder import SkipEncoder1d, SkipDecoder1d
import torch

# Symmetric encoder-decoder
encoder_channels = [32, 64, 128, 256]
encoder_timesteps = [100, 50, 25, 10]

encoder = SkipEncoder1d(channels=encoder_channels, timesteps=encoder_timesteps, kernel_size=5)

# Decoder reverses encoder structure
decoder = SkipDecoder1d(
    channels=encoder_channels[::-1],  # Reverse: [256, 128, 64, 32]
    timesteps=encoder_timesteps[::-1],  # Reverse: [10, 25, 50, 100]
    kernel_size=5,
)

# Encode-decode cycle
x = torch.randn(16, 32, 100)
latent = encoder(x)  # Shape: (16, 256, 10)
reconstructed = decoder(latent)  # Shape: (16, 32, 100)

# Check reconstruction quality
mse = torch.nn.functional.mse_loss(reconstructed, x)
```

**Implementation Notes**:
- Inherits all encoder functionality via `super().__init__`
- Adds final `nn.Conv1d` regression layer for output projection
- Regression layer uses same kernel_size and dilation as other layers
- Forward pass: `super().forward(x)` + `regression_layer(x)`
- Typically used with reversed channel/timestep sequences from encoder

---

### Component 3: `VectorToTensorBlock`

**Purpose**: Converts a flat latent vector into a 2D feature map (channels × timesteps) for decoder input. Uses dense layer followed by reshaping to target dimensions.

**Constructor Parameters**:
- `latent_dim: int` - Dimension of input latent vector
- `target_shape: tuple` - Target shape as (channels, timesteps) for output feature map

**Returns**: N/A (constructor)

**Forward Pass**:
- Input: `x: Tensor` - Shape (batch, latent_dim)
- Output: `Tensor` - Shape (batch, channels, timesteps) where (channels, timesteps) = target_shape

**Example**:
```python
from dlkit.domain.nn.encoder import VectorToTensorBlock
import torch

# Convert 128-d vector to (64, 10) feature map
vector_to_tensor = VectorToTensorBlock(
    latent_dim=128,
    target_shape=(64, 10),  # 64 channels, 10 timesteps
)

# Convert latent vectors
latent = torch.randn(32, 128)  # 32 samples, 128-d each
feature_map = vector_to_tensor(latent)  # Shape: (32, 64, 10)

# Ready to pass to decoder
print(feature_map.shape)  # torch.Size([32, 64, 10])
```

**Implementation Notes**:
- Uses `DenseBlock` to project latent_dim → (channels * timesteps)
- Reshapes using `.view(batch_size, channels, timesteps)`
- No activation in dense block (identity) - activation handled by decoder
- Learned transformation from compact vector to structured feature map

---

### Component 4: `TensorToVectorBlock`

**Purpose**: Converts a 2D feature map (channels × timesteps) into a flat latent vector using adaptive pooling and dense projection. Used in encoder to create bottleneck representation.

**Constructor Parameters**:
- `channels_in: int` - Number of input channels in feature map
- `latent_dim: int` - Dimension of output latent vector
- `transpose: bool = False` - Whether to transpose channels/timesteps before pooling

**Returns**: N/A (constructor)

**Forward Pass**:
- Input: `x: Tensor` - Shape (batch, channels, timesteps)
- Output: `Tensor` - Shape (batch, latent_dim)

**Example**:
```python
from dlkit.domain.nn.encoder import TensorToVectorBlock
import torch

# Convert (128, 25) feature map to 64-d vector
tensor_to_vector = TensorToVectorBlock(channels_in=128, latent_dim=64)

# Compress feature maps
feature_map = torch.randn(32, 128, 25)  # (batch, channels, timesteps)
latent = tensor_to_vector(feature_map)  # Shape: (32, 64)

# With transpose (pool over channels instead of timesteps)
tensor_to_vector_t = TensorToVectorBlock(channels_in=128, latent_dim=64, transpose=True)
latent_t = tensor_to_vector_t(feature_map)  # Shape: (32, 64)
```

**Implementation Notes**:
- Optional transpose: swaps channels and timesteps before pooling
- `AdaptiveAvgPool1d(1)` reduces temporal dimension to single value
- Flattens to (batch, channels_in)
- `DenseBlock` projects to latent_dim with identity activation
- GELU activation applied before pooling (hardcoded)
- Handles variable-length sequences via adaptive pooling

## Usage Patterns

### Common Use Case 1: Symmetric Encoder-Decoder Autoencoder
```python
from dlkit.domain.nn.encoder import (
    SkipEncoder1d,
    SkipDecoder1d,
    VectorToTensorBlock,
    TensorToVectorBlock,
)
import torch
import torch.nn as nn


class SymmetricAutoencoder(nn.Module):
    def __init__(self, input_channels=32, input_timesteps=100, latent_dim=64):
        super().__init__()

        # Define progressive compression
        channels = [input_channels, 64, 128, 256]
        timesteps = [input_timesteps, 50, 25, 10]

        # Encoder: compress spatial/temporal dimensions
        self.encoder = SkipEncoder1d(
            channels=channels, timesteps=timesteps, kernel_size=5, normalize="batch"
        )

        # Latent compression
        self.to_latent = TensorToVectorBlock(channels_in=channels[-1], latent_dim=latent_dim)

        # Latent decompression
        self.from_latent = VectorToTensorBlock(
            latent_dim=latent_dim, target_shape=(channels[-1], timesteps[-1])
        )

        # Decoder: reconstruct original dimensions
        self.decoder = SkipDecoder1d(
            channels=channels[::-1], timesteps=timesteps[::-1], kernel_size=5, normalize="batch"
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.to_latent(x)

    def decode(self, z):
        x = self.from_latent(z)
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))


# Usage
autoencoder = SymmetricAutoencoder()
x = torch.randn(16, 32, 100)
reconstructed = autoencoder(x)  # Shape: (16, 32, 100)
```

### Common Use Case 2: Multi-Scale Feature Extraction
```python
from dlkit.domain.nn.encoder import SkipEncoder1d
import torch
import torch.nn as nn


class MultiScaleEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Three encoders at different scales
        self.encoder_fine = SkipEncoder1d(
            channels=[32, 64, 128], timesteps=[100, 50, 25], kernel_size=3
        )

        self.encoder_medium = SkipEncoder1d(
            channels=[32, 64, 128], timesteps=[50, 25, 10], kernel_size=5
        )

        self.encoder_coarse = SkipEncoder1d(
            channels=[32, 64, 128], timesteps=[25, 10, 5], kernel_size=7
        )

    def forward(self, x):
        # Extract features at multiple temporal scales
        features_fine = self.encoder_fine(x)
        features_medium = self.encoder_medium(nn.functional.interpolate(x, size=50))
        features_coarse = self.encoder_coarse(nn.functional.interpolate(x, size=25))

        return {"fine": features_fine, "medium": features_medium, "coarse": features_coarse}


# Usage
encoder = MultiScaleEncoder()
x = torch.randn(8, 32, 100)
features = encoder(x)
# features["fine"]: (8, 128, 25)
# features["medium"]: (8, 128, 10)
# features["coarse"]: (8, 128, 5)
```

### Common Use Case 3: Custom Encoder with Attention
```python
from dlkit.domain.nn.encoder import SkipEncoder1d, TensorToVectorBlock
from dlkit.domain.nn.attention import SelfAttentionBlock
import torch
import torch.nn as nn


class AttentionEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        channels = [64, 128, 256, 512]
        timesteps = [200, 100, 50, 25]

        # Initial encoder
        self.encoder = SkipEncoder1d(
            channels=channels, timesteps=timesteps, normalize="layer", dropout=0.1
        )

        # Attention over encoded features
        self.attention = SelfAttentionBlock(embed_dim=channels[-1], num_heads=8)

        # Final latent compression
        self.to_latent = TensorToVectorBlock(channels_in=channels[-1], latent_dim=latent_dim)

    def forward(self, x):
        # Encode with skip connections
        x = self.encoder(x)  # (batch, 512, 25)

        # Apply self-attention
        x = self.attention(x)  # (batch, 512, 25)

        # Compress to latent vector
        z = self.to_latent(x)  # (batch, 128)
        return z


# Usage
encoder = AttentionEncoder(latent_dim=128)
x = torch.randn(32, 64, 200)
latent = encoder(x)  # Shape: (32, 128)
```

### Common Use Case 4: Asymmetric Encoder-Decoder
```python
from dlkit.domain.nn.encoder import (
    SkipEncoder1d,
    SkipDecoder1d,
    TensorToVectorBlock,
    VectorToTensorBlock,
)
import torch
import torch.nn as nn


class AsymmetricAutoencoder(nn.Module):
    """Encoder has more layers than decoder for aggressive compression."""

    def __init__(self):
        super().__init__()

        # Deep encoder (5 layers)
        encoder_channels = [32, 64, 128, 256, 512, 512]
        encoder_timesteps = [100, 50, 25, 12, 6, 3]

        self.encoder = SkipEncoder1d(
            channels=encoder_channels, timesteps=encoder_timesteps, kernel_size=5, dropout=0.1
        )

        self.to_latent = TensorToVectorBlock(512, latent_dim=64)

        # Shallow decoder (3 layers) - faster reconstruction
        decoder_channels = [512, 128, 32]
        decoder_timesteps = [3, 25, 100]

        self.from_latent = VectorToTensorBlock(64, target_shape=(512, 3))

        self.decoder = SkipDecoder1d(
            channels=decoder_channels, timesteps=decoder_timesteps, kernel_size=5
        )

    def forward(self, x):
        z = self.to_latent(self.encoder(x))
        return self.decoder(self.from_latent(z))


# Usage
ae = AsymmetricAutoencoder()
x = torch.randn(16, 32, 100)
output = ae(x)  # Shape: (16, 32, 100)
```

## Error Handling

**Exceptions Raised**:
- `RuntimeError`: If input dimensions don't match expected channel count
- `RuntimeError`: If interpolation fails due to invalid target size
- `ValueError`: If channels/timesteps sequences have different lengths
- `IndexError`: If channels/timesteps sequences are empty

**Error Handling Pattern**:
```python
from dlkit.domain.nn.encoder import SkipEncoder1d, TensorToVectorBlock
import torch

try:
    # Mismatched sequence lengths
    encoder = SkipEncoder1d(
        channels=[32, 64, 128],  # Length 3
        timesteps=[100, 50],  # Length 2 - ERROR!
    )
except (ValueError, IndexError) as e:
    print(f"Sequence length mismatch: {e}")
    # Fix: match lengths
    encoder = SkipEncoder1d(channels=[32, 64, 128], timesteps=[100, 50, 25])

try:
    # Wrong input channels
    x = torch.randn(16, 64, 100)  # 64 channels
    output = encoder(x)  # Expects 32 channels!
except RuntimeError as e:
    print(f"Channel mismatch: {e}")
    # Fix: match first element of channels
    x = torch.randn(16, 32, 100)
    output = encoder(x)

try:
    # Latent dimension mismatch
    tensor_to_vec = TensorToVectorBlock(channels_in=256, latent_dim=64)
    x = torch.randn(16, 128, 25)  # Wrong channels (128 != 256)
    latent = tensor_to_vec(x)
except RuntimeError as e:
    print(f"Dimension mismatch: {e}")
    # Fix: match channels_in
    x = torch.randn(16, 256, 25)
    latent = tensor_to_vec(x)
```

## Testing

### Test Coverage
- Unit tests: `tests/core/models/nn/test_encoder.py`
- Integration tests: Used extensively in CAE/VAE tests

### Key Test Scenarios
1. **Encoder forward pass**: Verify progressive downsampling
2. **Decoder forward pass**: Verify progressive upsampling
3. **Encode-decode symmetry**: Verify output shape matches input shape
4. **Skip connection gradients**: Verify gradient flow through residual paths
5. **Latent conversion**: Verify vector↔tensor conversions preserve information
6. **Variable sequence lengths**: Verify adaptive pooling handles different timesteps
7. **Normalization modes**: Verify batch/layer/instance norm produce different outputs
8. **Dropout behavior**: Verify dropout active in train, inactive in eval

### Fixtures Used
- Standard shape configurations for common encoder/decoder pairs
- Random seeds for reproducible initialization
- Gradient checking fixtures

## Performance Considerations
- Skip connections add ~10% parameter overhead but prevent vanishing gradients
- Adaptive pooling slightly slower than fixed pooling but handles variable lengths
- Batch normalization fastest for large batches; layer norm better for small batches
- Interpolation (upsampling/downsampling) is fast on GPUs
- Dilation increases receptive field without adding parameters
- Dense layers in latent conversion can be bottleneck for large feature maps
- Consider gradient checkpointing for very deep encoders (>10 layers)

## Future Improvements / TODOs
- [ ] Support for 2D and 3D encoders/decoders
- [ ] Strided convolution option instead of interpolation
- [ ] Transposed convolution option for decoder
- [ ] Attention-based skip connections
- [ ] Dynamic channel/timestep scheduling (not fixed linspace)
- [ ] Group convolutions for efficiency
- [ ] Depthwise separable convolutions
- [ ] Squeeze-and-excitation blocks in skip connections
- [ ] Learnable interpolation (instead of fixed nn.functional.interpolate)
- [ ] Support for irregular sampling (non-uniform timesteps)
- [ ] Positional encodings for temporal awareness
- [ ] Multi-head latent conversion (split channels)

## Related Modules
- `dlkit.domain.nn.cae`: Uses encoder/decoder blocks for autoencoders
- `dlkit.domain.nn.primitives`: Provides ConvolutionBlock1d, DenseBlock, SkipConnection
- `dlkit.domain.nn.attention`: Can be combined with encoders for attention-based models
- `dlkit.domain.nn.base`: Shape-aware model foundation

## Change Log
- **2025-10-03**: Initial documentation created
- **2024-XX-XX**: Added transpose option to TensorToVectorBlock
- **2024-XX-XX**: Made dilation increase per layer in SkipEncoder1d
- **2024-XX-XX**: Added regression layer to SkipDecoder1d for better output quality
