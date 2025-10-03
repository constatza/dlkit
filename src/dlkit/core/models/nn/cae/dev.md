# Convolutional Autoencoder (CAE) Module

## Overview
The CAE module provides convolutional autoencoder architectures for 1D temporal data, including standard autoencoders, skip-connection variants, and variational autoencoders (VAE). These models compress high-dimensional temporal sequences into compact latent representations and reconstruct them, useful for dimensionality reduction, denoising, and generative modeling.

## Architecture & Design Patterns
- **Abstract Base Class Pattern**: `CAE` defines the encode/decode interface that all autoencoders implement
- **Template Method Pattern**: Base `CAE.forward()` orchestrates encode→decode flow
- **Composition over Inheritance**: Models compose encoder/decoder modules from the encoder package
- **Shape-Aware Architecture**: Inherits from `ShapeAwareModel` for automatic shape validation and precision handling
- **Delegate Pattern**: `LinearCAE1d` delegates to `SkipCAE1d` for backward compatibility
- **Reparameterization Trick**: VAE uses standard Gaussian reparameterization for differentiable sampling

Key architectural decisions:
- All CAEs require shape specifications at initialization (no lazy initialization)
- Latent representation always bottlenecks through vector space
- Skip connections in encoder/decoder for better gradient flow
- Flexible activation, normalization, and dropout options
- VAE includes custom loss function combining reconstruction and KL divergence

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `CAE` | Abstract Class | Base class for all autoencoders | N/A |
| `SkipCAE1d` | Class | 1D convolutional autoencoder with skip connections | N/A |
| `LinearCAE1d` | Class | Simple linear autoencoder (delegates to SkipCAE1d) | N/A |
| `VAE1d` | Class | Variational autoencoder with KL divergence | N/A |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `reparameterize` | Function | VAE reparameterization trick (μ, logσ² → z) | `Tensor` |

### Protocols/Interfaces
| Name | Methods | Purpose |
|------|---------|---------|
| `CAE` (abstract) | `encode()`, `decode()`, `forward()`, `predict_step()` | Autoencoder interface |

## Dependencies

### Internal Dependencies
- `dlkit.core.models.nn.base`: `ShapeAwareModel` base class for shape handling
- `dlkit.core.shape_specs`: `IShapeSpec` protocol for shape specifications
- `dlkit.core.models.nn.encoder.skip`: `SkipEncoder1d`, `SkipDecoder1d` for feature extraction
- `dlkit.core.models.nn.encoder.latent`: `VectorToTensorBlock`, `TensorToVectorBlock` for latent space conversion
- `dlkit.core.datatypes.networks`: `NormalizerName` for normalization type hints

### External Dependencies
- `torch`: PyTorch tensor operations and neural network modules
- `torch.distributions.normal`: Normal distribution for VAE reparameterization
- `pydantic`: Runtime validation via `@validate_call` decorator

## Key Components

### Component 1: `CAE` (Abstract Base Class)

**Purpose**: Abstract base class defining the autoencoder interface with encode/decode abstraction. Provides common functionality for prediction steps and shape validation while enforcing implementation of encoding/decoding logic.

**Constructor Parameters**:
- `unified_shape: IShapeSpec` - Shape specification for input/output data (required)
- `**kwargs` - Additional parameters passed to `ShapeAwareModel`

**Abstract Methods**:
- `encode(*args, **kwargs) -> Any` - Encode input to latent space (must implement)
- `decode(*args, **kwargs) -> Any` - Decode latent representation (must implement)

**Concrete Methods**:
- `forward(x: Any) -> Any` - Full autoencoder pass (encode→decode)
- `predict_step(batch: Any, batch_idx: int) -> Dict[str, Tensor]` - Lightning prediction with latent output
- `accepts_shape(shape_spec: IShapeSpec) -> bool` - Validate shape compatibility

**Returns**:
- `forward()`: Reconstructed input
- `predict_step()`: Dict with keys "predictions" and "latent"

**Example**:
```python
from dlkit.core.models.nn.cae import SkipCAE1d
from dlkit.core.shape_specs import create_shape_spec

# Create shape spec for (channels=32, timesteps=100) input
shape_spec = create_shape_spec({"x": (32, 100)})

# Create autoencoder
cae = SkipCAE1d(
    shape_spec=shape_spec,
    latent_channels=8,
    latent_width=10,
    latent_size=64,
    num_layers=3
)

# Encode input to latent space
import torch
x = torch.randn(16, 32, 100)  # (batch, channels, timesteps)
latent = cae.encode(x)  # Shape: (16, 64)

# Decode latent back to original space
reconstructed = cae.decode(latent)  # Shape: (16, 32, 100)

# Full autoencoder pass
output = cae(x)  # Equivalent to cae.decode(cae.encode(x))
```

**Implementation Notes**:
- Shape validation happens automatically in `ShapeAwareModel.__init__`
- `predict_step` returns both reconstruction and latent for downstream analysis
- Handles tuple returns from encode/decode (needed for VAE)
- All tensors detached in prediction to save memory

---

### Component 2: `SkipCAE1d`

**Purpose**: 1D convolutional autoencoder with skip connections throughout encoder/decoder for improved gradient flow and feature preservation. The primary CAE implementation in DLKit.

**Constructor Parameters**:
- `shape_spec: IShapeSpec` - Required shape specification for input dimensions
- `latent_channels: int` - Number of channels in bottleneck feature map
- `latent_width: int` - Temporal width of bottleneck feature map (default: 1)
- `latent_size: int` - Dimension of compressed latent vector
- `num_layers: int` - Number of encoder/decoder layers (default: 3)
- `kernel_size: int` - Convolution kernel size (default: 3)
- `activation: Callable` - Activation function (default: `nn.functional.gelu`)
- `normalize: NormalizerName | None` - Normalization type: "batch", "layer", "instance", or None
- `dropout: float` - Dropout probability (default: 0.0)
- `transpose: bool` - Transpose before latent compression (default: False)
- `dilation: int` - Convolution dilation rate (default: 1)

**Returns**: N/A (constructor)

**Forward Methods**:
- `encode(x: Tensor) -> Tensor` - Encode to latent vector
- `decode(x: Tensor) -> Tensor` - Decode from latent vector

**Raises**:
- `ValueError`: If shape_spec is None
- `ValueError`: If input shape doesn't have at least 2 dimensions

**Example**:
```python
from dlkit.core.models.nn.cae import SkipCAE1d
from dlkit.core.shape_specs import create_shape_spec
import torch

# Create shape spec
shape_spec = create_shape_spec({"x": (64, 200)})  # 64 channels, 200 timesteps

# Create autoencoder with custom settings
cae = SkipCAE1d(
    shape_spec=shape_spec,
    latent_channels=16,
    latent_width=25,
    latent_size=128,
    num_layers=4,
    kernel_size=5,
    activation=torch.nn.functional.relu,
    normalize="batch",
    dropout=0.2
)

# Process data
x = torch.randn(32, 64, 200)
latent = cae.encode(x)  # Shape: (32, 128)
reconstructed = cae.decode(latent)  # Shape: (32, 64, 200)

# Full pass
output = cae(x)  # Shape: (32, 64, 200)
```

**Implementation Notes**:
- Uses `torch.linspace` to compute gradual channel/width reduction across layers
- Encoder: `SkipEncoder1d` + `TensorToVectorBlock` for latent compression
- Decoder: `VectorToTensorBlock` + `SkipDecoder1d` for reconstruction
- Skip connections in encoder/decoder prevent vanishing gradients
- Adaptive pooling in latent conversion handles variable input sizes
- All layers use same activation, normalization, and dropout settings

---

### Component 3: `LinearCAE1d`

**Purpose**: Simplified autoencoder interface that internally delegates to `SkipCAE1d` with linear activation. Provided for backward compatibility with older DLKit code.

**Constructor Parameters**:
- `shape_spec: IShapeSpec | None` - Shape specification (default: None)
- `latent_channels: int` - Bottleneck channels (default: 5)
- `latent_width: int` - Bottleneck width (default: 10)
- `latent_size: int` - Latent vector dimension (default: 10)
- `num_layers: int` - Number of layers (default: 3)
- `kernel_size: int` - Kernel size (default: 3)
- `**kwargs` - Additional arguments passed to base class

**Returns**: N/A (constructor)

**Example**:
```python
from dlkit.core.models.nn.cae import LinearCAE1d
from dlkit.core.shape_specs import create_shape_spec

shape_spec = create_shape_spec({"x": (32, 100)})
cae = LinearCAE1d(shape_spec=shape_spec, latent_size=64)

# Same interface as SkipCAE1d
import torch
x = torch.randn(16, 32, 100)
output = cae(x)
```

**Implementation Notes**:
- Composition pattern: wraps `SkipCAE1d` with identity activation
- All encode/decode calls delegated to internal `_impl` instance
- Deprecated architecture - prefer `SkipCAE1d` directly for new code
- Maintained for compatibility with existing configurations

---

### Component 4: `VAE1d`

**Purpose**: Variational autoencoder implementing the reparameterization trick for generative modeling. Learns a probabilistic latent space with Gaussian distribution, enabling sampling of new data.

**Constructor Parameters**:
- `shape_spec: IShapeSpec` - Required shape specification
- `latent_channels: int` - Bottleneck feature map channels
- `latent_width: int` - Bottleneck feature map width (default: 1)
- `latent_size: int` - Latent vector dimension
- `num_layers: int` - Number of encoder/decoder layers (default: 3)
- `kernel_size: int` - Convolution kernel size (default: 3)
- `activation: Callable` - Activation function (default: `gelu`)
- `normalize: NormalizerName | None` - Normalization type
- `dropout: float` - Dropout probability (default: 0.0)
- `scale_of_latent: int` - Multiplier for intermediate latent dimension (default: 4)
- `alpha: float` - Weight for reconstruction loss (default: 1.0)
- `beta: float` - Weight for KL divergence loss (default: 0.1)

**Returns**: N/A (constructor)

**Forward Methods**:
- `encode(x: Tensor) -> Tuple[Tensor, Tensor]` - Returns (mu, logvar)
- `decode(mu: Tensor, logvar: Tensor) -> Tuple[Tensor, Tensor, Tensor]` - Returns (reconstruction, mu, logvar)
- `forward(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]` - Full VAE pass
- `loss_function(predictions, targets, mu, logvar) -> Tensor` - Custom VAE loss (MSE + KL)
- `predict_step(batch, batch_idx) -> Dict[str, Tensor]` - Prediction with mean latent

**Raises**:
- `ValueError`: If shape_spec is None
- `ValueError`: If input shape doesn't have at least 2 dimensions

**Example**:
```python
from dlkit.core.models.nn.cae import VAE1d
from dlkit.core.shape_specs import create_shape_spec
import torch

# Create VAE
shape_spec = create_shape_spec({"x": (32, 100)})
vae = VAE1d(
    shape_spec=shape_spec,
    latent_channels=8,
    latent_width=10,
    latent_size=64,
    num_layers=3,
    alpha=1.0,  # Reconstruction weight
    beta=0.5    # KL divergence weight
)

# Encoding returns mu and logvar
x = torch.randn(16, 32, 100)
mu, logvar = vae.encode(x)  # Both shape: (16, 64)

# Decoding with reparameterization
reconstruction, mu, logvar = vae.decode(mu, logvar)  # reconstruction: (16, 32, 100)

# Full forward pass
recon, mu, logvar = vae(x)

# Compute VAE loss
loss = vae.loss_function(recon, x, mu, logvar)

# Generate new samples
with torch.no_grad():
    # Sample from standard normal
    z = torch.randn(16, 64)
    # Decode to generate new data
    generated, _, _ = vae.decode(z, torch.zeros_like(z))
```

**Implementation Notes**:
- Encoder outputs intermediate dimension `scale_of_latent * latent_size`
- Separate linear layers project to `mu` and `logvar`
- `reparameterize()` function implements z = μ + σ * ε where ε ~ N(0,1)
- Loss = α * MSE + β * KL(q(z|x) || N(0,1))
- KL divergence encourages latent to match standard normal
- During prediction, uses `mu` directly (no sampling) for consistency

---

### Component 5: `reparameterize` (Utility Function)

**Purpose**: Implements the reparameterization trick for variational autoencoders, enabling backpropagation through stochastic sampling.

**Parameters**:
- `mu: Tensor` - Mean of latent distribution
- `logvar: Tensor` - Log variance of latent distribution

**Returns**: `Tensor` - Sampled latent code z = μ + σ * ε

**Example**:
```python
from dlkit.core.models.nn.cae.vae import reparameterize
import torch

mu = torch.zeros(32, 64)  # Mean
logvar = torch.zeros(32, 64)  # Log variance (σ² = 1)

# Sample from N(mu, σ²)
z = reparameterize(mu, logvar)  # Shape: (32, 64)

# With non-zero parameters
mu = torch.ones(32, 64) * 2.0
logvar = torch.ones(32, 64) * 0.5  # σ² = exp(0.5) ≈ 1.65
z = reparameterize(mu, logvar)
```

**Implementation Notes**:
- Converts logvar to std: σ = exp(0.5 * logvar)
- Samples noise from standard normal: ε ~ N(0, 1)
- Applies reparameterization: z = μ + σ * ε
- Gradient flows through μ and logvar, not through random sampling

## Usage Patterns

### Common Use Case 1: Dimensionality Reduction with SkipCAE1d
```python
from dlkit.core.models.nn.cae import SkipCAE1d
from dlkit.core.shape_specs import create_shape_spec
import torch

# Reduce 128-dimensional time series to 32-dimensional latent
shape_spec = create_shape_spec({"x": (128, 200)})
autoencoder = SkipCAE1d(
    shape_spec=shape_spec,
    latent_channels=8,
    latent_width=1,
    latent_size=32,
    num_layers=4,
    normalize="batch",
    dropout=0.1
)

# Compress data
x = torch.randn(64, 128, 200)  # 64 samples
latent_codes = autoencoder.encode(x)  # Shape: (64, 32) - 4x compression

# Reconstruct from compressed representation
reconstructed = autoencoder.decode(latent_codes)  # Shape: (64, 128, 200)

# Measure reconstruction quality
mse = torch.nn.functional.mse_loss(reconstructed, x)
print(f"Reconstruction MSE: {mse.item():.4f}")
```

### Common Use Case 2: Denoising with CAE
```python
from dlkit.core.models.nn.cae import SkipCAE1d
from dlkit.core.shape_specs import create_shape_spec
import torch

# Train denoising autoencoder
shape_spec = create_shape_spec({"x": (64, 150)})
denoiser = SkipCAE1d(
    shape_spec=shape_spec,
    latent_channels=16,
    latent_width=5,
    latent_size=128,
    num_layers=3,
    activation=torch.nn.functional.gelu,
    dropout=0.2  # Regularization
)

# Add noise to clean data
clean_data = torch.randn(32, 64, 150)
noise = torch.randn_like(clean_data) * 0.1
noisy_data = clean_data + noise

# Denoise
denoised = denoiser(noisy_data)

# Training would minimize loss between denoised and clean_data
loss = torch.nn.functional.mse_loss(denoised, clean_data)
```

### Common Use Case 3: Generative Modeling with VAE
```python
from dlkit.core.models.nn.cae import VAE1d
from dlkit.core.shape_specs import create_shape_spec
import torch

# Create VAE for generative modeling
shape_spec = create_shape_spec({"x": (32, 100)})
vae = VAE1d(
    shape_spec=shape_spec,
    latent_channels=8,
    latent_width=10,
    latent_size=64,
    num_layers=3,
    alpha=1.0,  # Reconstruction importance
    beta=0.1    # Regularization strength
)

# Training loop
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
real_data = torch.randn(128, 32, 100)

for epoch in range(100):
    # Forward pass
    recon, mu, logvar = vae(real_data)

    # Compute VAE loss
    loss = vae.loss_function(recon, real_data, mu, logvar)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate new samples after training
vae.eval()
with torch.no_grad():
    # Sample from prior N(0, 1)
    z = torch.randn(16, 64)
    # Generate
    generated, _, _ = vae.decode(z, torch.zeros_like(z))
    # generated shape: (16, 32, 100)
```

### Common Use Case 4: Transfer Learning with Pretrained Encoder
```python
from dlkit.core.models.nn.cae import SkipCAE1d
from dlkit.core.shape_specs import create_shape_spec
import torch
import torch.nn as nn

# Load pretrained autoencoder
shape_spec = create_shape_spec({"x": (64, 200)})
pretrained_cae = SkipCAE1d(
    shape_spec=shape_spec,
    latent_channels=16,
    latent_width=10,
    latent_size=128,
    num_layers=4
)
# Assume pretrained_cae is trained on large unlabeled dataset
# pretrained_cae.load_state_dict(torch.load("pretrained_cae.pth"))

# Use encoder as feature extractor for downstream task
class DownstreamClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Add classification head
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # Extract features with pretrained encoder
        features = self.encoder.encode(x)
        # Classify
        logits = self.classifier(features)
        return logits

# Create classifier using pretrained encoder
classifier = DownstreamClassifier(pretrained_cae.encoder, num_classes=10)

# Train only the classification head
x = torch.randn(32, 64, 200)
labels = torch.randint(0, 10, (32,))
logits = classifier(x)
loss = nn.functional.cross_entropy(logits, labels)
```

## Error Handling

**Exceptions Raised**:
- `ValueError`: When shape_spec is None in SkipCAE1d or VAE1d
- `ValueError`: When input shape has fewer than 2 dimensions
- `ValueError`: When NormScaledFFNN base_model is None (wrong constructor usage)
- `TypeError`: When base_model is not an nn.Module
- `RuntimeError`: When tensor dimensions don't match expected shapes

**Error Handling Pattern**:
```python
from dlkit.core.models.nn.cae import SkipCAE1d
from dlkit.core.shape_specs import create_shape_spec
import torch

try:
    # Missing shape_spec - will fail
    cae = SkipCAE1d(
        shape_spec=None,
        latent_channels=8,
        latent_size=64
    )
except ValueError as e:
    print(f"Shape spec required: {e}")
    # Fix: provide shape spec
    shape_spec = create_shape_spec({"x": (32, 100)})
    cae = SkipCAE1d(shape_spec=shape_spec, latent_channels=8, latent_size=64)

try:
    # Invalid input dimensions - will fail
    shape_spec = create_shape_spec({"x": (10,)})  # Only 1D
    cae = SkipCAE1d(shape_spec=shape_spec, latent_channels=8, latent_size=64)
except ValueError as e:
    print(f"Insufficient dimensions: {e}")
    # Fix: provide at least 2D shape
    shape_spec = create_shape_spec({"x": (32, 100)})
    cae = SkipCAE1d(shape_spec=shape_spec, latent_channels=8, latent_size=64)

try:
    # Dimension mismatch at runtime
    x = torch.randn(16, 64, 100)  # Wrong number of channels
    output = cae(x)
except RuntimeError as e:
    print(f"Dimension mismatch: {e}")
    # Fix: match shape_spec
    x = torch.randn(16, 32, 100)
    output = cae(x)
```

## Testing

### Test Coverage
- Unit tests: `tests/core/models/nn/test_cae.py` (to be created)
- Integration tests: `tests/integration/test_cae_training.py` (to be created)

### Key Test Scenarios
1. **SkipCAE1d forward/backward**: Verify encode→decode produces correct shapes
2. **SkipCAE1d gradient flow**: Verify gradients propagate through skip connections
3. **VAE reparameterization**: Verify sampling is differentiable
4. **VAE loss function**: Verify KL divergence and reconstruction terms
5. **Shape validation**: Verify accepts_shape correctly rejects invalid specs
6. **Precision handling**: Verify model respects precision settings
7. **Normalization options**: Verify batch/layer/instance normalization work
8. **Dropout during training**: Verify dropout active in train mode, inactive in eval

### Fixtures Used
- `sample_shape_spec` (from `conftest.py`): Standard shape specifications
- `tmp_path` (pytest built-in): Temporary paths for model checkpoints
- Random seeds for reproducible initialization

## Performance Considerations
- Skip connections add minimal overhead but significantly improve gradient flow
- Batch normalization faster than layer normalization for large batches
- Adaptive pooling in latent conversion handles variable sequence lengths
- VAE's KL divergence computation is O(latent_size) per sample
- Dropout only active during training - no overhead at inference
- Consider reducing num_layers for faster training on small datasets
- Use smaller latent_size for faster encoding (but less expressive)
- Dilation can increase receptive field without adding parameters

## Future Improvements / TODOs
- [ ] Support for 2D and 3D convolutional autoencoders
- [ ] Conditional VAE variant (CVAE) with class labels
- [ ] Beta-VAE with adjustable β scheduling
- [ ] Disentangled VAE (β-TCVAE, FactorVAE)
- [ ] Vector Quantized VAE (VQ-VAE)
- [ ] Adversarial autoencoder (AAE) variant
- [ ] Denoising autoencoder (DAE) with explicit noise model
- [ ] Sparse autoencoder with L1 regularization
- [ ] Contractive autoencoder with Jacobian penalty
- [ ] Multi-scale autoencoder with multiple latent levels
- [ ] Attention mechanisms in encoder/decoder
- [ ] Checkpoint support for very deep autoencoders

## Related Modules
- `dlkit.core.models.nn.encoder`: Encoder/decoder building blocks used by CAEs
- `dlkit.core.models.nn.primitives`: Convolution and dense blocks
- `dlkit.core.models.nn.base`: `ShapeAwareModel` base class
- `dlkit.core.shape_specs`: Shape specification system
- `dlkit.core.models.wrappers`: Lightning wrappers for training CAEs

## Change Log
- **2025-10-03**: Initial documentation created
- **2024-XX-XX**: Migrated to unified shape specification system
- **2024-XX-XX**: Added VAE1d implementation with custom loss
- **2024-XX-XX**: Deprecated LinearCAE1d in favor of SkipCAE1d
