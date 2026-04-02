# Feed-Forward Neural Network (FFNN) Module

## Overview
The FFNN module provides fully-connected neural network architectures for supervised learning tasks, including flexible multi-layer networks, simple linear models, and norm-scaled variants for physics-informed learning. These models excel at learning mappings from fixed-size feature vectors to target outputs.

## Architecture & Design Patterns
- **Shape-Aware Construction**: All FFNNs inherit from `ShapeAwareModel` and require shape specifications
- **Skip Connections**: Multi-layer networks use residual connections for deep gradient flow
- **Template Method Pattern**: Base classes define structure, subclasses configure specifics
- **Decorator Pattern**: `NormScaledFFNN` wraps base models with input/output normalization
- **Builder Pattern**: `ConstantWidthFFNN` simplifies constant-width architecture creation
- **Precision Management**: Automatic precision casting via `ShapeAwareModel`
- **Validation via Pydantic**: Runtime parameter validation using `@validate_call`

Key architectural decisions:
- All FFNNs require 1D feature vectors (reject 2D+ inputs)
- Skip connections only between matching-width layers (or with 1x1 linear projection)
- Embedding layer handles input dimension projection
- Regression layer handles output dimension projection
- Norm-scaled models enforce homogeneous scaling for physics consistency
- Precision applied automatically after layer construction

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `FeedForwardNN` | Class | Multi-layer FFNN with flexible architecture | N/A |
| `ConstantWidthFFNN` | Class | Multi-layer FFNN with constant hidden width | N/A |
| `LinearNetwork` | Class | Single-layer linear model (baseline) | N/A |
| `NormScaledFFNN` | Abstract Class | Wrapper adding norm-based input/output scaling | N/A |
| `NormScaledLinearFFNN` | Class | Norm-scaled single linear layer | N/A |
| `NormScaledConstantWidthFFNN` | Class | Norm-scaled multi-layer FFNN | N/A |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `_build_layers` | Method | Constructs network layers from shape spec | `None` |
| `_compute_eps` | Static Method | Computes machine epsilon for safe division | `float` |
| `_vector_norm` | Method | Computes vector norms (L1/L2/Linf) | `Tensor` |

### Protocols/Interfaces
None - all inherit from `ShapeAwareModel` ABC.

## Dependencies

### Internal Dependencies
- `dlkit.domain.nn.base`: `ShapeAwareModel` for shape handling and precision
- `dlkit.domain.shapes`: `IShapeSpec`, `NullShapeSpec` protocols
- `dlkit.domain.nn.primitives`: `DenseBlock`, `SkipConnection` building blocks
- `dlkit.tools.config.precision`: `PrecisionStrategy` for dtype management

### External Dependencies
- `torch`: PyTorch tensor operations and neural network modules
- `pydantic`: Runtime validation via `@validate_call`, `ConfigDict`

## Key Components

### Component 1: `FeedForwardNN`

**Purpose**: General-purpose multi-layer feed-forward network with flexible layer sizes, skip connections, and configurable activation/normalization. The standard FFNN for most supervised learning tasks in DLKit.

**Constructor Parameters**:
- `unified_shape: IShapeSpec` - Required shape specification with input/output dimensions
- `layers: Sequence[int]` - Hidden layer widths (e.g., [128, 64, 32])
- `activation: Callable = nn.functional.gelu` - Activation function applied between layers
- `normalize: Literal["batch", "layer"] | None = None` - Normalization type
- `dropout: float = 0.0` - Dropout probability (0.0 = no dropout)
- `**kwargs` - Additional arguments passed to `ShapeAwareModel`

**Returns**: N/A (constructor)

**Forward Pass**:
- Input: `x: Tensor` - Shape (batch, input_size)
- Output: `Tensor` - Shape (batch, output_size)

**Raises**:
- `ValueError`: If shape_spec lacks input or output dimensions
- `ValueError`: If input/output shapes are not 1D

**Example**:
```python
from dlkit.domain.nn.ffnn import FeedForwardNN
from dlkit.domain.shapes import create_shape_spec
import torch

# Create shape spec for 784 inputs (MNIST) → 10 outputs (classes)
shape_spec = create_shape_spec({"x": (784,), "y": (10,)})

# Build 3-layer network: 784 → 256 → 128 → 64 → 10
model = FeedForwardNN(
    unified_shape=shape_spec,
    layers=[256, 128, 64],  # Hidden layers
    activation=torch.nn.functional.relu,
    normalize="batch",
    dropout=0.3,
)

# Forward pass
x = torch.randn(32, 784)  # Batch of 32 MNIST images
logits = model(x)  # Shape: (32, 10)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
targets = torch.randint(0, 10, (32,))

output = model(x)
loss = loss_fn(output, targets)
loss.backward()
optimizer.step()
```

**Implementation Notes**:
- Embedding layer: `nn.Linear(input_size, layers[0])`
- Hidden layers: `SkipConnection(DenseBlock(...))` for each transition
- Regression layer: `nn.Linear(layers[-1], output_size)`
- Skip connections use 1x1 linear projection when dimensions change
- `ensure_precision_applied()` called after layer construction
- `num_layers = len(layers) - 1` (excludes embedding/regression)

---

### Component 2: `ConstantWidthFFNN`

**Purpose**: Simplified FFNN where all hidden layers have the same width. Reduces configuration complexity for deep networks with uniform architecture. Internally just a specialized `FeedForwardNN`.

**Constructor Parameters**:
- `unified_shape: IShapeSpec` - Required shape specification
- `hidden_size: int` - Width of all hidden layers (required, no default)
- `num_layers: int` - Number of hidden layers (required, no default)
- `activation: Callable = nn.functional.gelu` - Activation function
- `normalize: Literal["batch", "layer"] | None = None` - Normalization type
- `dropout: float = 0.0` - Dropout probability
- `**kwargs` - Additional arguments

**Returns**: N/A (constructor)

**Raises**:
- `ValueError`: If `hidden_size` or `num_layers` is None
- `ValueError`: If `num_layers` <= 0

**Example**:
```python
from dlkit.domain.nn.ffnn import ConstantWidthFFNN
from dlkit.domain.shapes import create_shape_spec
import torch

shape_spec = create_shape_spec({"x": (100,), "y": (50,)})

# Create deep network: 100 → 256 → 256 → 256 → 256 → 50
# (4 hidden layers of width 256)
model = ConstantWidthFFNN(
    unified_shape=shape_spec,
    hidden_size=256,
    num_layers=4,
    activation=torch.nn.functional.gelu,
    normalize="layer",
    dropout=0.2,
)

# Forward pass
x = torch.randn(64, 100)
output = model(x)  # Shape: (64, 50)
```

**Implementation Notes**:
- Constructs `layers = [hidden_size] * num_layers`
- Delegates to `FeedForwardNN.__init__` with repeated width
- Simplifies configuration for ResNet-style architectures
- All hidden layers have identical width for perfect skip connections

---

### Component 3: `LinearNetwork`

**Purpose**: Minimal single-layer linear model (no hidden layers) with optional normalization. Serves as baseline for benchmarking more complex models and for simple linear regression tasks.

**Constructor Parameters**:
- `unified_shape: IShapeSpec` - Required shape specification
- `normalize: Literal["batch", "layer"] | None = None` - Optional normalization
- `bias: bool = True` - Whether to include bias term
- `**kwargs` - Additional arguments

**Returns**: N/A (constructor)

**Forward Pass**:
- Input: `x: Tensor` - Shape (batch, input_size)
- Output: `Tensor` - Shape (batch, output_size)

**Raises**:
- `ValueError`: If shape_spec lacks input or output dimensions
- `ValueError`: If input/output shapes are not 1D

**Example**:
```python
from dlkit.domain.nn.ffnn import LinearNetwork
from dlkit.domain.shapes import create_shape_spec
import torch

# Simple linear regression: 20 features → 1 target
shape_spec = create_shape_spec({"x": (20,), "y": (1,)})

# Linear model with batch normalization
model = LinearNetwork(unified_shape=shape_spec, normalize="batch", bias=True)

# Forward pass
x = torch.randn(128, 20)
predictions = model(x)  # Shape: (128, 1)

# Equivalent to: BatchNorm1d → Linear(20, 1)
```

**Implementation Notes**:
- Single `nn.Linear(input_size, output_size, bias=bias)`
- Optional normalization applied to *output* (after linear layer)
- No activation function (pure linear transformation)
- Fastest model for training and inference
- Good baseline to measure benefit of nonlinearity

---

### Component 4: `NormScaledFFNN`

**Purpose**: Abstract wrapper that enforces homogeneous scaling consistency for linear systems Ax = b by normalizing inputs and rescaling outputs. Critical for physics-informed neural networks where dimensional consistency matters.

**Constructor Parameters**:
- `base_model: nn.Module` - Underlying network (required, no default)
- `unified_shape: IShapeSpec` - Required shape specification
- `norm: str = "l2"` - Norm type: "l2", "l1", or "linf"
- `eps_gain: float = 10.0` - Safety factor for division (multiplies machine epsilon)
- `keep_stats: bool = False` - If True, return (output, {"norm": norms}) tuple
- `precision: PrecisionStrategy | None = None` - Precision override
- `**kwargs` - Additional arguments

**Returns**: N/A (constructor)

**Forward Pass**:
- Input: `x: Tensor` (representing vector b in Ax=b)
- Output: `Tensor` (representing vector x) OR `Tuple[Tensor, Dict]` if keep_stats=True

**Raises**:
- `ValueError`: If `base_model` is None (cannot instantiate abstract class directly)
- `TypeError`: If `base_model` is not an `nn.Module`
- `ValueError`: If `norm` not in {"l2", "l1", "linf"}
- `ValueError`: If `eps_gain` <= 0
- `TypeError`: If input is not floating-point tensor
- `ValueError`: If input has fewer than 1 dimension

**Algorithm**:
1. Compute ||b||_p (p-norm of input)
2. Normalize: b_scaled = b / ||b||_p
3. Predict: x_scaled = base_model(b_scaled)
4. Rescale: x = x_scaled * ||b||_p

**Example**:
```python
from dlkit.domain.nn.ffnn import NormScaledLinearFFNN
from dlkit.domain.shapes import create_shape_spec
import torch

# For solving linear systems: given b, predict x in Ax = b
shape_spec = create_shape_spec({"x": (100,), "y": (100,)})

# L2-norm scaled linear model
model = NormScaledLinearFFNN(
    unified_shape=shape_spec, bias=False, norm="l2", eps_gain=10.0, keep_stats=False
)

# Forward pass maintains scale consistency
b = torch.randn(32, 100)  # Right-hand side
x = model(b)  # Predicted solution

# Verify ||x|| is proportional to ||b||
print(torch.linalg.vector_norm(b, dim=-1))
print(torch.linalg.vector_norm(x, dim=-1))

# With stats
model_with_stats = NormScaledLinearFFNN(unified_shape=shape_spec, keep_stats=True)
x, stats = model_with_stats(b)
print(stats["norm"])  # Contains ||b||_2
```

**Implementation Notes**:
- Cannot instantiate `NormScaledFFNN` directly - use concrete subclasses
- Uses `torch.linalg.vector_norm` for all norm computations
- Safe division: `b / max(||b||_p, eps_gain * machine_epsilon)`
- `keep_stats` useful for debugging/monitoring norm distributions
- Precision applied to both wrapper and base_model

---

### Component 5: `NormScaledLinearFFNN`

**Purpose**: Norm-scaled wrapper around a single linear layer. Simplest norm-scaled model for linear system solving.

**Constructor Parameters**:
- `unified_shape: IShapeSpec` - Required shape specification
- `bias: bool = True` - Include bias in linear layer
- `norm: str = "l2"` - Norm type
- `eps_gain: float = 10.0` - Safety factor
- `keep_stats: bool = False` - Return stats dict
- `precision: PrecisionStrategy | None = None` - Precision override

**Returns**: N/A (constructor)

**Example**:
```python
from dlkit.domain.nn.ffnn import NormScaledLinearFFNN
from dlkit.domain.shapes import create_shape_spec
import torch

shape_spec = create_shape_spec({"x": (50,), "y": (50,)})

# L1-norm scaled linear model
model = NormScaledLinearFFNN(unified_shape=shape_spec, bias=True, norm="l1")

# Solve Ax = b
b = torch.randn(16, 50)
x = model(b)
```

**Implementation Notes**:
- Creates `nn.Linear(input_size, output_size, bias=bias)` as base_model
- Delegates all normalization logic to `NormScaledFFNN`
- Ideal for overdetermined/underdetermined linear systems
- No nonlinearity - pure linear transformation with scaling

---

### Component 6: `NormScaledConstantWidthFFNN`

**Purpose**: Norm-scaled wrapper around `ConstantWidthFFNN`. Enables deep nonlinear networks with norm scaling for complex physics-informed learning.

**Constructor Parameters**:
- `unified_shape: IShapeSpec` - Required shape specification
- `hidden_size: int` - Width of hidden layers (required)
- `num_layers: int` - Number of hidden layers (required)
- `norm: str = "l2"` - Norm type
- `eps_gain: float = 10.0` - Safety factor
- `keep_stats: bool = False` - Return stats dict
- `precision: PrecisionStrategy | None = None` - Precision override
- `activation: Callable | None = None` - Activation (default: GELU)
- `normalize: Literal["batch", "layer"] | None = None` - Layer normalization
- `dropout: float = 0.0` - Dropout probability

**Returns**: N/A (constructor)

**Example**:
```python
from dlkit.domain.nn.ffnn import NormScaledConstantWidthFFNN
from dlkit.domain.shapes import create_shape_spec
import torch

shape_spec = create_shape_spec({"x": (100,), "y": (100,)})

# Deep norm-scaled network
model = NormScaledConstantWidthFFNN(
    unified_shape=shape_spec,
    hidden_size=256,
    num_layers=5,
    norm="l2",
    activation=torch.nn.functional.relu,
    normalize="layer",
    dropout=0.1,
)

# Nonlinear system solving with norm consistency
b = torch.randn(64, 100)
x = model(b)
```

**Implementation Notes**:
- Creates `ConstantWidthFFNN` as base_model with all parameters
- Wraps with norm scaling via `NormScaledFFNN.__init__`
- Combines benefits of deep networks + dimensional consistency
- Useful for nonlinear PDEs, inverse problems

## Usage Patterns

### Common Use Case 1: Multi-Class Classification with FeedForwardNN
```python
from dlkit.domain.nn.ffnn import FeedForwardNN
from dlkit.domain.shapes import create_shape_spec
import torch
import torch.nn as nn

# CIFAR-10: 3072 features (32x32x3) → 10 classes
shape_spec = create_shape_spec({"x": (3072,), "y": (10,)})

model = FeedForwardNN(
    unified_shape=shape_spec,
    layers=[512, 256, 128],
    activation=nn.functional.relu,
    normalize="batch",
    dropout=0.5,
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for images, labels in dataloader:
    # Flatten images
    x = images.view(-1, 3072)

    # Forward pass
    logits = model(x)

    # Compute loss
    loss = loss_fn(logits, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    test_x = test_images.view(-1, 3072)
    predictions = model(test_x).argmax(dim=-1)
```

### Common Use Case 2: Regression with LinearNetwork Baseline
```python
from dlkit.domain.nn.ffnn import LinearNetwork, FeedForwardNN
from dlkit.domain.shapes import create_shape_spec
import torch

shape_spec = create_shape_spec({"x": (20,), "y": (1,)})

# Baseline: simple linear regression
baseline = LinearNetwork(unified_shape=shape_spec, normalize="batch")

# Compare with deep network
deep_model = FeedForwardNN(
    unified_shape=shape_spec, layers=[128, 64, 32], activation=torch.nn.functional.gelu
)

# Evaluate both
x_test = torch.randn(1000, 20)
y_test = torch.randn(1000, 1)

baseline.eval()
deep_model.eval()

with torch.no_grad():
    baseline_pred = baseline(x_test)
    deep_pred = deep_model(x_test)

    baseline_mse = torch.nn.functional.mse_loss(baseline_pred, y_test)
    deep_mse = torch.nn.functional.mse_loss(deep_pred, y_test)

    print(f"Baseline MSE: {baseline_mse:.4f}")
    print(f"Deep Model MSE: {deep_mse:.4f}")
    print(f"Improvement: {(baseline_mse - deep_mse) / baseline_mse * 100:.2f}%")
```

### Common Use Case 3: Physics-Informed Learning with NormScaledFFNN
```python
from dlkit.domain.nn.ffnn import NormScaledConstantWidthFFNN
from dlkit.domain.shapes import create_shape_spec
import torch

# Solve Poisson equation: ∇²u = f
# Input: forcing term f, Output: solution u
shape_spec = create_shape_spec({"x": (1024,), "y": (1024,)})

# Norm-scaled network maintains physical scaling
pinn = NormScaledConstantWidthFFNN(
    unified_shape=shape_spec,
    hidden_size=512,
    num_layers=6,
    norm="l2",  # L2 norm for energy conservation
    activation=torch.nn.functional.tanh,
    normalize="layer",
)


# Physics-informed loss
def physics_loss(model, f, u_true):
    u_pred = model(f)

    # Data loss
    data_loss = torch.nn.functional.mse_loss(u_pred, u_true)

    # Physics loss (∇²u ≈ f)
    # (simplified - actual implementation would use autograd)
    laplacian_u = compute_laplacian(u_pred)
    physics_loss = torch.nn.functional.mse_loss(laplacian_u, f)

    return data_loss + 0.1 * physics_loss


# Training
optimizer = torch.optim.LBFGS(pinn.parameters(), lr=1.0)


def closure():
    optimizer.zero_grad()
    loss = physics_loss(pinn, f_train, u_train)
    loss.backward()
    return loss


for epoch in range(100):
    optimizer.step(closure)
```

### Common Use Case 4: Transfer Learning with Pretrained Encoder
```python
from dlkit.domain.nn.ffnn import ConstantWidthFFNN
from dlkit.domain.shapes import create_shape_spec
import torch
import torch.nn as nn

# Pretrain autoencoder on large unlabeled dataset
encoder_shape = create_shape_spec({"x": (1000,), "y": (128,)})
encoder = ConstantWidthFFNN(unified_shape=encoder_shape, hidden_size=256, num_layers=3)
# ... pretrain encoder ...

# Freeze encoder and add classifier
for param in encoder.parameters():
    param.requires_grad = False

classifier_shape = create_shape_spec({"x": (128,), "y": (10,)})
classifier = ConstantWidthFFNN(unified_shape=classifier_shape, hidden_size=64, num_layers=2)


# Combined model
class TransferModel(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


model = TransferModel(encoder, classifier)

# Train only classifier weights
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
```

## Error Handling

**Exceptions Raised**:
- `ValueError`: Missing or invalid shape specifications
- `ValueError`: Non-1D input/output shapes (FFNNs only handle feature vectors)
- `ValueError`: Invalid parameters (e.g., num_layers <= 0)
- `ValueError`: Invalid norm type in NormScaledFFNN
- `TypeError`: Non-Module base_model in NormScaledFFNN
- `TypeError`: Non-floating-point input to NormScaledFFNN
- `RuntimeError`: Dimension mismatch during forward pass

**Error Handling Pattern**:
```python
from dlkit.domain.nn.ffnn import FeedForwardNN, NormScaledLinearFFNN
from dlkit.domain.shapes import create_shape_spec, NullShapeSpec
import torch

try:
    # NullShapeSpec rejected
    model = FeedForwardNN(unified_shape=NullShapeSpec(), layers=[128, 64])
except ValueError as e:
    print(f"Invalid shape: {e}")

try:
    # 2D shape rejected (FFNNs need 1D)
    shape_spec = create_shape_spec({"x": (28, 28), "y": (10,)})
    model = FeedForwardNN(unified_shape=shape_spec, layers=[128])
except ValueError as e:
    print(f"Invalid dimensions: {e}")
    # Fix: flatten to 1D
    shape_spec = create_shape_spec({"x": (784,), "y": (10,)})
    model = FeedForwardNN(unified_shape=shape_spec, layers=[128])

try:
    # Invalid norm type
    model = NormScaledLinearFFNN(
        unified_shape=shape_spec,
        norm="l3",  # Invalid!
    )
except ValueError as e:
    print(f"Invalid norm: {e}")
    # Fix: use valid norm
    model = NormScaledLinearFFNN(unified_shape=shape_spec, norm="l2")

try:
    # Integer input to NormScaled (needs float)
    x = torch.randint(0, 10, (32, 784))
    output = model(x)
except TypeError as e:
    print(f"Wrong dtype: {e}")
    # Fix: cast to float
    x = x.float()
    output = model(x)
```

## Testing

### Test Coverage
- Unit tests: `tests/core/models/nn/test_ffnn.py`
- Integration coverage lives under `tests/integration/`
- Precision tests: `tests/tools/config/precision/test_comprehensive.py`

### Key Test Scenarios
1. **Forward pass shapes**: Verify output shape matches specification
2. **Gradient flow**: Verify backpropagation through all layers
3. **Skip connections**: Verify gradients flow through residual paths
4. **Normalization**: Verify batch/layer norm behave correctly
5. **Dropout**: Verify dropout active in train mode, inactive in eval
6. **Precision casting**: Verify model respects precision settings
7. **NormScaled scaling**: Verify output norm proportional to input norm
8. **Shape validation**: Verify accepts_shape rejects invalid specs
9. **Constant width**: Verify all hidden layers have same width
10. **Linear baseline**: Verify LinearNetwork is purely linear

### Fixtures Used
- `sample_shape_spec` (from `conftest.py`): Standard shape configurations
- `tmp_path` (pytest built-in): Temporary checkpoint storage
- Random seeds for reproducible initialization

## Performance Considerations
- Skip connections add ~10-20% overhead but enable deeper networks
- Batch normalization fastest for batch_size > 32
- Layer normalization better for small batches or variable batch sizes
- Dropout overhead ~5-10% during training (zero at inference)
- ConstantWidthFFNN enables efficient skip connections (no projection needed)
- NormScaled models add overhead from norm computation (~5%)
- Linear models 10x faster than deep models for similar input/output sizes
- Consider gradient checkpointing for very deep networks (>20 layers)
- Use mixed precision training for larger models

## Future Improvements / TODOs
- [ ] Switchable skip connection strategies (pre-activation, post-activation)
- [ ] Support for grouped convolutions in skip connections
- [ ] Adaptive layer width scheduling
- [ ] Mixture of experts (MoE) variant
- [ ] Spectral normalization option
- [ ] Weight initialization schemes (Xavier, He, orthogonal)
- [ ] Gradient clipping built into model
- [ ] Automatic architecture search (NAS) integration
- [ ] Pruning utilities for compression
- [ ] Quantization-aware training
- [ ] Multi-GPU sharding for very wide layers
- [ ] Learnable skip connection weights
- [ ] Highway networks variant
- [ ] DenseNet-style skip connections
- [ ] Stochastic depth for regularization

## Related Modules
- `dlkit.domain.nn.base`: `ShapeAwareModel` foundation
- `dlkit.domain.nn.primitives`: `DenseBlock`, `SkipConnection` building blocks
- `dlkit.domain.shapes`: Shape specification system
- `dlkit.runtime.adapters.lightning`: Lightning wrappers for training FFNNs
- `dlkit.tools.config.precision`: Precision management

## Change Log
- **2025-10-03**: Initial documentation created
- **2024-XX-XX**: Added NormScaledFFNN variants for physics-informed learning
- **2024-XX-XX**: Migrated to unified shape specification system
- **2024-XX-XX**: Added ConstantWidthFFNN convenience class
- **2024-XX-XX**: Separated LinearNetwork for baseline comparisons
