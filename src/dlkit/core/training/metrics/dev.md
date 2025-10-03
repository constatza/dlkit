# Training Metrics Module

## Overview
The training metrics module provides a comprehensive, composable metrics system following SOLID principles. It implements protocol-based interfaces, strategy patterns for aggregation and normalization, and factory patterns for easy metric creation. All metrics support both method and callable function interfaces for flexible usage in functional programming contexts.

## Architecture & Design Patterns
- **Protocol-Based Interfaces (DIP)**: `IMetric`, `IAggregator`, `INormalizer` enable dependency inversion
- **Template Method Pattern**: `BaseMetric` defines algorithm structure with customizable steps
- **Strategy Pattern**: Composable aggregation and normalization strategies
- **Factory Pattern**: `MetricFactory` for dependency injection and metric instantiation
- **Registry Pattern**: Thread-safe registries for metrics, aggregators, and normalizers
- **Decorator Pattern**: `MetricDecorator` adds functionality to existing metrics
- **Composite Pattern**: `CompositeMetric` combines multiple metrics
- **Callable Interface**: Metrics are callable like functions for functional programming

Key architectural decisions:
- Metrics decouple computation logic from aggregation/normalization strategies
- Protocol-based design enables testing and alternative implementations
- Registry system allows dynamic metric registration and discovery
- Template method ensures consistent computation flow across all metrics
- Callable interface supports functional programming patterns (map, filter, lambda)
- Thread-safe registries support concurrent metric creation
- Factory handles dependency injection following SOLID principles

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `create_metric` | Function | Create metric using global factory | `IMetric` |
| `create_normalized_vector_norm_error` | Function | Create specialized vector norm metric | `NormalizedVectorNormErrorMetric` |
| `create_composite_metric` | Function | Combine multiple metrics | `CompositeMetric` |
| `MeanSquaredErrorMetric` | Class | MSE metric implementation | N/A |
| `MeanAbsoluteErrorMetric` | Class | MAE metric implementation | N/A |
| `RootMeanSquaredErrorMetric` | Class | RMSE metric implementation | N/A |
| `NormalizedVectorNormErrorMetric` | Class | Normalized vector error metric | N/A |
| `MSEOverVarianceMetric` | Class | MSE normalized by variance | N/A |
| `TemporalDerivativeMetric` | Class | Temporal derivative error metric | N/A |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `BaseMetric` | Class | Abstract base with template method | N/A |
| `CompositeMetric` | Class | Weighted metric combination | N/A |
| `MetricDecorator` | Class | Abstract decorator for metrics | N/A |
| `MetricRegistry` | Class | Thread-safe metric registry | N/A |
| `AggregatorRegistry` | Class | Thread-safe aggregator registry | N/A |
| `NormalizerRegistry` | Class | Thread-safe normalizer registry | N/A |
| `MetricFactory` | Class | Factory with dependency injection | N/A |

### Protocols/Interfaces
| Name | Methods | Purpose |
|------|---------|---------|
| `IMetric` | `compute()`, `__call__()`, `name`, `metadata` | Core metric interface |
| `IAggregator` | `aggregate()`, `name` | Aggregation strategy interface |
| `INormalizer` | `normalize()`, `name` | Normalization strategy interface |
| `IMetricRegistry` | `register()`, `get()`, `list_metrics()` | Registry interface |
| `IMetricFactory` | `create_metric()` | Factory interface |

## Dependencies

### Internal Dependencies
- None (foundational module)

### External Dependencies
- `torch`: Tensor operations and mathematical functions
- `threading`: Thread-safe registry implementation

## Key Components

### Component 1: `IMetric` Protocol

**Purpose**: Core interface defining the contract for all metrics. Enables dependency inversion and type-safe metric composition.

**Methods**:
- `compute(predictions: Tensor, targets: Tensor, **kwargs) -> Tensor` - Compute metric value
- `__call__(predictions: Tensor, targets: Tensor, **kwargs) -> Tensor` - Callable interface
- `name -> str` - Metric name property
- `metadata -> dict[str, Any]` - Metric metadata (dimensions, parameters)

**Example**:
```python
from dlkit.core.training.metrics import IMetric, create_metric
import torch

# Type-safe metric usage
def evaluate_model(metric: IMetric, predictions: Tensor, targets: Tensor) -> float:
    """Evaluate model with any metric implementing IMetric."""
    error = metric.compute(predictions, targets)
    return float(error.item())

# Create metrics
mse = create_metric("mse")
mae = create_metric("mae")

predictions = torch.tensor([1.0, 2.0, 3.0])
targets = torch.tensor([1.1, 1.9, 3.1])

# Both methods work
mse_error = mse.compute(predictions, targets)  # Method syntax
mae_error = mae(predictions, targets)  # Callable syntax

# Functional programming
metrics = [mse, mae]
errors = [m(predictions, targets) for m in metrics]
```

**Implementation Notes**:
- Protocol-based for structural typing (no inheritance required)
- Callable interface enables use in map, filter, lambda
- Metadata property supports metric introspection and logging
- All implementations must provide both compute and __call__

---

### Component 2: `BaseMetric` Template Method

**Purpose**: Abstract base class implementing the Template Method Pattern. Defines the computation algorithm structure while allowing subclasses to customize specific steps.

**Constructor Parameters**:
- `name: str` - Metric name
- `aggregator: IAggregator | None = None` - Aggregation strategy (default: mean)
- `normalizer: INormalizer | None = None` - Normalization strategy (default: none)
- `**kwargs` - Additional metric-specific parameters

**Template Method Steps**:
1. `_validate_inputs()` - Validate input tensors
2. `_compute_raw_error()` - Compute raw error (subclass implements)
3. `_should_normalize()` - Check if normalization needed
4. `_apply_normalization()` - Apply normalization strategy
5. `_apply_aggregation()` - Apply aggregation strategy
6. `_post_process()` - Post-process result (hook for subclasses)

**Example**:
```python
from dlkit.core.training.metrics.base import BaseMetric
from torch import Tensor
import torch

class CustomMetric(BaseMetric):
    """Custom metric with specialized error computation."""

    def __init__(self, power: float = 2.0, **kwargs):
        super().__init__(
            name=f"custom_power_{power}",
            power=power,
            **kwargs
        )

    def _compute_raw_error(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute error raised to custom power."""
        power = self._params["power"]
        return torch.pow(torch.abs(predictions - targets), power)

# Usage
metric = CustomMetric(power=3.0)
predictions = torch.tensor([1.0, 2.0, 3.0])
targets = torch.tensor([1.1, 1.9, 3.1])

error = metric.compute(predictions, targets)
print(f"Custom error: {error:.4f}")
```

**Implementation Notes**:
- Template method in `compute()` enforces consistent flow
- Subclasses only override `_compute_raw_error()` for core logic
- Aggregation and normalization injected via constructor (DIP)
- Hook methods allow customization without breaking template
- Shape validation ensures predictions and targets match
- Parameters stored in `_params` dict for subclass access

---

### Component 3: Aggregation Strategies

**Purpose**: Strategy pattern implementations for different aggregation methods. Enable flexible composition of metric computation logic.

**Built-in Aggregators**:
- `MeanAggregator` - Arithmetic mean
- `SumAggregator` - Sum aggregation
- `VectorNormAggregator(ord)` - Vector norm (L1, L2, etc.)
- `StdAggregator` - Standard deviation

**Standard Instances**:
- `MEAN_AGGREGATOR` - Default mean aggregation
- `SUM_AGGREGATOR` - Sum aggregation
- `L2_AGGREGATOR` - L2 norm (Euclidean)
- `L1_AGGREGATOR` - L1 norm (Manhattan)
- `STD_AGGREGATOR` - Standard deviation

**Example**:
```python
from dlkit.core.training.metrics import create_metric
import torch

# MSE with different aggregations
mse_mean = create_metric("mse", aggregator="mean")
mse_sum = create_metric("mse", aggregator="sum")
mse_l2 = create_metric("mse", aggregator="l2_norm")

predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
targets = torch.tensor([[1.1, 1.9], [3.2, 3.8]])

# Different aggregation results
mean_error = mse_mean(predictions, targets)  # Mean of squared errors
sum_error = mse_sum(predictions, targets)    # Sum of squared errors
l2_error = mse_l2(predictions, targets)      # L2 norm of squared errors

print(f"Mean: {mean_error:.4f}")
print(f"Sum: {sum_error:.4f}")
print(f"L2 Norm: {l2_error:.4f}")
```

**Implementation Notes**:
- Each aggregator implements `IAggregator` protocol
- Support optional `dim` parameter for dimension-specific aggregation
- VectorNormAggregator parameterized by norm order
- Singleton instances prevent redundant object creation
- Thread-safe registration in `AggregatorRegistry`

---

### Component 4: Normalization Strategies

**Purpose**: Strategy pattern implementations for normalizing metric values by reference statistics. Enable relative error measurement.

**Built-in Normalizers**:
- `VarianceNormalizer` - Normalize by target variance
- `StandardDeviationNormalizer` - Normalize by target standard deviation
- `VectorNormNormalizer(ord, dim)` - Normalize by vector norm
- `NaiveForecastNormalizer` - Normalize by naive forecast error (time series)

**Standard Instances**:
- `VARIANCE_NORMALIZER` - Normalize by variance
- `STD_NORMALIZER` - Normalize by standard deviation
- `L2_NORM_NORMALIZER` - Normalize by L2 norm
- `L1_NORM_NORMALIZER` - Normalize by L1 norm
- `NAIVE_FORECAST_NORMALIZER` - Normalize by naive forecast

**Example**:
```python
from dlkit.core.training.metrics import create_metric
import torch

# MSE with normalization
mse_raw = create_metric("mse")
mse_normalized = create_metric("mse", normalizer="variance")
mse_std = create_metric("mse", normalizer="std")

predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
targets = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5])

# Compare raw and normalized errors
raw_error = mse_raw(predictions, targets)
normalized_error = mse_normalized(predictions, targets)
std_error = mse_std(predictions, targets)

print(f"Raw MSE: {raw_error:.4f}")
print(f"MSE/Variance: {normalized_error:.4f}")
print(f"MSE/Std: {std_error:.4f}")
```

**Implementation Notes**:
- Each normalizer implements `INormalizer` protocol
- `eps` parameter prevents division by zero
- VectorNormNormalizer supports custom dimensions
- NaiveForecastNormalizer specialized for time series
- Normalization applied before aggregation in template method
- Singleton instances for common normalizers

---

### Component 5: `MeanSquaredErrorMetric`

**Purpose**: Compute mean squared error between predictions and targets. Most common metric for regression tasks.

**Constructor Parameters**:
- `aggregator: IAggregator | None = None` - Aggregation strategy (default: mean)
- `normalizer: INormalizer | None = None` - Normalization strategy (default: none)
- `**kwargs` - Additional parameters (dim, eps)

**Returns**: `Tensor` - Scalar or tensor based on aggregation

**Example**:
```python
from dlkit.core.training.metrics import MeanSquaredErrorMetric, create_metric
import torch

# Direct instantiation
mse = MeanSquaredErrorMetric()

# Via factory (recommended)
mse = create_metric("mse")

# Sample data
predictions = torch.tensor([1.0, 2.0, 3.0])
targets = torch.tensor([1.1, 1.9, 3.1])

# Compute MSE
error = mse(predictions, targets)
print(f"MSE: {error:.4f}")

# With custom aggregation
mse_sum = create_metric("mse", aggregator="sum")
total_error = mse_sum(predictions, targets)
print(f"Total squared error: {total_error:.4f}")
```

**Implementation Notes**:
- Computes `(predictions - targets)^2`
- Default aggregation is mean (hence "Mean" Squared Error)
- Supports alternative aggregations via strategy pattern
- Can be normalized by variance for relative error

---

### Component 6: `NormalizedVectorNormErrorMetric`

**Purpose**: Compute normalized vector norm error for multi-dimensional data. Each vector's error is normalized by the target vector's magnitude, enabling relative error measurement.

**Constructor Parameters**:
- `vector_dim: int = -1` - Dimension along which vectors are defined
- `norm_ord: int = 2` - Order of norm (1=L1, 2=L2, inf=L-infinity)
- `aggregator: IAggregator | None = None` - Aggregation strategy (default: mean)
- `eps: float = 1e-8` - Numerical stability epsilon

**Returns**: `Tensor` - Normalized errors (per-vector or aggregated)

**Example**:
```python
from dlkit.core.training.metrics import create_normalized_vector_norm_error
import torch

# Create metric for 2D vectors with L2 norm
metric = create_normalized_vector_norm_error(
    vector_dim=-1,  # Last dimension is vector dimension
    norm_ord=2,     # L2 norm
    aggregator="mean"
)

# Sample 2D vector data (each row is a vector)
predictions = torch.tensor([
    [1.0, 0.0],
    [0.0, 2.0],
    [1.0, 1.0]
])
targets = torch.tensor([
    [1.0, 1.0],
    [2.0, 0.0],
    [1.0, 1.0]
])

# Compute normalized error
# For each vector: ||pred - target|| / ||target||
error = metric(predictions, targets)
print(f"Mean normalized vector error: {error:.4f}")

# L1 norm variant
l1_metric = create_normalized_vector_norm_error(norm_ord=1)
l1_error = l1_metric(predictions, targets)
print(f"L1 normalized error: {l1_error:.4f}")
```

**Implementation Notes**:
- Computes `||pred_vector - target_vector||_ord / ||target_vector||_ord`
- Normalization happens per-vector, then aggregated
- Epsilon prevents division by zero for zero-magnitude targets
- Particularly useful for 2D spatial data or multi-feature predictions
- Validates tensor has at least 2 dimensions
- Normalization integrated in raw error computation (not via normalizer strategy)

---

### Component 7: `CompositeMetric`

**Purpose**: Combine multiple metrics into a single metric with optional weighting. Enables multi-objective optimization and custom loss functions.

**Constructor Parameters**:
- `name: str` - Composite metric name
- `metrics: list[IMetric]` - List of metrics to combine
- `weights: Tensor | None = None` - Optional weights for weighted combination

**Returns**: `Tensor` - Weighted sum or mean of component metrics

**Example**:
```python
from dlkit.core.training.metrics import create_metric, create_composite_metric
import torch

# Create component metrics
mse = create_metric("mse")
mae = create_metric("mae")

# Equal weight combination (simple average)
composite = create_composite_metric(
    name="mse_mae",
    metrics=[mse, mae]
)

predictions = torch.tensor([1.0, 2.0, 3.0])
targets = torch.tensor([1.1, 1.9, 3.1])

# Compute composite metric
combined_error = composite(predictions, targets)
print(f"Combined error: {combined_error:.4f}")

# Weighted combination (70% MSE, 30% MAE)
weighted_composite = create_composite_metric(
    name="weighted_error",
    metrics=[mse, mae],
    weights=[0.7, 0.3]
)

weighted_error = weighted_composite(predictions, targets)
print(f"Weighted error: {weighted_error:.4f}")

# Use in training loop
loss = weighted_composite(model_output, ground_truth)
loss.backward()
```

**Implementation Notes**:
- Computes each metric independently, then combines
- Weights broadcasted correctly for multi-dimensional results
- Equal weights default to arithmetic mean
- Metadata includes component metric names and weights
- Supports callable interface for functional usage
- Validates weights length matches metrics length

---

### Component 8: Registry and Factory System

**Purpose**: Thread-safe registration, discovery, and instantiation of metrics with dependency injection. Enables dynamic metric creation and extensibility.

**Registries**:
- `MetricRegistry` - Register and retrieve metric classes
- `AggregatorRegistry` - Register and retrieve aggregators
- `NormalizerRegistry` - Register and retrieve normalizers

**Factory**:
- `MetricFactory` - Create metrics with injected dependencies

**Global Access**:
- `get_global_metric_registry() -> MetricRegistry`
- `get_global_aggregator_registry() -> AggregatorRegistry`
- `get_global_normalizer_registry() -> NormalizerRegistry`
- `get_global_metric_factory() -> MetricFactory`

**Example**:
```python
from dlkit.core.training.metrics import (
    get_global_metric_registry,
    get_global_metric_factory,
    BaseMetric,
    create_metric
)
from torch import Tensor
import torch

# Register custom metric
class HuberLossMetric(BaseMetric):
    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(name="huber", delta=delta, **kwargs)

    def _compute_raw_error(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        delta = self._params["delta"]
        error = predictions - targets
        is_small = torch.abs(error) <= delta
        squared_loss = 0.5 * error ** 2
        linear_loss = delta * (torch.abs(error) - 0.5 * delta)
        return torch.where(is_small, squared_loss, linear_loss)

# Register with global registry
registry = get_global_metric_registry()
registry.register("huber", HuberLossMetric)

# List available metrics
available = registry.list_metrics()
print(f"Available metrics: {available}")

# Create via factory
factory = get_global_metric_factory()
huber = factory.create_metric("huber", delta=1.5)

# Or use convenience function
huber = create_metric("huber", delta=1.5)

# Use custom metric
predictions = torch.tensor([1.0, 2.0, 5.0])
targets = torch.tensor([1.1, 1.9, 3.0])
loss = huber(predictions, targets)
print(f"Huber loss: {loss:.4f}")
```

**Implementation Notes**:
- Thread-safe with `threading.RLock()`
- Global registries are singletons (double-checked locking)
- Factory handles dependency injection (aggregator, normalizer)
- Built-in metrics pre-registered at module import
- `create_metric()` function wraps factory for convenience
- Registry validates metric classes implement required methods

## Usage Patterns

### Common Use Case 1: Basic Metric Computation
```python
from dlkit.core.training.metrics import create_metric
import torch

# Create metrics
mse = create_metric("mse")
mae = create_metric("mae")
rmse = create_metric("rmse")

# Sample predictions and targets
predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
targets = torch.tensor([1.1, 1.9, 3.2, 3.8, 5.1])

# Compute metrics
mse_value = mse(predictions, targets)
mae_value = mae(predictions, targets)
rmse_value = rmse(predictions, targets)

print(f"MSE: {mse_value:.4f}")
print(f"MAE: {mae_value:.4f}")
print(f"RMSE: {rmse_value:.4f}")
```

### Common Use Case 2: Normalized Metrics
```python
from dlkit.core.training.metrics import create_metric
import torch

# Create normalized metrics
mse_normalized = create_metric("mse", normalizer="variance")
mae_normalized = create_metric("mae", normalizer="std")

predictions = torch.randn(100)
targets = torch.randn(100) * 2.0 + 1.0

# Normalized metrics account for target scale
norm_mse = mse_normalized(predictions, targets)
norm_mae = mae_normalized(predictions, targets)

print(f"Normalized MSE: {norm_mse:.4f}")
print(f"Normalized MAE: {norm_mae:.4f}")
```

### Common Use Case 3: Functional Programming
```python
from dlkit.core.training.metrics import create_metric
import torch

# Create multiple metrics
metric_names = ["mse", "mae", "rmse"]
metrics = [create_metric(name) for name in metric_names]

predictions = torch.tensor([1.0, 2.0, 3.0])
targets = torch.tensor([1.1, 1.9, 3.1])

# Functional style computation
errors = list(map(lambda m: m(predictions, targets), metrics))

# Or with list comprehension
errors = [metric(predictions, targets) for metric in metrics]

# Higher-order function
def evaluate_with_metrics(metrics, pred, tgt):
    return {m.name: float(m(pred, tgt)) for m in metrics}

results = evaluate_with_metrics(metrics, predictions, targets)
print(results)  # {'mse': 0.0233, 'mae': 0.1333, 'rmse': 0.1528}
```

### Common Use Case 4: Custom Metrics for Domain-Specific Tasks
```python
from dlkit.core.training.metrics import create_normalized_vector_norm_error
import torch

# Time series forecasting - multiple horizons
predictions = torch.randn(32, 10, 5)  # (batch, time, features)
targets = torch.randn(32, 10, 5)

# Create metric for feature vectors at each time step
vector_metric = create_normalized_vector_norm_error(
    vector_dim=-1,  # Features dimension
    norm_ord=2,     # L2 norm
    aggregator="mean"
)

# Compute error across all batches and time steps
error = vector_metric(predictions, targets)
print(f"Normalized vector error: {error:.4f}")
```

### Common Use Case 5: Composite Metrics for Multi-Objective Loss
```python
from dlkit.core.training.metrics import create_metric, create_composite_metric
import torch

# Create component metrics
reconstruction_loss = create_metric("mse")
regularization_loss = create_metric("mae")

# Combine with 90% reconstruction, 10% regularization
total_loss = create_composite_metric(
    name="vae_loss",
    metrics=[reconstruction_loss, regularization_loss],
    weights=[0.9, 0.1]
)

# Use in training loop
for batch in dataloader:
    predictions = model(batch)

    loss = total_loss(predictions, batch)
    loss.backward()
    optimizer.step()
```

## Error Handling

**Exceptions Raised**:
- `TypeError`: Non-tensor inputs to metric computation
- `ValueError`: Shape mismatch between predictions and targets
- `ValueError`: Invalid metric parameters (negative dimensions, etc.)
- `KeyError`: Metric, aggregator, or normalizer not found in registry
- `ValueError`: Registration of duplicate metric names

**Error Handling Pattern**:
```python
from dlkit.core.training.metrics import create_metric
import torch

# Handle shape mismatches
try:
    mse = create_metric("mse")
    predictions = torch.tensor([1.0, 2.0])
    targets = torch.tensor([1.0, 2.0, 3.0])  # Wrong shape
    error = mse(predictions, targets)
except ValueError as e:
    print(f"Shape mismatch: {e}")
    # Reshape or pad data

# Handle missing metrics
try:
    metric = create_metric("nonexistent_metric")
except KeyError as e:
    print(f"Metric not found: {e}")
    # Fall back to default
    metric = create_metric("mse")

# Handle invalid parameters
try:
    metric = create_metric("normalized_vector_norm_error", vector_dim=10)
    predictions = torch.tensor([[1.0, 2.0]])  # Only 2D
    error = metric(predictions, targets)
except ValueError as e:
    print(f"Invalid dimension: {e}")
    # Use correct dimension
    metric = create_metric("normalized_vector_norm_error", vector_dim=-1)
```

## Testing

### Test Coverage
- Unit tests: `tests/core/training/metrics/` (to be created)
- Protocol tests: Verify all metrics implement IMetric
- Integration tests: Metric composition and factory creation

### Key Test Scenarios
1. **Basic computation**: MSE, MAE, RMSE correctness
2. **Aggregation strategies**: Mean, sum, L1/L2 norms
3. **Normalization strategies**: Variance, std, vector norm
4. **Composite metrics**: Weighted and unweighted combinations
5. **Vector norm metrics**: Per-vector normalization
6. **Registry operations**: Register, retrieve, list metrics
7. **Factory creation**: Dependency injection correctness
8. **Callable interface**: Function-like usage
9. **Thread safety**: Concurrent registry access
10. **Edge cases**: Zero-magnitude targets, single-sample batches

### Fixtures Used
- `sample_predictions`: Standard prediction tensors
- `sample_targets`: Standard target tensors
- `vector_data`: Multi-dimensional vector datasets
- `time_series_data`: Sequential data for temporal metrics

## Performance Considerations
- Aggregation strategies use native PyTorch operations for GPU acceleration
- Singleton registry instances prevent object creation overhead
- Template method minimizes virtual function calls
- Normalization computed once and cached during computation
- Vectorized operations over loops for batch processing
- Thread-safe registries use RLock for minimal contention
- Factory caching could further optimize repeated metric creation

## Future Improvements / TODOs
- [ ] Add GPU benchmarks for large-scale metric computation
- [ ] Implement metric caching for repeated computations
- [ ] Add statistical metrics (R², correlation, etc.)
- [ ] Support for sparse tensor inputs
- [ ] Implement metric visualization utilities
- [ ] Add automatic metric selection based on task type
- [ ] Support for multi-task metrics (different metrics per output)
- [ ] Implement metric streaming for online learning
- [ ] Add confidence intervals for metrics
- [ ] Support for probabilistic metrics (calibration, etc.)

## Related Modules
- `dlkit.core.models.wrappers`: Model wrappers use metrics for training/validation
- `lightning.pytorch.callbacks`: Lightning callbacks log metric values
- `dlkit.runtime.workflows.strategies.tracking`: MLflow tracking logs metrics
- `dlkit.core.training.callbacks`: Callbacks compute and log metrics

## Change Log
- **2025-10-03**: Comprehensive documentation with enriched docstrings
- **2024-10-02**: Added temporal derivative metric
- **2024-10-01**: Implemented normalized vector norm error metric
- **2024-09-30**: Refactored to SOLID principles with protocols
- **2024-09-24**: Added registry and factory system
- **2024-09-20**: Initial metrics implementation with template method
