# Feed-Forward Neural Networks

Multi-layer perceptron architectures for regression and classification tasks.

## Overview

All models in this module inherit from `ShapeAwareModel` and require a `unified_shape: IShapeSpec` parameter for input/output dimension configuration.

| Model | File | Description |
|-------|------|-------------|
| `LinearNetwork` | `linear.py` | Single linear layer with optional normalization |
| `FeedForwardNN` | `simple.py` | Variable-width MLP with skip connections |
| `ConstantWidthFFNN` | `simple.py` | Constant-width MLP with residual blocks |
| `NormScaledFFNN` | `norm_scaled.py` | Abstract base for norm-scaled wrappers |
| `NormScaledLinearFFNN` | `norm_scaled.py` | Norm-scaled single linear layer |
| `NormScaledConstantWidthFFNN` | `norm_scaled.py` | Norm-scaled constant-width MLP |

---

## LinearNetwork

A minimal single-layer network for simple transformations.

### Architecture

```
y = Norm(Wx + b)
```

Where:
- `W ∈ ℝ^(output × input)` = weight matrix
- `b ∈ ℝ^output` = bias vector (optional)
- `Norm` = BatchNorm, LayerNorm, or Identity

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unified_shape` | `IShapeSpec` | required | Shape specification with input/output dims |
| `normalize` | `"batch" \| "layer" \| None` | `None` | Normalization type |
| `bias` | `bool` | `True` | Include bias term |

### Example

```python
from dlkit.core.models.nn.ffnn import LinearNetwork
from dlkit.core.shape_specs import ShapeSpec

shape = ShapeSpec(x=(128,), y=(10,))
model = LinearNetwork(unified_shape=shape, normalize="layer")
```

---

## FeedForwardNN

Multi-layer perceptron with variable layer widths and residual connections.

### Architecture

```
h₀ = W_emb · x                           (embedding layer)
hᵢ = SkipConnection(DenseBlock(hᵢ₋₁))   for i = 1, ..., N-1
y = W_out · h_{N-1}                      (output layer)
```

Where each `SkipConnection(DenseBlock(...))` computes:
```
hᵢ = W_skip · hᵢ₋₁ + Dropout(Linear(σ(Norm(hᵢ₋₁))))
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unified_shape` | `IShapeSpec` | required | Shape specification |
| `layers` | `Sequence[int]` | required | Hidden layer widths |
| `activation` | `Callable` | `F.gelu` | Activation function |
| `normalize` | `"batch" \| "layer" \| None` | `None` | Normalization type |
| `dropout` | `float` | `0.0` | Dropout probability |

### Example

```python
from dlkit.core.models.nn.ffnn import FeedForwardNN
from dlkit.core.shape_specs import ShapeSpec

shape = ShapeSpec(x=(128,), y=(10,))
model = FeedForwardNN(
    unified_shape=shape,
    layers=[256, 128, 64],  # Variable widths
    normalize="layer",
    dropout=0.1,
)
```

---

## ConstantWidthFFNN

Constant-width variant of FeedForwardNN with uniform hidden dimensions.

### Architecture

```
h₀ = W_emb · x                                      (embedding: input → hidden_size)
hᵢ = hᵢ₋₁ + DenseBlock(hᵢ₋₁)   for i = 1, ..., N   (residual blocks)
y = W_out · h_N                                     (output: hidden_size → output)
```

Since all hidden layers have the same width, the skip projection is `Identity()`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unified_shape` | `IShapeSpec` | required | Shape specification |
| `hidden_size` | `int` | required | Width of all hidden layers |
| `num_layers` | `int` | required | Number of hidden layers |
| `activation` | `Callable` | `F.gelu` | Activation function |
| `normalize` | `"batch" \| "layer" \| None` | `None` | Normalization type |
| `dropout` | `float` | `0.0` | Dropout probability |

### Example

```python
from dlkit.core.models.nn.ffnn import ConstantWidthFFNN
from dlkit.core.shape_specs import ShapeSpec

shape = ShapeSpec(x=(128,), y=(10,))
model = ConstantWidthFFNN(
    unified_shape=shape,
    hidden_size=256,
    num_layers=4,
    normalize="layer",
    dropout=0.1,
)
```

---

## NormScaledFFNN (Base Class)

Abstract wrapper that enforces homogeneous scaling consistency for problems of the form `Ax = b`.

### Architecture

```
x = f(b / ‖b‖_p) · ‖b‖_p
```

Where:
- `b` = input vector (e.g., right-hand side of linear system)
- `f` = base model operating on normalized input
- `‖·‖_p` = vector p-norm (L1, L2, or L∞)
- `x` = output scaled back to original magnitude

### Mathematical Motivation

For scale-equivariant problems where scaling the input should proportionally scale the output:
```
f(αb) = αf(b)  for all α > 0
```

This wrapper ensures the property by:
1. Normalizing: `b_scaled = b / ‖b‖_p`
2. Predicting: `x_scaled = base_model(b_scaled)`
3. Rescaling: `x = x_scaled · ‖b‖_p`

### Supported Norms

| Norm | Formula |
|------|---------|
| `"l2"` | `‖b‖₂ = √(Σ bᵢ²)` |
| `"l1"` | `‖b‖₁ = Σ |bᵢ|` |
| `"linf"` | `‖b‖∞ = max|bᵢ|` |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model` | `nn.Module` | required | Model operating on normalized input |
| `unified_shape` | `IShapeSpec` | required | Shape specification |
| `norm` | `"l2" \| "l1" \| "linf"` | `"l2"` | Vector norm type |
| `eps_gain` | `float` | `10.0` | Multiplier for epsilon (numerical stability) |
| `keep_stats` | `bool` | `False` | Return norm statistics with output |

---

## NormScaledLinearFFNN

Norm-scaled wrapper around a single linear layer.

### Architecture

```
x = (W · (b / ‖b‖_p) + bias) · ‖b‖_p
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unified_shape` | `IShapeSpec` | required | Shape specification |
| `bias` | `bool` | `True` | Include bias in linear layer |
| `norm` | `str` | `"l2"` | Vector norm type |
| `eps_gain` | `float` | `10.0` | Numerical stability factor |
| `keep_stats` | `bool` | `False` | Return norm statistics |

### Example

```python
from dlkit.core.models.nn.ffnn import NormScaledLinearFFNN
from dlkit.core.shape_specs import ShapeSpec

shape = ShapeSpec(x=(100,), y=(50,))
model = NormScaledLinearFFNN(unified_shape=shape, norm="l2")
```

---

## NormScaledConstantWidthFFNN

Norm-scaled wrapper around ConstantWidthFFNN.

### Architecture

```
x = ConstantWidthFFNN(b / ‖b‖_p) · ‖b‖_p
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unified_shape` | `IShapeSpec` | required | Shape specification |
| `hidden_size` | `int` | required | Hidden layer width |
| `num_layers` | `int` | required | Number of hidden layers |
| `norm` | `str` | `"l2"` | Vector norm type |
| `eps_gain` | `float` | `10.0` | Numerical stability factor |
| `keep_stats` | `bool` | `False` | Return norm statistics |
| `activation` | `Callable \| None` | `None` | Activation (default: GELU) |
| `normalize` | `"batch" \| "layer" \| None` | `None` | Normalization type |
| `dropout` | `float` | `0.0` | Dropout probability |

### Example

```python
from dlkit.core.models.nn.ffnn import NormScaledConstantWidthFFNN
from dlkit.core.shape_specs import ShapeSpec

shape = ShapeSpec(x=(100,), y=(50,))
model = NormScaledConstantWidthFFNN(
    unified_shape=shape,
    hidden_size=128,
    num_layers=3,
    norm="l2",
    normalize="layer",
)
```

---

## Shape Specification

All models use `IShapeSpec` for dimension configuration:

```python
from dlkit.core.shape_specs import ShapeSpec

# Standard 1D feature vectors
shape = ShapeSpec(x=(input_dim,), y=(output_dim,))

# Autoencoder (output = input)
shape = ShapeSpec(x=(dim,), y=(dim,))
```

### Shape Validation

Models validate shapes via `accepts_shape()`:
- Requires both input and output shapes
- Only supports 1D feature vectors (`len(shape) == 1`)
- Dimensions must be positive integers
- Rejects `NullShapeSpec`
