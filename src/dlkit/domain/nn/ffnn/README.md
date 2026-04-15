# Feed-Forward Neural Networks

Multi-layer perceptron architectures for regression and classification tasks.

## Overview

| Model | File | Description |
|-------|------|-------------|
| `LinearNetwork` | `linear.py` | Single linear layer with optional normalization |
| `FeedForwardNN` | `simple.py` | Variable-width MLP with skip connections |
| `ConstantWidthFFNN` | `simple.py` | Constant-width MLP with residual blocks |
| `SimpleFeedForwardNN` | `plain.py` | Variable-width MLP, no residual |
| `ConstantWidthSimpleFFNN` | `plain.py` | Constant-width MLP, no residual |
| `ScaleEquivariantFFNN` | `scale_equivariant.py` | Wrapper enforcing scale equivariance (wraps any nonlinear base) |
| `ScaleEquivariantConstantWidthFFNN` | `scale_equivariant.py` | Scale-equivariant constant-width MLP |
| `ConstantWidthParametricFFNN` | `parametric.py` | Constant-width MLP with constrained weight blocks |
| `EmbeddedParametricFFNN` | `parametric.py` | Embedded-width variant with constrained blocks |
| `ConstantWidthSPDFFNN` | `parametric_variants.py` | SPD-constrained blocks |
| `ConstantWidthSPDFactorizedFFNN` | `parametric_variants.py` | SPD + diagonal scale |
| `ConstantWidthFactorizedFFNN` | `parametric_variants.py` | Diagonal scale only |
| `EmbeddedSPDFFNN` | `parametric_variants.py` | Embedded SPD |
| `EmbeddedSPDFactorizedFFNN` | `parametric_variants.py` | Embedded SPD + diagonal scale |
| `EmbeddedFactorizedFFNN` | `parametric_variants.py` | Embedded diagonal scale |

> **Note on ScaleEquivariantFFNN and linear layers**: Linear maps are scale-equivariant by construction,
> so wrapping a purely linear layer in `ScaleEquivariantFFNN` adds no expressive power.
> `ScaleEquivariantFFNN` is only meaningful when the base model contains nonlinearities.

---

## LinearNetwork

A minimal single-layer network for simple transformations or baselines.

### Architecture

```
y = Norm(Wx + b)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | required | Input dimension |
| `out_features` | `int` | required | Output dimension |
| `normalize` | `"batch" \| "layer" \| None` | `None` | Normalization type |
| `bias` | `bool` | `True` | Include bias term |

---

## FeedForwardNN

Variable-width MLP with residual skip connections.

### Architecture

```
h₀ = W_emb · x
hᵢ = SkipConnection(DenseBlock(hᵢ₋₁))   for i = 1, …, N-1
y  = W_out · h_{N-1}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | required | Input dimension |
| `out_features` | `int` | required | Output dimension |
| `layers` | `list[int]` | required | Hidden layer widths |
| `activation` | `Callable` | `F.gelu` | Activation function |
| `normalize` | `"batch" \| "layer" \| None` | `None` | Normalization type |
| `dropout` | `float` | `0.0` | Dropout probability |

---

## ConstantWidthFFNN

Constant-width residual MLP; skip projection is Identity (no extra params).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | required | Input dimension |
| `out_features` | `int` | required | Output dimension |
| `hidden_size` | `int` | required | Width of all hidden layers |
| `num_layers` | `int` | required | Number of hidden layers |
| `activation` | `Callable` | `F.gelu` | Activation function |
| `normalize` | `"batch" \| "layer" \| None` | `None` | Normalization type |
| `dropout` | `float` | `0.0` | Dropout probability |

---

## NormScaledFFNN (Base Wrapper)

Wraps any `nn.Module` to enforce scale-equivariance:
```
output = base_model(x / ‖x‖_p) · ‖x‖_p
```

Meaningful only when `base_model` contains nonlinear activations.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model` | `nn.Module` | required | The nonlinear base network |
| `norm` | `"l2" \| "l1" \| "linf"` | `"l2"` | Vector norm type |
| `eps_gain` | `float` | `10.0` | Multiplier for machine epsilon (avoids div-by-zero) |
| `keep_stats` | `bool` | `False` | Return `(output, {"norm": norms})` when True |

---

## NormScaledConstantWidthFFNN

Convenience wrapper: `NormScaledFFNN(ConstantWidthFFNN(...))`.

### Parameters

All of `ConstantWidthFFNN` plus `norm`, `eps_gain`, `keep_stats`.
