# Neural Network Primitives

Building blocks for constructing neural network architectures in DLKit.

## Overview

This module provides fundamental components that serve as building blocks for larger architectures:

| Component | File | Purpose |
|-----------|------|---------|
| `DenseBlock` | `dense.py` | Pre-activation dense layer with normalization |
| `SkipConnection` | `skip.py` | Residual connection wrapper with flexible aggregation |
| `ConvolutionBlock1d` | `convolutional.py` | 1D convolution with normalization and dropout |
| `DeconvolutionBlock1d` | `convolutional.py` | 1D transposed convolution for upsampling |
| `TransformMixin` | `transform.py` | Lightning callback for transform chains |

---

## DenseBlock

A pre-activation dense layer following the pattern from ResNet v2.

### Architecture

```
y = Dropout(Linear(σ(Norm(x))))
```

Where:
- `Norm` = LayerNorm, BatchNorm, or Identity (configurable)
- `σ` = activation function (default: GELU)
- `Linear` = fully connected layer
- `Dropout` = dropout regularization (optional)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | required | Input dimension |
| `out_features` | `int` | required | Output dimension |
| `activation` | `Callable` | `F.gelu` | Activation function |
| `normalize` | `"layer" \| "batch" \| None` | `None` | Normalization type |
| `dropout` | `float` | `0.0` | Dropout probability |

### Example

```python
from dlkit.core.models.nn.primitives import DenseBlock

block = DenseBlock(
    in_features=128,
    out_features=64,
    activation=F.relu,
    normalize="layer",
    dropout=0.1,
)
```

---

## SkipConnection

Residual connection wrapper that adds skip paths around any module.

### Architecture

**Sum aggregation** (default):
```
y = σ(W_skip · x + f(x))
```

**Concat aggregation**:
```
y = σ([W_skip · x ‖ f(x)])
```

Where:
- `f(x)` = output of the wrapped module
- `W_skip` = projection matrix (Identity if dimensions match, otherwise 1×1 conv or linear)
- `σ` = activation function (default: Identity)
- `‖` = concatenation along channel dimension

### Dimension Matching

The skip projection `W_skip` is selected automatically:

| Condition | Projection Layer |
|-----------|-----------------|
| `in_channels == out_channels` | `Identity()` |
| `layer_type == "conv1d"` | `Conv1d(in, out, kernel=1)` |
| `layer_type == "conv2d"` | `Conv2d(in, out, kernel=1)` |
| `layer_type == "linear"` | `Linear(in, out, bias=False)` |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `nn.Module` | required | Module to wrap |
| `how` | `"sum" \| "concat"` | `"sum"` | Aggregation method |
| `layer_type` | `"conv1d" \| "conv2d" \| "linear"` | `"conv1d"` | Projection layer type |
| `activation` | `Callable` | `Identity()` | Post-aggregation activation |
| `in_channels` | `int \| None` | `None` | Input channels (auto-detected from module) |
| `out_channels` | `int \| None` | `None` | Output channels (auto-detected from module) |
| `stride` | `int` | `1` | Stride for projection layer |
| `bias` | `bool` | `True` | Include bias in projection |

### Example

```python
from dlkit.core.models.nn.primitives import SkipConnection, DenseBlock

# Wrap a DenseBlock with residual connection
residual_block = SkipConnection(
    DenseBlock(128, 128, normalize="layer"),
    how="sum",
    layer_type="linear",
)

# With dimension change and activation
residual_block = SkipConnection(
    DenseBlock(128, 256, normalize="layer"),
    how="sum",
    layer_type="linear",
    activation=nn.GELU(),
)
```

---

## ConvolutionBlock1d

1D convolutional block with pre-activation design.

### Architecture

```
y = Dropout(σ(Conv1d(Norm(x))))
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | `int` | required | Input channels |
| `out_channels` | `int` | required | Output channels |
| `in_timesteps` | `int` | required | Input sequence length |
| `kernel_size` | `int` | `3` | Convolution kernel size |
| `stride` | `int` | `1` | Convolution stride |
| `padding` | `str \| int` | `"same"` | Padding mode |
| `activation` | `Callable` | `F.gelu` | Activation function |
| `normalize` | `"layer" \| "batch" \| "instance" \| None` | `None` | Normalization type |
| `dropout` | `float` | `0.0` | Dropout probability |
| `dilation` | `int` | `1` | Dilation rate |
| `groups` | `int` | `1` | Convolution groups |

### Output Size Calculation

For explicit padding:
```
out_size = floor((in_size + 2×padding - dilation×(kernel-1) - 1) / stride + 1)
```

For `padding="same"`: output size equals input size.

---

## DeconvolutionBlock1d

1D transposed convolution for upsampling operations.

### Architecture

```
y = ConvTranspose1d(σ(x))
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | `int` | required | Input channels |
| `out_channels` | `int` | required | Output channels |
| `in_timesteps` | `int` | required | Input sequence length |
| `kernel_size` | `int` | `3` | Kernel size |
| `stride` | `int` | `1` | Stride |
| `padding` | `str \| int` | `"same"` | Padding mode |
| `output_padding` | `int` | `0` | Additional output padding |
| `dilation` | `int` | `1` | Dilation rate |
| `groups` | `int` | `1` | Convolution groups |

---

## TransformMixin

Lightning callback mixin for applying transform chains to features and targets.

### Usage

Used with Lightning modules to apply fitted transforms during training:

```python
class MyModel(TransformMixin, LightningModule):
    def __init__(self, settings):
        TransformMixin.__init__(self, settings)
        LightningModule.__init__(self)
        # ... model initialization
```

### Behavior

1. **Initialization**: Creates `TransformChain` for features and targets from settings
2. **on_train_start**: Moves chains to device and fits on training data
3. **forward**: Applies feature transforms before model, optionally inverts target transforms after

### Requirements

- `settings.shape.x` must be defined (feature shape)
- `settings.shape.y` must be defined for non-autoencoder models
- For autoencoders (`settings.is_autoencoder=True`), target chain reuses feature chain
