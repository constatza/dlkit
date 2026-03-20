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

**Code notation:**
```
y = Dropout(Linear(σ(Norm(x))))
```

**Mathematical form:**

$$\mathbf{y} = \text{Dropout}\!\left(W\,\sigma\!\left(\text{Norm}(\mathbf{x})\right) + \mathbf{b}\right)$$

Where:
- $\text{Norm}$ — LayerNorm, BatchNorm, or Identity (configurable)
- $\sigma$ — activation function (default: GELU)
- $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, $\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}$ — weight matrix and bias of the linear layer
- $\text{Dropout}$ — dropout regularization (optional)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | required | Input dimension $d_{\text{in}}$ |
| `out_features` | `int` | required | Output dimension $d_{\text{out}}$ |
| `activation` | `Callable` | `F.gelu` | Activation function $\sigma$ |
| `normalize` | `"layer" \| "batch" \| None` | `None` | Normalization type |
| `dropout` | `float` | `0.0` | Dropout probability $p$ |

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

The skip path is **purely linear** — it passes through only a projection layer (no activation). No post-aggregation activation is used in any callsite in this codebase; the `activation` parameter defaults to `nn.Identity()` and remains unused in practice.

**Sum aggregation** (default):

Code notation:
```
skip = W_skip · x          # linear-only path, no activation
y = skip + f(x)
```

Mathematical form:

$$\mathbf{y} = W_{\text{skip}}\,\mathbf{x} + f(\mathbf{x})$$

**Concat aggregation:**

Code notation:
```
skip = W_skip · x          # linear-only path, no activation
y = [skip ‖ f(x)]
```

Mathematical form:

$$\mathbf{y} = \begin{bmatrix} W_{\text{skip}}\,\mathbf{x} \\ f(\mathbf{x}) \end{bmatrix}$$

Where:
- $f(\mathbf{x})$ — output of the wrapped module (main path)
- $W_{\text{skip}}\,\mathbf{x}$ — skip path: linear projection only, no activation
- $[\cdot \;\|\; \cdot]$ — concatenation along the channel dimension

### Dimension Matching

The skip projection $W_{\text{skip}}$ is selected automatically:

| Condition | Projection Layer |
|-----------|-----------------|
| $C_{\text{in}} = C_{\text{out}}$ | $\text{Identity}()$ |
| `layer_type == "conv1d"` | $\text{Conv1d}(C_{\text{in}}, C_{\text{out}}, k{=}1)$ |
| `layer_type == "conv2d"` | $\text{Conv2d}(C_{\text{in}}, C_{\text{out}}, k{=}1)$ |
| `layer_type == "linear"` | $\text{Linear}(C_{\text{in}}, C_{\text{out}},\; \text{bias=False})$ |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `nn.Module` | required | Module to wrap |
| `how` | `"sum" \| "concat"` | `"sum"` | Aggregation method |
| `layer_type` | `"conv1d" \| "conv2d" \| "linear"` | `"conv1d"` | Projection layer type |
| `activation` | `Callable` | `Identity()` | Post-aggregation activation (unused in practice — defaults to no-op) |
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

**Code notation:**
```
y = Dropout(σ(Conv1d(Norm(x))))
```

**Mathematical form:**

$$\mathbf{y} = \text{Dropout}\!\left(\sigma\!\left(\mathbf{W} * \text{Norm}(\mathbf{x}) + \mathbf{b}\right)\right)$$

Where $*$ denotes the 1D convolution operation, $\mathbf{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k}$ is the filter bank, and $k$ is the kernel size.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | `int` | required | Input channels $C_{\text{in}}$ |
| `out_channels` | `int` | required | Output channels $C_{\text{out}}$ |
| `in_timesteps` | `int` | required | Input sequence length $L$ |
| `kernel_size` | `int` | `3` | Convolution kernel size $k$ |
| `stride` | `int` | `1` | Convolution stride $s$ |
| `padding` | `str \| int` | `"same"` | Padding mode $p$ |
| `activation` | `Callable` | `F.gelu` | Activation function $\sigma$ |
| `normalize` | `"layer" \| "batch" \| "instance" \| None` | `None` | Normalization type |
| `dropout` | `float` | `0.0` | Dropout probability |
| `dilation` | `int` | `1` | Dilation rate $d$ |
| `groups` | `int` | `1` | Convolution groups |

### Output Size Calculation

For explicit integer padding $p$:

**Code notation:**
```
out_size = floor((in_size + 2×padding - dilation×(kernel-1) - 1) / stride + 1)
```

**Mathematical form:**

$$L_{\text{out}} = \left\lfloor \frac{L_{\text{in}} + 2p - d(k - 1) - 1}{s} + 1 \right\rfloor$$

For `padding="same"`: $L_{\text{out}} = L_{\text{in}}$.

---

## DeconvolutionBlock1d

1D transposed convolution for upsampling operations.

### Architecture

**Code notation:**
```
y = ConvTranspose1d(σ(x))
```

**Mathematical form:**

$$\mathbf{y} = \mathbf{W}^{\top} \star \sigma(\mathbf{x}) + \mathbf{b}$$

Where $\star$ denotes the 1D transposed convolution (fractionally-strided convolution) and $\mathbf{W}^{\top}$ refers to the transposed filter bank applied in decoder fashion.

### Output Size Calculation

$$L_{\text{out}} = (L_{\text{in}} - 1) \cdot s - 2p + d(k - 1) + p_{\text{out}} + 1$$

Where $p_{\text{out}}$ is `output_padding`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | `int` | required | Input channels $C_{\text{in}}$ |
| `out_channels` | `int` | required | Output channels $C_{\text{out}}$ |
| `in_timesteps` | `int` | required | Input sequence length $L$ |
| `kernel_size` | `int` | `3` | Kernel size $k$ |
| `stride` | `int` | `1` | Stride $s$ |
| `padding` | `str \| int` | `"same"` | Padding $p$ |
| `output_padding` | `int` | `0` | Additional output padding $p_{\text{out}}$ |
| `dilation` | `int` | `1` | Dilation rate $d$ |
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

---

## Parametrized Linear Layers

All layers live in `parametrized_layers.py`. Constrained layers use PyTorch's
`torch.nn.utils.parametrize` machinery unless noted.

### Layer Reference

| Layer | Constraint | Mathematical Form | Square required |
|---|---|---|---|
| `SymmetricLinear` | Hard (parametrize) | $W = W^\top$ | Yes |
| `SPDLinear` | Hard (parametrize) | SPD via Gershgorin | Yes |
| `SPDFactorizedLinear` | Hard (parametrize) | $W = D\,\text{SPD}(A)\,D$ | Yes |
| `FactorizedLinear` | Modelling choice (plain Module) | $W = \text{diag}(\phi(\mathbf{s}))\,A$ | No |
| `SymmetricFactorizedLinear` | Hard (parametrize) | $W = D\,\text{Sym}(A)\,D$ | Yes |

### `SPDLinear`

Uses `register_spd`, which chains **two** parametrizations applied in order to the raw learnable parameter $X$:

$$X \;\xrightarrow{\;\texttt{Symmetric}\;}\; A \;\xrightarrow{\;\texttt{SPD}\;}\; W$$

**Step 1 — Symmetrisation** (`Symmetric`):

$$A = \text{triu}(X) + \text{triu}(X,\,1)^\top$$

The unconstrained parameter $X \in \mathbb{R}^{n \times n}$ is folded into a symmetric matrix $A = A^\top$ by mirroring the upper triangle. Only the $\tfrac{n(n+1)}{2}$ upper-triangular entries are free; the lower triangle is not independently learned.

**Step 2 — Positive-definiteness** (`SPD`):

The `SPD` module expects a symmetric input (it validates this at runtime) and rewrites only the diagonal:

$$W_{ij} = A_{ij}, \quad i \neq j$$

$$W_{ii} = \phi(A_{ii}) + \sum_{j \neq i} |A_{ij}| + \epsilon_{\min}$$

where $\phi$ is `pos_fn` (default: softplus) and $\epsilon_{\min}$ is `min_diag`.

**Gershgorin guarantee**: $W_{ii} > \sum_{j \neq i} |W_{ij}|$ for all $i$, so all eigenvalues are positive and $W \succ 0$.

**Full pipeline:**

$$W_{ii} = \phi\!\Bigl(\text{triu}(X)_{ii}\Bigr) + \sum_{j \neq i} \bigl|\text{triu}(X)_{\min(i,j),\max(i,j)}\bigr| + \epsilon_{\min}, \qquad W_{ij} = X_{\min(i,j),\,\max(i,j)} \text{ for } i \neq j$$

**Key parameters**:
- `min_diag` (float, default `1e-4`): positive floor $\epsilon_{\min}$ on each diagonal entry.
- `pos_fn` (callable, default `F.softplus`): element-wise map $\phi : \mathbb{R} \to \mathbb{R}_{>0}$.

### `SPDFactorizedLinear`

Chained parametrizations: `Symmetric → SPD → PositiveSandwichScale`.

**Mathematical form:**

$$W = D\,\text{SPD}(A)\,D, \qquad D = \text{diag}\!\left(e^{\mathbf{s}}\right)$$

The sandwich scale $D$ preserves both symmetry and positive definiteness: if $M \succ 0$ then $D M D \succ 0$ for any invertible diagonal $D$.

**Additional parameters**:
- `mean`, `std`: initialisation of the log-scale $\mathbf{s}$ ($\text{mean}=0 \Rightarrow D \approx I$).
- `pos_fn`: forwarded to the `SPD` stage.

### `FactorizedLinear`

Plain `nn.Module` (no `parametrize`). Stores `base_weight` $A$ and `log_scale` $\mathbf{s}$
separately for a flat, transparent state dict.

**Mathematical form:**

$$W = \text{diag}\!\left(\phi(\mathbf{s})\right) A, \qquad \phi = \exp \text{ (default)}$$

Equivalently: row $i$ of $W$ is $\phi(s_i)$ times row $i$ of $A$.

**Key parameters**:
- `mean`, `std`: initialisation of `log_scale` $\mathbf{s}$.
- `pos_fn` (callable, default `torch.exp`): maps $\mathbf{s}$ to positive row scales. Alternatives: `F.softplus`, `lambda x: F.relu(x) + 1e-6`.

### Configuring `pos_fn`

`SPDLinear` (and all SPD variants) require a square weight matrix, so `in_features`
must equal `out_features`. Both arguments are kept to match `nn.Linear`'s signature,
but passing different values raises a `ValueError` at construction time.

```python
import torch
import torch.nn.functional as F
from dlkit.core.models.nn.primitives import SPDLinear, FactorizedLinear

n = 16  # in_features == out_features (square weight matrix)

# Default: softplus (smooth, non-saturating)
SPDLinear(n, n)

# Exponential (sharper gradient near zero)
SPDLinear(n, n, pos_fn=torch.exp)

# Bounded alternative
SPDLinear(n, n, pos_fn=lambda x: F.relu(x) + 1e-4)

# FactorizedLinear: rectangular matrices are allowed (not square-constrained)
FactorizedLinear(16, 32)

# FactorizedLinear with softplus
FactorizedLinear(16, 32, pos_fn=F.softplus)
```

### Parametrization Modules

Each module is a composable building block applied via `torch.nn.utils.parametrize`.

---

#### `Symmetric`

Folds the upper triangle onto the lower triangle so that $W = W^\top$.

$$W_{ij} = X_{\min(i,j),\;\max(i,j)}$$

Equivalently, using PyTorch indexing:

$$W = \underbrace{\text{triu}(X)}_{\text{upper + diagonal}} + \underbrace{\text{triu}(X,\,1)^\top}_{\text{strict upper, mirrored}}$$

- **Input**: any unconstrained square matrix $X \in \mathbb{R}^{n \times n}$
- **Output**: symmetric matrix using only the upper-triangular values of $X$
- **Preserved**: $W^\top = W$

---

#### `SPD`

Expects a **symmetric** input and rewrites the diagonal to enforce strict diagonal dominance. By the **Gershgorin circle theorem**, this guarantees all eigenvalues are positive.

**Off-diagonal entries** (unchanged):

$$W_{ij} = X_{ij}, \quad i \neq j$$

**Diagonal rewrite:**

$$W_{ii} = \phi(X_{ii}) + \underbrace{\sum_{j \neq i} |X_{ij}|}_{\text{row off-diag. }L^1\text{ norm}} + \epsilon_{\min}$$

where $\phi : \mathbb{R} \to \mathbb{R}_{>0}$ is `pos_fn` (default: softplus) and $\epsilon_{\min}$ is `min_diag`.

**Gershgorin guarantee**: since $W_{ii} > \sum_{j \neq i} |W_{ij}|$ for all $i$, every eigenvalue $\lambda_i > 0$, therefore $W \succ 0$.

- **Input**: symmetric matrix $X$
- **Output**: symmetric positive-definite matrix $W \succ 0$
- **Preserved**: $W^\top = W$, $\lambda_{\min}(W) \geq \epsilon_{\min}$

---

#### `PositiveRowScale`

Scales each row $i$ by an independent positive factor $e^{s_i}$.

$$W_{ij} = e^{s_i}\, A_{ij}$$

Matrix form:

$$W = \text{diag}(e^{\mathbf{s}})\, A, \qquad \mathbf{s} \in \mathbb{R}^{n_{\text{rows}}} \text{ (learnable)}$$

- **Input**: base weight $A$
- **Output**: row-rescaled weight $W$; each row's $L^2$ norm is multiplied by $e^{s_i}$
- **Preserved**: sparsity pattern; column relationships within each row

---

#### `PositiveColumnScale`

Scales each column $j$ by an independent positive factor $e^{s_j}$.

$$W_{ij} = A_{ij}\, e^{s_j}$$

Matrix form:

$$W = A\,\text{diag}(e^{\mathbf{s}}), \qquad \mathbf{s} \in \mathbb{R}^{n_{\text{cols}}} \text{ (learnable)}$$

- **Input**: base weight $A$
- **Output**: column-rescaled weight $W$; each column's $L^2$ norm is multiplied by $e^{s_j}$
- **Preserved**: row relationships within each column

---

#### `PositiveSandwichScale`

Scales both rows and columns symmetrically with the **same** vector $\mathbf{s}$.

$$W_{ij} = e^{s_i}\, A_{ij}\, e^{s_j}$$

Matrix form:

$$W = D\, A\, D, \qquad D = \text{diag}(e^{\mathbf{s}}),\quad \mathbf{s} \in \mathbb{R}^{n} \text{ (learnable)}$$

- **Input**: base square matrix $A \in \mathbb{R}^{n \times n}$
- **Output**: sandwich-scaled matrix $W$
- **Preserved**: symmetry (if $A = A^\top$ then $W = W^\top$) and positive definiteness (if $A \succ 0$ then $W \succ 0$, since $D$ is invertible)

---

#### `PositiveScalarScale`

Applies a single global positive scale factor to the entire weight matrix.

$$W = e^{s}\, A, \qquad s \in \mathbb{R} \text{ (learnable scalar)}$$

- **Input**: base tensor $A$ of any shape
- **Output**: positively scaled tensor $W$; all entries scaled by the same factor $e^s$
- **Preserved**: relative magnitudes and signs of all entries

---

**Summary table:**

| Class | Element-wise formula | Matrix form | Constraint preserved |
|---|---|---|---|
| `Symmetric` | $W_{ij} = X_{\min(i,j),\max(i,j)}$ | $\text{triu}(X) + \text{triu}(X,1)^\top$ | $W = W^\top$ |
| `SPD` | $W_{ii} = \phi(X_{ii}) + \sum_{j\neq i}\lvert X_{ij}\rvert + \epsilon$ | diagonal rewrite | $W \succ 0$ |
| `PositiveRowScale` | $W_{ij} = e^{s_i} A_{ij}$ | $\text{diag}(e^{\mathbf{s}})\,A$ | positive row norms |
| `PositiveColumnScale` | $W_{ij} = A_{ij}\,e^{s_j}$ | $A\,\text{diag}(e^{\mathbf{s}})$ | positive column norms |
| `PositiveSandwichScale` | $W_{ij} = e^{s_i} A_{ij} e^{s_j}$ | $D A D,\ D{=}\text{diag}(e^{\mathbf{s}})$ | symmetry + PD |
| `PositiveScalarScale` | $W_{ij} = e^s\, A_{ij}$ | $e^s\, A$ | sign pattern |

### Registration Helpers

```python
register_symmetric(module, tensor_name="weight")
register_spd(module, tensor_name="weight", *, min_diag=1e-4, pos_fn=F.softplus)
register_symmetric_factorized(module, size, *, mean=0.0, std=0.1)
register_spd_factorized(module, size, *, min_diag=1e-4, mean=0.0, std=0.1, pos_fn=F.softplus)
```
