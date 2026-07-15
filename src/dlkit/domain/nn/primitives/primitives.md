# Neural Network Primitives

Building blocks for constructing neural network architectures in DLKit.

## Overview

This module provides fundamental components that serve as building blocks for larger architectures.

Blocks that take an `activation` kwarg (`DenseBlock`, `ConvolutionBlock1d`,
`DeconvolutionBlock1d`, `UVGate`) initialize their `Linear`/`Conv` weights via
`domain.nn.init.initialize_`, matched to that activation — see `nn.md`.

| Component | File | Purpose |
|-----------|------|---------|
| `DenseBlock` | `dense.py` | Pre-activation dense layer with normalization |
| `SkipConnection` | `skip.py` | Residual connection wrapper with flexible aggregation |
| `SparseMoE`, `TopKRouter` | `moe.py` | Feature-last sparse mixture-of-experts primitives |
| `HyperConnection`, `GraphHyperConnection` | `hyper.py` | Multi-lane residual wrappers with identity-biased lane mixing |
| `HyperSequential`, `GraphHyperSequential`, `MoESequential` | `stacks.py` | Sequential composition helpers for Hyper-Connection and residual sparse MoE stacks |
| `FactorizedInit`, `resolve_factorized_init` | `factorized_init.py` | Architecture-level factorized initialization policy helper |
| `ConvolutionBlock1d` | `convolutional.py` | 1D convolution with normalization and dropout |
| `DeconvolutionBlock1d` | `convolutional.py` | 1D transposed convolution for upsampling |
| `ScaleEquivariantWrapper` | `scale_equivariant.py` | Shared norm-scaled wrapper for positive scale equivariance |
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
| `normalize` | `"layer" \| "batch" \| None` | `"layer"` | Normalization type |
| `dropout` | `float` | `0.0` | Dropout probability $p$ |
| `bias` | `bool` | `True` | Whether the linear layer has a bias term |

### Example

```python
from dlkit.domain.nn.primitives import DenseBlock

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

The skip path is **purely linear** — it passes through only a projection layer (no activation). There is no post-aggregation activation; the aggregated output is returned directly.

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
| `in_channels` | `int \| None` | `None` | Input channels (auto-detected from module) |
| `out_channels` | `int \| None` | `None` | Output channels (auto-detected from module) |
| `stride` | `int` | `1` | Stride for projection layer |
| `bias` | `bool` | `True` | Include bias in projection |
| `branch_scale` | `float` | `1.0` | Multiplier on the wrapped module's output before aggregation. Set to `1/sqrt(2*num_layers)` (GPT-2 appendix) when stacking many of these to counteract residual variance growth across depth |

### Example

```python
from dlkit.domain.nn.primitives import SkipConnection, DenseBlock

# Wrap a DenseBlock with residual connection
residual_block = SkipConnection(
    DenseBlock(128, 128, normalize="layer"),
    how="sum",
    layer_type="linear",
)

# Concat mode: output channels = 2 × out_channels (skip ‖ main)
# DenseBlock(128, 128) → SkipConnection output width is 256
residual_block = SkipConnection(
    DenseBlock(128, 128, normalize="layer"),
    how="concat",
    layer_type="linear",
)
```

---

## ScaleEquivariantWrapper

Reusable wrapper that enforces positive scale equivariance by normalizing the
input by a per-sample norm, delegating to a wrapped module, and rescaling the
output by the original norm.

Code notation:
```
norm = ||x||
y = norm * f(x / max(norm, eps))
```

The wrapper is intentionally model-agnostic. Dense and coordinate spectral-bias
models reuse the same implementation rather than duplicating norm-scaling logic.

---

## SparseMoE

`moe.py` provides a composable sparse mixture-of-experts layer for feature-last
tensors. It is deliberately independent of FFNN and graph model families:
experts are plain `nn.Module` instances and the router only sees the final
feature dimension.

In the FFNN composites, those experts are FFN sublayers/blocks rather than full
embedded FFNN models. This follows Shazeer et al.'s sparsely gated MoE and the
GShard/Switch Transformer convention: top-k routing selects FFN experts, and the
Transformer residual path wraps the routed FFN sublayer/block. DLKit's linear
MoE FFNNs provide `ParametricDenseBlock` experts backed by `nn.Linear`;
factorized MoE FFNNs keep the same sublayer shape and replace only those
kernels with `FactorizedLinear`.

Key pieces:
- `TopKRouter`: maps each flattened token/sample to top-k expert ids and
  normalized route weights.
- `SparseMoE`: dispatches selected tokens to selected experts and combines
  expert outputs with route weights.
- `RoutingStats`: exposes `aux_loss`, `expert_counts`, `router_probs`, and
  `selected_experts` when callers request diagnostics.

Defaults follow a practical sparse MoE v1: `top_k=2`, softmax token routing,
no capacity drop, no shared experts, and tensor-only forward output unless
`return_stats=True`.

Example:

```python
from torch import nn
from dlkit.domain.nn.primitives import SparseMoE

experts = [nn.Linear(64, 64) for _ in range(4)]
moe = SparseMoE(in_features=64, experts=experts, top_k=2)
out, stats = moe(x, return_stats=True)
```

Shared experts can be supplied with `shared_experts=[...]`; they run for every
token and are added to the routed expert mixture.

---

## Sequential Hyper/MoE Stacks

`stacks.py` owns stack-level lifecycle concerns without defining expert
routing or factorized initialization:
- `HyperSequential`: expands a normal feature tensor into lanes once, applies a
  sequence of `HyperConnection` layers, then reduces lanes unless requested
  otherwise.
- `GraphHyperSequential`: same lane lifecycle for graph modules with
  `forward(x, edge_index, edge_attr=None)`.
- `MoESequential`: applies shape-preserving `SparseMoE` layers as a scaled
  residual stack and aggregates per-layer `RoutingStats` when requested.
- `residual_branch_scale`: returns the default branch scaling used by stack
  wrappers.

These helpers are the preferred way to build sequential Hyper-Connection or
MoE bodies because they keep lane/routing lifecycle outside concrete model
classes.

---

## HyperConnection

`hyper.py` provides multi-lane residual primitives inspired by Hyper-Connections.
The wrappers keep the residual mechanism modular: any feature-last module can be
wrapped without becoming a new concrete model family.

Key pieces:
- `LaneExpand`: expands `Tensor[..., features]` into
  `Tensor[..., lanes, features]`.
- `LaneReduce`: collapses lanes with identity-biased learned weights.
- `HyperConnection`: applies a wrapped module per lane with learnable pre/post
  lane mixing, initialized to identity.
- `GraphHyperConnection`: forwards `edge_index` and `edge_attr` to a graph
  module per lane while sharing graph structure.
- `LaneMixingStats`: exposes lane entropy, dominant-lane fraction, and
  off-diagonal mixing norm.

Example:

```python
from torch import nn
from dlkit.domain.nn.primitives import HyperConnection

block = HyperConnection(nn.Linear(64, 64), num_lanes=4, branch_scale=0.5)
out = block(x)
lanes = block(x, return_lanes=True)
```

`GraphHyperConnection` is intended for node-feature graph modules with the
signature `forward(x, edge_index, edge_attr=None)`.

---

## Factorized Initialization Policy

`factorized_init.py` centralizes architecture-level defaults for
`FactorizedLinear` construction:
- `log_scale_mean = 0.0`, giving unit scale at init because `exp(0) = 1`.
- `log_scale_std = 0.1`, fixed internally rather than exposed by public model
  constructors.
- `kaiming_a` is resolved from the selected activation for the factorized base
  weight initializer.

Low-level `FactorizedLinear` remains an expert primitive and still accepts its
literal initialization knobs. Public composite FFNN constructors pass only an
activation and let this helper resolve the factorized init policy.

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

> **Constraint**: `padding="same"` is only supported when `stride=1`.  Passing
> `padding="same"` with `stride != 1` raises `ValueError` at construction time.
> Use an explicit integer padding value when upsampling with stride > 1.

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

`parametrized_layers.py` currently exposes the rectangular
`FactorizedLinear` primitive. Positive-scale parametrization modules live in
`parametrizations.py` for direct composition with PyTorch modules when a caller
wants to register parametrizations manually.

### Layer Reference

| Layer | Constraint | Mathematical Form | Square required |
|---|---|---|---|
| `FactorizedLinear` | Modelling choice (plain Module) | $W = \text{diag}(e^{\mathbf{s}})\,A$ | No |

### `FactorizedLinear`

Plain `nn.Module` (no `parametrize`). Stores `base_weight` $A$ and `log_scale` $\mathbf{s}$
separately for a flat, transparent state dict.

Implements Random Weight Factorization (RWF), per Wang, Wang, Seidman & Perdikaris,
["Random Weight Factorization Improves the Training of Continuous Neural
Representations"](https://arxiv.org/abs/2210.01274) (2022).

**Mathematical form:**

$$W = \text{diag}\!\left(e^{\mathbf{s}}\right) A$$

Equivalently: row $i$ of $W$ is $e^{s_i}$ times row $i$ of $A$.

**Key parameters**:
- `mean`, `std`: literal Gaussian parameters used to sample `log_scale`
  $\mathbf{s}$ (shipped default: `mean=0.0`, `std=0.1` — unit scale at init,
  since $e^0 = 1$).
- `kaiming_a` (`float`, default `0.0`): the `a` gain passed to
  `nn.init.kaiming_uniform_` for `base_weight`.

```python
from dlkit.domain.nn.primitives import FactorizedLinear

# FactorizedLinear: paper-style exponential RWF
FactorizedLinear(16, 32)
```

### Parametrization Modules

Each module is a composable building block applied via `torch.nn.utils.parametrize`.

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
| `PositiveRowScale` | $W_{ij} = e^{s_i} A_{ij}$ | $\text{diag}(e^{\mathbf{s}})\,A$ | positive row norms |
| `PositiveColumnScale` | $W_{ij} = A_{ij}\,e^{s_j}$ | $A\,\text{diag}(e^{\mathbf{s}})$ | positive column norms |
| `PositiveSandwichScale` | $W_{ij} = e^{s_i} A_{ij} e^{s_j}$ | $D A D,\ D{=}\text{diag}(e^{\mathbf{s}})$ | symmetry + PD |
| `PositiveScalarScale` | $W_{ij} = e^s\, A_{ij}$ | $e^s\, A$ | sign pattern |

---

## Gating Mechanisms

`gated.py` provides a protocol and four concrete gate classes, plus two gated
convolutional blocks built on top of them.

### `IGatingMechanism`

`@runtime_checkable` protocol.  Any gate must implement:

```python
def forward(self, h: Tensor, x: Tensor) -> Tensor: ...
```

`h` is the hidden state; `x` is the context (some gates ignore it).

### Gate classes

| Class | Formula | Context `x` used? | Key params |
|-------|---------|-------------------|------------|
| `GLUGate` | `a ⊙ σ(b)` where `[a, b] = proj(h)` (torch-native `F.glu`) | No | `hidden_size` |
| `SwiGLUGate` | `content(h) ⊙ silu(gate(h))` | No | `hidden_size`, `bias` |
| `GRNGate` | `LayerNorm(h + dropout(content(eta1) ⊙ σ(gate(eta1))))` where `eta2 = ELU(W2(h) + ctx(x))`, `eta1 = bottleneck(eta2)` | Yes | `hidden_size`, `context_size`, `dropout` |
| `UVGate` | `σ(gate(h)) ⊙ σ(U(x)) + (1−σ(gate(h))) ⊙ σ(V(x))` | Yes | `in_features`, `hidden_size`, `activation` |

`GLUGate` and `SwiGLUGate` accept but ignore `x`; they satisfy the protocol
for uniform use in `GatedMLP`.

`SwiGLUGate` (Shazeer 2020): `bias=False` matches the paper's convention of
omitting bias on GLU-variant projections; defaults to `True` to match
`nn.Linear`'s own default.

`GRNGate` (Lim et al. 2021, TFT) — matches the paper's four-linear structure
(`W1` bottleneck, `W2`/`W3` pre-activation, `W4`/`W5` gate/content):
- `context_size=None` means `x` is expected to have `hidden_size` features.
- Explicit `context_size` projects `x` from `context_size → hidden_size`.

`UVGate` (Wang et al. 2022):
- `in_features` must match the `in_features` of the enclosing model, because
  `x` is the raw network input forwarded from `GatedMLP.forward`.

### GatedConvolutionBlock1d

```
Norm(x) → Conv1d(in → 2·out) → split → content ⊙ σ(gate) → Dropout
```

Output shape: `(B, out_channels, T')`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | `int` | required | Input channels |
| `out_channels` | `int` | required | Output channels |
| `in_timesteps` | `int` | required | Sequence length (for LayerNorm) |
| `kernel_size` | `int` | `3` | Kernel size |
| `stride` | `int` | `1` | Stride |
| `padding` | `str \| int` | `"same"` | Padding |
| `normalize` | `NormalizerName \| None` | `None` | Normalisation before conv |
| `dropout` | `float` | `0.0` | Dropout after gating |
| `dilation` | `int` | `1` | Dilation rate |
| `groups` | `int` | `1` | Grouped convolution |

### GatedDeconvolutionBlock1d

```
Norm(x) → ConvTranspose1d(in → 2·out) → split → content ⊙ σ(gate) → Dropout
```

Same parameter table as `GatedConvolutionBlock1d` plus `output_padding: int = 0`.

> **Constraint**: `padding="same"` is only valid when `stride=1`.  `ValueError`
> is raised otherwise — use an explicit integer padding for strided upsampling.
