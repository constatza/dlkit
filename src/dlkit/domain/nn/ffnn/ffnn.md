# Feed-Forward Neural Networks

`dlkit.domain.nn.ffnn` groups flat-input neural networks by architecture first.
The package distinguishes:
- residual vs plain
- dense vs constrained linear bodies
- standard vs scale-equivariant wrappers
- constant-width bodies vs embedded input/output projections

## Module layout

| File | Purpose |
|---|---|
| `linear.py` | Linear baseline |
| `simple.py` | Plain dense FFNNs without skip connections |
| `residual.py` | Residual dense FFNNs with skip connections |
| `constrained.py` | Constrained linear FFNN builders and explicit plain/residual variants |
| `scale_equivariant.py` | Class-based scale-equivariant wrappers for dense and constrained FFNNs |
| `gated.py` | Pluggable-gate feed-forward network (`GatedMLP`) |

## Variant matrix

### Dense

| Architecture | Plain | Residual | Scale-equivariant plain | Scale-equivariant residual |
|---|---|---|---|---|
| Variable-width | `SimpleFeedForwardNN` | `FeedForwardNN` | `ScaleEquivariantSimpleFeedForwardNN` | `ScaleEquivariantFeedForwardNN` |
| Constant-width | `ConstantWidthSimpleFFNN` | `ConstantWidthFFNN` | `ScaleEquivariantConstantWidthSimpleFFNN` | `ScaleEquivariantConstantWidthFFNN` |

### Constrained constant-width

| Layer family | Plain | Residual | Scale-equivariant plain | Scale-equivariant residual |
|---|---|---|---|---|
| Factorized | `ConstantWidthSimpleFactorizedFFNN` | `ConstantWidthFactorizedFFNN` | `ScaleEquivariantConstantWidthSimpleFactorizedFFNN` | `ScaleEquivariantConstantWidthFactorizedFFNN` |
| SPD | `ConstantWidthSimpleSPDFFNN` | `ConstantWidthSPDFFNN` | `ScaleEquivariantConstantWidthSimpleSPDFFNN` | `ScaleEquivariantConstantWidthSPDFFNN` |
| SPD-factorized | `ConstantWidthSimpleSPDFactorizedFFNN` | `ConstantWidthSPDFactorizedFFNN` | `ScaleEquivariantConstantWidthSimpleSPDFactorizedFFNN` | `ScaleEquivariantConstantWidthSPDFactorizedFFNN` |

### Constrained embedded

| Layer family | Plain | Residual | Scale-equivariant plain | Scale-equivariant residual |
|---|---|---|---|---|
| Factorized | `EmbeddedSimpleFactorizedFFNN` | `EmbeddedFactorizedFFNN` | `ScaleEquivariantEmbeddedSimpleFactorizedFFNN` | `ScaleEquivariantEmbeddedFactorizedFFNN` |
| SPD | `EmbeddedSimpleSPDFFNN` | `EmbeddedSPDFFNN` | `ScaleEquivariantEmbeddedSimpleSPDFFNN` | `ScaleEquivariantEmbeddedSPDFFNN` |
| SPD-factorized | `EmbeddedSimpleSPDFactorizedFFNN` | `EmbeddedSPDFactorizedFFNN` | `ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN` | `ScaleEquivariantEmbeddedSPDFactorizedFFNN` |

## Low-level constrained builders

`constrained.py` also keeps the reusable builder-oriented classes:
- `ParametricDenseBlock`
- `ConstantWidthParametricFFNN` — residual body (no `residual:` param)
- `ConstantWidthSimpleParametricFFNN` — plain body (no `residual:` param)
- `EmbeddedParametricFFNN` — residual embedded body (no `residual:` param)
- `EmbeddedSimpleParametricFFNN` — plain embedded body (no `residual:` param)

These remain available for custom compositions, but the preferred public model
surface is the explicit plain/residual class matrix above.

## Naming rules

- No `Simple` prefix means the model uses residual/skip connections.
- `Simple...` means the model is plain and does not use skip connections.
- `ConstantWidth...` means the network body is square and width-preserving.
- `Embedded...` means the network has input embedding and output projection layers.
- `ScaleEquivariant...` means the model wraps a base FFNN with norm-based
  input/output scaling.

## Shape-aware construction

The classes that naturally map dataset feature counts to `in_features` and
`out_features` implement `from_shape(shape, **kwargs)`. This includes:
- dense FFNNs
- embedded constrained FFNNs
- scale-equivariant variable-width dense FFNNs (`ScaleEquivariantFeedForwardNN`, `ScaleEquivariantSimpleFeedForwardNN`)
- scale-equivariant embedded FFNNs
- scale-equivariant constant-width dense FFNNs

When `from_shape()` is used, `shape.in_features` and `shape.out_features`
take precedence. Duplicate `in_features` or `out_features` values passed
through config kwargs are ignored rather than forwarded twice.

Square constant-width constrained bodies use `size`, so they are configured
explicitly rather than through `from_shape()`.

## Configuration guidance

For config-driven construction, prefer the top-level package export surface:

```toml
[MODEL]
name = "EmbeddedSimpleFactorizedFFNN"
module_path = "dlkit.domain.nn"
hidden_size = 64
num_layers = 3
```

```toml
[MODEL]
name = "ConstantWidthFFNN"
module_path = "dlkit.domain.nn"
hidden_size = 128
num_layers = 4
```

```toml
[MODEL]
name = "ScaleEquivariantFeedForwardNN"
module_path = "dlkit.domain.nn"
layers = [128, 128]
```

---

## Gated Networks

### GatedMLP

Feed-forward network where each hidden layer is a pluggable gating unit.  The
raw input `x` is forwarded as context into every gate, enabling
context-sensitive gates (GRN, UV) to modulate hidden states against the
original features.

**Architecture:**

```
h = Linear(x)                    # embedding, no activation
for gate, norm, drop in layers:
    h = drop(norm(gate(h, x)))   # x forwarded as context
return Linear(h)                 # output projection
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | required | Input dimension |
| `out_features` | `int` | required | Output dimension |
| `hidden_size` | `int` | required | Width of all hidden layers |
| `num_layers` | `int` | required | Number of gated hidden layers (>= 1) |
| `gate_factory` | `Callable[[], IGatingMechanism]` | required | Zero-arg factory called once per layer |
| `normalize` | `NormalizerName \| None` | `None` | Normalisation after each gate |
| `dropout` | `float` | `0.0` | Dropout after normalisation |

Raises `ValueError` if `num_layers < 1`.  Supports `from_shape(shape, **kwargs)`.

**Example — context-free gating with SwiGLU:**

```python
from dlkit.domain.nn.ffnn import GatedMLP
from dlkit.domain.nn.primitives import SwiGLUGate

model = GatedMLP(
    in_features=64,
    out_features=16,
    hidden_size=128,
    num_layers=3,
    gate_factory=lambda: SwiGLUGate(hidden_size=128),
)
```

**Example — context-sensitive gating with UVGate:**

```python
from dlkit.domain.nn.primitives import UVGate

model = GatedMLP(
    in_features=64,
    out_features=16,
    hidden_size=128,
    num_layers=3,
    gate_factory=lambda: UVGate(in_features=64, hidden_size=128),
)
```

> **Note on context-sensitive gates**: `GRNGate` and `UVGate` receive
> `x` of shape `(batch, in_features)` from `GatedMLP.forward`.  Construct
> them with `in_features` (for `UVGate`) or `context_size=in_features`
> (for `GRNGate` with explicit context) matching the model's `in_features`.
> `GLUGate` and `SwiGLUGate` ignore `x` and only require `hidden_size`.
