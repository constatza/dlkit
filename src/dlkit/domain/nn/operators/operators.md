# Neural Operator Architectures

Networks that approximate operators mapping between infinite-dimensional function spaces.

## Why neural operators?

Standard neural networks approximate functions ℝⁿ → ℝᵐ.  A *neural operator*
approximates an *operator* 𝒢 : 𝒰 → 𝒱 mapping one function to another, e.g.
the solution operator of a PDE.  Key property: the approximation is
discretisation-invariant — a model trained on a coarse grid generalises to
finer grids at inference time.

## Interface hierarchy

```
IOperatorNetwork          (capability marker)
  ├── IGridOperator       forward(x) → y         [FNO]
  └── IQueryOperator      forward(u, y) → v      [DeepONet]
```

`IGridOperator` and `IQueryOperator` have incompatible `forward` signatures,
so they cannot share a callable protocol without violating LSP.
`IOperatorNetwork` is the shared marker for `isinstance` checks.

## Module layout

| Symbol | File | Description |
|--------|------|-------------|
| `IOperatorNetwork` | `base.py` | Marker protocol: is this a neural operator? |
| `IGridOperator` | `base.py` | Protocol for grid-to-grid operators |
| `IQueryOperator` | `base.py` | Protocol for query-point operators |
| `FourierNeuralOperator1d` | `fno.py` | FNO for 1-D spatial domains |
| `DeepONet` | `deeponet.py` | Branch + trunk operator network |

---

## FourierNeuralOperator1d

Implements `IGridOperator`.  Input and output live on the same spatial grid.

### Architecture

```
Lifting   : Conv1d(in_channels → width, 1)
Body      : n_layers × FourierLayer(width, n_modes)   [from spectral/]
Projection: Conv1d(width → out_channels, 1)
```

Each `FourierLayer` performs:
```
y = activation(SpectralConv1d(x) + Conv1d(x, kernel=1))
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | `int` | required | Input channel count |
| `out_channels` | `int` | required | Output channel count |
| `width` | `int` | `64` | Latent channel width throughout the body |
| `n_modes` | `int` | `16` | Fourier modes retained per layer |
| `n_layers` | `int` | `4` | Number of FNO body blocks |
| `activation` | `Callable` | `F.gelu` | Pointwise activation |

### Example

```python
from dlkit.domain.nn.operators import FourierNeuralOperator1d

model = FourierNeuralOperator1d(
    in_channels=2,   # e.g. [u(x), x]
    out_channels=1,
    width=64,
    n_modes=16,
    n_layers=4,
)
y = model(x)  # x: (B, 2, L) → y: (B, 1, L)
```

### TOML configuration

```toml
[model]
name = "FourierNeuralOperator1d"
out_channels = 1
width = 64
n_modes = 16
n_layers = 4
```

`FourierNeuralOperator1d` implements `from_shape(shape, **kwargs)`, so the
shared model factory builds it explicitly from the dataset-derived channel
summary.

---

## DeepONet

Implements `IQueryOperator`.  Evaluates the operator at arbitrary query coordinates.

### Architecture

```
branch_out = BranchNet(u)         # FeedForwardNN: (B, n_sensors) → (B, trunk_width * out_features)
trunk_out  = TrunkNet(y_i)        # FeedForwardNN: (n_queries, n_coords) → (n_queries, trunk_width * out_features)
v_i        = Σ branch_out * trunk_out / trunk_width + bias
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | required | Sensor count (branch input) |
| `out_features` | `int` | required | Output values per query point |
| `n_coords` | `int` | required | Spatial coordinate dimension for trunk |
| `trunk_width` | `int` | `64` | Dot-product latent space per output channel |
| `branch_depth` | `int` | `4` | Hidden layers in branch net |
| `trunk_depth` | `int` | `4` | Hidden layers in trunk net |
| `hidden_size` | `int` | `128` | Width of hidden layers in both sub-networks |
| `activation` | `Callable` | `F.gelu` | Activation for both sub-networks |

### Multi-input DataEntry configuration

DeepONet requires two inputs.  Use positional `model_input` values:

```toml
[[data.features]]
name = "u"
model_input = 0   # branch input

[[data.features]]
name = "y"
model_input = 1   # trunk input (query coordinates)
```

`MLPDeepONet` implements `from_shape(shape, **kwargs)` and expects the shape
summary to contain both input entries in branch/trunk order.

### Example

```python
from dlkit.domain.nn.operators import DeepONet

model = DeepONet(
    in_features=100,   # 100 sensor locations
    out_features=1,
    n_coords=1,        # 1-D query coordinate
    trunk_width=64,
    hidden_size=128,
)
v = model(u, y)   # u: (B, 100), y: (B, Q, 1) → v: (B, Q, 1)
```

---

## Using both architectures with NormScaledFFNN

Both FNO and DeepONet can be wrapped in `NormScaledFFNN` for scale-equivariant
problems, since they contain nonlinear activations.

```python
from dlkit.domain.nn.ffnn.norm_scaled import NormScaledFFNN
from dlkit.domain.nn.operators import FourierNeuralOperator1d

model = NormScaledFFNN(
    base_model=FourierNeuralOperator1d(in_channels=1, out_channels=1),
    norm="l2",
)
```
