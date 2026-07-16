# Feed-Forward Neural Networks

`dlkit.domain.nn.ffnn` groups flat-input neural networks by architecture.
The package distinguishes:
- residual vs plain
- dense vs factorized linear bodies
- standard vs scale-equivariant wrappers
- projected vs body-only structured variants

The plain-`nn.Linear` embedding/regression layers in `residual.py` and
`film.py` initialize their weights via `domain.nn.init.initialize_`, matched to
the network's `activation` — see `nn.md`.

## Module layout

| File | Purpose |
|---|---|
| `linear.py` | Linear baselines: `LinearNetwork` and single-layer parametrized variants |
| `residual.py` | Dense FFNNs with `skip` and `project` selectors |
| `constrained.py` | Factorized FFNNs with `skip` and `project` selectors |
| `hyper_moe.py` | Hyper-Connection and sparse-MoE FFNN composites |
| `scale_equivariant.py` | Class-based scale-equivariant wrappers for dense and constrained FFNNs |
| `gated.py` | Pluggable-gate feed-forward network (`GatedMLP`) |
| `film.py` | FiLM-conditioned FFNNs: `FiLMBlock`, `FiLMResidualBlock`, `VarWidthFiLMFFNN`, `FiLMFFNN`, `FiLMEmbeddedFFNN` |

## Single-layer linear baselines (`linear.py`)

All classes are keyword-only, expose `in_features` and `out_features`, and implement `from_entries(input_shapes, output_shapes, **kwargs)`.

| Class | Primitive | Constraint | Shape |
|---|---|---|---|
| `LinearNetwork` | `nn.Linear` | none | rectangular |
| `FactorizedLinearNetwork` | `FactorizedLinear` | row-wise scale factorization | rectangular |

## Variant matrix

### Dense

| Architecture | Plain (`skip=False`) | Residual (`skip=True`, default) | Scale-equivariant |
|---|---|---|---|
| Variable-width | `VarWidthFFNN(skip=False)` | `VarWidthFFNN` | — |
| Constant-width | `FFNN(skip=False)` | `FFNN` | `ScaleEquivariantFFNN` |
| Body-only constant-width | `FFNN(skip=False, project=False)` | `FFNN(project=False)` | — |

### Constrained — Factorized layer types

Public factorized FFNNs use the exponential `FactorizedLinear` primitive only.
If a model family name contains `Factorized`, every linear kernel owned by that
model is a `FactorizedLinear`. Names without `Factorized` use ordinary
`nn.Linear` kernels for their owned projections.

Each factorized layer uses an explicit positive row scale on top of a base
weight matrix. In `FactorizedLinear`, the effective weight is
`exp(log_scale).unsqueeze(1) * base_weight`, matching the paper-style random
weight factorization `diag(exp(s)) @ V`. Public factorized architecture
constructors expose `mean` and `std` for the factorized log-scale
initialisation while keeping `kaiming_a` internal.

Public factorized model rules:
- `FactorizedFFNN(skip=True)` keeps the non-embedded first-block → body → head
  shape; `skip=False` removes residual connections from the body.
- `EmbeddedFactorizedFFNN(project=True)` uses factorized input/output
  projections around a constant-width body; `project=False` removes those
  projections and requires `in_features == out_features == hidden_size`.
- Hyper/MoE composites use `EmbeddedHyperFFNN` and `EmbeddedMoEFFNN`.
  `linear_kind="linear" | "factorized"` selects kernels, and `project=False`
  selects the body-only square form.

This matches Shazeer-style sparsely gated MoE, GShard, and Switch Transformer
semantics: top-k routers select FFN experts, and Transformer-style residual
placement wraps the routed FFN sublayer/block rather than making each expert a
full embedded FFNN.

| Variant | Plain | Residual/stacked | Notes |
|---|---|---|---|
| **Factorized non-embedded** | `FactorizedFFNN(skip=False)` | `FactorizedFFNN` | first block bridges to hidden width |
| **Factorized projected/body-only** | `EmbeddedFactorizedFFNN(skip=False, project=...)` | `EmbeddedFactorizedFFNN(project=...)` | `project=False` is square body-only |
| **Hyper linear/factorized** | — | `EmbeddedHyperFFNN(linear_kind=...)` | `project=False` is square body-only |
| **MoE linear/factorized** | — | `EmbeddedMoEFFNN(linear_kind=...)` | `project=False` is square body-only |

Scale-equivariant public wrappers are kept for dense FFNNs, FiLM FFNNs, and
the projected/body-only factorized surface:
`ScaleEquivariantEmbeddedFactorizedFFNN(skip=..., project=...)`.

> Note: `VarWidthFFNN` and `FFNN` both accept `skip: bool = True`. Pass `skip=False` to get plain (no skip connection) behavior without needing a separate class.
> They also accept `project: bool = True`; pass `project=False` for a body-only
> stack when `in_features`/`out_features` already match the body widths.
> Their hidden transition block is selected by `block_kind: DenseBlockKind`.
> The default `"dense"` keeps the single-projection dense branch; `"mlp"`,
> `"glu"`, `"geglu"`, and `"swiglu"` select bibliography-style FFN/gated FFN
> blocks. `linear_kind` selects the projection kernel: `"linear"` by default,
> or `"factorized"` for `FactorizedLinear` projections inside the selected
> block topology. Python callers may pass `block_factory` for a custom module
> factory with the same `in_features`/`out_features` constructor contract.

Dense shape intuition:

```text
VarWidthFFNN
  (B, in_features)
  -> embed to layers[0]
  -> layers[1], ..., layers[-1]
  -> (B, out_features)

FFNN
  (B, in_features)
  -> embed to hidden_size
  -> constant-width hidden transitions repeated by num_layers
  -> (B, out_features)

FFNN(project=False)
  (B, hidden_size)
  -> constant-width hidden transitions repeated by num_layers
  -> (B, hidden_size)
```

## Low-level constrained builders

`constrained.py` also keeps reusable builder-oriented classes:
- `ParametricDenseBlock` — a single norm → act → injected-layer → dropout block with the same `in_features`/`out_features` constructor contract as the other dense primitives
The preferred public model surface is `FactorizedFFNN` and
`EmbeddedFactorizedFFNN`.

The Hyper/MoE composites also expose `block_kind` for the
wrapped hidden branch or expert block. Their default is `"parametric"` to keep
the existing linear-kernel branch shape; pass `"mlp"`, `"geglu"`, or `"swiglu"`
to use Transformer/MoE-style FFN experts. Use `linear_kind="factorized"` to
make the selected topology use `FactorizedLinear` kernels.

## Naming rules

| Token | Meaning |
|---|---|
| `VarWidth...` | explicit per-layer width list required (`layers: Sequence[int]`) |
| no width prefix | constant-width implied — specify `hidden_size` + `num_layers` |
| `skip=False` | plain, no skip connections |
| no `Simple` prefix | residual/skip connections active (`skip=True` default) |
| `project=True` | has dedicated input/output projection layers around the body |
| `project=False` | structured layers act directly from the input and require square body dimensions |
| `ScaleEquivariant...` | wraps a base model with norm-based input/output scaling |

For constrained layer types, `Embedded...` means the class can project around a
body; the `project` flag decides whether those projection layers are present.

Unless stated otherwise, `num_layers` counts learned hidden blocks on the model's main path. Dedicated embedding/setup layers and terminal readout layers are excluded from that count.

## Shape-based construction

All constrained FFNNs implement `from_entries(input_shapes, output_shapes, **kwargs)` where
`input_shapes` and `output_shapes` are `Mapping[str, tuple[int, ...]]`.

- **Body-only mode** (`project=False`): requires the first input and output
  shapes to be equal.
- **Projected mode** (`project=True`): extracts `in_features` from the first
  input shape and `out_features` from the first output shape.

`from_entries` does **not** filter kwargs — passing duplicate `in_features` or `out_features` raises a `TypeError`.

## Configuration guidance

```toml
[model]
name = "EmbeddedFactorizedFFNN"
module_path = "dlkit.domain.nn"
in_features = 64
out_features = 64
project = false
num_layers = 3
```

```toml
[model]
name = "EmbeddedHyperFFNN"
module_path = "dlkit.domain.nn"
hidden_size = 64
num_layers = 3
num_lanes = 4
linear_kind = "factorized"
```

```toml
[model]
name = "EmbeddedMoEFFNN"
module_path = "dlkit.domain.nn"
hidden_size = 64
num_layers = 3
num_experts = 8
top_k = 2
linear_kind = "factorized"
```

---

## Gated Networks

### GatedMLP

Feed-forward network where each hidden layer is a pluggable gating unit. The
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

Raises `ValueError` if `num_layers < 1`. Supports `from_entries(input_shapes, output_shapes, **kwargs)`.

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

---

## FiLM-conditioned Networks

All FiLM classes accept a conditioning vector alongside the primary input via `forward(x, condition)`.
The FiLM modulation formula is `(1 + γ(c)) * x + β(c)` where `γ` and `β` are linear projections of
the condition, zero-initialised so the layer is identity at the start of training.

### Variant matrix

| Class | Body style | Mirrors |
|---|---|---|
| `VarWidthFiLMFFNN` | Variable-width (`layers` list); embedding `Linear` → N `FiLMBlock`s → output `Linear` | `VarWidthFFNN` |
| `FiLMFFNN` | Constant-width (`hidden_size`, `num_layers`); embedding `Linear` → N `FiLMBlock`s → output `Linear` | `FFNN` |
| `FiLMEmbeddedFFNN` | Constant-width residual body; `Linear` embed → N `FiLMResidualBlock`s (each scaled by `residual_branch_scale(num_layers)`, per-block skip) wrapped in `ConditionedResidualSequential` (identity end-to-end skip) → `Linear` head | `FFNN(project=True)` |
| `ScaleEquivariantVarWidthFiLMFFNN` | `ConditionedScaleEquivariantWrapper` around `VarWidthFiLMFFNN`; `f(αx, c) = α·f(x, c)` | `VarWidthFiLMFFNN` |
| `ScaleEquivariantFiLMFFNN` | `ConditionedScaleEquivariantWrapper` around `FiLMFFNN`; `f(αx, c) = α·f(x, c)` | `ScaleEquivariantFFNN` |
| `ScaleEquivariantFiLMEmbeddedFFNN` | `ConditionedScaleEquivariantWrapper` around `FiLMEmbeddedFFNN` | — |

Scale equivariance applies to the features branch only; the condition vector passes through unchanged.

### Low-level FiLM blocks

| Class | Role |
|---|---|
| `FiLMBlock` | Single dense block (`Norm → Act → Lin → Drop`) followed by `FiLMLayer` modulation |
| `FiLMResidualBlock` | Two dense blocks + `FiLMLayer` + identity residual skip (square: `in_features == out_features`); `branch_scale` (default `1.0`) multiplies the FiLM-modulated branch before adding the shortcut — `FiLMEmbeddedFFNN` sets this to `residual_branch_scale(num_layers)` on every block it constructs |

### Parameters

All FiLM network classes require `condition_dim` in addition to the standard FFNN knobs.

| Parameter | Applies to | Description |
|---|---|---|
| `condition_dim` | all FiLM classes | Dimensionality of the external conditioning vector |
| `layers` | `VarWidthFiLMFFNN`, `ScaleEquivariantVarWidthFiLMFFNN` | Explicit per-layer width list (same role as `VarWidthFFNN`) |
| `hidden_size` | `FiLMFFNN`, `ScaleEquivariantFiLMFFNN`, `FiLMEmbeddedFFNN`, `ScaleEquivariantFiLMEmbeddedFFNN` | Constant hidden width |
| `num_layers` | `FiLMFFNN`, `ScaleEquivariantFiLMFFNN` | Number of hidden `FiLMBlock` transitions (>= 1) |
| `num_layers` | `FiLMEmbeddedFFNN`, `ScaleEquivariantFiLMEmbeddedFFNN` | Number of `FiLMResidualBlock`s in the body (>= 1) |

### Shape-based construction

All FiLM network classes implement `from_entries(input_shapes, output_shapes, condition_dim, **kwargs)`:

```python
from dlkit.domain.nn.ffnn.film import FiLMFFNN

model = FiLMFFNN.from_entries(
    {"x": (16,)}, {"y": (4,)}, condition_dim=8, hidden_size=64, num_layers=3
)
```

`from_entries` extracts `in_features` and `out_features` from the first input and output shapes; passing them again as kwargs raises `TypeError`.

### Configuration guidance

```toml
[model]
name = "FiLMFFNN"
module_path = "dlkit.domain.nn.ffnn"
condition_dim = 8
hidden_size = 64
num_layers = 3
```

```toml
[model]
name = "VarWidthFiLMFFNN"
module_path = "dlkit.domain.nn.ffnn"
condition_dim = 8
layers = [128, 64, 64]
```

```toml
[model]
name = "FiLMEmbeddedFFNN"
module_path = "dlkit.domain.nn.ffnn"
condition_dim = 8
hidden_size = 64
num_layers = 4
```
