# Feed-Forward Neural Networks

`dlkit.domain.nn.ffnn` groups flat-input neural networks by architecture.
The package distinguishes:
- residual vs plain
- dense vs constrained linear bodies (SPD, SPDFactorized, Factorized)
- standard vs scale-equivariant wrappers
- embedded vs non-embedded structured variants

## Module layout

| File | Purpose |
|---|---|
| `linear.py` | Linear baselines: `LinearNetwork` and single-layer parametrized variants |
| `simple.py` | Plain dense FFNNs without skip connections |
| `residual.py` | Residual dense FFNNs with skip connections |
| `constrained.py` | Constrained linear FFNN builders and explicit plain/residual variants |
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
| Embedded constant-width | — | `EmbeddedFFNN` | — |

### Constrained — Factorized layer types

Factorized layers can be rectangular. These networks expose `in_features`, `hidden_size`, and `out_features` as independent parameters.

Each factorized layer uses an explicit positive scaling factor on top of a base
weight matrix. In `FactorizedLinear`, the effective weight is
`exp(log_scale).unsqueeze(1) * base_weight`, matching the paper-style random
weight factorization `diag(exp(s)) @ V`. The public rectangular `Factorized*`
family uses this exponential parameterization by default with literal RWF
initialization semantics (`mean=mu`, `std=sigma`, default `mu=0.0`,
`sigma=0.1` (unit scale at init: `exp(0) = 1.0`)). A softplus-based alternative remains an advanced low-level
primitive, not the public FFNN default.

**Embedded**: `Linear(in→h)` → `[FactorizedLinear(h→h) × num_layers]` with act → `Linear(h→out)`.

**Non-embedded**: `FactorizedLinear(in→h)` (no skip) → `[FactorizedLinear(h→h) × (num_layers-1)]` with act → `Linear(h→out)`.

**Constant-width (square)**: `[FactorizedLinear(n→n) × num_layers]` all-residual — **no** embedding or regression `nn.Linear`. Every layer including the last is `FactorizedLinear`. Requires `in==out`. For asymmetric inputs use the Embedded variants instead.

| Variant | Plain | Residual | Notes |
|---|---|---|---|
| Non-embedded (rectangular) | `SimpleFactorizedFFNN` | `FactorizedFFNN` | `first_block(in→h, no skip)` + body + `Linear(h→out)` |
| Embedded (rectangular) | `EmbeddedSimpleFactorizedFFNN` | `EmbeddedFactorizedFFNN` | `Linear(in→h)` + body + `Linear(h→out)` |
| Embedded softplus (rectangular) | `EmbeddedSimpleSoftplusFactorizedFFNN` | `EmbeddedSoftplusFactorizedFFNN` | same as Embedded but body uses `SoftplusFactorizedLinear` |
| **FactorizedEnd exp (rectangular)** | `EmbeddedSimpleFactorizedEndFFNN` | `EmbeddedFactorizedEndFFNN` | `Linear(in→h)` + body + `FactorizedLinear(h→out)` |
| **FactorizedEnd softplus (rectangular)** | `EmbeddedSimpleSoftplusFactorizedEndFFNN` | `EmbeddedSoftplusFactorizedEndFFNN` | `Linear(in→h)` + softplus body + `SoftplusFactorizedLinear(h→out)` |
| **FullyFactorized exp (rectangular)** | `EmbeddedSimpleFullyFactorizedFFNN` | `EmbeddedFullyFactorizedFFNN` | `FactorizedLinear(in→h)` + body + `FactorizedLinear(h→out)`; no plain `nn.Linear` |
| **FullyFactorized softplus (rectangular)** | `EmbeddedSimpleFullySoftplusFactorizedFFNN` | `EmbeddedFullySoftplusFactorizedFFNN` | `SoftplusFactorizedLinear(in→h)` + softplus body + `SoftplusFactorizedLinear(h→out)`; no plain `nn.Linear` |
| **Constant-width exp (square)** | `ConstantWidthSimpleFactorizedFFNN` | `ConstantWidthFactorizedFFNN` | pure body; `in==out`; `exp` scale fn; GELU default |
| **Constant-width softplus (square)** | — | `ConstantWidthSoftplusFactorizedFFNN` | pure body; `in==out`; `softplus` scale fn; unit-scale correction at init |

Scale-equivariant wrappers:
- Rectangular exp: `ScaleEquivariantFactorizedFFNN`, `ScaleEquivariantSimpleFactorizedFFNN`, `ScaleEquivariantEmbeddedFactorizedFFNN`, `ScaleEquivariantEmbeddedSimpleFactorizedFFNN`
- Rectangular softplus: `ScaleEquivariantEmbeddedSoftplusFactorizedFFNN`, `ScaleEquivariantEmbeddedSimpleSoftplusFactorizedFFNN`
- FactorizedEnd exp: `ScaleEquivariantEmbeddedFactorizedEndFFNN`, `ScaleEquivariantEmbeddedSimpleFactorizedEndFFNN`
- FactorizedEnd softplus: `ScaleEquivariantEmbeddedSoftplusFactorizedEndFFNN`, `ScaleEquivariantEmbeddedSimpleSoftplusFactorizedEndFFNN`
- FullyFactorized exp: `ScaleEquivariantEmbeddedFullyFactorizedFFNN`, `ScaleEquivariantEmbeddedSimpleFullyFactorizedFFNN`
- FullyFactorized softplus: `ScaleEquivariantEmbeddedFullySoftplusFactorizedFFNN`, `ScaleEquivariantEmbeddedSimpleFullySoftplusFactorizedFFNN`
- Square exp: `ScaleEquivariantConstantWidthFactorizedFFNN`, `ScaleEquivariantConstantWidthSimpleFactorizedFFNN`
- Square softplus: `ScaleEquivariantConstantWidthSoftplusFactorizedFFNN`

> Note: `VarWidthFFNN` and `FFNN` both accept `skip: bool = True`. Pass `skip=False` to get plain (no skip connection) behavior without needing a separate class.

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

EmbeddedFFNN
  (B, in_features)
  -> input embedding to hidden_size
  -> fixed-width residual body repeated by num_layers
  -> output projection to (B, out_features)
```

## Low-level constrained builders

`constrained.py` also keeps reusable builder-oriented classes:
- `ParametricDenseBlock` — a single norm → act → layer → dropout block; accepts `in_size` when the block's input and output dimensions differ
- `EmbeddedParametricFFNN` — residual body with `Linear` embedding/regression projections (no `residual:` param)
- `EmbeddedSimpleParametricFFNN` — plain body with `Linear` embedding/regression projections (no `residual:` param)

These remain available for custom compositions. The preferred public model surface is the explicit plain/residual class matrix above.

## Embedded dense convenience model

`EmbeddedFFNN` is the standard embedded dense residual variant used by the
DeepONet operator family.

Architecture:

```text
Linear(in_features -> hidden_size)
-> residual constant-width body
-> Linear(hidden_size -> out_features)
```

It exposes the same dense-network knobs as `FFNN`:
- `in_features`
- `out_features`
- `hidden_size`
- `num_layers`
- optional `activation`, `normalize`, `dropout`, `bias`

## Naming rules

| Token | Meaning |
|---|---|
| `VarWidth...` | explicit per-layer width list required (`layers: Sequence[int]`) |
| no width prefix | constant-width implied — specify `hidden_size` + `num_layers` |
| `Simple...` | plain, no skip connections (or use `skip=False` on `FFNN`/`VarWidthFFNN`) |
| no `Simple` prefix | residual/skip connections active (`skip=True` default) |
| `Embedded...` | has a dedicated initial linear projection layer before the body |
| no `Embedded` prefix | structured layers act directly from the input |
| `ScaleEquivariant...` | wraps a base model with norm-based input/output scaling |

For constrained layer types, "Embedded" means the network has a dedicated initial projection layer before the body (and a regression layer after it).

Unless stated otherwise, `num_layers` counts learned hidden blocks on the model's main path. Dedicated embedding/setup layers and terminal readout layers are excluded from that count.

## Shape-based construction

All constrained FFNNs implement `from_entries(input_shapes, output_shapes, **kwargs)` where
`input_shapes` and `output_shapes` are `Mapping[str, tuple[int, ...]]`.

- **Square-type classes** (`SPDFFNN`, `EmbeddedSPDFFNN`, etc.): require the first input and output shapes to be equal; extract `in_features` from the input shape's leading dim.
- **Rectangular-type classes** (`FactorizedFFNN`, `EmbeddedFactorizedFFNN`, etc.): extract `in_features` from the first input shape and `out_features` from the first output shape.

`from_entries` does **not** filter kwargs — passing duplicate `in_features` or `out_features` raises a `TypeError`.

## Configuration guidance

```toml
[model]
name = "SPDFFNN"
module_path = "dlkit.domain.nn"
in_features = 8
num_layers = 3
```

```toml
[model]
name = "EmbeddedSPDFFNN"
module_path = "dlkit.domain.nn"
in_features = 8
num_layers = 2
```

```toml
[model]
name = "FactorizedFFNN"
module_path = "dlkit.domain.nn"
hidden_size = 64
num_layers = 3
```

```toml
[model]
name = "EmbeddedFactorizedFFNN"
module_path = "dlkit.domain.nn"
hidden_size = 64
num_layers = 3
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
| `FiLMEmbeddedFFNN` | Constant-width residual body; `Linear` embed → N `FiLMResidualBlock`s (with per-block skip) wrapped in `ConditionedResidualSequential` (end-to-end skip) → `Linear` head | `EmbeddedFFNN` |
| `ScaleEquivariantVarWidthFiLMFFNN` | `ConditionedScaleEquivariantWrapper` around `VarWidthFiLMFFNN`; `f(αx, c) = α·f(x, c)` | `VarWidthFiLMFFNN` |
| `ScaleEquivariantFiLMFFNN` | `ConditionedScaleEquivariantWrapper` around `FiLMFFNN`; `f(αx, c) = α·f(x, c)` | `ScaleEquivariantFFNN` |
| `ScaleEquivariantFiLMEmbeddedFFNN` | `ConditionedScaleEquivariantWrapper` around `FiLMEmbeddedFFNN` | — |

Scale equivariance applies to the features branch only; the condition vector passes through unchanged.

### Low-level FiLM blocks

| Class | Role |
|---|---|
| `FiLMBlock` | Single dense block (`Norm → Act → Lin → Drop`) followed by `FiLMLayer` modulation |
| `FiLMResidualBlock` | Two dense blocks + `FiLMLayer` + identity residual skip (square: `in_features == out_features`) |

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
