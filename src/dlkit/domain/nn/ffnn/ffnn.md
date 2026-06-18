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
| `SymmetricLinearNetwork` | `SymmetricLinear` | W = Wᵀ | **square only** |
| `SPDLinearNetwork` | `SPDLinear` | W symmetric positive-definite | **square only** |
| `SymmetricFactorizedLinearNetwork` | `SymmetricFactorizedLinear` | W = D·Sym(A)·D | **square only** |
| `SPDFactorizedLinearNetwork` | `SPDFactorizedLinear` | W = D·SPD(A)·D | **square only** |

Square-only classes raise `ValueError` at construction if `in_features != out_features`.

## Variant matrix

### Dense

| Architecture | Plain (`skip=False`) | Residual (`skip=True`, default) | Scale-equivariant |
|---|---|---|---|
| Variable-width | `VarWidthFFNN(skip=False)` | `VarWidthFFNN` | — |
| Constant-width | `FFNN(skip=False)` | `FFNN` | `ScaleEquivariantFFNN` |
| Embedded constant-width | — | `EmbeddedFFNN` | — |

### Constrained — square layer types (SPD, SPDFactorized)

Square layer types (SPD, SPDFactorized) are always square. These networks expose **only `in_features`** — `hidden_size` and `out_features` are always equal to `in_features`.

All layers in these networks belong to the same constrained type; no plain `nn.Linear` appears inside the model.

**Embedded**: `StructuredLayer(n)` no-act → `[StructuredLayer(n) × num_layers]` with act → `StructuredLayer(n)` no-act. Requires `num_layers >= 0`.

**Non-embedded**: `[StructuredLayer(n) × num_layers]` with act → `StructuredLayer(n)` no-act. Requires `num_layers >= 0`.

| Layer family | Plain (non-embedded) | Residual (non-embedded) | Plain (embedded) | Residual (embedded) |
|---|---|---|---|---|
| SPD | `SimpleSPDFFNN` | `SPDFFNN` | `EmbeddedSimpleSPDFFNN` | `EmbeddedSPDFFNN` |
| SPD-factorized | `SimpleSPDFactorizedFFNN` | `SPDFactorizedFFNN` | `EmbeddedSimpleSPDFactorizedFFNN` | `EmbeddedSPDFactorizedFFNN` |

Scale-equivariant wrappers follow the same naming: `ScaleEquivariant[Embedded]SPDFFNN` etc.

### Constrained — rectangular layer types (Factorized)

Factorized layers can be rectangular. These networks expose `in_features`, `hidden_size`, and `out_features` as independent parameters.

**Embedded**: `Linear(in→h)` → `[FactorizedLinear(h→h) × num_layers]` with act → `Linear(h→out)`.

**Non-embedded**: `FactorizedLinear(in→h)` (no skip) → `[FactorizedLinear(h→h) × (num_layers-1)]` with act → `Linear(h→out)`.

| Variant | Plain (non-embedded) | Residual (non-embedded) | Plain (embedded) | Residual (embedded) |
|---|---|---|---|---|
| Factorized | `SimpleFactorizedFFNN` | `FactorizedFFNN` | `EmbeddedSimpleFactorizedFFNN` | `EmbeddedFactorizedFFNN` |

Scale-equivariant wrappers: `ScaleEquivariant[Embedded]FactorizedFFNN`, `ScaleEquivariantSimple[Embedded]FactorizedFFNN`.

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

For square layer types, "Embedded" means the initial projection is also a structured (SPD/Symmetric) layer without activation — not a plain `nn.Linear`.

Unless stated otherwise, `num_layers` counts learned hidden blocks on the model's main path. Dedicated embedding/setup layers and terminal readout layers are excluded from that count.

## Shape-based construction

All constrained FFNNs implement `from_entries(input_shapes, output_shapes, **kwargs)` where
`input_shapes` and `output_shapes` are `Mapping[str, tuple[int, ...]]`.

- **Square-type classes** (`SPDFFNN`, `EmbeddedSPDFFNN`, etc.): require the first input and output shapes to be equal; extract `in_features` from the input shape's leading dim.
- **Rectangular-type classes** (`FactorizedFFNN`, `EmbeddedFactorizedFFNN`, etc.): extract `in_features` from the first input shape and `out_features` from the first output shape.

`from_entries` does **not** filter kwargs — passing duplicate `in_features` or `out_features` raises a `TypeError`.

## Configuration guidance

```toml
[MODEL]
name = "SPDFFNN"
module_path = "dlkit.domain.nn"
in_features = 8
num_layers = 3
```

```toml
[MODEL]
name = "EmbeddedSPDFFNN"
module_path = "dlkit.domain.nn"
in_features = 8
num_layers = 2
```

```toml
[MODEL]
name = "FactorizedFFNN"
module_path = "dlkit.domain.nn"
hidden_size = 64
num_layers = 3
```

```toml
[MODEL]
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
[MODEL]
name = "FiLMFFNN"
module_path = "dlkit.domain.nn.ffnn"
condition_dim = 8
hidden_size = 64
num_layers = 3
```

```toml
[MODEL]
name = "VarWidthFiLMFFNN"
module_path = "dlkit.domain.nn.ffnn"
condition_dim = 8
layers = [128, 64, 64]
```

```toml
[MODEL]
name = "FiLMEmbeddedFFNN"
module_path = "dlkit.domain.nn.ffnn"
condition_dim = 8
hidden_size = 64
num_layers = 4
```
