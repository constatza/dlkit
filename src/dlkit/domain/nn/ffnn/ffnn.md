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

## Single-layer linear baselines (`linear.py`)

All classes are keyword-only, expose `in_features` and `out_features`, and implement `from_contract(contract, **kwargs)`.

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

| Architecture | Plain | Residual | Scale-equivariant plain | Scale-equivariant residual |
|---|---|---|---|---|
| Variable-width | `SimpleFeedForwardNN` | `FeedForwardNN` | `ScaleEquivariantSimpleFeedForwardNN` | `ScaleEquivariantFeedForwardNN` |
| Constant-width | `ConstantWidthSimpleFFNN` | `ConstantWidthFFNN` | `ScaleEquivariantConstantWidthSimpleFFNN` | `ScaleEquivariantConstantWidthFFNN` |

### Constrained — square layer types (SPD, SPDFactorized)

Square layer types (SPD, SPDFactorized) are always square. These networks expose **only `in_features`** — `hidden_size` and `out_features` are always equal to `in_features`.

All layers in these networks belong to the same constrained type; no plain `nn.Linear` appears inside the model.

**Embedded**: `StructuredLayer(n)` no-act → `[StructuredLayer(n) × (num_layers-2)]` with act → `StructuredLayer(n)` no-act. Requires `num_layers >= 2`.

**Non-embedded**: `[StructuredLayer(n) × (num_layers-1)]` with act → `StructuredLayer(n)` no-act. Requires `num_layers >= 1`.

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

## Low-level constrained builders

`constrained.py` also keeps reusable builder-oriented classes:
- `ParametricDenseBlock` — a single norm → act → layer → dropout block; accepts `in_size` when the block's input and output dimensions differ
- `EmbeddedParametricFFNN` — residual body with `Linear` embedding/regression projections (no `residual:` param)
- `EmbeddedSimpleParametricFFNN` — plain body with `Linear` embedding/regression projections (no `residual:` param)

These remain available for custom compositions. The preferred public model surface is the explicit plain/residual class matrix above.

## Naming rules

| Token | Meaning |
|---|---|
| No `Simple` prefix | residual/skip connections active |
| `Simple...` | plain, no skip connections |
| `Embedded...` | has a dedicated initial projection layer before the body |
| No `Embedded` | structured layers act directly from the input |
| `ScaleEquivariant...` | wraps a base model with norm-based input/output scaling |

For square layer types, "Embedded" means the initial projection is also a structured (SPD/Symmetric) layer without activation — not a plain `nn.Linear`.

## Contract-based construction

All constrained FFNNs implement `from_contract(contract, **kwargs)` via `TabulaRSpec`.

- **Square-type classes** (`SPDFFNN`, `EmbeddedSPDFFNN`, etc.): require `in_shape == out_shape`; extract `in_features = in_shape[0]`.
- **Rectangular-type classes** (`FactorizedFFNN`, `EmbeddedFactorizedFFNN`, etc.): extract `in_features = in_shape[0]`, `out_features = out_shape[0]`.

`from_contract` does **not** filter kwargs — passing duplicate `in_features` or `out_features` raises a `TypeError`.

## Configuration guidance

```toml
[MODEL]
name = "SPDFFNN"
module_path = "dlkit.domain.nn"
in_features = 8
num_layers = 4
```

```toml
[MODEL]
name = "EmbeddedSPDFFNN"
module_path = "dlkit.domain.nn"
in_features = 8
num_layers = 4
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

Raises `ValueError` if `num_layers < 1`. Supports `from_contract(contract, **kwargs)`.

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
