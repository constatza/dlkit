# Domain Neural Networks

`dlkit.domain.nn` owns pure ML logic: model families, factory helpers, and
semantic parameter contracts. No engine orchestration belongs here.

## Model families

| Sub-package | Contents |
|---|---|
| `ffnn/` | Dense and constrained feed-forward networks, including plain/residual and scale-equivariant variants |
| `cae/` | Convolutional autoencoders |
| `encoder/` | Encoder modules |
| `attention/` | Self-attention and transformer blocks |
| `graph/` | PyG-based graph neural networks |
| `generative/` | VAE and generative samplers |
| `spectral/` | Spectral convolution, Fourier-enhanced models, and coordinate spectral-bias networks |
| `operators/` | Physics-informed operator networks |
| `primitives/` | Reusable low-level blocks, constrained linear layers, gating mechanisms, and conditioning primitives |

## Conditioned network primitives

`primitives/conditioning.py` provides the building blocks for externally-conditioned modules:

| Class | Role |
|---|---|
| `IConditionedModule` | Abstract base; enforces `forward(x, condition) -> Tensor` |
| `AsConditioned` | Adapter that wraps an unconditional `nn.Module` to satisfy `IConditionedModule`; discards the condition |
| `FiLMLayer` | Feature-wise Linear Modulation: `(1 + γ(c)) * x + β(c)`; zero-initialised so it is identity at init |
| `ConditionedSequential` | Sequential chain forwarding the same condition to every block |
| `ConditionedResidualSequential` | `ConditionedSequential` with an end-to-end skip connection (`body(x, c) + shortcut(x)`) |

`primitives/skip.py` also contains `ResidualSequential` — an unconditional sequential with an end-to-end skip: `chain(x) + shortcut(x)`.

`primitives/scale_equivariant.py` contains `ConditionedScaleEquivariantWrapper`, which extends the norm-scaling pattern to conditioned modules: `base_model(x / ||x||, condition) * ||x||`. Scale equivariance applies to the features input only; the condition passes through unchanged.

## FFNN surface

The FFNN family is organized symmetrically around architecture and naming:
- `VarWidth...` means explicit per-layer width list required; no prefix means constant-width (`hidden_size` + `num_layers`)
- `Simple...` means plain, no skip connections; no `Simple` prefix means residual/skip connections active
- `FFNN` and `VarWidthFFNN` both accept `skip: bool = True` — use `skip=False` instead of a separate `Simple*` class
- `Embedded...` means the network has a dedicated initial projection layer before the body; `EmbeddedFFNN` is the dense constant-width version
- `ScaleEquivariant...` means norm-scaled wrapper behavior
- Square layer types (SPD, SPDFactorized) expose only `in_features`; rectangular types (Factorized) expose `in_features`, `hidden_size`, and `out_features`
- `num_layers` counts learned hidden blocks on the model's main path; pure embedding and readout projections are not included

Representative exports from `dlkit.domain.nn` include:
- dense: `VarWidthFFNN`, `FFNN`, `EmbeddedFFNN`
- FiLM-conditioned: `VarWidthFiLMFFNN`, `FiLMFFNN`, `FiLMEmbeddedFFNN`, `ScaleEquivariantVarWidthFiLMFFNN`, `ScaleEquivariantFiLMFFNN`, `ScaleEquivariantFiLMEmbeddedFFNN`
- constrained SPD (square): `SPDFFNN`, `SimpleSPDFFNN`, `EmbeddedSPDFFNN`, `EmbeddedSimpleSPDFFNN`
- constrained Factorized (rectangular): `FactorizedFFNN`, `SimpleFactorizedFFNN`, `EmbeddedFactorizedFFNN`, `EmbeddedSimpleFactorizedFFNN`
- constrained Factorized (square constant-width): `ConstantWidthFactorizedFFNN`, `ConstantWidthSimpleFactorizedFFNN`, `ScaleEquivariantConstantWidthFactorizedFFNN`, `ScaleEquivariantConstantWidthSimpleFactorizedFFNN`
- coordinate spectral-bias: `FourierFeatureNetwork`, `FactorizedFourierFeatureNetwork`, `Siren`, `ModifiedMLP`, `ScaleEquivariantFourierFeatureNetwork`, `ScaleEquivariantFactorizedFourierFeatureNetwork`
- scale-equivariant: `ScaleEquivariantFFNN`, `ScaleEquivariantSPDFFNN`, `ScaleEquivariantEmbeddedSPDFactorizedFFNN`, `ScaleEquivariantFactorizedFFNN`
- gated: `GatedMLP`

`dlkit.nn` is the user-facing shim for this non-graph NN surface and re-exports the same
families, including the FiLM-conditioned variants above.

For the full matrix, see `ffnn/ffnn.md`.

## Residual/plain family matrix

| Family | Residual (default) | Plain (`skip=False`) |
|---|---|---|
| Dense (variable-width) | `VarWidthFFNN` | `VarWidthFFNN(skip=False)` |
| Dense (constant-width) | `FFNN` | `FFNN(skip=False)` |
| Graph (GAT) | `GATv2Projection`, `ScaledGATv2Projection` | `SimpleGATv2Projection`, `ScaledSimpleGATv2Projection` |

## Graph NN surface

The graph family follows the same residual/plain naming convention as FFNN:
- No `Simple` prefix means residual connections active
- `Simple...` prefix means plain, no residual connections
- `Scaled...` means column-wise input scaling applied

Representative exports from `dlkit.gnn` and `dlkit.domain.nn.graph` include:
- residual: `GATv2Projection`, `ScaledGATv2Projection`
- plain: `SimpleGATv2Projection`, `ScaledSimpleGATv2Projection`

`dlkit.domain.nn` and `dlkit.nn` intentionally do not re-export graph classes.
That keeps broad non-graph imports free of optional PyG side effects.

For the full matrix, see `graph/graph.md`.

## Model factory

`dlkit.domain.nn.factory.build_model` constructs any `nn.Module`:

- If both `input_shapes` and `output_shapes` are provided and the model implements
  `from_entries` (i.e. inherits `StandardEntryConsumer`), it calls
  `model_cls.from_entries(input_shapes, output_shapes, **kwargs)`.
- Otherwise it calls `model_cls(**kwargs)` directly.

## Shape-providing protocol

All built-in model families implement the **entry-consumer pattern** defined in
`contracts.py`. Understanding it is essential when adding a new model family.

### Three components

| Component | Where | Role |
|-----------|-------|------|
| `InputSpec` | per-model inner class | Declares expected entry names (field names = `forward` arg names) |
| `_constructor_dims` | classmethod hook | Maps `(InputShapes, OutputShapes)` → `{constructor_kwarg: int}` |
| `_SHAPE_KWARG_NAMES` | class attribute | Names the kwargs that `_constructor_dims` supplies (for checkpoint stripping) |

### `StandardEntryConsumer` — the base mixin

Provides a sealed `from_entries` classmethod (Template Method pattern).

```python
class StandardEntryConsumer:
    _SHAPE_KWARG_NAMES: frozenset[str] = frozenset({"in_features", "out_features"})

    @classmethod
    def _constructor_dims(cls, input_shapes, output_shapes) -> dict[str, int]:
        # Default: take the first feature dim and first output dim.
        return {
            "in_features": next(iter(input_shapes.values()))[0],
            "out_features": next(iter(output_shapes.values()))[0],
        }

    @classmethod
    def from_entries(cls, input_shapes, output_shapes, **kwargs) -> Self:
        # 1. Validate that all entries declared in InputSpec are present.
        # 2. Extract constructor dims via the hook.
        # 3. Construct cls(**dims, **kwargs).
        ...
```

`SquareEntryConsumer` is a subclass whose `_constructor_dims` validates that
`in_shape == out_shape` (used for SPD-family models).

### Entry name validation (early error before PyTorch runs)

`from_entries` checks `InputSpec.model_fields` against the available
`input_shapes` keys **before** calling `_constructor_dims`. If a required entry
is missing, it raises immediately:

```
ValueError: MySPDModel requires entries ['x'] but only ['y'] are available
```

### Shape dimensionality validation in `_constructor_dims`

`_constructor_dims` is also the right place to validate the **rank** of an input
shape — e.g. a spectral model that requires at least a 2-D sample `(C, L)` would
fail with a cryptic PyTorch error deep in the forward pass if fed a 1-D sample.
Catch it here instead:

```python
@classmethod
def _constructor_dims(cls, input_shapes, output_shapes):
    in_shape = next(iter(input_shapes.values()))
    if len(in_shape) < 2:
        raise ValueError(
            f"{cls.__name__} requires at least 2-D input shape (C, L) "
            f"but got shape {in_shape} — check your feature entry configuration."
        )
    return {"in_channels": in_shape[0], "seq_len": in_shape[1], ...}
```

Any shape invariant a model needs (minimum rank, minimum size, parity, square
constraint) should be validated in `_constructor_dims`, not in `__init__`. That
way the engine catches the problem at model-build time, before any tensor ever
flows through the network.

### Adding a new model family

```python
class MyModel(StandardEntryConsumer, nn.Module):
    # 1. Declare what entries forward() expects.
    class InputSpec(InputSpec):
        x: Shape         # forward(self, x: Tensor)
        context: Shape   # forward(self, x: Tensor, context: Tensor)

    # 2. Override _SHAPE_KWARG_NAMES to match your constructor kwargs.
    _SHAPE_KWARG_NAMES: frozenset[str] = frozenset({
        "in_features", "context_dim", "out_features"
    })

    # 3. Override _constructor_dims to extract and VALIDATE dims.
    @classmethod
    def _constructor_dims(cls, input_shapes, output_shapes):
        x_shape = input_shapes["x"]
        ctx_shape = input_shapes["context"]
        if len(x_shape) != 1:
            raise ValueError(
                f"{cls.__name__} requires 1-D 'x' shape but got {x_shape}"
            )
        return {
            "in_features": x_shape[0],
            "context_dim": ctx_shape[0],
            "out_features": next(iter(output_shapes.values()))[0],
        }

    def __init__(self, *, in_features, context_dim, out_features, ...): ...
```

Rules:
- `InputSpec` field names must match `forward` parameter names exactly.
- `_SHAPE_KWARG_NAMES` must list every key that `_constructor_dims` returns.
- Shape rank/size validation belongs in `_constructor_dims`.
- `__init__` may repeat simple range checks (`if n < 0: raise`) but should not
  re-validate shape contracts — that's `_constructor_dims`' job.

### Checkpoint round-trip

`engine/inference/model_builder.py` strips `_SHAPE_KWARG_NAMES` from the
checkpoint hyperparams before calling `from_entries`, preventing "got multiple
values for keyword argument" errors when both the checkpoint and `_constructor_dims`
supply the same key. If you add a new kwarg that comes from shapes, add it to
`_SHAPE_KWARG_NAMES` or the round-trip will break.

## Attention blocks

`attention.transformer.TransformerEncoderBlock` keeps its current public
constructor surface but disables the nested-tensor fast path whenever the live
encoder-layer configuration cannot use it. With the current pre-LN
(`norm_first=True`) encoder setup, that means nested tensors are disabled
explicitly instead of relying on PyTorch to warn at runtime.

## Parameter role contracts

Domain defines the semantic vocabulary used by the engine's optimization
subsystem. Runtime classification belongs to the engine.

### `ParameterRole` (`parameter_roles.py`)

```python
class ParameterRole(Enum):
    INPUT
    HIDDEN
    OUTPUT
    BIAS
    NORMALIZATION
    EMBEDDING
    ENCODER
    DECODER
    UNKNOWN
```

`UNKNOWN` is the safe fallback: the engine assigns those parameters to the
general-purpose optimizer rather than the Muon-specialized path.

The default classifier is graph-based and model-agnostic:
- it uses official `nn.Module` structure APIs
- it uses `torch.fx` to classify executed parameter-owning sites
- it does not require model-side protocols or naming conventions
- it traces through composite wrappers to the fundamental parameter-owning
  sublayers that actually sit on the input/output boundary
- ambiguous or unsupported cases remain `UNKNOWN`

### What must not live here

- Runtime role inference logic
- Role partitioning or overlap validation
- Optimizer construction or stepping

Those concerns belong to `dlkit.engine.training.optimization`.
