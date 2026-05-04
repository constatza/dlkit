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
| `spectral/` | Spectral convolution and Fourier-enhanced MLPs |
| `operators/` | Physics-informed operator networks |
| `primitives/` | Reusable low-level blocks and constrained linear layers |

## FFNN surface

The FFNN family is organized symmetrically around architecture and naming:
- `Simple...` means plain, no skip connections
- no `Simple` prefix means residual/skip connections
- `ConstantWidth...` means a square width-preserving body
- `Embedded...` means input embedding plus output projection
- `ScaleEquivariant...` means norm-scaled wrapper behavior

Representative exports from `dlkit.domain.nn` include:
- dense: `FeedForwardNN`, `SimpleFeedForwardNN`, `ConstantWidthFFNN`, `ConstantWidthSimpleFFNN`
- constrained: `ConstantWidthFactorizedFFNN`, `ConstantWidthSimpleFactorizedFFNN`, `EmbeddedSPDFFNN`, `EmbeddedSimpleSPDFFNN`
- scale-equivariant: `ScaleEquivariantConstantWidthFFNN`, `ScaleEquivariantConstantWidthSimpleFFNN`, `ScaleEquivariantEmbeddedSPDFactorizedFFNN`, `ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN`

For the full matrix, see `ffnn/ffnn.md`.

## Model factory

`dlkit.domain.nn.factory.build_model` constructs any `nn.Module` from a class
and an optional `ShapeSummary`.

- If the model exposes `from_shape(shape, **kwargs)`, the factory calls that
  explicit shape-aware constructor.
- Otherwise the factory calls the model directly with `**kwargs` and performs
  no implicit shape injection.

Built-in flat-input, operator, spectral, and 1-D convolutional entrypoints
implement `from_shape()` where dataset-driven construction is part of the
public contract.

## Parameter role contracts

Two thin modules expose the semantic vocabulary used by the engine's
optimization subsystem. Domain defines what roles exist; engine defines how
to infer them.

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
general-purpose optimizer rather than specialized heuristics.

### `IParameterRoleProvider` (`role_provider.py`)

Optional `@runtime_checkable` protocol. Models can implement it to provide an
explicit parameter-role mapping and bypass engine-side inference heuristics.

### What must not live here

- Runtime role inference logic
- Role partitioning or overlap validation
- Optimizer construction or stepping

Those concerns belong to `dlkit.engine.training.optimization`.
