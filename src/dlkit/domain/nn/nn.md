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
| `primitives/` | Reusable low-level blocks, constrained linear layers, and gating mechanisms |

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
- scale-equivariant: `ScaleEquivariantFeedForwardNN`, `ScaleEquivariantSimpleFeedForwardNN`, `ScaleEquivariantConstantWidthFFNN`, `ScaleEquivariantConstantWidthSimpleFFNN`, `ScaleEquivariantEmbeddedSPDFactorizedFFNN`, `ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN`
- gated: `GatedMLP`

For the full matrix, see `ffnn/ffnn.md`.

## Residual/plain family matrix

| Family | Residual | Plain |
|---|---|---|
| Dense (variable-width) | `FeedForwardNN` | `SimpleFeedForwardNN` |
| Dense (constant-width) | `ConstantWidthFFNN` | `ConstantWidthSimpleFFNN` |
| Graph (GAT) | `GATv2Projection`, `ScaledGATv2Projection` | `SimpleGATv2Projection`, `ScaledSimpleGATv2Projection` |

## Graph NN surface

The graph family follows the same residual/plain naming convention as FFNN:
- No `Simple` prefix means residual connections active
- `Simple...` prefix means plain, no residual connections
- `Scaled...` means column-wise input scaling applied

Representative exports from `dlkit.domain.nn` include:
- residual: `GATv2Projection`, `ScaledGATv2Projection`
- plain: `SimpleGATv2Projection`, `ScaledSimpleGATv2Projection`

For the full matrix, see `graph/graph.md`.

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
