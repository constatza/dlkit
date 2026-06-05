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
- `VarWidth...` means explicit per-layer width list required; no prefix means constant-width (`hidden_size` + `num_layers`)
- `Simple...` means plain, no skip connections; no `Simple` prefix means residual/skip connections active
- `FFNN` and `VarWidthFFNN` both accept `skip: bool = True` — use `skip=False` instead of a separate `Simple*` class
- `Embedded...` means the network has a dedicated initial projection layer before the body; `EmbeddedFFNN` is the dense constant-width version
- `ScaleEquivariant...` means norm-scaled wrapper behavior
- Square layer types (SPD, SPDFactorized) expose only `in_features`; rectangular types (Factorized) expose `in_features`, `hidden_size`, and `out_features`

Representative exports from `dlkit.domain.nn` include:
- dense: `VarWidthFFNN`, `FFNN`, `EmbeddedFFNN`
- constrained SPD (square): `SPDFFNN`, `SimpleSPDFFNN`, `EmbeddedSPDFFNN`, `EmbeddedSimpleSPDFFNN`
- constrained Factorized (rectangular): `FactorizedFFNN`, `SimpleFactorizedFFNN`, `EmbeddedFactorizedFFNN`, `EmbeddedSimpleFactorizedFFNN`
- scale-equivariant: `ScaleEquivariantFFNN`, `ScaleEquivariantSPDFFNN`, `ScaleEquivariantEmbeddedSPDFactorizedFFNN`, `ScaleEquivariantFactorizedFFNN`
- gated: `GatedMLP`

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

Representative exports from `dlkit.domain.nn` include:
- residual: `GATv2Projection`, `ScaledGATv2Projection`
- plain: `SimpleGATv2Projection`, `ScaledSimpleGATv2Projection`

For the full matrix, see `graph/graph.md`.

## Model factory

`dlkit.domain.nn.factory.build_model` constructs any `nn.Module` from a class,
optional `ModelContractSpec`, and `**kwargs`.

- If the model implements `ContractConsumer` (exposes `from_contract(contract, **kwargs)`),
  the factory calls that contract-aware constructor.
- Otherwise the factory calls the model directly with `**kwargs`.

Built-in model families implement `from_contract()` accepting the appropriate
`ModelContractSpec` variant (`TabulaRSpec`, `GridOperatorSpec`, `SequenceSpec`,
`BranchTrunkSpec`, or `GraphContractSpec`).

## Contract ↔ geometry mapping

`contract_resolver.resolve_contract(geometry, output_shapes)` maps a `GeometrySpec`
to the right `ModelContractSpec` variant.  The dispatch table:

| Geometry kind | Has TARGET_COORDINATES? | Contract produced |
|---------------|------------------------|-------------------|
| `TABULAR` | no | `TabulaRSpec(in_shape, out_shape)` |
| `TABULAR` | yes | `BranchTrunkSpec(branch_shape, query_shape, out_features)` |
| `SEQUENCE` | — | `SequenceSpec(in_channels, seq_len, out_channels)` |
| `REGULAR_GRID` / `POINT_CLOUD` | no | `GridOperatorSpec(in_channels, out_channels, spatial_shape)` |
| `REGULAR_GRID` / `POINT_CLOUD` | yes | `BranchTrunkSpec(...)` |
| `GRAPH` | — | `GraphContractSpec(in_channels, out_channels, edge_dim)` |

`output_shapes` must contain the target field shapes in config order; the first
element supplies the output dimension(s).  For tabular models the engine infers
`out_shape` from the dataset at training time — if none is available at inference
the resolver raises `WorkflowError` (prefer retraining over silent fallbacks).

For `BranchTrunkSpec`, the output-shape rules follow DeepONet semantics:
- canonical query-mode scalar targets: `(n_queries,)` -> `out_features = 1`
- canonical query-mode vector targets: `(n_queries, out_features)` -> `out_features = out_features`
- paired single-query targets: `(out_features,)` -> `out_features = out_features`

## Contract checkpoint serialization

Contracts are serialized to `dlkit_metadata["contract"]` in Lightning checkpoints.
This makes checkpoints self-sufficient: inference reconstructs the exact
`from_contract(contract, **kwargs)` call without re-running geometry inference.

```python
from dlkit.domain.nn.contracts import serialize_contract, deserialize_contract, TabulaRSpec

spec = TabulaRSpec(in_shape=(16,), out_shape=(4,))
data = serialize_contract(spec)
# {"_type": "TabulaRSpec", "in_shape": (16,), "out_shape": (4,)}

restored = deserialize_contract(data)
assert restored == spec
```

`deserialize_contract` handles list→tuple coercion from JSON round-trips and
returns `None` for unrecognized `_type` values rather than raising.

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
