# Domain Neural Networks

`dlkit.domain.nn` owns pure ML logic: model families, factory, and semantic
parameter contracts. No runtime orchestration or engine concerns belong here.

---

## Model Families

| Sub-package | Contents |
|---|---|
| `ffnn/` | Feed-forward networks (FFNN, autoencoder variants) |
| `cae/` | Convolutional autoencoders |
| `encoder/` | Encoder modules |
| `attention/` | Self-attention and transformer blocks |
| `graph/` | PyG-based graph neural networks (`BaseGraphNetwork`) |
| `generative/` | VAE and generative samplers |
| `spectral/` | Spectral convolution / Fourier layers |
| `operators/` | Physics-informed operator networks |
| `primitives/` | Reusable building blocks (MLP, ResBlock, …) |

---

## Model Factory

`dlkit.domain.nn.factory.build_model` constructs any `nn.Module` from a class
and a `ShapeSummary`. It selects one of four strategies automatically:

| Strategy | Condition |
|---|---|
| **no-shape** | `shape is None` — passes kwargs only |
| **lazy** | `LazyModuleMixin` subclass — injects output-dim alias |
| **ffnn** | constructor has `in_features` / `input_dim` / `input_size` |
| **conv** | constructor has `in_channels` |

Explicit user kwargs always override shape-injected values.

---

## Parameter Role Contracts

Two thin modules expose the semantic vocabulary used by the engine's optimization
subsystem. Domain defines *what roles exist*; engine defines *how to infer them*.

### `ParameterRole` (`parameter_roles.py`)

```python
class ParameterRole(Enum):
    INPUT        # First-layer weights / input projections
    HIDDEN       # Interior weight matrices
    OUTPUT       # Final-layer weights / heads
    BIAS         # Any bias vector
    NORMALIZATION  # LayerNorm / BatchNorm scale and shift
    EMBEDDING    # Token or positional embedding tables
    ENCODER      # Encoder-specific weights in AE / seq2seq
    DECODER      # Decoder-specific weights
    UNKNOWN      # Could not be classified → defaults to conservative optimizer
```

`UNKNOWN` is the safe fallback: the engine assigns `UNKNOWN` parameters to the
general-purpose optimizer (AdamW) rather than specialised ones like Muon.

### `IParameterRoleProvider` (`role_provider.py`)

Optional `@runtime_checkable` protocol. Implement it on any model to bypass
engine-side role-inference heuristics:

```python
class MyModel(nn.Module, IParameterRoleProvider):
    def parameter_roles(self) -> dict[str, ParameterRole]:
        return {
            "encoder.weight": ParameterRole.ENCODER,
            "decoder.weight": ParameterRole.DECODER,
            "head.weight":    ParameterRole.OUTPUT,
        }
```

When the engine's `OptimizationProgramBuilder` assembles a live program it
checks `isinstance(model, IParameterRoleProvider)` first. If the model
implements the protocol its declared roles take precedence; engine-side
inference strategies (`CompositeParameterRoleInferenceStrategy`) run only as
a fallback for models that do not.

### What must NOT go here

- Role inference logic (inspects live `nn.Module` at runtime → engine concern)
- Role assignment / partitioning / overlap validation (engine concern)
- Any optimizer construction or stepping

These all live in `dlkit.engine.training.optimization`.
