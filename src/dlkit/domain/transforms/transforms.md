# Domain Transforms

`dlkit.domain.transforms` provides protocol-based, chain-composable tensor transformations
built on `torch.nn.Module`. All transforms are checkpointable (fitted state persists via
`state_dict`/`load_state_dict`) and device-aware.

## Available transforms

| Class | Constructor parameters | Description |
|---|---|---|
| `StandardScaler` | `*, dim=0` | Z-score normalization: subtracts mean, divides by std |
| `MinMaxScaler` | `*, dim=0` | Scales each feature to [-1, 1] |
| `SampleNormL2` | `*, eps=1e-8, feature_dims=None` | Per-sample L2 normalization along specified feature dims |
| `SpectralRadiusNorm` | — | Divides the tensor by its spectral radius (largest singular value) |
| `PCA` | `*, n_components, n_power_iterations=2` | Reduces last dim via `torch.pca_lowrank`; requires full data in memory |
| `IncrementalPCA` | `*, n_components, batch_size=256` | Streaming PCA via sklearn `partial_fit`; use for large datasets |
| `TruncatedSVD` | `*, n_components, n_iter=4` | SVD without mean-centering; prefer for sparse or non-negative data |
| `ICA` | `*, n_components, fun="logcosh", max_iter=200, random_state=0` | FastICA via sklearn; extracts statistically independent components |
| `Permutation` | `*, dims` | Reorders tensor dimensions (e.g., `dims=(0, 2, 1)`) |
| `TensorSubset` | `*, keep, dim=1` | Slices a fixed index set along a tensor dimension |

## PCA vs alternatives

| Transform | When to use |
|---|---|
| `PCA` | Data fits comfortably in memory; fastest option (one-shot torch SVD) |
| `IncrementalPCA` | Dataset is too large for memory; streams batches via sklearn `partial_fit` |
| `TruncatedSVD` | Sparse or non-negative data where the mean offset is meaningful (no centering) |
| `ICA` | Independent source separation; optimises statistical independence, not variance |

## Protocols

| Protocol | Methods / properties | Description |
|---|---|---|
| `FittableTransform` | `fit(data)`, `fitted: bool` | Transform that learns statistics from training data before use |
| `IncrementalFittableTransform` | `reset_fit_state()`, `update_fit(batch)`, `finalize_fit()` | Streaming fit without materialising the full dataset |
| `InvertibleTransform` | `inverse_transform(x)` | Transform that can be reversed (e.g. for target de-normalisation at inference) |
| `ShapeAwareTransform` | `configure_shape(shape_spec, entry_name)` | Receives shape information for eager buffer allocation (performance optimisation) |

All protocols are `runtime_checkable`; use `isinstance(t, FittableTransform)` to inspect
capabilities at runtime.

## TOML configuration

`module_path` can be omitted from any transform entry; the engine defaults to
`dlkit.domain.transforms`. Only specify `module_path` for custom transforms defined in
other modules.

### Example 1 — StandardScaler + PCA on features

```toml
[[DATASET.features]]
name = "x"
path = "data/features.npy"

[[DATASET.features.transforms]]
name = "StandardScaler"

[[DATASET.features.transforms]]
name = "PCA"
n_components = 50
```

### Example 2 — PCA on targets

```toml
[[DATASET.targets]]
name = "y"
path = "data/targets.npy"

[[DATASET.targets.transforms]]
name = "PCA"
n_components = 10
```

### Example 3 — PCA on both features and targets simultaneously

```toml
[[DATASET.features]]
name = "x"
path = "data/features.npy"

[[DATASET.features.transforms]]
name = "StandardScaler"

[[DATASET.features.transforms]]
name = "PCA"
n_components = 50

[[DATASET.targets]]
name = "y"
path = "data/targets.npy"

[[DATASET.targets.transforms]]
name = "PCA"
n_components = 10
```

### Example 4 — IncrementalPCA for a large features dataset

```toml
[[DATASET.features]]
name = "x"
path = "data/large_features.npy"

[[DATASET.features.transforms]]
name = "IncrementalPCA"
n_components = 50
batch_size = 512
```

## Adding a custom transform

Subclass `Transform` and implement `forward()`. If the transform participates in
geometry/shape inference, also implement `infer_output_shape(self, in_shape)`:

```python
import torch
from dlkit.domain.transforms import Transform


class MyTransform(Transform):
    def __init__(self, *, n_components: int) -> None:
        super().__init__()
        self.n_components = n_components

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : self.n_components]

    def infer_output_shape(self, in_shape: tuple[int, ...]) -> tuple[int, ...]:
        return in_shape[:-1] + (self.n_components,)
```

Optionally implement any subset of the protocol methods (`fit`, `inverse_transform`,
`configure_shape`, `reset_fit_state` / `update_fit` / `finalize_fit`) — the engine
detects capabilities via `isinstance` checks against the runtime-checkable protocols.

## TransformChain

`TransformChain` composes an ordered sequence of transforms into a single `nn.Module`.
It handles fitting (sequential `fit` / incremental `update_fit` calls), forward application,
and inverse traversal in reverse order.

```python
from dlkit.domain.transforms import TransformChain, StandardScaler, PCA

chain = TransformChain([StandardScaler(), PCA(n_components=32)])
chain.fit(train_data)      # fits each FittableTransform in order
output = chain(train_data) # applies all transforms sequentially
```
