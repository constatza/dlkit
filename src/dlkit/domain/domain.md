# Domain Layer

`dlkit.domain` owns pure model-adjacent logic with no runtime orchestration.

## Current Modules
- `nn`
- `shapes`
- `transforms`
- `metrics`
- `losses`

Energy-norm metrics and the corresponding loss helpers are dense-only at the
domain boundary: callers must provide batched dense `(B, D, D)` matrices rather
than sparse tensors.

## Shapes
The shape subsystem is split into focused modules:
- `inference_engine.py`
- `inference_strategies.py`
- `specification_base.py`
- `shape_specifications.py`
- `validation_engine.py`
- `serialization_types.py`
- `shape_serializer.py`
- `shape_migrator.py`

Dataset-driven shape inference normalizes scalar tensors to `(1,)` so class-label
targets and other zero-rank samples produce valid shape entries instead of
falling back to defaults.
Serialized shape payloads use the versioned metadata wrapper; runtime deserialization
expects that canonical format.

## Transforms
The transform pipeline surface uses `TransformContext` as the runtime-facing
value object around feature/target transform chains. Capability checks are
Protocol-based only; `dlkit.domain.transforms.base` is the canonical home for
`FittableTransform`, `InvertibleTransform`, and `ShapeAwareTransform`.

### Available transforms

| Class | Algorithm | fit() | Invertible |
|-------|-----------|-------|------------|
| `StandardScaler` | z-score normalisation | full batch | yes |
| `MinMaxScaler` | min-max normalisation | full batch | yes |
| `PCA` | principal component analysis via `torch.pca_lowrank` | full batch | yes |
| `TruncatedSVD` | truncated SVD without mean-centering | full batch | yes |
| `ICA` | independent component analysis (sklearn `FastICA` in fit only) | full batch | yes |
| `IncrementalPCA` | streaming PCA via sklearn `partial_fit` | incremental | yes |
| `SampleNormL2` | per-sample L2 normalisation | no fit | yes |

`IncrementalPCA` implements `IncrementalFittableTransform` and is natively
supported by `TransformChain.fit_from_dataloader`. All other fittable transforms
use the materialising path: batches are collected, composed through prior
transforms, then passed in a single `fit(full_data)` call.

`ShapeAwareTransform.configure_shape(shape_spec: GeometrySpec, entry_name: str)`
receives a `GeometrySpec` and uses `get_shape(entry_name)` for eager buffer
pre-allocation. sklearn is used only inside `fit()` — `forward()` and
`inverse_transform()` are always pure torch with no numpy round-trips.

These modules provide value-level and algorithmic behavior that runtime services
can consume without pulling orchestration back into the domain layer.
