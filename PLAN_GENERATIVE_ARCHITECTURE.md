# Flow Matching / Generative Architecture Plan

## Context

Flow matching (FM) requires a principled extension of DLKit's abstractions — not
just new implementations, but new **interfaces** that capture what is fundamentally
different about generative workflows:

1. **Stochastic coupled transforms**: supervision builder creates both the new
   feature `(xt, t)` AND the new target `(ut,)` from a single source `x1` —
   something the current per-slot `Transform` framework cannot express.

2. **Generator lifecycle**: randomness must be explicit, seedable, and routed
   differently in training (global RNG), validation (fixed seed per batch), and
   generation (user-controlled).

3. **Generation ≠ prediction**: inference runs an iterative ODE integrator, not a
   single forward pass. The `IPredictor` interface is wrong for this; we need a
   separate `IGenerator` protocol.

The strategy is:
- **Introduce new interfaces** where behaviour is fundamentally different
  (`IBatchTransform`, `IGeneratorFactory`, `IGenerator`)
- **Generalise with a new intermediate base wrapper** (`GenerativeLightningWrapper`)
  that any generative model type can extend
- **Implement FM as a special case** under those interfaces, placed inside
  `core/models/nn/generative/` alongside other model families (ffnn, cae, graph…)
- **Zero breaking changes** to existing code

---

## Fundamental Differences vs Standard Supervised Learning

| Aspect | Standard DLKit | Flow Matching |
|--------|---------------|---------------|
| Dataset | `(x, y)` pairs | `x1` samples only |
| Transform target | Per-slot, independent, deterministic | Coupled: `x1 → (xt,t)` + `(ut,)` simultaneously |
| Randomness | None in hot path | Stochastic per batch (noise + time sampling) |
| Inference | One forward pass | Iterative ODE integration loop |
| Loss target | Label from dataset | Computed dynamically from `(x0, x1, t)` |
| Generator control | Not needed | Different per phase (train/val/generate) |

---

## New Interfaces (Phase 1 — generalise first)

### `core/training/transforms/interfaces.py` — additions only

```python
class IBatchTransform(Protocol):
    """Coupled transform operating on an entire Batch.

    Captures the case where features and targets must be transformed
    together (e.g. stochastic supervision builders, multi-modal augmentation).
    Receives an optional Generator for reproducible randomness.
    """
    def __call__(
        self,
        batch: Batch,
        generator: torch.Generator | None = None,
    ) -> Batch: ...


class IGeneratorFactory(Protocol):
    """Single responsibility: produce a torch.Generator (or None) per batch.

    Decouples generator policy from wrapper logic.
    Three canonical implementations: Null, Deterministic, Fixed.
    """
    def __call__(self, batch_idx: int) -> torch.Generator | None: ...
```

### `core/models/wrappers/generator_factories.py` — new file (alongside wrappers)

```python
# Three implementations — each has one responsibility

class NullGeneratorFactory:
    """Returns None → PyTorch global RNG. Used during training."""
    def __call__(self, batch_idx: int) -> None:
        return None

class DeterministicGeneratorFactory:
    """Returns fresh seeded generator. Used during validation."""
    def __init__(self, seed: int = 42) -> None: ...
    def __call__(self, batch_idx: int) -> torch.Generator:
        return torch.Generator().manual_seed(self._seed + batch_idx)

class FixedGeneratorFactory:
    """Returns the same generator every call. Used for exact reproduction."""
    def __init__(self, generator: torch.Generator) -> None: ...
    def __call__(self, batch_idx: int) -> torch.Generator:
        return self._generator
```

### `interfaces/inference/protocols/generative.py` — new file (ISP: separate from IPredictor)

```python
@dataclass(frozen=True)
class GenerationResult:
    samples: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class IGenerator(Protocol):
    """Protocol for generative inference (generation from noise).

    Intentionally separate from IPredictor — discriminative and generative
    inference are fundamentally different contracts.
    """
    def generate(
        self,
        n_samples: int,
        *,
        context: dict[str, Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> GenerationResult: ...

    def is_loaded(self) -> bool: ...
    def unload(self) -> None: ...
```

---

## New Intermediate Wrapper: `GenerativeLightningWrapper`

### `core/models/wrappers/generative.py` — new file

```python
class GenerativeLightningWrapper(ProcessingLightningWrapper):
    """Intermediate base for generative models.

    Single responsibility: apply IBatchTransform pipeline before model invocation,
    routing the correct IGeneratorFactory per training phase.

    Does NOT know about FM paths, noise schedules, or ODE solvers.
    Extends ProcessingLightningWrapper directly — not StandardLightningWrapper,
    since generative models use batch transforms, not per-slot transform chains.
    """

    def __init__(
        self,
        *args,
        batch_transforms: list[IBatchTransform],
        train_generator_factory: IGeneratorFactory,
        val_generator_factory: IGeneratorFactory,
        **kwargs,
    ) -> None: ...

    # Pure fold over batch transforms — functional, no mutation
    def _apply_batch_transforms(
        self,
        batch: Batch,
        generator: torch.Generator | None,
    ) -> Batch:
        return reduce(lambda b, t: t(b, generator), self._batch_transforms, batch)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        gen = self._train_generator_factory(batch_idx)
        batch = self._apply_batch_transforms(batch, gen)
        predictions = self._invoke_model(batch)
        loss = self._compute_loss(predictions, batch.targets)
        self._log_stage_outputs("train", loss)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        gen = self._val_generator_factory(batch_idx)
        batch = self._apply_batch_transforms(batch, gen)
        predictions = self._invoke_model(batch)
        loss = self._compute_loss(predictions, batch.targets)
        self._log_stage_outputs("val", loss)
        self._update_metrics(predictions, batch.targets, "val")

    # left abstract — generative subclasses define generation strategy
    @abstractmethod
    def predict_step(
        self, batch: Batch, batch_idx: int
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]: ...
```

---

## Flow Matching Implementation — `core/models/nn/generative/`

Flow matching components are a model family, placed alongside `ffnn/`, `cae/`,
`graph/`, etc. under `core/models/nn/`.

### Package layout

```
core/models/nn/generative/
  __init__.py
  interfaces.py          # narrow FM-specific protocols (ITimeSampler, INoiseSampler, …)
  functions/
    __init__.py
    broadcast.py         # _broadcast_time() — pure helper
    paths.py             # linear_path(), noise_schedule_path() — pure
    targets.py           # displacement_target() — pure
    loss.py              # velocity_mse() — pure
    solvers.py           # euler_step(), heun_step() — pure
  samplers/
    __init__.py
    time.py              # UniformTimeSampler(nn.Module)
    noise.py             # GaussianNoiseSampler(nn.Module)
  paths.py               # LinearInterpolationPath, NoiseSchedulePath (nn.Module)
  targets.py             # DisplacementTarget, ScheduleTarget (nn.Module)
  adapters.py            # KwargContextAdapter, PositionalContextAdapter (nn.Module)
  supervision.py         # SupervisionBuilder(IBatchTransform, nn.Module)
  sampler.py             # FlowMatchSampler(nn.Module)
```

### `interfaces.py` — narrow FM protocols (ISP — one method each)

```python
class ITimeSampler(Protocol):
    def __call__(self, x: Tensor, generator: Generator | None = None) -> Tensor: ...

class INoiseSampler(Protocol):
    def __call__(self, x: Tensor, generator: Generator | None = None) -> Tensor: ...

class IInterpolationPath(Protocol):
    """Pure: (x0, x1, t) -> xt."""
    def __call__(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor: ...

class IVelocityTarget(Protocol):
    """Pure: (x0, x1, xt, t) -> ut."""
    def __call__(
        self, x0: Tensor, x1: Tensor, xt: Tensor, t: Tensor
    ) -> Tensor: ...

class IModelAdapter(Protocol):
    """Normalize velocity model calling convention."""
    def __call__(
        self,
        model: nn.Module,
        xt: Tensor,
        t: Tensor,
        context: dict[str, Tensor] | None,
    ) -> Tensor: ...

class IFixedStepSolver(Protocol):
    """Pure: one ODE step."""
    def __call__(
        self,
        model_fn: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
        t: Tensor,
        dt: float,
    ) -> Tensor: ...
```

### `functions/` — pure functions only (no classes, no side effects)

```python
# broadcast.py
def _broadcast_time(t: Tensor, ref: Tensor) -> Tensor:
    """Expand t:[B] to [B, 1, ...] matching ref.ndim."""
    return t.view(t.shape[0], *([1] * (ref.ndim - 1)))

# paths.py
def linear_path(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    t_b = _broadcast_time(t, x1)
    return (1.0 - t_b) * x0 + t_b * x1

def noise_schedule_path(x1: Tensor, eps: Tensor, t: Tensor) -> Tensor:
    t_b = _broadcast_time(t, x1)
    return torch.cos(t_b * torch.pi / 2) * x1 + torch.sin(t_b * torch.pi / 2) * eps

# targets.py
def displacement_target(x0: Tensor, x1: Tensor, xt: Tensor, t: Tensor) -> Tensor:
    return x1 - x0

# loss.py
def velocity_mse(vt: Tensor, ut: Tensor) -> Tensor:
    return F.mse_loss(vt, ut)

# solvers.py — pure ODE step functions; model_fn injected as argument (DIP)
def euler_step(
    model_fn: Callable[[Tensor, Tensor], Tensor],
    x: Tensor, t: Tensor, dt: float,
) -> Tensor:
    return x + dt * model_fn(x, t)

def heun_step(
    model_fn: Callable[[Tensor, Tensor], Tensor],
    x: Tensor, t: Tensor, dt: float,
) -> Tensor:
    v1 = model_fn(x, t)
    x_mid = x + dt * v1
    v2 = model_fn(x_mid, t + dt)
    return x + (dt / 2.0) * (v1 + v2)
```

### `samplers/` — nn.Module wrappers, thin delegates to pure functions

```python
class GaussianNoiseSampler(nn.Module):
    def forward(self, x: Tensor, generator: Generator | None = None) -> Tensor:
        return torch.randn_like(x, generator=generator)

class UniformTimeSampler(nn.Module):
    def forward(self, x: Tensor, generator: Generator | None = None) -> Tensor:
        return torch.rand(x.shape[0], device=x.device, dtype=x.dtype, generator=generator)
```

### `paths.py` — thin delegate modules

```python
class LinearInterpolationPath(nn.Module):
    def forward(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        return linear_path(x0, x1, t)  # pure function

class NoiseSchedulePath(nn.Module):
    def forward(self, x1: Tensor, eps: Tensor, t: Tensor) -> Tensor:
        return noise_schedule_path(x1, eps, t)
```

### `adapters.py`

```python
class KwargContextAdapter(nn.Module):
    def __call__(self, model, xt, t, context) -> Tensor:
        return model(xt, t, **(context or {}))

class PositionalContextAdapter(nn.Module):
    def __call__(self, model, xt, t, context) -> Tensor:
        return model(xt, t, context)
```

### `supervision.py` — `SupervisionBuilder` implements `IBatchTransform`

```python
class SupervisionBuilder(nn.Module):
    """Single responsibility: transform Batch(features=(x1,)) into
    Batch(features=(xt, t_broadcast), targets=(ut,)).

    Composes four injected components; all math delegated to them.
    Implements IBatchTransform structurally (Protocol duck-typing).
    """
    def __init__(
        self,
        time_sampler: ITimeSampler,
        noise_sampler: INoiseSampler,
        path: IInterpolationPath,
        target: IVelocityTarget,
    ) -> None: ...

    def forward(
        self,
        batch: Batch,
        generator: torch.Generator | None = None,
    ) -> Batch:
        x1 = batch.features[0]
        t  = self._time_sampler(x1, generator)
        x0 = self._noise_sampler(x1, generator)
        xt = self._path(x0, x1, t)
        ut = self._target(x0, x1, xt, t)
        return Batch(features=(xt, _broadcast_time(t, x1)), targets=(ut,))
```

### `sampler.py` — `FlowMatchSampler` (model injected, not owned)

```python
class FlowMatchSampler(nn.Module):
    """Single responsibility: integrate ODE from x0~N(0,I) to x1_hat.

    Receives velocity model as a call argument — does NOT own it.
    This decouples integration config from model serialization.
    """
    def __init__(
        self,
        adapter: IModelAdapter,
        x0_sampler: INoiseSampler,
        solver: IFixedStepSolver,
        n_steps: int = 100,
    ) -> None: ...

    def forward(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        context: dict[str, Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        device = next(model.parameters()).device
        dtype  = next(model.parameters()).dtype
        x = self._x0_sampler(torch.empty(shape, device=device, dtype=dtype), generator)
        dt = 1.0 / self._n_steps
        t_grid = torch.linspace(0.0, 1.0 - dt, self._n_steps, device=device)
        for t_val in t_grid:
            t = torch.full((shape[0],), t_val, device=device, dtype=dtype)
            x = self._solver(lambda x, t: self._adapter(model, x, t, context), x, t, dt)
        return x
```

---

## `FlowMatchingLightningWrapper`

### `core/models/wrappers/flowmatching.py` — new file

```python
class FlowMatchingLightningWrapper(GenerativeLightningWrapper):
    """Flow matching training + generation wrapper.

    Single responsibility: FM-specific loss override + ODE-based predict_step.
    Batch transform and generator routing fully inherited from GenerativeLightningWrapper.
    """
    def __init__(
        self,
        *args,
        sampler: FlowMatchSampler,
        velocity_loss: Callable[[Tensor, Tensor], Tensor] = velocity_mse,
        val_seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            train_generator_factory=NullGeneratorFactory(),
            val_generator_factory=DeterministicGeneratorFactory(val_seed),
            **kwargs,
        )
        self._sampler = sampler
        self._velocity_loss = velocity_loss
        self._data_shape: tuple[int, ...] | None = None

    def _compute_loss(self, predictions: Tensor, targets: tuple[Tensor, ...]) -> Tensor:
        return self._velocity_loss(predictions, targets[0])

    def on_fit_start(self) -> None:
        for batch in self.trainer.datamodule.train_dataloader():
            self._data_shape = batch.features[0].shape[1:]
            break

    def predict_step(
        self, batch: Batch, batch_idx: int
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
        n = batch.features[0].shape[0]
        assert self._data_shape is not None
        x1_hat = self._sampler(self.model, (n, *self._data_shape))
        return (x1_hat,), batch.targets, ()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["dlkit_metadata"]["flow_matching"] = {
            "n_steps": self._sampler._n_steps,
            "data_shape": list(self._data_shape or []),
        }
```

---

## Config

### `tools/config/flowmatching_settings.py` — new file

```python
class FlowMatchingSettings(BaseModel):
    enabled: bool = False
    path_type: Literal["linear", "noise_schedule"] = "linear"
    target_type: Literal["displacement", "schedule"] = "displacement"
    solver: Literal["euler", "heun"] = "euler"
    n_inference_steps: int = 100
    val_seed: int = 42
```

### `tools/config/general_settings.py` — add optional field

```python
FLOW_MATCHING: FlowMatchingSettings = Field(default_factory=FlowMatchingSettings)
```

---

## BuildFactory Extension

### `runtime/factories/build_factory.py` — add `FlowMatchingBuildStrategy`

```python
class FlowMatchingBuildStrategy(IBuildStrategy):
    """Single responsibility: build FM components from config.

    Delegates all dataset/datamodule/shape logic to FlexibleBuildStrategy,
    overrides only the wrapper construction step.
    """

    def can_handle(self, settings: GeneralSettings) -> bool:
        return settings.FLOW_MATCHING.enabled

    def _build_core(self, settings: GeneralSettings) -> BuildComponents:
        components = FlexibleBuildStrategy()._build_core(settings)
        fm_model = WrapperFactory.create_flow_matching_wrapper(
            model_settings=settings.MODEL,
            settings=_build_wrapper_settings(settings),
            entry_configs=components.meta["entry_configs"],
            shape_summary=components.shape_spec,
            supervision_builder=_build_supervision_builder(settings.FLOW_MATCHING),
            sampler=_build_sampler(settings.FLOW_MATCHING),
        )
        return replace(components, model=fm_model)


# Pure builder functions — single responsibility each
def _build_supervision_builder(fm: FlowMatchingSettings) -> SupervisionBuilder:
    path   = LinearInterpolationPath() if fm.path_type == "linear" else NoiseSchedulePath()
    target = DisplacementTarget() if fm.target_type == "displacement" else ScheduleTarget()
    return SupervisionBuilder(
        time_sampler=UniformTimeSampler(),
        noise_sampler=GaussianNoiseSampler(),
        path=path,
        target=target,
    )

def _build_sampler(fm: FlowMatchingSettings) -> FlowMatchSampler:
    solver = euler_step if fm.solver == "euler" else heun_step
    return FlowMatchSampler(
        adapter=KwargContextAdapter(),
        x0_sampler=GaussianNoiseSampler(),
        solver=solver,
        n_steps=fm.n_inference_steps,
    )
```

Register `FlowMatchingBuildStrategy` before `FlexibleBuildStrategy` in
`BuildFactory.__init__`.

---

## `WrapperFactory` addition

### `core/models/wrappers/factories.py`

```python
@staticmethod
def create_flow_matching_wrapper(
    model_settings: ModelComponentSettings,
    settings: WrapperComponentSettings,
    supervision_builder: SupervisionBuilder,
    sampler: FlowMatchSampler,
    entry_configs: tuple[DataEntry, ...] | None = None,
    shape_summary: ShapeSummary | None = None,
    **kwargs,
) -> FlowMatchingLightningWrapper:
    return FlowMatchingLightningWrapper(
        model_settings=model_settings,
        settings=settings,
        entry_configs=entry_configs or (),
        shape_summary=shape_summary,
        batch_transforms=[supervision_builder],
        sampler=sampler,
        **kwargs,
    )
```

---

## `GenerativePredictor` (inference side)

### `interfaces/inference/predictor.py` — new class alongside `CheckpointPredictor`

```python
class GenerativePredictor:
    """Stateful generative inference — implements IGenerator.

    Single responsibility: load FM checkpoint, expose clean generate() API.
    Kept separate from CheckpointPredictor (ISP — different contracts).
    """
    def load(self) -> None:
        checkpoint = load_checkpoint(self._config.checkpoint_path)
        self._velocity_model = build_model_from_checkpoint(checkpoint)
        self._sampler = _restore_sampler_from_checkpoint(checkpoint)
        self._data_shape = checkpoint["dlkit_metadata"]["flow_matching"]["data_shape"]
        self._velocity_model.to(self._config.device).eval()

    def generate(
        self,
        n_samples: int,
        *,
        context: dict[str, Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> GenerationResult:
        shape = (n_samples, *self._data_shape)
        with torch.no_grad():
            samples = self._sampler(self._velocity_model, shape, context, generator)
        return GenerationResult(samples=samples)
```

### `interfaces/inference/api.py` — add alongside `load_predictor()`

```python
def load_generative_predictor(
    checkpoint_path: Path | str,
    device: str = "auto",
    n_steps: int | None = None,
) -> GenerativePredictor: ...
```

---

## SOLID / Functional Compliance

| Principle | Application |
|-----------|-------------|
| **SRP** | `NullGeneratorFactory`, `DeterministicGeneratorFactory`, `FixedGeneratorFactory` — one policy each; each FM component computes one mathematical quantity; pure functions in `functions/` have zero side effects |
| **OCP** | `FlowMatchingBuildStrategy` + new wrappers extend capability without modifying existing strategies or `StandardLightningWrapper` |
| **LSP** | `FlowMatchingLightningWrapper ⊂ GenerativeLightningWrapper ⊂ ProcessingLightningWrapper` — each level satisfies its parent's contract |
| **ISP** | `ITimeSampler`, `INoiseSampler`, `IInterpolationPath`, `IVelocityTarget`, `IFixedStepSolver` — one-method protocols; `IGenerator` separate from `IPredictor` |
| **DIP** | `SupervisionBuilder` injects protocols; `GenerativeLightningWrapper` depends on `IBatchTransform` + `IGeneratorFactory`; `FlowMatchSampler` receives model per call |
| **Functional** | `_apply_batch_transforms` is a pure `reduce` fold; all `functions/` are referentially transparent; nn.Module components delegate to pure functions |
| **Actions vs pure** | Actions: `GaussianNoiseSampler.forward`, `UniformTimeSampler.forward`, `FlowMatchSampler.forward`; Pure: everything in `functions/`, all path/target/adapter `forward()` calls |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/models/wrappers/generator_factories.py` | Null / Deterministic / Fixed factories |
| `core/models/wrappers/generative.py` | `GenerativeLightningWrapper` |
| `core/models/wrappers/flowmatching.py` | `FlowMatchingLightningWrapper` |
| `core/models/nn/generative/__init__.py` | Package (new model family) |
| `core/models/nn/generative/interfaces.py` | ITimeSampler, INoiseSampler, IInterpolationPath, IVelocityTarget, IModelAdapter, IFixedStepSolver |
| `core/models/nn/generative/functions/broadcast.py` | `_broadcast_time()` |
| `core/models/nn/generative/functions/paths.py` | `linear_path()`, `noise_schedule_path()` |
| `core/models/nn/generative/functions/targets.py` | `displacement_target()` |
| `core/models/nn/generative/functions/loss.py` | `velocity_mse()` |
| `core/models/nn/generative/functions/solvers.py` | `euler_step()`, `heun_step()` |
| `core/models/nn/generative/samplers/time.py` | `UniformTimeSampler` |
| `core/models/nn/generative/samplers/noise.py` | `GaussianNoiseSampler` |
| `core/models/nn/generative/paths.py` | `LinearInterpolationPath`, `NoiseSchedulePath` |
| `core/models/nn/generative/targets.py` | `DisplacementTarget`, `ScheduleTarget` |
| `core/models/nn/generative/adapters.py` | `KwargContextAdapter`, `PositionalContextAdapter` |
| `core/models/nn/generative/supervision.py` | `SupervisionBuilder(IBatchTransform)` |
| `core/models/nn/generative/sampler.py` | `FlowMatchSampler` |
| `tools/config/flowmatching_settings.py` | `FlowMatchingSettings` |
| `interfaces/inference/protocols/generative.py` | `IGenerator`, `GenerationResult` |

## Files to Modify

| File | Change |
|------|--------|
| `core/training/transforms/interfaces.py` | Add `IBatchTransform`, `IGeneratorFactory` |
| `core/models/wrappers/factories.py` | Add `create_flow_matching_wrapper()` |
| `runtime/factories/build_factory.py` | Add `FlowMatchingBuildStrategy`, register before Flexible |
| `tools/config/general_settings.py` | Add `FLOW_MATCHING: FlowMatchingSettings` |
| `interfaces/inference/api.py` | Add `load_generative_predictor()` |
| `interfaces/inference/predictor.py` | Add `GenerativePredictor` class |

---

## Verification

1. **Pure function tests**: Every function in `functions/` — shapes, dtypes, no side effects
2. **Generator reproducibility**: Same `DeterministicGeneratorFactory` seed → identical `(xt, ut)` across calls
3. **Wrapper integration**: `FlowMatchingLightningWrapper.training_step` produces correct loss;
   `predict_step` returns tensors of correct shape
4. **Generator routing**: training uses global RNG (non-deterministic), validation uses fixed seed
5. **Checkpoint round-trip**: Save → load → generate produces tensors of correct shape
6. **BuildFactory routing**: `FlowMatchingBuildStrategy.can_handle()` only true when `FLOW_MATCHING.enabled=True`
7. **Regression**: Full `pytest` — zero failures (all changes are additive)
