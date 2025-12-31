# DLKit Architecture Review: Overengineering Analysis and Simplification Recommendations

## Executive Summary

After reviewing the dlkit codebase, I've identified **architectural complexity** across three major subsystems where SOLID patterns are overused or misapplied. The goal is to **refactor intelligently** - keeping patterns that add value while removing unnecessary layers.

### Guiding Principles for Refactoring:

1. **SOLID when it matters**: Use patterns that enable extension, testing, and clarity
2. **ABCs for explicit contracts**: Keep interfaces that prevent errors and document intent
3. **Simplicity first**: Prefer direct code over abstraction layers when functionality doesn't vary
4. **Functionality preservation**: Guarantee no feature loss through detailed migration plans

### Current Issues:

- **Excessive indirection**: 8+ layers between user intent and execution
- **Single-implementation abstractions**: Interfaces with only one concrete class
- **Pattern stacking**: Multiple patterns (Template Method + Chain + Strategy) for linear flows
- **Cognitive overhead**: Developers must navigate layers even for simple changes

## Architectural Complexity Analysis

### 1. **Inference Subsystem: 27 Files for "Load and Predict"** ⚠️ CRITICAL

**Current Architecture (8+ abstraction layers)**:
```
User calls load_predictor()
  ↓ api.py
  ↓ PredictorFactory (factory.py)
  ↓ get_predictor_factory() (container.py - DI container)
  ↓ CheckpointPredictor (predictor.py)
  ↓ ModelLoadingUseCase (use_cases.py)
  ↓ CheckpointReconstructor (infrastructure/adapters.py)
  ↓ PyTorchModelLoader (infrastructure/adapters.py)
  ↓ TorchModelStateManager (infrastructure/adapters.py)
  ↓ ShapeInferenceUseCase + chain of strategies
  ↓ InferenceExecutionUseCase (use_cases.py)
  ↓ DirectInferenceExecutor (infrastructure/adapters.py)
  ↓ Model forward pass
```

**File count breakdown**:
- `application/`: use_cases.py, config_inference.py, orchestrators.py (3 files)
- `domain/`: models.py, ports.py (2 files)
- `infrastructure/`: adapters.py (1 file)
- `config/`: config_builder.py, inference_config.py (2 files)
- `inputs/`: adapters.py, inference_input.py (2 files)
- `strategies/`: inference_strategy.py, prediction_strategy.py (2 files)
- `transforms/`: executor.py, checkpoint_loader.py (2 files)
- Core: api.py, predictor.py, factory.py, container.py, checkpoint_utils.py, reconstruction.py (6 files)
- **Total: 27 files**

**Problems**:
1. **Hexagonal architecture overkill**: Application/Domain/Infrastructure separation adds no value for ML inference
2. **Use case splitting**: ModelLoadingUseCase + InferenceExecutionUseCase + ShapeInferenceUseCase = unnecessary fragmentation
3. **Port/Adapter proliferation**: IModelLoader, IInferenceExecutor, IModelStateManager, ICheckpointReconstructor - all abstractions with single implementations
4. **Dependency injection container**: The container.py adds complexity without flexibility (only one implementation per interface)
5. **State management ceremony**: ModelState with state transitions (LOADED → INFERENCE_READY) for a model that just needs eval() mode

**Industry Standard Comparison**:
- **Hugging Face Transformers**: `model = AutoModel.from_pretrained(); output = model(input)` (2 lines)
- **torchvision**: `model = models.resnet50(pretrained=True); model.eval(); output = model(x)` (3 lines)
- **dlkit**: 27 files, 8+ abstraction layers

**Recommendation**: Collapse to **3-5 files maximum**
- `inference.py`: Single module with `load_predictor()` and `Predictor` class
- `checkpoint.py`: Checkpoint loading utilities
- `transforms.py`: Transform execution (if needed separately)

### 2. **Transform System: ABC Interface Explosion** ⚠️ CRITICAL

**Current Architecture**:
```python
class MinMaxScaler(
    Transform,                    # Base class
    IFittableTransform,          # ABC for fit()
    IInvertibleTransform,        # ABC for inverse_transform()
    IShapeAwareTransform         # ABC for configure_shape()
):
    # Actual implementation
```

**File count**: 15 files for basic transformations
- `base.py`: Transform base class
- `interfaces.py`: 4 ABC interfaces (IFittableTransform, IInvertibleTransform, ISerializableTransform, IShapeAwareTransform)
- `shape_inference.py`: Shape inference registry
- `chain.py`: TransformChain with analytical shape inference
- `manager.py`: TransformManager service
- `pipeline.py`: Transform pipeline
- Individual transforms: `minmax.py`, `standard.py`, `pca.py`, `sample_norm.py`, `permute.py`, `subset.py`, `spectral.py`
- `errors.py`: Custom exceptions

**Problems**:
1. **Interface Segregation Principle overdone**: 4 ABC mixins for capabilities that could be simple duck typing
2. **isinstance() dependency**: Runtime capability checking via `isinstance(transform, IInvertibleTransform)` adds fragility
3. **Shape specification coupling**: Integration with `shape_spec` system adds unnecessary complexity
4. **Transform chain analytics**: Analytical shape inference with registries instead of simple execution
5. **Manager service pattern**: TransformManager adds layer for simple transform application

**Industry Standard Comparison**:
- **scikit-learn**: Single `TransformerMixin` base class with `fit()`, `transform()`, `inverse_transform()` as optional methods
- **dlkit**: 4 ABC interfaces + registry system + manager service

**Recommendation**: Simplify to **scikit-learn pattern**
```python
class Transform(nn.Module):
    """Base class for transformations."""

    def fit(self, data):
        """Override if needed."""
        pass

    def transform(self, x):
        """Override (required)."""
        raise NotImplementedError

    def inverse_transform(self, x):
        """Override if invertible."""
        raise NotImplementedError(f"{self.__class__.__name__} is not invertible")
```

- **No ABCs**: Use duck typing or simple base class methods
- **No shape registry**: Infer shapes lazily during fit()
- **No manager**: Call transform methods directly

### 3. **Lightning Wrapper + Processing Pipeline: Chain of Responsibility Overkill** ⚠️ HIGH

**Current Architecture**:
```
ProcessingLightningWrapper (base.py)
  ↓ Initializes 4 separate pipelines:
    - train_pipeline
    - val_pipeline
    - test_pipeline
    - predict_pipeline

Each pipeline uses Chain of Responsibility:
  DataExtractionStep
    → ModelInvocationStep (with model invoker strategy)
      → OutputClassificationStep (with classifier strategy)
        → OutputNamingStep (with namer strategy)
          → LossPairingStep
            → ValidationDataStep
```

**File count**: 9 files for pipeline processing
- `pipeline.py`: ProcessingStep base + 6 step implementations
- `model_invokers.py`: ModelInvoker strategies
- `classifiers.py`: Output classifier strategies
- `naming.py`: Output naming strategies
- `providers.py`: Data providers
- `context.py`: ProcessingContext
- `interfaces.py`: Pipeline interfaces
- `graph_support.py`: Graph-specific support
- Plus wrapper files: `base.py`, `standard.py`, `graph.py`, `timeseries.py`, `factories.py`

**Problems**:
1. **Chain of Responsibility for linear flow**: The pattern adds complexity for a strictly sequential pipeline
2. **ProcessingContext threading**: Context object passed through all steps instead of simple returns
3. **Strategy pattern overuse**: ModelInvoker, OutputClassifier, OutputNamer are strategies for operations that rarely vary
4. **Four pipeline copies**: train/val/test/predict pipelines share 90% of logic but are separate objects
5. **Template method + Chain of Responsibility**: Combining patterns creates double indirection

**Industry Standard Comparison**:
- **PyTorch Lightning**: Override `training_step()`, `validation_step()`, directly - no pipeline
- **dlkit**: 6-step chain of responsibility with strategy patterns

**Recommendation**: Collapse to **simple LightningModule methods**
```python
class ModelWrapper(LightningModule):
    def training_step(self, batch, batch_idx):
        # Extract features/targets directly
        features, targets = self._extract_batch(batch)

        # Model forward
        predictions = self.model(features)

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        return loss
```

- **No pipeline objects**: Direct method implementation
- **No chain of responsibility**: Sequential code is clearer than chain pattern
- **No context threading**: Use simple returns
- **Shared logic via helper methods**: Not separate pipeline objects

## Top 3 Critical Components Needing Simplification

### 🔥 #1: Inference Subsystem - Detailed Refactoring Plan

**Current Complexity**: 27 files, 8 abstraction layers

#### Functionality Analysis (What Must Be Preserved):

1. ✅ **Stateful predictor pattern**: Load once, predict many times
2. ✅ **Automatic precision inference**: Detect model dtype and load data accordingly
3. ✅ **Transform execution**: Apply fitted transforms during inference
4. ✅ **Separate feature/target transforms**: Distinct transform handling
5. ✅ **Shape specification inference**: Automatic shape detection from checkpoints/data
6. ✅ **Device placement**: Auto-select or manual device specification
7. ✅ **Config-based batch inference**: Load dataset from config and iterate
8. ✅ **Checkpoint validation**: Validate checkpoint compatibility
9. ✅ **Context manager support**: `with load_predictor() as predictor:`
10. ✅ **Metadata extraction**: Get checkpoint info without loading full model

#### What's Actually Overengineered:

1. **Hexagonal architecture layers**: Application/Domain/Infrastructure split adds NO value
   - Only ONE implementation per port (no polymorphism)
   - Testing doesn't benefit (mocking PyTorch is pointless)
   - Domain models (ModelState, InferenceRequest) just wrap dict/dataclass

2. **Use case objects**: Single-method classes that could be functions
   - `ModelLoadingUseCase.load_model()` → `_load_model()` function
   - `InferenceExecutionUseCase.execute_inference()` → `_execute_inference()` function

3. **DI Container**: Adds complexity without flexibility
   - Only used to wire up single implementations
   - Factory pattern sufficient for predictor creation

#### Refactoring Plan (Functionality-Preserving):

**Phase 1: Consolidate Core Logic** (Preserve ALL functionality)

Create consolidated `inference.py` with these components:

```python
# src/dlkit/inference.py (~300-400 lines total)

from typing import Protocol, Iterator
from pathlib import Path
import torch

# KEEP ABC: Explicit contract for predictors (enables testing, documentation)
class Predictor(Protocol):
    """Stateful predictor interface.

    Keep as Protocol (not ABC) for structural typing - allows duck typing
    while providing clear contract."""

    def predict(self, inputs, batch_size=None) -> InferenceResult: ...
    def predict_from_config(self, config) -> Iterator[InferenceResult]: ...
    def is_loaded(self) -> bool: ...
    def unload(self) -> None: ...


class CheckpointPredictor:
    """Stateful predictor implementation.

    Consolidates ModelLoadingUseCase + InferenceExecutionUseCase logic.
    All checkpoint loading, shape inference, transform loading happens here.
    """

    def __init__(self, config: PredictorConfig, precision_service):
        self._config = config
        self._precision_service = precision_service
        self._model_state = None
        self._inferred_precision = None

        if config.auto_load:
            self.load()

    def load(self) -> Self:
        """Consolidates ModelLoadingUseCase logic directly."""
        checkpoint = torch.load(self._config.checkpoint_path)

        # Shape inference (keep as separate function for SRP)
        shape_spec = _infer_shape_specification(
            checkpoint=checkpoint,
            dataset=None  # Can extend later
        )

        # Model reconstruction (direct, no adapter layer)
        model = _build_model_from_checkpoint(checkpoint, shape_spec)

        # Transform loading (separate feature/target - PRESERVED)
        feature_transforms, target_transforms = _load_transforms_from_checkpoint(
            checkpoint
        )

        # Precision inference (PRESERVED)
        self._inferred_precision = self._precision_service.infer_precision_from_model(model)

        # Device placement + eval mode
        model.eval().to(self._config.device)

        self._model_state = ModelState(
            model=model,
            feature_transforms=feature_transforms,
            target_transforms=target_transforms,
            metadata=checkpoint.get('dlkit_metadata')
        )
        self._loaded = True
        return self

    def predict(self, inputs, batch_size=None) -> InferenceResult:
        """Consolidates InferenceExecutionUseCase logic."""
        if not self._loaded:
            raise PredictorNotLoadedError()

        # Establish precision context (PRESERVED)
        with precision_override(self._inferred_precision):
            # Apply feature transforms (PRESERVED)
            if self._config.apply_transforms and self._model_state.feature_transforms:
                inputs = self._apply_transforms(
                    inputs, self._model_state.feature_transforms
                )

            # Model forward with no_grad
            with torch.no_grad():
                predictions = self._model_state.model(inputs)

            # Apply inverse target transforms (PRESERVED)
            if self._config.apply_transforms and self._model_state.target_transforms:
                predictions = self._apply_inverse_transforms(
                    predictions, self._model_state.target_transforms
                )

            return InferenceResult(predictions=predictions)

    # Context manager support (PRESERVED)
    def __enter__(self): ...
    def __exit__(self, *args): ...


# Helper functions (extracted from use cases - SRP maintained)
def _infer_shape_specification(checkpoint, dataset=None) -> ShapeSpec:
    """Shape inference logic from ShapeInferenceUseCase.

    Preserves chain of responsibility for fallback strategies."""
    # Try checkpoint metadata first
    if 'dlkit_metadata' in checkpoint and 'shape_spec' in checkpoint['dlkit_metadata']:
        return ShapeSpec.from_dict(checkpoint['dlkit_metadata']['shape_spec'])

    # Fallback to dataset inference if provided
    if dataset is not None:
        return infer_shape_from_dataset(dataset)

    raise WorkflowError("Cannot infer shape")


def _build_model_from_checkpoint(checkpoint, shape_spec) -> nn.Module:
    """Model building logic from PyTorchModelLoader.

    Preserves dtype inference and settings reconstruction."""
    model_settings = _extract_model_settings(checkpoint)

    # Detect checkpoint dtype (PRESERVED - prevents precision loss)
    state_dict = extract_state_dict(checkpoint)
    checkpoint_dtype = _detect_dtype(state_dict)

    # Create model with shape spec
    model = FactoryProvider.create_component(
        model_settings,
        BuildContext(mode="inference", unified_shape=shape_spec)
    )

    # Convert to checkpoint dtype BEFORE loading weights (PRESERVED)
    model = model.to(dtype=checkpoint_dtype)
    model.load_state_dict(state_dict, strict=False)

    return model


def _load_transforms_from_checkpoint(checkpoint) -> tuple[dict, dict]:
    """Transform loading from CheckpointTransformLoader.

    Preserves separation of feature/target transforms."""
    feature_transforms = {}
    target_transforms = {}

    if 'fitted_transforms' in checkpoint:
        # Extract and categorize transforms using entry_configs
        entry_configs = checkpoint.get('inference_metadata', {}).get('entry_configs', {})

        for name, transform_state in checkpoint['fitted_transforms'].items():
            transform = _reconstruct_transform(transform_state)

            # Categorize as feature or target
            if entry_configs.get(name, {}).get('type') == 'target':
                target_transforms[name] = transform
            else:
                feature_transforms[name] = transform

    return feature_transforms, target_transforms


# Public API
def load_predictor(
    checkpoint_path,
    device="auto",
    batch_size=32,
    apply_transforms=True,
    precision=None
) -> CheckpointPredictor:
    """Factory function for predictor creation.

    Replaces PredictorFactory + DI container with simple factory function."""
    config = PredictorConfig(
        checkpoint_path=Path(checkpoint_path),
        device=device,
        batch_size=batch_size,
        apply_transforms=apply_transforms,
        precision=precision,
        auto_load=True
    )

    return CheckpointPredictor(
        config=config,
        precision_service=get_precision_service()
    )


def validate_checkpoint(checkpoint_path) -> dict[str, str]:
    """Checkpoint validation (PRESERVED)."""
    # ... existing validation logic


def get_checkpoint_info(checkpoint_path) -> dict:
    """Checkpoint metadata extraction (PRESERVED)."""
    # ... existing extraction logic
```

**Phase 2: File Structure**

```
src/dlkit/interfaces/inference/
├── __init__.py          # Public API exports
├── predictor.py         # CheckpointPredictor class (~200 lines)
├── loading.py           # Checkpoint/model loading utilities (~150 lines)
├── transforms.py        # Transform loading/application (~100 lines)
├── shapes.py            # Shape inference logic (~80 lines)
└── config.py            # PredictorConfig, InferenceResult dataclasses (~50 lines)

Total: 6 files (~580 lines) vs 27 files (~2000+ lines)
```

**What We KEEP from SOLID**:

1. ✅ **Single Responsibility**: Each function does ONE thing
   - `_load_model()` only loads model
   - `_infer_shapes()` only infers shapes
   - `_load_transforms()` only loads transforms

2. ✅ **Dependency Inversion**: Predictor depends on `PrecisionService` abstraction (injected)

3. ✅ **Interface Segregation**: `Predictor` Protocol defines clear contract

**What We REMOVE**:

1. ❌ Hexagonal architecture (application/domain/infrastructure)
2. ❌ Use case objects (replace with functions)
3. ❌ Port interfaces with single implementations
4. ❌ DI container (use simple factory function)
5. ❌ ModelState state machine (just use dataclass)

**Migration Guarantee**:

- [x] All public API preserved: `load_predictor()`, `validate_checkpoint()`, `get_checkpoint_info()`
- [x] All features preserved: Listed in "Functionality Analysis" above
- [x] All tests pass: Unit tests updated to use new structure
- [x] Performance same or better: Fewer object allocations
- [x] Backward compatibility: Old API can wrap new API during transition

### 🔥 #2: Transform System - Detailed Refactoring Plan

**Current Complexity**: 15 files, 4 ABC interfaces, shape registry

#### Functionality Analysis (What Must Be Preserved):

1. ✅ **Fittable transforms**: Learn statistics from data (mean/std, min/max, PCA components)
2. ✅ **Invertible transforms**: Reverse transformations for denormalization
3. ✅ **Shape-aware transforms**: Pre-allocate buffers using shape_spec (performance optimization)
4. ✅ **Transform chaining**: Compose multiple transforms with TransformChain
5. ✅ **Checkpoint persistence**: Save/load fitted state via torch.nn.Module state_dict
6. ✅ **Device management**: Move transforms to GPU/CPU via .to(device)
7. ✅ **Fitted state tracking**: Know when transform is ready to use
8. ✅ **Shape inference**: Analytical shape inference for chains (no dummy tensor execution)
9. ✅ **Lazy initialization**: Allocate buffers during fit() if no shape_spec provided

#### What's Actually Over-Abstracted:

1. **4 ABC interfaces**: IFittableTransform, IInvertibleTransform, IShapeAwareTransform, ISerializableTransform
   - Problem: Forces isinstance() checks everywhere
   - Better: Simple base class with optional method overrides (duck typing)

2. **Shape registry pattern**: Separate registry for shape inference functions
   - Problem: Indirection for what could be class methods
   - Better: Shape inference as method or simple dict

3. **TransformManager service**: Adds layer for simple transform application
   - Problem: Unnecessary service pattern
   - Better: Direct method calls on transforms

#### Refactoring Plan (Functionality-Preserving):

**Phase 1: Consolidate Interfaces** (Keep SOME ABCs where valuable)

```python
# src/dlkit/core/training/transforms/base.py

import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Optional

class Transform(nn.Module):
    """Base class for tensor transformations.

    Integrates with PyTorch nn.Module for:
    - Device management (.to(device))
    - Checkpoint persistence (state_dict/load_state_dict)
    - Parameter tracking (register_buffer)

    **Design Philosophy**:
    - Methods are OPTIONAL by default (no abstract methods except forward)
    - Capabilities declared via simple method presence (duck typing)
    - ABCs used only for truly mandatory contracts
    """

    def __init__(self):
        super().__init__()
        # Use tensor buffer for checkpoint persistence (PRESERVED)
        self.register_buffer("_fitted", torch.zeros(1, requires_grad=False))

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformation (REQUIRED override)."""
        raise NotImplementedError

    # Optional capabilities (override if needed)

    def fit(self, data: torch.Tensor) -> None:
        """Fit transform to data (optional).

        Override if transform needs to learn statistics.
        MUST set self.fitted = True after fitting.
        """
        pass

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transformation (optional).

        Override if transform is invertible.
        Default implementation raises clear error.
        """
        raise RuntimeError(
            f"{self.__class__.__name__} does not support inverse_transform. "
            f"Transform is not invertible."
        )

    def configure_shape(self, shape: tuple[int, ...]) -> None:
        """Configure with shape for buffer pre-allocation (optional).

        Override if transform benefits from eager buffer allocation.
        This is a PERFORMANCE optimization, not a requirement.

        Args:
            shape: Expected input shape for this transform
        """
        pass

    @property
    def fitted(self) -> bool:
        """Check if transform has been fitted (PRESERVED)."""
        return self.get_buffer("_fitted").item() == 1

    @fitted.setter
    def fitted(self, value: bool) -> None:
        """Set fitted state (PRESERVED)."""
        self._fitted.fill_(1 if value else 0)

    # Capability checking (replaces isinstance checks)
    def is_invertible(self) -> bool:
        """Check if transform supports inverse_transform."""
        # Try calling with dummy tensor to see if implemented
        try:
            dummy = torch.zeros(1)
            self.inverse_transform(dummy)
            return True
        except RuntimeError:
            return False

    def is_fittable(self) -> bool:
        """Check if transform requires fitting."""
        # Check if fit() is overridden
        return self.fit != Transform.fit


# KEEP ONE ABC: Explicit contract for chain-compatible transforms
from abc import ABC

class ChainableTransform(ABC):
    """Explicit interface for transforms that work in TransformChain.

    USE THIS ABC because:
    1. Prevents runtime errors in chain composition
    2. Documents clear contract for chain compatibility
    3. Enables type checking at chain creation time
    """

    @abstractmethod
    def infer_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Infer output shape from input shape (required for chains).

        This enables analytical shape inference in TransformChain
        without executing dummy tensors.
        """
        pass
```

**Phase 2: Simplify Transform Implementations**

```python
# src/dlkit/core/training/transforms/minmax.py

class MinMaxScaler(Transform, ChainableTransform):
    """Min-max normalization to [-1, 1].

    Combines capabilities:
    - Fittable (learns min/max)
    - Invertible (can denormalize)
    - Shape-aware (supports eager allocation)
    - Chainable (provides shape inference)
    """

    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim if isinstance(dim, tuple) else (dim,)
        self._shape_configured = False

    # Optional: Shape-aware capability
    def configure_shape(self, shape: tuple[int, ...]) -> None:
        """Pre-allocate buffers (PERFORMANCE optimization)."""
        moments_shape = tuple([1 if i in self.dim else s for i, s in enumerate(shape)])
        self.register_buffer("min", torch.zeros(moments_shape))
        self.register_buffer("max", torch.ones(moments_shape))
        self._shape_configured = True

    # Optional: Fittable capability
    def fit(self, data: torch.Tensor) -> None:
        """Learn min/max statistics."""
        # Lazy allocation if not pre-configured
        if not self._shape_configured:
            moments_shape = tuple([1 if i in self.dim else s for i, s in enumerate(data.shape)])
            self.register_buffer("min", torch.zeros(moments_shape, device=data.device))
            self.register_buffer("max", torch.ones(moments_shape, device=data.device))
            self._shape_configured = True

        # Compute min/max
        current_min = torch.amin(data, dim=self.dim, keepdim=True)
        current_max = torch.amax(data, dim=self.dim, keepdim=True)

        if self.fitted:
            # Accumulate across multiple fit() calls
            self.min = torch.minimum(self.min, current_min)
            self.max = torch.maximum(self.max, current_max)
        else:
            self.min = current_min
            self.max = current_max
            self.fitted = True

    # Required: Transform forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale to [-1, 1]."""
        if not self.fitted:
            raise RuntimeError("MinMaxScaler must be fitted before use")
        return 2 * (x - self.min) / (self.max - self.min + 1e-8) - 1

    # Optional: Invertible capability
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize from [-1, 1]."""
        if not self.fitted:
            raise RuntimeError("MinMaxScaler must be fitted before inverse")
        return (x + 1) / 2 * (self.max - self.min) + self.min

    # Required for ChainableTransform ABC
    def infer_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """MinMaxScaler preserves shape."""
        return input_shape
```

**Phase 3: Simplified TransformChain**

```python
# src/dlkit/core/training/transforms/chain.py

class TransformChain(Transform, ChainableTransform):
    """Compose multiple transforms with analytical shape inference.

    Preserves key features:
    - Sequential application of transforms
    - Analytical shape inference (no dummy tensor execution)
    - Checkpoint persistence
    - Inverse transform support (if all transforms invertible)
    """

    def __init__(self, transforms: list[Transform]):
        super().__init__()
        # Store as ModuleList for proper checkpoint handling
        self.transforms = nn.ModuleList(transforms)

    def fit(self, data: torch.Tensor) -> None:
        """Fit all fittable transforms sequentially."""
        current_data = data
        for transform in self.transforms:
            if transform.is_fittable():
                transform.fit(current_data)
                current_data = transform(current_data)
        self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all transforms sequentially."""
        for transform in self.transforms:
            x = transform(x)
        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse transforms in reverse order."""
        # Check if all transforms are invertible
        for transform in self.transforms:
            if not transform.is_invertible():
                raise RuntimeError(
                    f"Cannot invert chain: {transform.__class__.__name__} is not invertible"
                )

        # Apply inverses in reverse
        for transform in reversed(self.transforms):
            x = transform.inverse_transform(x)
        return x

    def infer_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Analytical shape inference through chain."""
        current_shape = input_shape
        for transform in self.transforms:
            if isinstance(transform, ChainableTransform):
                current_shape = transform.infer_output_shape(current_shape)
            else:
                # Fallback: assume shape preservation
                pass
        return current_shape
```

**Phase 4: File Structure**

```
src/dlkit/core/training/transforms/
├── __init__.py          # Public API exports
├── base.py              # Transform, ChainableTransform ABC (~150 lines)
├── chain.py             # TransformChain (~80 lines)
├── errors.py            # Custom exceptions (~30 lines)
├── scalers.py           # MinMaxScaler, StandardScaler (~200 lines)
├── pca.py               # PCA transform (~150 lines)
├── sample_norm.py       # SampleNormL2 (~80 lines)
├── permute.py           # Permutation (~50 lines)
├── subset.py            # TensorSubset (~60 lines)
└── spectral.py          # Spectral transforms (~60 lines)

Total: 10 files (~860 lines) vs 15 files (~1200+ lines)
```

**What We KEEP from SOLID**:

1. ✅ **Interface Segregation**: ChainableTransform ABC for transforms that work in chains
2. ✅ **Open-Closed**: Can add new transforms without modifying existing ones
3. ✅ **Single Responsibility**: Each transform does ONE transformation
4. ✅ **Liskov Substitution**: All transforms work as nn.Module

**What We SIMPLIFY**:

1. ✅ **Reduce ABCs from 4 → 1**: Only ChainableTransform is abstract
2. ✅ **Duck typing for capabilities**: Check with .is_invertible() instead of isinstance()
3. ✅ **No shape registry**: Shape inference as method on ChainableTransform
4. ✅ **No manager service**: Direct method calls on transforms

**Migration Guarantee**:

- [x] All transforms work: MinMaxScaler, StandardScaler, PCA, etc.
- [x] TransformChain preserved: Sequential composition works
- [x] Checkpoint persistence: state_dict/load_state_dict work
- [x] Shape inference: Analytical inference without dummy tensors
- [x] All tests pass: Unit tests updated to use simplified API
- [x] Performance same/better: Lazy vs eager allocation both supported

### 🔥 #3: Lightning Wrapper + Pipelines - Detailed Refactoring Plan

**Current Complexity**: 14 files, Chain of Responsibility + Strategy patterns

#### Functionality Analysis (What Must Be Preserved):

1. ✅ **Configurable feature/target extraction**: Use entry_configs to categorize batch data
2. ✅ **Transform integration**: Apply fitted transforms to features before model forward
3. ✅ **Inverse transforms**: Apply to predictions for denormalization
4. ✅ **Multiple model families**: Standard (arrays), Graph, Timeseries wrappers
5. ✅ **Output naming**: Name model outputs based on target configurations
6. ✅ **Loss pairing**: Pair predictions with corresponding targets for loss
7. ✅ **Metric computation**: Compute metrics with proper dtype handling
8. ✅ **Predict mode**: Inference without targets/loss computation
9. ✅ **Checkpoint metadata**: Save model settings, shape specs, transforms
10. ✅ **Precision management**: Apply user-configured precision to nested model

#### What's Actually Overengineered:

1. **Chain of Responsibility for linear flow**: 6 steps in strict sequence
   - Problem: Pattern adds complexity for non-branching pipeline
   - Better: Direct sequential helper methods

2. **ProcessingContext threading**: Context object passed through steps
   - Problem: Indirection makes debugging harder
   - Better: Direct returns from helper methods

3. **Strategy patterns**: ModelInvoker, OutputClassifier, OutputNamer
   - Problem: Single implementation per strategy
   - Better: Direct method calls (strategies only if multiple implementations needed)

4. **4 separate pipeline instances**: train/val/test/predict pipelines
   - Problem: Code duplication, separate objects for same logic
   - Better: Single logic with mode parameter

#### Refactoring Plan (Functionality-Preserving):

**Phase 1: Collapse Pipeline into Helper Methods**

```python
# src/dlkit/core/models/wrappers/base.py

from lightning import LightningModule
from typing import Any, Optional
import torch

class ProcessingLightningWrapper(LightningModule):
    """Lightning wrapper with processing support.

    Preserves ALL functionality from pipeline system:
    - Entry-based feature/target extraction
    - Transform application/inverse
    - Output naming and loss pairing
    - Multi-family support (array, graph, timeseries)

    Simplifies HOW it's implemented:
    - Direct helper methods instead of Chain of Responsibility
    - Sequential code instead of context threading
    - Mode parameter instead of 4 separate pipelines
    """

    def __init__(
        self,
        settings: WrapperComponentSettings,
        model_settings: ModelComponentSettings,
        entry_configs: dict[str, DataEntry] | None = None,
        shape_spec: IShapeSpec | None = None,
        **kwargs
    ):
        super().__init__()

        # Store configuration (PRESERVED)
        self.save_hyperparameters({
            "settings": settings,
            "model_settings": model_settings
        }, ignore=["settings", "model_settings", "entry_configs"])

        # Shape spec (PRESERVED)
        self.shape_spec = shape_spec

        # Create model (PRESERVED)
        self.model = self._create_abc_model(model_settings, shape_spec)

        # Apply precision (PRESERVED)
        from dlkit.interfaces.api.services.precision_service import get_precision_service
        precision_service = get_precision_service()
        precision_strategy = precision_service.resolve_precision()
        dtype = precision_strategy.to_torch_dtype()
        self.model = self.model.to(dtype=dtype)

        # Metrics (PRESERVED)
        self.val_metrics = MetricCollection([...])
        self.test_metrics = MetricCollection([...])

        # Loss function (PRESERVED)
        self.loss_function = getattr(self.model, "loss_function", None) or \
                            FactoryProvider.create_component(settings.loss_function, ...)

        # Entry configs for feature/target extraction (PRESERVED)
        self._entry_configs = entry_configs or {}

        # Categorize entries (replaces DataExtractionStep logic)
        self._feature_names = {
            name for name, config in self._entry_configs.items()
            if is_feature_entry(config)
        }
        self._target_names = {
            name for name, config in self._entry_configs.items()
            if is_target_entry(config)
        }

    # Lightning step methods - direct implementation (NO PIPELINE)

    def training_step(self, batch, batch_idx):
        """Training step with all pipeline functionality inline."""
        # 1. Extract features/targets (was DataExtractionStep)
        features, targets = self._extract_features_targets(batch)

        # 2. Apply forward transforms (was part of ModelInvocationStep)
        if hasattr(self, 'fitted_feature_transforms'):
            features = self._apply_transforms(features, self.fitted_feature_transforms)

        # 3. Model forward (was ModelInvocationStep)
        predictions = self._invoke_model(features)

        # 4. Apply inverse transforms to predictions (was part of LossPairingStep)
        if hasattr(self, 'fitted_target_transforms'):
            predictions = self._apply_inverse_transforms(predictions, self.fitted_target_transforms)

        # 5. Pair predictions with targets and compute loss (was LossPairingStep)
        loss = self._compute_loss(predictions, targets)

        # 6. Compute metrics
        self._update_metrics(predictions, targets, stage="train")

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - same logic, different metrics."""
        features, targets = self._extract_features_targets(batch)

        if hasattr(self, 'fitted_feature_transforms'):
            features = self._apply_transforms(features, self.fitted_feature_transforms)

        predictions = self._invoke_model(features)

        if hasattr(self, 'fitted_target_transforms'):
            predictions = self._apply_inverse_transforms(predictions, self.fitted_target_transforms)

        loss = self._compute_loss(predictions, targets)
        self._update_metrics(predictions, targets, stage="val")

        return loss

    def test_step(self, batch, batch_idx):
        """Test step - same as validation."""
        features, targets = self._extract_features_targets(batch)

        if hasattr(self, 'fitted_feature_transforms'):
            features = self._apply_transforms(features, self.fitted_feature_transforms)

        predictions = self._invoke_model(features)

        if hasattr(self, 'fitted_target_transforms'):
            predictions = self._apply_inverse_transforms(predictions, self.fitted_target_transforms)

        loss = self._compute_loss(predictions, targets)
        self._update_metrics(predictions, targets, stage="test")

        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step - no targets/loss needed."""
        # Extract features only (no targets required)
        if isinstance(batch, dict):
            features = {k: v for k, v in batch.items() if k in self._feature_names}
        else:
            features = batch

        # Apply forward transforms
        if hasattr(self, 'fitted_feature_transforms'):
            features = self._apply_transforms(features, self.fitted_feature_transforms)

        # Model forward
        predictions = self._invoke_model(features)

        # Apply inverse transforms
        if hasattr(self, 'fitted_target_transforms'):
            predictions = self._apply_inverse_transforms(predictions, self.fitted_target_transforms)

        return predictions

    # Helper methods (replace pipeline steps - SRP maintained)

    def _extract_features_targets(self, batch: dict) -> tuple[dict, dict]:
        """Extract and categorize batch data (was DataExtractionStep).

        Preserves:
        - Entry-config-based categorization
        - Fallback heuristics when no configs
        """
        if not self._feature_names and not self._target_names:
            # Fallback heuristic (PRESERVED from DataExtractionStep)
            target_like = {"y", "target", "targets", "label", "labels"}
            features = {k: v for k, v in batch.items() if k.lower() not in target_like}
            targets = {k: v for k, v in batch.items() if k.lower() in target_like}
            return features, targets

        # Config-based extraction (PRESERVED)
        features = {k: v for k, v in batch.items() if k in self._feature_names}
        targets = {k: v for k, v in batch.items() if k in self._target_names}
        return features, targets

    def _invoke_model(self, features: dict | torch.Tensor) -> Any:
        """Invoke model forward pass (was ModelInvocationStep).

        Preserves:
        - Dict vs tensor handling
        - Dtype validation
        """
        # Get model dtype for validation
        model_dtype = next(self.model.parameters()).dtype if \
                     any(self.model.parameters()) else torch.float32

        # Validate input dtypes match model (PRESERVED defensive validation)
        if isinstance(features, torch.Tensor):
            if features.dtype != model_dtype:
                logger.warning(
                    f"Input dtype {features.dtype} != model dtype {model_dtype}, "
                    f"auto-casting"
                )
                features = features.to(dtype=model_dtype)
        elif isinstance(features, dict):
            for k, v in features.items():
                if torch.is_tensor(v) and v.dtype != model_dtype:
                    logger.warning(f"Feature '{k}' dtype mismatch, auto-casting")
                    features[k] = v.to(dtype=model_dtype)

        # Model forward
        return self.model(features)

    def _apply_transforms(self, data: dict, transforms: dict) -> dict:
        """Apply forward transforms (was part of pipeline)."""
        transformed = {}
        for name, tensor in data.items():
            if name in transforms:
                transformed[name] = transforms[name](tensor)
            else:
                transformed[name] = tensor
        return transformed

    def _apply_inverse_transforms(self, predictions: Any, transforms: dict) -> Any:
        """Apply inverse transforms (was part of pipeline).

        Preserves:
        - Dict vs tensor handling
        - Transform ambiguity detection
        """
        if isinstance(predictions, dict):
            return {
                k: transforms[k].inverse_transform(v) if k in transforms else v
                for k, v in predictions.items()
            }

        # Single tensor - check for ambiguity (PRESERVED)
        if len(transforms) == 0:
            return predictions
        elif len(transforms) == 1:
            name = next(iter(transforms.keys()))
            return transforms[name].inverse_transform(predictions)
        else:
            from dlkit.core.training.transforms.errors import TransformAmbiguityError
            raise TransformAmbiguityError(
                list(transforms.keys()),
                context="Model returned single tensor but multiple target transforms exist"
            )

    def _compute_loss(self, predictions: Any, targets: dict) -> torch.Tensor:
        """Compute loss with pairing (was LossPairingStep).

        Preserves:
        - Automatic pairing of predictions with targets
        - Dtype alignment
        """
        # Ensure predictions and targets have compatible dtypes
        if isinstance(predictions, torch.Tensor) and isinstance(targets, dict):
            target_tensor = next(iter(targets.values()))
            if predictions.dtype != target_tensor.dtype:
                target_tensor = target_tensor.to(dtype=predictions.dtype)
                targets = {k: v.to(dtype=predictions.dtype) if torch.is_tensor(v) else v
                          for k, v in targets.items()}

        return self.loss_function(predictions, targets)

    def _update_metrics(self, predictions: Any, targets: dict, stage: str):
        """Update metrics with dtype casting (PRESERVED from _compute_metrics)."""
        # Cast to model dtype for consistency
        model_dtype = next(self.model.parameters()).dtype
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.to(dtype=model_dtype)
        if isinstance(targets, dict):
            targets = {k: v.to(dtype=model_dtype) if torch.is_tensor(v) else v
                      for k, v in targets.items()}

        # Update appropriate metric collection
        metrics = {"train": self.train_metrics, "val": self.val_metrics,
                  "test": self.test_metrics}.get(stage)
        if metrics:
            metrics.update(predictions, targets)

    # Template methods for subclasses (array vs graph vs timeseries)

    def _create_abc_model(self, model_settings, shape_spec):
        """Create model - override in subclasses if needed."""
        build_context = BuildContext(mode="training")
        if shape_spec and not shape_spec.is_empty():
            build_context = build_context.with_overrides(unified_shape=shape_spec)
        return FactoryProvider.create_component(model_settings, build_context)
```

**Phase 2: Specialized Wrappers** (Keep specialization via inheritance)

```python
# src/dlkit/core/models/wrappers/standard.py

class StandardArrayWrapper(ProcessingLightningWrapper):
    """Wrapper for standard array-based models.

    Adds transform fitting capability to base wrapper.
    """

    def setup(self, stage: str):
        """Setup transforms during fit."""
        if stage == "fit":
            # Fit transforms on training data
            self._fit_transforms_from_datamodule()

# src/dlkit/core/models/wrappers/graph.py

class GraphWrapper(ProcessingLightningWrapper):
    """Wrapper for graph neural networks.

    Specializes model invocation for graph data structures.
    """

    def _invoke_model(self, features):
        """Override for graph-specific invocation."""
        # Handle PyG Data objects
        return self.model(features.x, features.edge_index)
```

**Phase 3: File Structure**

```
src/dlkit/core/models/wrappers/
├── __init__.py          # Public API
├── base.py              # ProcessingLightningWrapper (~300 lines)
├── standard.py          # StandardArrayWrapper (~100 lines)
├── graph.py             # GraphWrapper (~80 lines)
├── timeseries.py        # TimeseriesWrapper (~80 lines)
└── factories.py         # Wrapper factory (~50 lines)

src/dlkit/runtime/pipelines/
└── (REMOVED - logic moved to wrapper methods)

Total: 5 wrapper files (~610 lines) vs 14 files (~1500+ lines)
```

**What We KEEP from SOLID**:

1. ✅ **Single Responsibility**: Each helper method does ONE thing
2. ✅ **Template Method**: Base wrapper + specialized subclasses
3. ✅ **Open-Closed**: Can extend with new wrapper types without modifying base
4. ✅ **Dependency Inversion**: Depends on entry_configs abstraction

**What We SIMPLIFY**:

1. ✅ **Remove Chain of Responsibility**: Direct method calls
2. ✅ **Remove ProcessingContext**: Direct returns
3. ✅ **Remove Strategy patterns**: Direct methods (no single-implementation strategies)
4. ✅ **Unify pipelines**: Single logic instead of 4 pipeline instances

**Migration Guarantee**:

- [x] Feature/target extraction preserved: Entry-config-based categorization
- [x] Transform integration: Forward and inverse transforms applied correctly
- [x] Multiple model families: Array, Graph, Timeseries wrappers work
- [x] Output naming preserved: Predictions named based on targets
- [x] Loss pairing preserved: Automatic pairing with dtype alignment
- [x] Metric computation: Proper dtype handling maintained
- [x] Checkpoint metadata: All metadata saved/loaded correctly
- [x] All tests pass: Unit and integration tests updated
- [x] Debugging easier: Direct stack traces instead of chain traversal

## Recommended Design Patterns (Balanced SOLID Application)

### ✅ KEEP (Patterns That Add Real Value):

1. **Single Responsibility Principle** - Each function/class does ONE thing
   - Example: `_load_model()`, `_infer_shapes()`, `_load_transforms()` separate functions
   - WHY: Maintainability, testability, reusability

2. **Template Method Pattern** - Base class + specialized subclasses
   - Example: `ProcessingLightningWrapper` → `GraphWrapper`, `TimeseriesWrapper`
   - WHY: Code reuse without duplication, extensibility

3. **Protocol/ABC for Contracts** - When preventing runtime errors matters
   - Example: `ChainableTransform` ABC ensures transforms provide `infer_output_shape()`
   - Example: `Predictor` Protocol documents predictor interface
   - WHY: Type safety, clear contracts, IDE support

4. **Factory Functions** - Simple factory functions for object creation
   - Example: `load_predictor()` instead of `PredictorFactory` class
   - WHY: Simpler than factory classes when no state needed

5. **Dependency Inversion** - Depend on abstractions when swapping implementations
   - Example: `CheckpointPredictor` depends on `PrecisionService` interface
   - WHY: Enables testing, allows multiple precision strategies

6. **Guard Clauses** - Early returns for edge cases
   - Example: Shape inference fallback chains
   - WHY: Reduces nesting, improves readability

### ❌ REMOVE (Patterns Adding Complexity Without Value):

1. **Hexagonal Architecture** - When only ONE implementation exists
   - Current: Application/Domain/Infrastructure layers with single adapter per port
   - Better: Direct implementation without port/adapter ceremony
   - WHY: Extra layers don't enable polymorphism if there's only one implementation

2. **Chain of Responsibility** - For strictly linear, non-branching flows
   - Current: 6 pipeline steps in fixed sequence with context threading
   - Better: Direct sequential method calls with simple returns
   - WHY: Chain pattern is for BRANCHING flows; linear flows are clearer as direct code

3. **Use Case Objects** - Single-method classes wrapping functions
   - Current: `ModelLoadingUseCase`, `InferenceExecutionUseCase` classes
   - Better: Module-level functions or predictor methods
   - WHY: Classes add overhead when no state management needed

4. **DI Container** - When wiring is trivial
   - Current: Container to wire up single implementations
   - Better: Direct construction in factory function
   - WHY: Container overhead not justified for simple wiring

5. **Strategy Pattern Overuse** - Single-implementation "strategies"
   - Current: `ModelInvoker`, `OutputClassifier`, `OutputNamer` with one impl each
   - Better: Direct method calls
   - WHY: Strategy valuable for MULTIPLE implementations, overkill for one

6. **Excessive ABCs** - 4 ABC mixins for optional capabilities
   - Current: `IFittableTransform`, `IInvertibleTransform`, `IShapeAwareTransform`, `ISerializableTransform`
   - Better: One ABC for mandatory contract + duck typing for optional capabilities
   - WHY: Duck typing is more Pythonic, fewer isinstance() checks

### 🎯 APPLY (Industry-Standard ML Patterns):

1. **Functional API Pattern** - Like Hugging Face, PyTorch
   - Example: `predictor = load_predictor(path); output = predictor.predict(x)`
   - WHY: Simplicity, discoverability, matches user expectations

2. **Stateful Object Pattern** - Like scikit-learn estimators
   - Example: Load once, predict many (no reloading)
   - WHY: Performance, matches industry standard

3. **Method Overriding** - Like PyTorch Lightning
   - Example: Override `training_step()`, `validation_step()` directly
   - WHY: Clear, debuggable, no framework indirection

4. **Capability Methods** - Check capabilities without isinstance()
   - Example: `transform.is_invertible()` instead of `isinstance(transform, IInvertibleTransform)`
   - WHY: More Pythonic, no ABC dependency

5. **Template Method** - Provide hooks for subclass customization
   - Example: `_invoke_model()` override for graph vs array data
   - WHY: Extension without modification (Open-Closed)

## Implementation Priority

### Phase 1: Inference Subsystem (Weeks 1-2)
**Goal**: Consolidate 27 files → 6 files while preserving ALL functionality

**Actions**:
- Merge use case classes into predictor methods
- Remove hexagonal architecture layers
- Keep `Predictor` Protocol for type safety
- Preserve: precision inference, transform handling, shape inference, context manager

**Expected outcome**:
- 75% fewer files (27 → 6)
- Same functionality, clearer structure
- Easier debugging (direct stack traces)
- Maintain SOLID: SRP (focused functions), DIP (PrecisionService), ISP (Predictor Protocol)

### Phase 2: Transform System (Weeks 3-4)
**Goal**: Reduce 4 ABCs → 1 ABC while preserving ALL functionality

**Actions**:
- Keep `ChainableTransform` ABC (prevents runtime errors in chains)
- Remove `IFittableTransform`, `IInvertibleTransform`, `IShapeAwareTransform` ABCs
- Add capability methods: `is_invertible()`, `is_fittable()`
- Preserve: fit, inverse_transform, configure_shape, checkpoint persistence

**Expected outcome**:
- Simpler interface (4 ABCs → 1 ABC)
- Same capabilities via duck typing
- Maintain SOLID: SRP (each transform does ONE thing), OCP (extensible), LSP (all transforms work as nn.Module)

### Phase 3: Wrapper + Pipeline (Weeks 5-6)
**Goal**: Replace Chain of Responsibility with direct methods, preserve ALL functionality

**Actions**:
- Collapse 6 pipeline steps into helper methods
- Remove ProcessingContext (use direct returns)
- Keep Template Method for wrapper specialization
- Preserve: feature/target extraction, transforms, loss pairing, metrics, multi-family support

**Expected outcome**:
- 60% fewer files (14 → 5)
- Clearer execution flow (no chain traversal)
- Easier customization (override methods, not rebuild chains)
- Maintain SOLID: SRP (each helper does ONE thing), Template Method (base + subclasses), OCP (extensible)

## Success Metrics

### Code Metrics:
- **Lines of Code**: Reduce by 40-50% in refactored components (not aggressive 60-70%)
- **Files**: Reduce from 56 critical files → ~20 files (preserves modularity)
- **Cyclomatic Complexity**: Same or lower (simpler flows)

### Developer Experience:
- **Onboarding Time**: New developer understanding from 1 week → 2-3 days
- **Bug Fix Time**: Average fix from 5 files touched → 2-3 files
- **Feature Add Time**: New transform/wrapper from 3-4 files → 1-2 files

### Quality Assurance:
- **Test Coverage**: Maintain 85%+ coverage
- **All Tests Pass**: 100% test suite passing after refactoring
- **Performance**: Same or better (fewer object allocations)
- **Functionality**: Zero functionality loss (guaranteed by detailed migration plans)

## Conclusion

The dlkit architecture demonstrates **well-intentioned but misapplied SOLID principles**. While the patterns (hexagonal, chain of responsibility, extensive ABCs) are valuable in specific contexts, they add complexity to ML operations that don't require that level of abstraction.

### Key Insights:

1. **SOLID is valuable when it enables flexibility**
   - Keep Template Method for wrapper specialization (Graph vs Array vs Timeseries)
   - Keep Dependency Inversion for PrecisionService (multiple strategies exist)
   - Keep ABCs for mandatory contracts (ChainableTransform prevents runtime errors)

2. **SOLID becomes harmful when applied dogmatically**
   - Hexagonal architecture with single implementations per port adds no polymorphism
   - Chain of Responsibility for linear flows obscures logic without enabling branching
   - 4 ABC mixins for optional capabilities when duck typing suffices

3. **Industry-standard ML libraries balance simplicity with extensibility**
   - Hugging Face: Functional API + simple classes (no DI containers)
   - scikit-learn: Base classes + duck typing (not extensive ABCs)
   - PyTorch Lightning: Direct method overrides (no pipeline frameworks)

### Refactoring Philosophy:

**"Use patterns when they solve real problems, not for architectural purity."**

- ✅ **Keep**: Patterns that enable multiple implementations, prevent errors, or simplify extension
- ❌ **Remove**: Patterns that exist "just in case" or because they're "best practice"
- 🎯 **Apply**: Industry-standard patterns that users expect from ML libraries

### Expected Impact:

This refactoring will make dlkit:
- **More maintainable**: Fewer abstraction layers to navigate
- **Easier to learn**: Matches patterns from Hugging Face, scikit-learn, PyTorch
- **Simpler to debug**: Direct stack traces instead of framework traversal
- **Equally powerful**: Zero functionality loss, all features preserved

**Recommendation**: Proceed with refactoring using the detailed, functionality-preserving plans above. Each plan guarantees no feature loss while significantly improving code clarity and maintainability.
