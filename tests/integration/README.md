# DLKit Integration Tests

This directory contains comprehensive end-to-end integration tests for the dlkit training and inference workflows.

## Overview

The integration tests verify the complete pipeline from settings configuration to final results across all major dlkit workflows:

1. **Training with vanilla strategy** - Basic training without external services
2. **Training with MLflow strategy** - Training with experiment tracking
3. **Optuna optimization strategy** - Hyperparameter optimization workflows
4. **Inference workflow** - Model loading and prediction generation

## Test Structure

```
tests/integration/
├── README.md                                    # This file
├── conftest.py                                  # Shared fixtures and test data
├── test_basic_integration.py                    # Simple, working integration tests
├── test_mlflow_training_integration.py          # MLflow workflow tests
├── test_optuna_optimization_integration.py      # Optuna optimization tests
└── test_inference_workflow_integration.py       # Inference workflow tests
```

## Key Design Principles

### Fixtures and Test Data
- **All test data is created via fixtures** - Never create data inside test functions
- **Modular, composable fixtures** in `conftest.py` for maximum reusability
- **Uses `tmp_path` fixture** exclusively for temporary paths (no `tempfile` package)
- **Minimal datasets** (50-100 samples) for fast execution
- **Small models** (4 input dims, 2 output dims, 8 hidden units) for speed

### Test Categories

#### Fast Tests (default)
- Use `fast_dev_run=True` for Lightning trainer
- Minimal epochs (1-2) and trials (3-5)
- Small datasets and models
- Target execution time: < 30 seconds per test

#### Slow Tests (`@pytest.mark.slow`)
- Longer training with more epochs
- More comprehensive optimization trials  
- Target execution time: 1-5 minutes per test
- Run with: `pytest -m slow`
- Skip with: `pytest -m "not slow"`

### Configuration Patterns

All tests use TOML configurations that follow this pattern:

```toml
[SESSION]
name = "test_name"
inference = false  # or true for inference tests
seed = 42

[DATASET]
name = "FlexibleDataset" 
root_dir = "/tmp/test_data"

[[DATASET.features]]
name = "X"
path = "/tmp/test_data/features.npy"

[[DATASET.targets]]
name = "y"
path = "/tmp/test_data/targets.npy"

[DATASET.split]
filepath = "/tmp/test_data/split.txt"

[MODEL]
name = "ConstantWidthFFNN"
module_path = "dlkit.core.models.nn.ffnn.simple"
# checkpoint = "/path/to/model.ckpt"  # for inference

[MODEL.params]
input_dim = 4
output_dim = 2
hidden_dims = [8]

[TRAINING.trainer]
fast_dev_run = true
enable_progress_bar = false
enable_model_summary = false

# Strategy-specific sections:
[MLFLOW]      # for MLflow tests
[OPTUNA]      # for optimization tests
```

## Test Files

### `test_basic_integration.py`
Simple, robust tests that manually create all test data and configurations. These tests:
- Avoid complex fixture dependencies
- Use absolute paths in configurations
- Focus on end-to-end workflow verification
- Should always pass if the core system is functional

**Key Tests:**
- `test_vanilla_training_end_to_end()` - Basic training workflow
- `test_mlflow_training_basic()` - MLflow integration
- `test_inference_basic_workflow()` - Inference pipeline

### `test_mlflow_training_integration.py`
Comprehensive MLflow training integration tests including:
- Complete MLflow training pipeline
- Model registration (optional)
- Fallback behavior when MLflow unavailable
- Server health checking
- Auto-detection from settings
- Training metrics preservation
- Invalid configuration handling

### `test_optuna_optimization_integration.py` 
Optuna hyperparameter optimization tests including:
- Complete optimization pipeline with study creation
- Custom sampler/pruner configuration  
- Study persistence with storage backends
- MLflow + Optuna integration (nested runs)
- Objective direction handling
- Model hyperparameter sampling

### `test_inference_workflow_integration.py`
End-to-end inference workflow tests including:
- Inference from pre-trained checkpoints
- Train-then-infer workflows
- Error handling (missing/corrupted checkpoints)
- Batch prediction generation
- Performance with larger datasets

## Fixtures Reference

### Core Fixtures (`conftest.py`)

#### Data Creation
- `minimal_dataset(tmp_path)` - Creates small synthetic dataset (100 samples, 4 features, 2 targets)
- `minimal_model_checkpoint(tmp_path)` - Creates simple model checkpoint for inference

#### Configuration Generation  
- `base_config_content(minimal_dataset, tmp_path)` - Base TOML configuration
- `mlflow_config_content(base_config_content, tmp_path)` - Adds MLflow configuration
- `optuna_config_content(base_config_content, tmp_path)` - Adds Optuna configuration
- `inference_config_content(base_config_content, checkpoint)` - Inference configuration

#### Settings Objects
- `training_settings(...)` - Loaded GeneralSettings for training
- `mlflow_settings(...)` - GeneralSettings with MLflow enabled
- `optuna_settings(...)` - GeneralSettings with Optuna enabled
- `inference_settings(...)` - GeneralSettings for inference mode

#### Test Helpers
- `expected_training_metrics()` - Expected result structure validation
- `expected_inference_result()` - Expected inference result structure
- `create_test_training_result()` - Factory for TrainingResult instances

## Running the Tests

### All Integration Tests
```bash
# Run all integration tests
uv run pytest tests/integration/ -v

# Run with coverage
uv run pytest tests/integration/ --cov=dlkit.interfaces.api --cov=dlkit.runtime.workflows

# Skip slow tests
uv run pytest tests/integration/ -m "not slow"
```

### Specific Test Categories
```bash
# Only basic tests (fastest)
uv run pytest tests/integration/test_basic_integration.py -v

# Only MLflow tests
uv run pytest tests/integration/test_mlflow_training_integration.py -v

# Only slow tests
uv run pytest tests/integration/ -m slow -v
```

### Debugging Failed Tests
```bash
# Verbose output with full tracebacks
uv run pytest tests/integration/test_basic_integration.py -v --tb=long

# Stop on first failure
uv run pytest tests/integration/ -x

# Run specific test with output capture disabled
uv run pytest tests/integration/test_basic_integration.py::TestBasicIntegration::test_vanilla_training_end_to_end -v -s
```

## Current Status

### ✅ Completed
- Integration test directory structure
- Comprehensive fixture system with proper dependency management
- Complete test suites for all major workflows
- Proper pytest configuration with custom marks
- Basic integration tests that avoid complex fixture dependencies

### ⚠️ Known Issues
- Some tests may fail due to Python type union operators (`str | None` syntax)
- Complex fixture dependencies in the comprehensive test files need debugging
- MLflow and Optuna tests depend on external package availability

### 🔄 Recommended Next Steps
1. **Fix type union issues** - Update code to use `Union[str, None]` for Python < 3.10 compatibility
2. **Debug fixture dependencies** - Resolve path resolution and dependency chain issues
3. **Add more error scenarios** - Test network failures, corrupted configs, etc.
4. **Performance benchmarks** - Add timing assertions for performance regression detection
5. **CI/CD integration** - Configure these tests to run in continuous integration

## Test Design Philosophy

These integration tests follow several key principles:

- **Good-path first**: Focus on successful workflow execution before error cases
- **Fast by default**: Use minimal data and quick configurations unless marked `@pytest.mark.slow`
- **Functional separation**: Separate data creation (fixtures) from test logic (test functions)
- **SOLID principles**: Single responsibility per test, dependency inversion via fixtures
- **No external dependencies**: Tests create all necessary data and don't rely on external services
- **Clear assertions**: Test outcomes, not internal implementation details

The integration tests complement the unit tests by verifying that all components work together correctly in realistic usage scenarios.
