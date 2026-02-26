# DLKit Settings Test Suite

This comprehensive test suite provides thorough coverage of the DLKit settings module using SOLID principles, pytest fixtures, and property-based testing with Hypothesis.

## Test Structure

The test suite mirrors the settings module structure following best practices:

```
tests/settings/
├── README.md                          # This file
├── conftest.py                        # Global fixtures
├── test_general_settings.py           # General settings tests
├── test_session_settings.py           # Session settings tests  
├── test_mlflow_settings.py           # MLflow settings tests
├── test_hyperparameter_properties.py  # Property-based hyperparameter tests
├── test_integration.py               # Integration tests
├── test_standalone_core.py           # Standalone core functionality tests
├── core/
│   ├── conftest.py                    # Core module fixtures
│   ├── test_base_settings.py         # Base settings classes tests
│   ├── test_context.py               # BuildContext tests
│   └── test_factories.py             # Factory pattern tests
└── components/
    ├── conftest.py                    # Component fixtures
    └── test_model_components.py       # Model component tests
```

## Key Testing Principles

### 1. SOLID Principles Applied to Tests
- **Single Responsibility**: Each test function tests one specific behavior
- **Open/Closed**: Tests can be extended without modifying existing code
- **Dependency Inversion**: Uses fixtures and dependency injection
- **Interface Segregation**: Tests specific interfaces and contracts
- **Liskov Substitution**: Ensures subclasses work correctly

### 2. Fixture-Based Architecture
- **NEVER create data inside test functions**
- **ALWAYS use fixtures for test data**
- **Modular, composable fixtures** in conftest.py files
- **Separate data creation from test logic**

### 3. Property-Based Testing
- Uses Hypothesis for generating test cases
- Tests invariants across all hyperparameter types
- Ensures robust behavior with edge cases
- Validates mathematical properties and constraints

### 4. Functional Programming Style
- **Data fixtures**: Pure data creation without side effects
- **Pure functions**: Assertions and validation logic
- **Side-effectful actions**: Isolated in specific test functions

## Test Categories

### Core Infrastructure Tests (`core/`)

#### `test_base_settings.py`
- **BasicSettings**: Immutability, validation, serialization
- **ComponentSettings**: Dynamic component configuration 
- **HyperParameterSettings**: Optuna integration, hyperparameter sampling

#### `test_context.py`
- **BuildContext**: Dependency injection, override management
- Context creation and modification
- Override chaining and immutability

#### `test_factories.py`
- **ComponentFactory**: Abstract factory pattern
- **DefaultComponentFactory**: Standard component creation
- **ComponentRegistry**: Factory registration and management
- **FactoryProvider**: Global singleton access

### Settings Class Tests

#### `test_general_settings.py`
- Complete configuration loading from files
- Mode detection and validation
- Feature flag properties
- Configuration access methods
- Parser-backed TOML integration

#### `test_session_settings.py`
- Session mode management (training/inference/testing)
- Mode-specific validation
- Configuration access patterns
- Backward compatibility

#### `test_mlflow_settings.py`
- MLflow server and client configuration
- Tracking URI validation and defaults
- Active state detection
- Property accessors

#### `test_model_components.py`
- Model component configuration
- Hyperparameter specifications
- Checkpoint handling
- Shape inference
- Metric and loss components
- Wrapper component configuration

### Advanced Testing

#### `test_hyperparameter_properties.py`
- **Property-based tests** using Hypothesis
- Hyperparameter sampling invariants
- Type preservation across sampling
- Error handling for invalid specifications
- Integration across all settings classes

#### `test_integration.py`
- End-to-end workflows
- Factory pattern integration
- Settings loading and validation
- Cross-component interaction
- Build context application

#### `test_standalone_core.py`
- **Standalone functionality verification**
- Works without full package imports
- Core feature validation
- Dependency-free testing

## Running Tests

### Full Test Suite
```bash
# Run all settings tests
uv run pytest tests/settings/ -v

# Run with coverage
uv run pytest tests/settings/ --cov=dlkit.tools.config --cov-report=html
```

### Specific Test Categories
```bash
# Core functionality
uv run pytest tests/settings/core/ -v

# Component tests
uv run pytest tests/settings/components/ -v

# Property-based tests
uv run pytest tests/settings/test_hyperparameter_properties.py -v

# Integration tests
uv run pytest tests/settings/test_integration.py -v
```

### Standalone Testing
```bash
# Test core functionality without dependencies
python tests/settings/test_standalone_core.py
```

## Fixtures Reference

### Global Fixtures (`conftest.py`)

#### Data Fixtures
- `sample_config_data()`: Complete configuration dictionary
- `sample_model_config_data()`: Model configuration
- `sample_hyperparameter_data()`: Hyperparameter specifications
- `config_file_content()`: TOML configuration content

#### Utility Fixtures  
- `config_file(tmp_path)`: Temporary configuration file
- `mock_trial()`: Mock Optuna trial for hyperparameter testing
- `sample_build_context()`: BuildContext for dependency injection

#### Hypothesis Strategies
- `basic_settings_data()`: Generate BasicSettings data
- `hyperparameter_spec()`: Generate hyperparameter specifications
- `session_mode_strategy()`: Generate SessionMode values

### Core Fixtures (`core/conftest.py`)

#### Test Implementations
- `TestBasicSettings`: Concrete BasicSettings implementation
- `TestComponentSettings`: Concrete ComponentSettings implementation
- `TestHyperParameterSettings`: Concrete HyperParameterSettings implementation

#### Data Fixtures
- `basic_settings_data()`: BasicSettings configuration
- `component_settings_data()`: ComponentSettings configuration
- `hyperparameter_settings_data()`: HyperParameterSettings configuration
- `build_context_data()`: BuildContext configuration

#### Hypothesis Strategies
- `valid_component_name()`: Generate valid component names
- `hyperparameter_int_spec()`: Generate integer hyperparameter specs
- `hyperparameter_float_spec()`: Generate float hyperparameter specs
- `hyperparameter_categorical_spec()`: Generate categorical hyperparameter specs

### Component Fixtures (`components/conftest.py`)

#### Model Component Data
- `model_component_data()`: Basic model configuration
- `model_component_with_checkpoint_data()`: Model with checkpoint
- `hyperparameter_model_data()`: Model with hyperparameters

#### Other Component Data
- `metric_component_data()`: Metric configuration
- `loss_component_data()`: Loss function configuration
- `wrapper_component_data()`: Wrapper configuration
- `complex_wrapper_data()`: Advanced wrapper configuration

### Settings Package Fixtures (`settings/conftest.py`)

#### General Settings Data
- `sample_general_settings_data()`: Complete GeneralSettings configuration
- `inference_config_data()`: Inference mode configuration
- `invalid_inference_config_data()`: Invalid inference configuration

#### Advanced Configuration
- `sample_toml_config_advanced()`: Complex TOML configuration
- `malformed_toml_config()`: Invalid TOML for error testing
- `optuna_model_config()`: Configuration with Optuna model settings

## Best Practices Demonstrated

### 1. Test Data Management
```python
# ✅ Good: Using fixtures
def test_settings_creation(sample_config_data: Dict[str, Any]) -> None:
    settings = GeneralSettings(**sample_config_data)
    assert settings.SESSION.name == "test_session"

# ❌ Bad: Creating dataflow in test
def test_settings_creation() -> None:
    data = {"SESSION": {"name": "test_session"}}  # Don't do this
    settings = GeneralSettings(**data)
```

### 2. File Path Management
```python
# ✅ Good: Using tmp_path fixture
def test_config_loading(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    # Use tmp_path directly

# ❌ Bad: Using tempfile
def test_config_loading() -> None:
    import tempfile  # Don't do this
```

### 3. Property-Based Testing
```python
# ✅ Good: Testing invariants
@given(hyperparameter_spec())
def test_sampling_invariant(spec: Dict[str, Any]) -> None:
    # Test that sampling preserves expected properties
    
# ❌ Bad: Only testing specific cases
def test_sampling_specific() -> None:
    # Only tests one specific case
```

### 4. Error Testing
```python
# ✅ Good: Testing error conditions
def test_invalid_config_raises_error() -> None:
    with pytest.raises(ValidationError, match="specific message"):
        GeneralSettings(invalid_config)

# ❌ Bad: Not testing error cases
```

## Coverage Goals

- **Line Coverage**: >95% for all settings classes
- **Branch Coverage**: >90% for all conditional logic
- **Property Coverage**: All hyperparameter invariants tested
- **Integration Coverage**: All factory patterns tested
- **Error Coverage**: All validation errors tested

## Extending the Tests

When adding new settings classes or features:

1. **Add fixtures** in appropriate conftest.py
2. **Create property tests** for invariants
3. **Add integration tests** for cross-component interaction
4. **Test error conditions** thoroughly
5. **Follow SOLID principles** in test design

## Dependencies

- `pytest`: Test framework
- `hypothesis`: Property-based testing
- `pydantic`: Settings validation
- `pathlib`: File path handling (no tempfile!)

## Notes

- All tests use type hints following Google docstring style
- Tests are designed to run in any order (no dependencies)
- Fixtures provide clean separation between data and test logic
- Property-based tests catch edge cases traditional tests miss
- Integration tests verify SOLID principle compliance
- The standalone test verifies core functionality without package dependencies
