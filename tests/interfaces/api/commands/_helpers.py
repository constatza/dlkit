"""Pure helper functions for convert command tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.interfaces.api.domain import WorkflowError


def create_shape_spec(dims: list[int]) -> str:
    """Create shape specification string from dimensions.

    Args:
        dims: List of dimension sizes

    Returns:
        str: Shape specification string (e.g., "3,224,224")
    """
    return ",".join(str(d) for d in dims)


def create_multi_input_shape_spec(shape_lists: list[list[int]]) -> str:
    """Create multi-input shape specification string.

    Args:
        shape_lists: List of dimension lists for multiple inputs

    Returns:
        str: Multi-input shape specification (e.g., "3,224,224;100")
    """
    return ";".join(create_shape_spec(dims) for dims in shape_lists)


def parse_expected_shapes(shape_spec: str, batch_size: int) -> list[tuple[int, ...]]:
    """Parse shape specification into expected tuples with batch dimension.

    Args:
        shape_spec: Shape specification string
        batch_size: Batch size to prepend

    Returns:
        list: List of shape tuples with batch dimension
    """
    parts = [p.strip() for p in shape_spec.split(";") if p.strip()]
    shapes: list[tuple[int, ...]] = []

    for part in parts:
        dims = [d.strip() for d in part.replace("x", ",").split(",") if d.strip()]
        int_dims = [int(d) for d in dims]
        shapes.append(tuple([batch_size, *int_dims]))

    return shapes


def is_valid_onnx_path(path: Path) -> bool:
    """Check if path is suitable for ONNX output.

    Args:
        path: Path to check

    Returns:
        bool: True if path is valid for ONNX output
    """
    return not path.is_dir()


def extract_batch_dimensions(shapes: list[tuple[int, ...]]) -> set[int]:
    """Extract unique batch dimensions from shape tuples.

    Args:
        shapes: List of shape tuples

    Returns:
        set: Set of unique batch dimensions (first dimension of each shape)
    """
    return {shape[0] for shape in shapes}


def validate_dynamic_axes_structure(
    input_names: list[str], output_names: list[str] = None
) -> dict[str, dict[int, str]]:
    """Create expected dynamic axes structure for ONNX export.

    Args:
        input_names: List of input tensor names
        output_names: List of output tensor names (defaults to ["output"])

    Returns:
        dict: Dynamic axes dictionary for ONNX export
    """
    if output_names is None:
        output_names = ["output"]

    dynamic_axes = {}

    # Add dynamic batch dimension for each input
    for name in input_names:
        dynamic_axes[name] = {0: "batch"}

    # Add dynamic batch dimension for each output
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    return dynamic_axes


# Validation command helpers


def create_strategy_detection_test_cases() -> list[tuple[str | None, bool, bool, str]]:
    """Create test cases for strategy auto-detection.

    Returns:
        list: List of (input_strategy, mlflow_active, optuna_active, expected_strategy) tuples
    """
    return [
        # Explicit strategy cases
        ("vanilla", False, False, "vanilla"),
        ("mlflow", False, False, "mlflow"),
        ("optuna", False, False, "optuna"),
        ("mlflow_training", False, False, "mlflow_training"),
        ("optuna_optimization", False, False, "optuna_optimization"),
        # Auto-detection cases (precedence: optuna > mlflow > vanilla)
        (None, False, False, "vanilla"),
        (None, True, False, "mlflow"),
        (None, False, True, "optuna"),
        (None, True, True, "optuna"),  # Optuna takes precedence
    ]


def create_validation_error_test_cases() -> list[tuple[str, str, dict[str, str]]]:
    """Create test cases for validation error scenarios.

    Returns:
        list: List of (section_name, expected_error_message, context_expectations) tuples
    """
    return [
        ("MODEL", "[MODEL] section is required", {"command": "validate_config"}),
        ("DATASET", "[DATASET] section is required", {"command": "validate_config"}),
        ("DATAMODULE", "[DATAMODULE] section is required", {"command": "validate_config"}),
        ("TRAINING", "[TRAINING] section is required for training", {"command": "validate_config"}),
        (
            "MODEL.checkpoint",
            "[MODEL.checkpoint] is required for inference mode",
            {"command": "validate_config"},
        ),
    ]


def normalize_strategy_name(strategy: str | None) -> str | None:
    """Normalize strategy name for comparison (case insensitive).

    Args:
        strategy: Strategy name to normalize

    Returns:
        str | None: Normalized strategy name or None
    """
    return strategy.lower() if strategy else None


def is_inference_mode(session_mock: Any) -> bool:
    """Check if settings indicate inference mode.

    Args:
        session_mock: Mock SESSION object

    Returns:
        bool: True if in inference mode, False otherwise
    """
    return session_mock and hasattr(session_mock, "inference") and session_mock.inference


def should_require_training_section(session_mock: Any) -> bool:
    """Determine if TRAINING section should be required.

    Args:
        session_mock: Mock SESSION object

    Returns:
        bool: True if TRAINING section is required, False otherwise
    """
    # TRAINING required if not in inference mode and SESSION exists
    return not is_inference_mode(session_mock)


def should_require_checkpoint(session_mock: Any) -> bool:
    """Determine if MODEL.checkpoint should be required.

    Args:
        session_mock: Mock SESSION object

    Returns:
        bool: True if checkpoint is required, False otherwise
    """
    return is_inference_mode(session_mock)


def create_import_error_scenarios() -> list[tuple[str, str, type]]:
    """Create test scenarios for import error handling.

    Returns:
        list: List of (strategy, module_name, exception_type) tuples
    """
    return [
        ("mlflow", "mlflow", ImportError),
        ("optuna", "optuna", ImportError),
        ("mlflow_training", "mlflow", ImportError),
        ("optuna_optimization", "optuna", ImportError),
        ("mlflow", "mlflow", ModuleNotFoundError),
        ("optuna", "optuna", ModuleNotFoundError),
        ("mlflow", "mlflow", AttributeError),
        ("optuna", "optuna", AttributeError),
    ]


def verify_workflow_error_structure(
    error: WorkflowError, expected_message_parts: list[str], expected_context_keys: list[str]
) -> bool:
    """Verify WorkflowError has expected structure and content.

    Args:
        error: WorkflowError instance to verify
        expected_message_parts: Parts that should be in error message
        expected_context_keys: Keys that should be in error context

    Returns:
        bool: True if error structure is valid, False otherwise
    """
    # Check message content
    for part in expected_message_parts:
        if part not in error.message:
            return False

    # Check context structure
    if not isinstance(error.context, dict):
        return False

    for key in expected_context_keys:
        if key not in error.context:
            return False

    return True


def create_settings_permutations() -> list[dict[str, Any]]:
    """Create various settings permutations for comprehensive testing.

    Returns:
        list: List of settings configuration dictionaries
    """
    base_sections = ["MODEL", "DATASET", "DATAMODULE"]

    permutations = []

    # Create permutations of missing required sections
    for i in range(len(base_sections)):
        config = {section: True for section in base_sections}
        config[base_sections[i]] = False  # Missing this section
        config["TRAINING"] = True
        config["inference"] = False
        permutations.append(config)

    # Training mode without TRAINING section
    config = {section: True for section in base_sections}
    config["TRAINING"] = False
    config["inference"] = False
    permutations.append(config)

    # Inference mode without checkpoint
    config = {section: True for section in base_sections}
    config["TRAINING"] = False
    config["inference"] = True
    config["checkpoint"] = False
    permutations.append(config)

    return permutations


def create_expected_input_names(num_inputs: int) -> list[str]:
    """Create expected input names for ONNX export.

    Args:
        num_inputs: Number of inputs

    Returns:
        list: List of input names
    """
    if num_inputs == 1:
        return ["input"]
    return [f"input{i}" for i in range(num_inputs)]


def validate_tensor_shape_consistency(tensor_shapes: list[tuple[int, ...]]) -> bool:
    """Validate that tensor shapes have consistent batch dimensions.

    Args:
        tensor_shapes: List of tensor shape tuples

    Returns:
        bool: True if all shapes have the same batch dimension
    """
    if not tensor_shapes:
        return True

    batch_dims = extract_batch_dimensions(tensor_shapes)
    return len(batch_dims) == 1


def create_mock_tensor_data(shape: tuple[int, ...]) -> dict[str, Any]:
    """Create mock tensor dataflow for testing.

    Args:
        shape: Tensor shape

    Returns:
        dict: Mock tensor attributes
    """
    return {"shape": shape, "dtype": "float32", "device": "cpu"}
