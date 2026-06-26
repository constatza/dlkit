"""Pure helper functions for CLI tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def create_toml_config(
    output_path: Path,
    *,
    experiment_name: str = "test_session",
    model_name: str = "TestModel",
    data_dir: str = "./test_data",
    enable_mlflow: bool = False,
    enable_optuna: bool = False,
    max_epochs: int = 10,
    additional_sections: dict[str, Any] | None = None,
) -> None:
    """Create a TOML configuration file with specified parameters.

    Args:
        output_path: Path where to write the config file
        experiment_name: Name for the experiment
        model_name: Name for the model
        data_dir: Data directory path
        enable_mlflow: Whether to enable MLflow
        enable_optuna: Whether to enable Optuna
        max_epochs: Number of training epochs
        additional_sections: Additional TOML sections to include
    """
    config_content = f"""
[run]
type = "{"search" if enable_optuna else "train"}"
seed = 42
precision = "medium"

[experiment]
name = "{experiment_name}"

[model]
name = "{model_name}"
module_path = "test.module"

[data]
root = "{data_dir}"

[training]
loss = "mse"

[training.trainer]
max_epochs = {max_epochs}
accelerator = "auto"
default_root_dir = "./outputs"
"""

    if enable_mlflow:
        config_content += """
[tracking]
backend = "mlflow"
uri = "sqlite:///test.db"
"""

    if enable_optuna:
        config_content += """
[search]
n_trials = 50
direction = "minimize"
objective = "val/loss"
space = { "training.optimizer.lr" = { type = "log_float", low = 1e-5, high = 1e-3 } }
"""

    if additional_sections:
        for section_name, section_data in additional_sections.items():
            config_content += f"\n[{section_name}]\n"
            for key, value in section_data.items():
                if isinstance(value, str):
                    config_content += f'{key} = "{value}"\n'
                elif isinstance(value, bool):
                    config_content += f"{key} = {str(value).lower()}\n"
                else:
                    config_content += f"{key} = {value}\n"

    output_path.write_text(config_content.strip())


def create_invalid_toml_config(output_path: Path) -> None:
    """Create an invalid TOML configuration file for error testing.

    Args:
        output_path: Path where to write the invalid config file
    """
    invalid_content = """
[INVALID_SECTION
missing_bracket = true
unclosed_string = "this is not closed
invalid_key with spaces = "value"
"""
    output_path.write_text(invalid_content)


def create_minimal_valid_config(output_path: Path) -> None:
    """Create the minimal valid configuration needed for testing.

    Args:
        output_path: Path where to write the config file
    """
    minimal_content = """
[run]
type = "train"

[experiment]
name = "minimal_test"
"""
    output_path.write_text(minimal_content)


def simulate_api_response(
    *,
    success: bool = True,
    execution_time: int = 1000,
    error_message: str | None = None,
    error_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate an API response for testing purposes.

    Args:
        success: Whether the API call was successful
        execution_time: Simulated execution time in milliseconds
        error_message: Error message if success=False
        error_context: Error context if success=False

    Returns:
        Dictionary representing an API response
    """
    if success:
        return {
            "is_success": True,
            "execution_time_ms": execution_time,
            "value": {"status": "completed", "message": "Operation completed successfully"},
        }
    return {
        "is_success": False,
        "execution_time_ms": execution_time,
        "error": {
            "message": error_message or "Operation failed",
            "context": error_context or {},
        },
    }


def create_batch_input_files(
    input_dir: Path, count: int = 3, file_extension: str = ".csv"
) -> list[Path]:
    """Create sample input files for batch processing tests.

    Args:
        input_dir: Directory to create files in
        count: Number of files to create
        file_extension: File extension to use

    Returns:
        List of paths to created files
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    for i in range(count):
        input_file = input_dir / f"input_{i}{file_extension}"
        if file_extension == ".csv":
            input_file.write_text(f"dataflow,label\n{i + 1},{i + 2}\n{i + 3},{i + 4}\n")
        elif file_extension == ".json":
            input_file.write_text(f'{{"id": {i}, "dataflow": [1, 2, 3]}}')
        else:
            input_file.write_text(f"Sample dataflow file {i}")

        created_files.append(input_file)

    return created_files


# Legacy IOResult helpers removed; direct exceptions/results are used now.
