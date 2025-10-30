"""Configuration service for template generation and validation following SOLID principles.

This service handles configuration template generation by introspecting Pydantic models
rather than using static dictionaries. It follows the Single Responsibility Principle
by focusing solely on configuration-related operations.
"""

from __future__ import annotations

from typing import Any, Literal, get_origin, get_args
from pathlib import Path
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from tomlkit import document, table, dumps, comment


TemplateKind = Literal["training", "inference", "mlflow", "optuna"]


class ConfigurationService:
    """Service for configuration template generation and validation.

    Follows SOLID principles:
    - Single Responsibility: Configuration template operations only
    - Open/Closed: Extensible for new template types
    - Dependency Inversion: Works with Settings abstractions
    """

    @classmethod
    def generate_template(cls, template_type: TemplateKind) -> str:
        """Generate configuration template from Pydantic models.

        Args:
            template_type: Type of template to generate

        Returns:
            str: TOML configuration template with comments and placeholders
        """
        template_dict = cls._build_template_dict(template_type)
        return cls._render_toml(template_dict, kind=template_type)

    @classmethod
    def _build_template_dict(cls, template_type: TemplateKind) -> dict[str, Any]:
        """Build template dictionary with essential user-configurable fields only.

        Skips complex model fields with extra='allowed' that have tons of dynamic fields.
        """
        # Build minimal essential sections manually to avoid model complexity
        template_dict = {}

        # Add SESSION section with only essential fields
        template_dict["SESSION"] = {
            "name": "my_session",
            "inference": False,  # Will be customized per template type
            "seed": 42,
            "precision": "medium",
        }

        # Add essential MODEL section (simplified)
        template_dict["MODEL"] = {"name": "your.model.class"}

        # Add a free-form EXTRAS section for user helpers
        template_dict["EXTRAS"] = {"example_key": "user_defined_value"}

        # Add other sections based on template type
        if template_type == "training":
            return cls._customize_for_training(template_dict)
        elif template_type == "inference":
            return cls._customize_for_inference(template_dict)
        elif template_type == "mlflow":
            return cls._customize_for_mlflow(template_dict)
        elif template_type == "optuna":
            return cls._customize_for_optuna(template_dict)
        else:
            raise ValueError(f"Unknown template type: {template_type}")

    @classmethod
    def _extract_model_fields(cls, model_class: type[BaseModel]) -> dict[str, Any]:
        """Extract fields from a Pydantic model with appropriate placeholder values."""
        result = {}

        for field_name, field_info in model_class.model_fields.items():
            # Convert field name to uppercase for TOML sections
            toml_name = field_name.upper()

            # Handle nested models (check if annotation has model_fields)
            annotation = field_info.annotation

            # Handle Optional types (Union[X, None])
            origin = get_origin(annotation)
            if origin is not None:
                args = get_args(annotation)
                if len(args) == 2 and type(None) in args:
                    # This is Optional[X], get the non-None type
                    annotation = next(arg for arg in args if arg is not type(None))

            # Check if this is a nested Pydantic model
            if hasattr(annotation, "model_fields"):
                # This is a nested Pydantic model
                nested_dict = cls._extract_model_fields(annotation)
                if nested_dict:  # Only include if it has content
                    result[toml_name] = nested_dict
            else:
                # Generate placeholder value for simple fields
                placeholder = cls._generate_placeholder_value(field_info)
                if placeholder is not None:
                    result[toml_name] = placeholder

        return result

    @classmethod
    def _generate_placeholder_value(cls, field_info: FieldInfo) -> Any:
        """Generate type-appropriate placeholder values since None isn't valid TOML."""
        from pydantic_core import PydanticUndefined

        annotation = field_info.annotation
        default = field_info.default

        # Handle Optional types (Union[X, None])
        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            if len(args) == 2 and type(None) in args:
                # This is Optional[X], get the non-None type
                annotation = next(arg for arg in args if arg is not type(None))

        # Skip fields with no default (will be handled as required fields)
        if default is PydanticUndefined:
            return None

        # Return actual default if it's not None or a factory
        if default is not None and not callable(default):
            return default

        # Generate type-appropriate placeholders
        if annotation is str:
            return "placeholder_value"
        elif annotation is int:
            return 0
        elif annotation is float:
            return 0.0
        elif annotation is bool:
            return False
        elif annotation is Path or getattr(annotation, "__module__", "").startswith("pathlib"):
            return "./path/placeholder"
        else:
            # For complex types, return a placeholder string
            return (
                f"<{annotation.__name__ if hasattr(annotation, '__name__') else str(annotation)}>"
            )

    @classmethod
    def _customize_for_training(cls, base_dict: dict) -> dict:
        """Customize template for training workflow."""
        result = base_dict.copy()

        # Set training-specific values
        if "SESSION" in result and isinstance(result["SESSION"], dict):
            result["SESSION"]["inference"] = False

        # Add essential training sections
        result["TRAINING"] = {"trainer": {"max_epochs": 100, "accelerator": "auto"}}

        result["DATAMODULE"] = {"name": "your.datamodule.class"}

        result["DATASET"] = {"name": "your.dataset.class"}

        # Paths resolve relative to DLKIT_ROOT_DIR or CWD; no PATHS section required.

        return result

    @classmethod
    def _customize_for_inference(cls, base_dict: dict) -> dict:
        """Customize template for inference workflow."""
        result = base_dict.copy()

        # Set inference-specific values
        if "SESSION" in result and isinstance(result["SESSION"], dict):
            result["SESSION"]["inference"] = True

        # Add checkpoint for inference
        result["MODEL"]["checkpoint"] = "./model.ckpt"

        # No special inference section - use SESSION.inference boolean only

        # Paths resolve relative to DLKIT_ROOT_DIR or CWD; no PATHS section required.

        return result

    @classmethod
    def _customize_for_mlflow(cls, base_dict: dict) -> dict:
        """Customize template for MLflow tracking workflow."""
        # Start with training template
        result = cls._customize_for_training(base_dict)

        # Add MLflow configuration
        result["MLFLOW"] = {
            "enabled": True,
            "server": {"host": "localhost", "port": 5000},
            "client": {
                "experiment_name": "my_experiment",
                "run_name": "my_run",
                "register_model": True,
            },
        }

        return result

    @classmethod
    def _customize_for_optuna(cls, base_dict: dict) -> dict:
        """Customize template for Optuna optimization workflow."""
        # Start with MLflow template (includes training)
        result = cls._customize_for_mlflow(base_dict)

        # Add Optuna configuration
        result["OPTUNA"] = {
            "enabled": True,
            "n_trials": 100,
            "study_name": "my_study",
            "storage": cls._get_default_optuna_storage(),
        }

        return result

    @classmethod
    def _get_default_optuna_storage(cls) -> str:
        """Get default Optuna storage URL under .dlkit directory.

        Returns:
            Default Optuna storage URL
        """
        from dlkit.tools.config.environment import DLKitEnvironment
        from dlkit.interfaces.servers.path_resolution import ServerPathResolver

        path_resolver = ServerPathResolver(DLKitEnvironment())
        return path_resolver.get_default_optuna_storage_url()

    @classmethod
    def _get_environment_aware_output_dir(cls) -> str:
        """Get environment-aware default output directory.

        Returns:
            Default output directory path respecting environment configuration
        """
        from dlkit.tools.config.environment import DLKitEnvironment

        env = DLKitEnvironment()
        return str(env.get_root_path() / "output")

    @classmethod
    def _get_field_comments(cls, template_type: TemplateKind) -> dict[str, str]:
        """Get field comments based on template type."""
        base_comments = {
            "SESSION.name": "Human-readable run/session name (for logs and tracking)",
            "SESSION.inference": "Run in inference mode when true",
            "SESSION.seed": "Random seed for reproducibility",
            "SESSION.precision": "Computation precision preset (e.g., medium, high)",
            "MODEL.name": "Model class path or registry alias",
            "TRAINING.trainer.max_epochs": "Maximum number of epochs (Lightning Trainer)",
            "TRAINING.trainer.accelerator": "Hardware accelerator: cpu | gpu | auto | tpu",
            "DATAMODULE.name": "DataModule class path or alias (dataflow loading)",
            "DATASET.name": "Dataset class path or alias",
            "EXTRAS.example_key": "Free-form user settings for custom scripts; ignored by core",
        }

        if template_type == "inference":
            base_comments.update({
                "MODEL.checkpoint": "Path to trained model checkpoint (required for inference)",
            })

        if template_type in ["mlflow", "optuna"]:
            base_comments.update({
                "MLFLOW.enabled": "Enable MLflow experiment tracking",
                "MLFLOW.server.host": "MLflow server host",
                "MLFLOW.server.port": "MLflow server port",
                "MLFLOW.client.experiment_name": "MLflow experiment name",
                "MLFLOW.client.run_name": "MLflow run name",
            })

        if template_type == "optuna":
            base_comments.update({
                "OPTUNA.enabled": "Enable Optuna hyperparameter optimization",
                "OPTUNA.n_trials": "Number of optimization trials to run",
                "OPTUNA.study_name": "Name for the Optuna study",
                "OPTUNA.storage": "Database URL for Optuna study storage",
            })

        return base_comments

    @classmethod
    def _render_toml(cls, template: dict, *, kind: TemplateKind) -> str:
        """Render template dictionary as TOML with comments."""
        # Deterministic section order
        section_order = [
            "SESSION",
            "MODEL",
            "MLFLOW",
            "OPTUNA",
            "TRAINING",
            "DATAMODULE",
            "DATASET",
            "EXTRAS",
            "SESSION.inference",
        ]

        doc = document()
        comments = cls._get_field_comments(kind)

        for section_key in section_order:
            if section_key not in template:
                continue

            content = template[section_key]
            if content is None:
                continue

            # Handle dotted sections like SESSION.inference
            if "." in section_key:
                parent, child = section_key.split(".", 1)
                if parent not in doc:
                    doc.add(parent, table())
                parent_table = doc[parent]
                child_table = table()

                if isinstance(content, dict):
                    for k, v in content.items():
                        dotted_key = f"{section_key}.{k}"
                        if dotted_key in comments:
                            child_table.add(comment(comments[dotted_key]))
                        child_table.add(k, v)

                parent_table.add(child, child_table)
                continue

            # Regular sections
            if isinstance(content, dict):
                section_table = table()

                # Add scalar fields first
                for k, v in content.items():
                    if not isinstance(v, dict):
                        dotted_key = f"{section_key}.{k}"
                        if dotted_key in comments:
                            section_table.add(comment(comments[dotted_key]))
                        section_table.add(k, v)

                doc.add(section_key, section_table)

                # Add nested subtables
                for k, v in content.items():
                    if isinstance(v, dict):
                        subtable = table()
                        for subk, subv in v.items():
                            dotted_key = f"{section_key}.{k}.{subk}"
                            if dotted_key in comments:
                                subtable.add(comment(comments[dotted_key]))
                            subtable.add(subk, subv)
                        doc[section_key].add(k, subtable)

        return dumps(doc)
