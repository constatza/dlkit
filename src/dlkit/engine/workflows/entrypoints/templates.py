"""Runtime-owned template generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast, get_args, get_origin

from pydantic import BaseModel
from pydantic import BaseModel as PydanticBaseModel
from pydantic.fields import FieldInfo
from tomlkit import comment, document, dumps, table

TemplateKind = Literal["training", "inference", "mlflow", "optuna"]


def generate_template(template_type: TemplateKind = "training") -> str:
    """Generate a TOML configuration template."""
    template_dict = _build_template_dict(template_type)
    return _render_toml(template_dict, kind=template_type)


def validate_template(
    template_content: str,
    template_type: TemplateKind | None = None,
) -> dict[str, Any]:
    """Validate TOML template content against GeneralSettings."""
    from dlkit.infrastructure.config import GeneralSettings

    errors: list[str] = []
    try:
        import tomlkit

        parsed = tomlkit.loads(template_content)
        try:
            GeneralSettings.model_validate(dict(parsed))
        except Exception as exc:
            errors.append(f"Settings validation failed: {exc}")
    except Exception as exc:
        errors.append(f"TOML parsing failed: {exc}")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "template_type": template_type,
    }


def _build_template_dict(template_type: TemplateKind) -> dict[str, Any]:
    template_dict: dict[str, Any] = {}
    template_dict["SESSION"] = {
        "name": "my_session",
        "inference": False,
        "seed": 42,
        "precision": "medium",
    }
    template_dict["MODEL"] = {"name": "your.model.class"}
    template_dict["EXTRAS"] = {"example_key": "user_defined_value"}

    match template_type:
        case "training":
            return _customize_for_training(template_dict)
        case "inference":
            return _customize_for_inference(template_dict)
        case "mlflow":
            return _customize_for_mlflow(template_dict)
        case "optuna":
            return _customize_for_optuna(template_dict)
        case _:
            raise ValueError(f"Unknown template type: {template_type}")


def _extract_model_fields(model_class: type[BaseModel]) -> dict[str, Any]:
    result = {}
    for field_name, field_info in model_class.model_fields.items():
        toml_name = field_name.upper()
        annotation = field_info.annotation

        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            if len(args) == 2 and type(None) in args:
                annotation = next(arg for arg in args if arg is not type(None))

        if isinstance(annotation, type) and issubclass(annotation, PydanticBaseModel):
            nested_dict = _extract_model_fields(annotation)
            if nested_dict:
                result[toml_name] = nested_dict
        else:
            placeholder = _generate_placeholder_value(field_info)
            if placeholder is not None:
                result[toml_name] = placeholder
    return result


def _generate_placeholder_value(field_info: FieldInfo) -> Any:
    from pydantic_core import PydanticUndefined

    annotation = field_info.annotation
    default = field_info.default

    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if len(args) == 2 and type(None) in args:
            annotation = next(arg for arg in args if arg is not type(None))

    if default is PydanticUndefined:
        return None
    if default is not None and not callable(default):
        return default
    if annotation is str:
        return "placeholder_value"
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    if annotation is Path or getattr(annotation, "__module__", "").startswith("pathlib"):
        return "./path/placeholder"
    name = getattr(annotation, "__name__", None) or repr(annotation)
    return f"<{name}>"


def _customize_for_training(base_dict: dict[str, Any]) -> dict[str, Any]:
    result = base_dict.copy()
    if "SESSION" in result and isinstance(result["SESSION"], dict):
        result["SESSION"]["inference"] = False
    result["TRAINING"] = {"trainer": {"max_epochs": 100, "accelerator": "auto"}}
    result["DATAMODULE"] = {"name": "your.datamodule.class"}
    result["DATASET"] = {"name": "your.dataset.class"}
    return result


def _customize_for_inference(base_dict: dict[str, Any]) -> dict[str, Any]:
    result = base_dict.copy()
    if "SESSION" in result and isinstance(result["SESSION"], dict):
        result["SESSION"]["inference"] = True
    result["MODEL"]["checkpoint"] = "./model.ckpt"
    return result


def _customize_for_mlflow(base_dict: dict[str, Any]) -> dict[str, Any]:
    result = _customize_for_training(base_dict)
    result["MLFLOW"] = {
        "experiment_name": "my_experiment",
        "run_name": "my_run",
        "register_model": True,
    }
    return result


def _customize_for_optuna(base_dict: dict[str, Any]) -> dict[str, Any]:
    result = _customize_for_mlflow(base_dict)
    result["OPTUNA"] = {
        "enabled": True,
        "n_trials": 100,
        "study_name": "my_study",
        "storage": _get_default_optuna_storage(),
    }
    return result


def _get_default_optuna_storage() -> str:
    from dlkit.infrastructure.io import locations

    return locations.optuna_storage_uri()


def _get_field_comments(template_type: TemplateKind) -> dict[str, str]:
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
        base_comments.update(
            {
                "MODEL.checkpoint": "Path to trained model checkpoint (required for inference)",
            }
        )
    if template_type in ["mlflow", "optuna"]:
        base_comments.update(
            {
                "MLFLOW.experiment_name": "MLflow experiment name",
                "MLFLOW.run_name": "MLflow run name",
            }
        )
    if template_type == "optuna":
        base_comments.update(
            {
                "OPTUNA.enabled": "Enable Optuna hyperparameter optimization",
                "OPTUNA.n_trials": "Number of optimization trials to run",
                "OPTUNA.study_name": "Name for the Optuna study",
                "OPTUNA.storage": "Database URL for Optuna study storage",
            }
        )
    return base_comments


def _render_toml(template: dict[str, Any], *, kind: TemplateKind) -> str:
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
    comments = _get_field_comments(kind)

    for section_key in section_order:
        if section_key not in template:
            continue

        content = template[section_key]
        if content is None:
            continue

        if "." in section_key:
            parent, child = section_key.split(".", 1)
            if parent not in doc:
                doc.add(parent, table())
            parent_table = cast(Any, doc[parent])
            child_table = table()

            if isinstance(content, dict):
                for key, value in content.items():
                    dotted_key = f"{section_key}.{key}"
                    if dotted_key in comments:
                        child_table.add(comment(comments[dotted_key]))
                    child_table.add(key, value)

            parent_table.add(child, child_table)
            continue

        if isinstance(content, dict):
            section_table = table()
            for key, value in content.items():
                if not isinstance(value, dict):
                    dotted_key = f"{section_key}.{key}"
                    if dotted_key in comments:
                        section_table.add(comment(comments[dotted_key]))
                    section_table.add(key, value)
            doc.add(section_key, section_table)

            for key, value in content.items():
                if isinstance(value, dict):
                    subtable = table()
                    for subkey, subvalue in value.items():
                        dotted_key = f"{section_key}.{key}.{subkey}"
                        if dotted_key in comments:
                            subtable.add(comment(comments[dotted_key]))
                        subtable.add(subkey, subvalue)
                    cast(Any, doc[section_key]).add(key, subtable)

    return dumps(doc)
