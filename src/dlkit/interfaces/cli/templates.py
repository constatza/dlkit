"""Centralized config template builders and TOML rendering.

Provides dict builders for each template and a renderer using tomlkit.
This is the single source of truth used by the CLI and sync tools.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from tomlkit import aot, comment, document, dumps, table
from tomlkit.items import AoT, Table
from tomlkit.toml_document import TOMLDocument


def _get_default_optuna_storage() -> str:
    """Get default Optuna storage URL with environment awareness.

    Returns:
        Default Optuna storage URL resolved through environment settings
    """
    from dlkit.infrastructure.io import locations

    return locations.optuna_storage_uri()


def _get_environment_aware_output_dir() -> str:
    """Get environment-aware default output directory for templates.

    Returns:
        Default output directory path respecting environment configuration
    """
    from dlkit.infrastructure.config.environment import EnvironmentSettings

    env = EnvironmentSettings()
    return str(env.get_root_path() / "output")


TemplateKind = Literal["training", "inference", "mlflow", "optuna"]


def _dataset_template(*, include_targets: bool) -> dict[str, Any]:
    dataset: dict[str, Any] = {
        "name": "FlexibleDataset",
        "root_dir": "./data",
        "features": [
            {
                "name": "x",
                "path": "features.npy",
                "data_role": "feature",
                "field_role": "feature",
            }
        ],
    }
    if include_targets:
        dataset["targets"] = [
            {
                "name": "y",
                "path": "targets.npy",
                "data_role": "target",
                "field_role": "target",
            }
        ]
    return dataset


def build_training_template_dict() -> dict:
    return {
        "SESSION": {
            "name": "my_training_session",
            "workflow": "train",
            "seed": 42,
            "precision": "32",
            "root_dir": "./",
        },
        "MODEL": {
            "name": "your.model.class",
        },
        "TRAINING": {
            "trainer": {
                "max_epochs": 100,
                "accelerator": "auto",
            },
        },
        "DATAMODULE": {
            "name": "your.datamodule.class",
        },
        "DATASET": _dataset_template(include_targets=True),
    }


def build_inference_template_dict() -> dict:
    return {
        "SESSION": {
            "name": "my_inference_session",
            "workflow": "inference",
            "seed": 42,
            "precision": "32",
            "root_dir": "./",
        },
        "MODEL": {
            "name": "your.model.class",
            "checkpoint": "./model.ckpt",
        },
        "DATASET": _dataset_template(include_targets=False),
    }


def build_mlflow_template_dict() -> dict:
    return {
        "SESSION": {
            "name": "my_mlflow_session",
            "workflow": "train",
            "seed": 42,
            "precision": "32",
            "root_dir": "./",
        },
        "MODEL": {
            "name": "your.model.class",
        },
        "MLFLOW": {
            "experiment_name": "my_experiment",
            "run_name": "my_run",
            "register_model": True,
        },
        "TRAINING": {
            "trainer": {
                "max_epochs": 100,
                "accelerator": "auto",
            },
        },
        "DATAMODULE": {
            "name": "your.datamodule.class",
        },
        "DATASET": _dataset_template(include_targets=True),
    }


def build_optuna_template_dict() -> dict:
    return {
        "SESSION": {
            "name": "my_optuna_session",
            "workflow": "optimize",
            "seed": 42,
            "precision": "32",
            "root_dir": "./",
        },
        "MODEL": {
            "name": "your.model.class",
        },
        "MLFLOW": {
            "experiment_name": "my_experiment",
        },
        "OPTUNA": {
            "enabled": True,
            "n_trials": 100,
            "study_name": "my_study",
            "storage": _get_default_optuna_storage(),
        },
        "TRAINING": {
            "trainer": {
                "max_epochs": 100,
                "accelerator": "auto",
            },
        },
        "DATAMODULE": {
            "name": "your.datamodule.class",
        },
        "DATASET": _dataset_template(include_targets=True),
    }


def get_template_dict(kind: TemplateKind) -> dict:
    if kind == "training":
        return build_training_template_dict()
    if kind == "inference":
        return build_inference_template_dict()
    if kind == "mlflow":
        return build_mlflow_template_dict()
    if kind == "optuna":
        return build_optuna_template_dict()
    raise ValueError(f"Unknown template kind: {kind}")


def _comments_for(kind: TemplateKind) -> dict[str, str]:
    """Return a mapping of dotted keys to human-readable comments."""
    base = {
        "SESSION.name": "Human-readable run/session name (for logs and tracking)",
        "SESSION.workflow": "Workflow mode: 'train', 'optimize', or 'inference'",
        "SESSION.seed": "Random seed for reproducibility",
        "SESSION.precision": "Computation precision preset (e.g., '32', '16-mixed')",
        "MODEL.name": "Model class path or registry alias",
        "SESSION.root_dir": "Optional root directory for path resolution",
        "TRAINING.trainer.max_epochs": "Maximum number of epochs (Lightning Trainer)",
        "TRAINING.trainer.accelerator": "Hardware accelerator: cpu | gpu | auto | tpu",
        "DATAMODULE.name": "DataModule class path or alias (dataflow loading)",
        "DATASET.name": "Dataset class path or alias",
        "DATASET.root_dir": "Root directory used to resolve relative dataset entry paths",
        "DATASET.features": "Feature entries loaded into the batch TensorDict",
        "DATASET.targets": "Target entries loaded into the batch TensorDict",
    }
    if kind == "inference":
        base.update(
            {
                "MODEL.checkpoint": "Path to trained model checkpoint (required for inference)",
            }
        )
    return base


def _build_table(
    content: dict[str, Any],
    *,
    comments: dict[str, str],
    prefix: str,
) -> tuple[Table, list[tuple[str, dict[str, Any] | list[dict[str, Any]]]]]:
    tbl = table()
    nested_items: list[tuple[str, dict[str, Any] | list[dict[str, Any]]]] = []
    for key, value in content.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) or _is_array_of_tables(value):
            nested_items.append((key, value))
            continue
        if dotted in comments:
            tbl.add(comment(comments[dotted]))
        tbl.add(key, value)
    return tbl, nested_items


def _is_array_of_tables(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _build_aot(
    entries: list[dict[str, Any]],
    *,
    comments: dict[str, str],
    prefix: str,
) -> AoT:
    array = aot()
    for entry in entries:
        entry_tbl, nested_items = _build_table(entry, comments=comments, prefix=prefix)
        for key, value in nested_items:
            dotted = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                child_tbl, grand_nested = _build_table(value, comments=comments, prefix=dotted)
                if grand_nested:
                    raise ValueError(
                        f"Nested arrays of tables are not supported in templates: {dotted}"
                    )
                entry_tbl.add(key, child_tbl)
            elif _is_array_of_tables(value):
                raise ValueError(
                    f"Nested arrays of tables are not supported in templates: {dotted}"
                )
        array.append(entry_tbl)
    return array


def render_toml(template: dict, *, kind: TemplateKind = "training") -> str:
    # Deterministic section order
    order = (
        "SESSION",
        "MODEL",
        "MLFLOW",
        "OPTUNA",
        "TRAINING",
        "DATAMODULE",
        "DATASET",
        "SESSION.workflow",
    )

    doc: TOMLDocument = document()
    need_parent_headers: set[str] = set()
    for key in order:
        if key not in template:
            continue
        content = template[key]
        if content is None:
            continue
        comments = _comments_for(kind)
        # Dotted section like SESSION.workflow: add as nested table
        if "." in key:
            parent, child = key.split(".", 1)
            if parent not in doc:
                doc.add(parent, table())
            parent_tbl = cast(Table, doc[parent])
            child_tbl, nested_items = _build_table(content, comments=comments, prefix=key)
            if nested_items:
                raise ValueError(f"Nested content is not supported under dotted section {key}")
            parent_tbl.add(child, child_tbl)
            continue

        parent_tbl, nested_items = _build_table(content, comments=comments, prefix=key)
        doc.add(key, parent_tbl)
        if nested_items:
            if len(nested_items) == len(content):
                need_parent_headers.add(key)
            section_tbl = cast(Table, doc[key])
            for child_key, child_value in nested_items:
                dotted = f"{key}.{child_key}"
                if isinstance(child_value, dict):
                    child_tbl, grand_nested = _build_table(
                        child_value, comments=comments, prefix=dotted
                    )
                    if grand_nested:
                        raise ValueError(
                            f"Nested arrays of tables are not supported in templates: {dotted}"
                        )
                    section_tbl.add(child_key, child_tbl)
                    continue
                if dotted in comments:
                    section_tbl.add(comment(comments[dotted]))
                section_tbl.add(
                    child_key, _build_aot(child_value, comments=comments, prefix=dotted)
                )
    rendered = dumps(doc)
    # Ensure explicit parent headers for purely nested sections (e.g., [TRAINING])
    for sec in need_parent_headers:
        if f"[{sec}]\n" not in rendered and f"[{sec}." in rendered:
            # Insert header before the first nested occurrence
            idx = rendered.find(f"[{sec}.")
            if idx != -1:
                rendered = rendered[:idx] + f"[{sec}]\n\n" + rendered[idx:]
    return rendered


def render_template(kind: TemplateKind) -> str:
    return render_toml(get_template_dict(kind), kind=kind)
