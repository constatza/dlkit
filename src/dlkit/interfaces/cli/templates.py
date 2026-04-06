"""Centralized config template builders and TOML rendering.

Provides dict builders for each template and a renderer using tomlkit.
This is the single source of truth used by the CLI and sync tools.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from tomlkit import comment, document, dumps, table


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


def build_training_template_dict() -> dict:
    return {
        "SESSION": {
            "name": "my_training_session",
            "inference": False,
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
        "DATASET": {
            "name": "your.dataset.class",
        },
    }


def build_inference_template_dict() -> dict:
    return {
        "SESSION": {
            "name": "my_inference_session",
            "inference": True,
            "seed": 42,
            "precision": "32",
            "root_dir": "./",
        },
        "MODEL": {
            "name": "your.model.class",
            "checkpoint": "./model.ckpt",
        },
    }


def build_mlflow_template_dict() -> dict:
    return {
        "SESSION": {
            "name": "my_mlflow_session",
            "inference": False,
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
        "DATASET": {
            "name": "your.dataset.class",
        },
    }


def build_optuna_template_dict() -> dict:
    return {
        "SESSION": {
            "name": "my_optuna_session",
            "inference": False,
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
        "DATASET": {
            "name": "your.dataset.class",
        },
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
        "SESSION.inference": "Inference mode flag (false = training)",
        "SESSION.seed": "Random seed for reproducibility",
        "SESSION.precision": "Computation precision preset (e.g., '32', '16-mixed')",
        "MODEL.name": "Model class path or registry alias",
        "SESSION.root_dir": "Optional root directory for path resolution",
        "TRAINING.trainer.max_epochs": "Maximum number of epochs (Lightning Trainer)",
        "TRAINING.trainer.accelerator": "Hardware accelerator: cpu | gpu | auto | tpu",
        "DATAMODULE.name": "DataModule class path or alias (dataflow loading)",
        "DATASET.name": "Dataset class path or alias",
    }
    if kind == "inference":
        base.update(
            {
                "MODEL.checkpoint": "Path to trained model checkpoint (required for inference)",
            }
        )
    return base


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
        "SESSION.inference",
    )

    doc = document()
    need_parent_headers: set[str] = set()
    for key in order:
        if key not in template:
            continue
        content = template[key]
        if content is None:
            continue
        comments = _comments_for(kind)
        # Dotted section like SESSION.inference: add as nested table
        if "." in key:
            parent, child = key.split(".", 1)
            if parent not in doc:
                doc.add(parent, table())
            parent_tbl = cast(Any, doc[parent])
            child_tbl = table()
            for k, v in content.items():
                dotted = f"{key}.{k}"
                if dotted in comments:
                    parent_tbl.add(comment(comments[dotted]))
                child_tbl.add(k, v)
            parent_tbl.add(child, child_tbl)
            continue

        # For regular sections: if any nested dicts exist, add an explicit parent section
        has_nested = any(isinstance(v, dict) for v in content.values())
        parent_tbl = table()
        # Scalars on the section
        for k, v in content.items():
            if not isinstance(v, dict):
                dotted = f"{key}.{k}"
                if dotted in comments:
                    parent_tbl.add(comment(comments[dotted]))
                parent_tbl.add(k, v)
        doc.add(key, parent_tbl)
        # Now add nested subtables
        if has_nested:
            if all(isinstance(v, dict) for v in content.values()) and len(content) > 0:
                need_parent_headers.add(key)
            for k, v in content.items():
                if isinstance(v, dict):
                    child_tbl = table()
                    for ck, cv in v.items():
                        dotted = f"{key}.{k}.{ck}"
                        if dotted in comments:
                            child_tbl.add(comment(comments[dotted]))
                        child_tbl.add(ck, cv)
                    cast(Any, doc[key]).add(k, child_tbl)
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
