"""Shared template builders and TOML rendering for JobConfig examples."""

from __future__ import annotations

from typing import Any, Literal, cast

from tomlkit import aot, comment, document, dumps, table
from tomlkit.items import AoT, Table
from tomlkit.toml_document import TOMLDocument

TemplateKind = Literal["training", "inference", "mlflow", "optuna"]


def _get_default_optuna_storage() -> str:
    """Return a deterministic placeholder Optuna storage URI for templates."""
    return "sqlite:///optuna.db"


def _data_features_template() -> list[dict[str, Any]]:
    return [{"name": "x", "path": "features.npy", "data_role": "feature"}]


def _data_targets_template() -> list[dict[str, Any]]:
    return [{"name": "y", "path": "targets.npy", "data_role": "target"}]


def build_training_template_dict() -> dict[str, Any]:
    """Build canonical training job config template dict."""
    return {
        "run": {"type": "train", "seed": 42, "precision": "32"},
        "experiment": {"name": "my-experiment"},
        "model": {"name": "your.model.class"},
        "data": {
            "root": "./data",
            "batch_size": 32,
            "num_workers": 0,
            "features": _data_features_template(),
            "targets": _data_targets_template(),
            "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
        },
        "training": {
            "loss": "mse",
            "stopping": {"monitor": "val/loss", "patience": 10, "direction": "min"},
            "trainer": {
                "max_epochs": 100,
                "accelerator": "auto",
                "default_root_dir": "./lightning",
            },
            "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 1e-4},
        },
    }


def build_inference_template_dict() -> dict[str, Any]:
    """Build canonical inference job config template dict."""
    return {
        "run": {"type": "predict", "seed": 42, "precision": "32"},
        "experiment": {"name": "my-inference-experiment"},
        "model": {"name": "your.model.class", "checkpoint": "./model.ckpt"},
        "data": {
            "root": "./data",
            "batch_size": 32,
            "num_workers": 0,
            "features": _data_features_template(),
        },
    }


def build_mlflow_template_dict() -> dict[str, Any]:
    """Build training job config template dict with MLflow tracking."""
    base = build_training_template_dict()
    base["tracking"] = {"backend": "mlflow", "uri": "http://localhost:5000"}
    base["experiment"] = {
        "name": "my-mlflow-experiment",
        "run_name": "my-run",
        "register_model": True,
    }
    return base


def build_search_template_dict() -> dict[str, Any]:
    """Build canonical HPO search job config template dict."""
    base = build_training_template_dict()
    base["run"]["type"] = "search"
    base["search"] = {
        "n_trials": 20,
        "direction": "minimize",
        "objective": "val/loss",
        "space": {
            "training.optimizer.lr": {"type": "log_float", "low": 1e-5, "high": 1e-1},
            "model.hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        },
    }
    return base


def get_template_dict(kind: TemplateKind) -> dict[str, Any]:
    """Return the template dict for the given kind."""
    match kind:
        case "training":
            return build_training_template_dict()
        case "inference":
            return build_inference_template_dict()
        case "mlflow":
            return build_mlflow_template_dict()
        case "optuna":
            return build_search_template_dict()
        case _:
            raise ValueError(f"Unknown template kind: {kind}")


def _comments_for(kind: TemplateKind) -> dict[str, str]:
    base: dict[str, str] = {
        "run.type": "Workflow type: 'train', 'predict', or 'search'",
        "run.seed": "Random seed for reproducibility",
        "run.precision": "Computation precision preset (e.g., '32', '16-mixed')",
        "experiment.name": "Human-readable experiment name (for logs and tracking)",
        "model.name": "Model class path or registry alias",
        "data.root": "Root directory used to resolve relative dataset entry paths",
        "data.batch_size": "DataLoader batch size",
        "data.features": "Feature entries loaded into the batch TensorDict",
        "data.targets": "Target entries loaded into the batch TensorDict",
        "training.trainer.max_epochs": "Maximum number of epochs (Lightning Trainer)",
        "training.trainer.accelerator": "Hardware accelerator: cpu | gpu | auto | tpu",
        "training.trainer.default_root_dir": "Local Lightning work directory when tracking is disabled",
        "tracking.backend": "Tracking backend: 'mlflow' or 'none'",
        "tracking.uri": "Tracking server URI (for MLflow)",
    }
    if kind == "inference":
        base["model.checkpoint"] = "Path to trained model checkpoint (required for inference)"
    if kind == "optuna":
        base["search.n_trials"] = "Number of hyperparameter optimization trials"
        base["search.space"] = "Hyperparameter search space keyed by dotted config path"
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
    """Render a template dictionary as TOML."""
    order = ("run", "experiment", "model", "data", "training", "search", "tracking")

    doc: TOMLDocument = document()
    need_parent_headers: set[str] = set()
    for key in order:
        if key not in template:
            continue
        content = template[key]
        if content is None:
            continue
        comments = _comments_for(kind)
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
                    for grand_key, grand_value in grand_nested:
                        grand_dotted = f"{dotted}.{grand_key}"
                        if isinstance(grand_value, dict):
                            grand_tbl, _ = _build_table(
                                grand_value, comments=comments, prefix=grand_dotted
                            )
                            child_tbl.add(grand_key, grand_tbl)
                        elif _is_array_of_tables(grand_value):
                            raise ValueError(
                                f"Nested arrays of tables are not supported in templates: {grand_dotted}"
                            )
                    section_tbl.add(child_key, child_tbl)
                    continue
                if dotted in comments:
                    section_tbl.add(comment(comments[dotted]))
                section_tbl.add(
                    child_key, _build_aot(child_value, comments=comments, prefix=dotted)
                )
    rendered = dumps(doc)
    for sec in need_parent_headers:
        if f"[{sec}]\n" not in rendered and f"[{sec}." in rendered:
            idx = rendered.find(f"[{sec}.")
            if idx != -1:
                rendered = rendered[:idx] + f"[{sec}]\n\n" + rendered[idx:]
    return rendered


def render_template(kind: TemplateKind) -> str:
    """Render the canonical template for the given kind."""
    return render_toml(get_template_dict(kind), kind=kind)
