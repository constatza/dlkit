"""Shared template builders and TOML rendering for JobConfig examples."""

from __future__ import annotations

import inspect
from typing import Any, Literal, cast

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from tomlkit import aot, comment, document, dumps, table
from tomlkit.items import AoT, Table
from tomlkit.toml_document import TOMLDocument

from dlkit.common.metric_stages import DEFAULT_VAL_LOSS_METRIC
from dlkit.infrastructure.config.convergence_settings import ConvergenceSettings
from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.job_config import MultiRunJobConfig
from dlkit.infrastructure.config.optimizer_component import AdamWSettings
from dlkit.infrastructure.config.search_settings import SearchSettings
from dlkit.infrastructure.config.split_settings import IndexSplitSettings
from dlkit.infrastructure.config.trainer_settings import TrainerSettings
from dlkit.infrastructure.config.training_settings import StoppingSettings

TemplateKind = Literal[
    "training", "inference", "mlflow", "search", "fit", "convergence", "multirun"
]


def _default(cls: type[BaseModel], field: str) -> Any:
    """Read a settings class's real field default, failing loudly if it has none.

    Args:
        cls: The Pydantic settings class that declares the field.
        field: Field name on ``cls``.

    Returns:
        The field's real default value.

    Raises:
        ValueError: If the field has no plain default (e.g. it's `default_factory`-only).
    """
    info = cls.model_fields[field]
    if info.default is PydanticUndefined:
        raise ValueError(f"{cls.__name__}.{field} has no default; must be hand-authored")
    return info.default


def _data_features_template() -> list[dict[str, Any]]:
    return [{"name": "x", "path": "features.npy"}]


def _data_targets_template() -> list[dict[str, Any]]:
    return [{"name": "y", "path": "targets.npy"}]


def build_training_template_dict() -> dict[str, Any]:
    """Build canonical training job config template dict."""
    return {
        "run": {"type": "train", "seed": 42, "precision": "32"},
        "experiment": {"name": "my-experiment"},
        "model": {"class": "your.model.class"},
        "data": {
            "root": "./data",
            "batch_size": _default(DataSettings, "batch_size"),
            "num_workers": _default(DataSettings, "num_workers"),
            "features": _data_features_template(),
            "targets": _data_targets_template(),
            "splits": {
                "val": _default(IndexSplitSettings, "val_ratio"),
                "test": _default(IndexSplitSettings, "test_ratio"),
            },
        },
        "training": {
            "loss": "mse",
            "stopping": {
                "monitor": DEFAULT_VAL_LOSS_METRIC,
                "patience": _default(StoppingSettings, "patience"),
                "direction": _default(StoppingSettings, "direction"),
            },
            "trainer": {
                "max_epochs": _default(TrainerSettings, "max_epochs"),
                "accelerator": _default(TrainerSettings, "accelerator"),
            },
            "optimizer": {
                "name": "AdamW",
                "lr": _default(AdamWSettings, "lr"),
                "weight_decay": _default(AdamWSettings, "weight_decay"),
            },
        },
    }


def build_inference_template_dict() -> dict[str, Any]:
    """Build canonical inference job config template dict."""
    return {
        "run": {"type": "predict", "seed": 42, "precision": "32"},
        "experiment": {"name": "my-inference-experiment"},
        "model": {"class": "your.model.class", "checkpoint": "./model.ckpt"},
        "data": {
            "root": "./data",
            "batch_size": _default(DataSettings, "batch_size"),
            "num_workers": _default(DataSettings, "num_workers"),
            "features": _data_features_template(),
        },
    }


def build_mlflow_template_dict() -> dict[str, Any]:
    """Build training job config template dict with MLflow tracking."""
    base = build_training_template_dict()
    base["tracking"] = {"backend": "mlflow", "uri": "http://localhost:5000"}
    base["experiment"] = {"name": "my-mlflow-experiment", "run_name": "my-run"}
    return base


def build_search_template_dict() -> dict[str, Any]:
    """Build canonical HPO search job config template dict."""
    base = build_training_template_dict()
    base["run"]["type"] = "search"
    base["search"] = {
        "n_trials": _default(SearchSettings, "n_trials"),
        "direction": _default(SearchSettings, "direction"),
        "objective": DEFAULT_VAL_LOSS_METRIC,
        "space": {
            "training.optimizer.lr": {"type": "log_float", "low": 1e-5, "high": 1e-1},
            "model.hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        },
    }
    return base


def build_fit_template_dict() -> dict[str, Any]:
    """Build canonical one-shot fit job config template dict.

    ``training`` is intentionally omitted: ``FitJobConfig`` covers models
    whose entire "training" is one deterministic, non-gradient call, so
    nothing downstream needs optimizer/loss wiring.
    """
    base = build_training_template_dict()
    base["run"]["type"] = "fit"
    del base["training"]
    return base


def build_convergence_template_dict() -> dict[str, Any]:
    """Build canonical sample-size convergence study job config template dict."""
    base = build_training_template_dict()
    base["run"]["type"] = "convergence"
    base["convergence"] = {
        "sizes": [100, 500, 1000],
        "repeats": _default(ConvergenceSettings, "repeats"),
        "target_metric": DEFAULT_VAL_LOSS_METRIC,
        "c": _default(ConvergenceSettings, "c"),
    }
    return base


def build_multirun_template_dict() -> dict[str, Any]:
    """Build canonical multirun sweep job config template dict."""
    return {
        "run": {"type": "multirun"},
        "multirun": {
            "experiment_name": "my-sweep",
            "parent_run_name": "sweep-parent",
            "failure_policy": _default(MultiRunJobConfig, "failure_policy"),
            "runs": [
                {
                    "id": "a",
                    "label": "Run A",
                    "files": ["jobs/base.toml", "jobs/variant_a.toml"],
                    "patches": {"run.seed": 7},
                },
                {"id": "variants", "files": "jobs/variants/*.toml"},
            ],
        },
    }


def _resolve_model_class(model: str, module_path: str | None) -> type:
    """Resolve a model name (+ optional module_path) to an actual class.

    Reuses the same registry/import resolution every ``[model]`` block goes
    through at build time (``resolve_component``), rather than a new
    import mechanism.

    Args:
        model: Class name or dotted path.
        module_path: Fallback module to import from when ``model`` isn't
            already registered.

    Returns:
        The resolved class.

    Raises:
        TypeError: If the resolved object isn't a class.
    """
    from dlkit.infrastructure.registry.resolve import resolve_component

    resolved = resolve_component("model", model, module_path=module_path)
    if not inspect.isclass(resolved):
        raise TypeError(f"Resolved model {model!r} is not a class: {type(resolved)!r}")
    return resolved


def _model_fields_from_signature(cls: type) -> dict[str, Any]:
    """Derive illustrative ``[model]`` fields from a model class's ``__init__`` signature.

    Real Python defaults are sourced directly. Required params with no default
    get an explicit ``"TODO"`` placeholder rather than a guessed value. Params
    the class declares as shape-inferred (``shape_kwarg_names()``) are
    excluded entirely — those are resolved automatically at build time from
    dataset shapes, not user-configured.

    Args:
        cls: The resolved model class.

    Returns:
        Dict of constructor parameter name to illustrative value.
    """
    shape_kwargs: frozenset[str] = frozenset()
    shape_kwarg_names = getattr(cls, "shape_kwarg_names", None)
    if callable(shape_kwarg_names):
        shape_kwargs = shape_kwarg_names()

    toml_scalar_types = (bool, int, float, str)
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(cls.__init__).parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in shape_kwargs:
            continue
        if param.default is inspect.Parameter.empty:
            fields[name] = "TODO"
        elif isinstance(param.default, toml_scalar_types):
            # Non-scalar defaults (None, tuples, ...) aren't TOML-representable
            # and aren't worth restating — the model's own default already
            # applies when the field is simply omitted.
            fields[name] = param.default
    return fields


def _entries_from_spec(
    spec_cls: type[BaseModel] | None, *, path_suffix: str
) -> list[dict[str, Any]] | None:
    """Build one entry per field name declared on a model's ``InputSpec``/``OutputSpec``.

    Args:
        spec_cls: The model's ``InputSpec`` or ``OutputSpec`` class, if declared.
        path_suffix: File extension for the placeholder path (e.g. ``"npy"``).

    Returns:
        One entry dict per spec field, or ``None`` when the model declares no
        spec — callers fall back to the existing generic single-entry
        placeholder in that case.
    """
    if spec_cls is None or not spec_cls.model_fields:
        return None
    return [
        {"name": name, "path": f"{name}.{path_suffix}", "transforms": []}
        for name in spec_cls.model_fields
    ]


def apply_model_fields(
    template: dict[str, Any], model: str, *, module_path: str | None, kind: TemplateKind
) -> dict[str, Any]:
    """Layer a resolved model class's own fields/entries onto a workflow-kind template.

    Args:
        template: A workflow-kind template dict (from ``build_*_template_dict``).
        model: Class name or dotted path to resolve.
        module_path: Fallback module for resolution.
        kind: The template's workflow kind, for error messages.

    Returns:
        The same template dict, mutated in place with model-specific content.

    Raises:
        ValueError: If ``kind`` has no ``[model]`` section (e.g. ``multirun``).
    """
    if "model" not in template:
        raise ValueError(
            f"Template kind {kind!r} has no [model] section; --model is not applicable"
        )

    cls = _resolve_model_class(model, module_path)
    model_fields = _model_fields_from_signature(cls)
    template["model"] = {
        "class": model,
        **({"module_path": module_path} if module_path else {}),
        **model_fields,
    }

    if "data" in template:
        features = _entries_from_spec(getattr(cls, "InputSpec", None), path_suffix="npy")
        if features is not None:
            template["data"]["features"] = features
        if "targets" in template["data"]:
            targets = _entries_from_spec(getattr(cls, "OutputSpec", None), path_suffix="npy")
            if targets is not None:
                template["data"]["targets"] = targets

    return template


def get_template_dict(
    kind: TemplateKind, *, model: str | None = None, module_path: str | None = None
) -> dict[str, Any]:
    """Return the template dict for the given kind.

    Args:
        kind: Workflow template kind.
        model: Optional model class name/path to introspect and layer in.
        module_path: Optional fallback module for resolving ``model``.

    Returns:
        The template dict.

    Raises:
        ValueError: If ``kind`` is unrecognized.
    """
    match kind:
        case "training":
            template = build_training_template_dict()
        case "inference":
            template = build_inference_template_dict()
        case "mlflow":
            template = build_mlflow_template_dict()
        case "search":
            template = build_search_template_dict()
        case "fit":
            template = build_fit_template_dict()
        case "convergence":
            template = build_convergence_template_dict()
        case "multirun":
            template = build_multirun_template_dict()
        case _:
            raise ValueError(f"Unknown template kind: {kind}")
    if model is not None:
        template = apply_model_fields(template, model, module_path=module_path, kind=kind)
    return template


def _comments_for(kind: TemplateKind) -> dict[str, str]:
    base: dict[str, str] = {
        "run.type": "Workflow type: 'train', 'predict', 'search', 'convergence', 'multirun', or 'fit'",
        "run.seed": "Random seed for reproducibility",
        "run.precision": "Computation precision preset (e.g., '32', '16-mixed')",
        "experiment.name": "Human-readable experiment name (for logs and tracking)",
        "model.class": "Model class path or registry alias",
        "data.root": "Root directory used to resolve relative dataset entry paths",
        "data.batch_size": "DataLoader batch size",
        "data.features": "Feature entries loaded into the batch TensorDict",
        "data.targets": "Target entries loaded into the batch TensorDict",
        "data.features.transforms": (
            "Add transforms here if your model's input shapes need reshaping (e.g. Unsqueeze)"
        ),
        "data.targets.transforms": (
            "Add transforms here if your model's output shapes need reshaping (e.g. Unsqueeze)"
        ),
        "training.trainer.max_epochs": "Maximum number of epochs (Lightning Trainer)",
        "training.trainer.accelerator": "Hardware accelerator: cpu | gpu | auto | tpu",
        "tracking.backend": "Tracking backend: 'mlflow' or 'none'",
        "tracking.uri": "Tracking server URI (for MLflow)",
    }
    if kind == "inference":
        base["model.checkpoint"] = "Path to trained model checkpoint (required for inference)"
    if kind == "search":
        base["search.n_trials"] = "Number of hyperparameter optimization trials"
        base["search.space"] = "Hyperparameter search space keyed by dotted config path"
    if kind == "convergence":
        base["convergence.sizes"] = "Explicit list of training-set sizes to evaluate"
    if kind == "multirun":
        base["multirun.experiment_name"] = "MLflow experiment name shared by parent and children"
        base["multirun.runs"] = "Ordered child run sources"
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
    # `all(...)` over an empty list is vacuously True, so an explicit `and value`
    # guard is required: an empty list is a plain TOML array, not an AoT.
    return isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)


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
    order = (
        "run",
        "experiment",
        "model",
        "data",
        "training",
        "search",
        "convergence",
        "multirun",
        "tracking",
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


def render_template(
    kind: TemplateKind, *, model: str | None = None, module_path: str | None = None
) -> str:
    """Render the canonical template for the given kind.

    Args:
        kind: Workflow template kind.
        model: Optional model class name/path to introspect and layer in.
        module_path: Optional fallback module for resolving ``model``.

    Returns:
        Rendered TOML text.
    """
    return render_toml(get_template_dict(kind, model=model, module_path=module_path), kind=kind)
