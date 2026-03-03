"""Pure feature-dependency analysis for flexible workflow builds."""

from __future__ import annotations

from typing import Any

from dlkit.tools.config.data_entries import DataEntry

FeatureDependencyMap = dict[str, set[str]]


def _add_reason(dependencies: FeatureDependencyMap, name: str | None, reason: str) -> None:
    """Add a reason tag for a feature name."""
    if not isinstance(name, str) or not name:
        return
    dependencies.setdefault(name, set()).add(reason)


def _parse_feature_name_from_batch_key(key: Any) -> str | None:
    """Extract ``<name>`` from ``features.<name>`` keys."""
    if not isinstance(key, str):
        return None
    namespace, separator, name = key.partition(".")
    if separator != "." or namespace != "features" or not name:
        return None
    return name


def _is_model_input_enabled(entry: DataEntry) -> bool:
    """Return True when the feature participates in model dispatch."""
    model_input = getattr(entry, "model_input", True)
    return model_input is not False and model_input is not None


def collect_feature_dependencies(
    features: tuple[DataEntry, ...],
    targets: tuple[DataEntry, ...],
    loss_spec: Any,
    metric_specs: tuple[Any, ...],
) -> FeatureDependencyMap:
    """Collect feature dependencies from model, loss, metrics, and target refs."""
    dependencies: FeatureDependencyMap = {}

    for feature in features:
        name = getattr(feature, "name", None)
        if _is_model_input_enabled(feature):
            _add_reason(dependencies, name, "model_input")
        if getattr(feature, "loss_input", None):
            _add_reason(dependencies, name, "loss_input")

    if loss_spec is not None:
        _add_reason(
            dependencies,
            _parse_feature_name_from_batch_key(getattr(loss_spec, "target_key", None)),
            "loss_target",
        )
        for ref in tuple(getattr(loss_spec, "extra_inputs", ()) or ()):
            _add_reason(
                dependencies,
                _parse_feature_name_from_batch_key(getattr(ref, "key", None)),
                "loss_extra",
            )

    for metric_spec in metric_specs:
        _add_reason(
            dependencies,
            _parse_feature_name_from_batch_key(getattr(metric_spec, "target_key", None)),
            "metric_target",
        )
        for ref in tuple(getattr(metric_spec, "extra_inputs", ()) or ()):
            _add_reason(
                dependencies,
                _parse_feature_name_from_batch_key(getattr(ref, "key", None)),
                "metric_extra",
            )

    for target in targets:
        _add_reason(dependencies, getattr(target, "feature_ref", None), "target_feature_ref")

    return dependencies


def select_required_features(
    features: tuple[DataEntry, ...],
    dependencies: FeatureDependencyMap,
) -> tuple[DataEntry, ...]:
    """Select only features that appear in the dependency map."""
    required_names = set(dependencies.keys())
    return tuple(
        feature for feature in features if getattr(feature, "name", None) in required_names
    )


def validate_feature_selection(
    selected_features: tuple[DataEntry, ...],
    dependencies: FeatureDependencyMap,
) -> None:
    """Raise when a dependency references a missing feature entry."""
    selected_names = {
        entry.name for entry in selected_features if isinstance(getattr(entry, "name", None), str)
    }
    missing = sorted(set(dependencies.keys()) - selected_names)
    if not missing:
        return

    missing_reasons = {name: sorted(dependencies[name]) for name in missing}
    raise ValueError(
        "Feature dependency resolution failed: referenced feature entries are missing from "
        f"configuration: {missing_reasons}"
    )
