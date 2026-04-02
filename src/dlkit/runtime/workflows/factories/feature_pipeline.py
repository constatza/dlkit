"""Feature-selection pipeline for flexible dataset builds."""

from __future__ import annotations

from dataclasses import dataclass

from dlkit.tools.config.data_entries import DataEntry

from .feature_dependencies import (
    collect_feature_dependencies,
    select_required_features,
    validate_feature_selection,
)


@dataclass(frozen=True, slots=True)
class FeatureSelection:
    """Selected dataset entries plus dependency metadata."""

    features: tuple[DataEntry, ...]
    targets: tuple[DataEntry, ...]
    dependency_reasons: dict[str, list[str]]
    dropped_feature_names: list[str]


class FeaturePipeline:
    """Compute the minimal feature set required by wrapper routes."""

    def select(
        self,
        configured_features: tuple[DataEntry, ...],
        configured_targets: tuple[DataEntry, ...],
        loss_function: object | None,
        metrics: tuple[object, ...],
    ) -> FeatureSelection:
        """Select required features and preserve all targets."""

        feature_dependencies = collect_feature_dependencies(
            configured_features,
            configured_targets,
            loss_function,
            metrics,
        )
        selected_features = select_required_features(configured_features, feature_dependencies)
        validate_feature_selection(selected_features, feature_dependencies)

        selected_feature_names = {
            entry.name for entry in selected_features if isinstance(entry.name, str) and entry.name
        }
        dropped_feature_names = [
            entry.name
            for entry in configured_features
            if isinstance(entry.name, str)
            and entry.name
            and entry.name not in selected_feature_names
        ]
        dependency_reasons = {
            name: sorted(reasons)
            for name, reasons in feature_dependencies.items()
            if name in selected_feature_names
        }
        return FeatureSelection(
            features=selected_features,
            targets=tuple(configured_targets),
            dependency_reasons=dependency_reasons,
            dropped_feature_names=sorted(dropped_feature_names),
        )
