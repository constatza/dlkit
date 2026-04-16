from __future__ import annotations

import numpy as np
import pytest

from dlkit.engine.workflows.factories.feature_dependencies import (
    collect_feature_dependencies,
    select_required_features,
    validate_feature_selection,
)
from dlkit.infrastructure.config.data_entries import AutoencoderTarget, Feature, Target
from dlkit.infrastructure.config.model_components import (
    LossComponentSettings,
    LossInputRef,
    MetricComponentSettings,
    MetricInputRef,
)


def test_collect_feature_dependencies_captures_all_dependency_sources(tmp_path) -> None:
    x_path = tmp_path / "x.npy"
    matrix_path = tmp_path / "matrix.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.zeros((3, 2), dtype=np.float32))
    np.save(matrix_path, np.zeros((3, 2, 2), dtype=np.float32))
    np.save(y_path, np.zeros((3, 1), dtype=np.float32))

    features = (
        Feature(name="x", path=x_path),
        Feature(name="matrix", path=matrix_path, model_input=False, loss_input="matrix"),
    )
    targets = (
        Target(name="y", path=y_path),
        AutoencoderTarget(name="recon", feature_ref="matrix", path=y_path),
    )
    loss_spec = LossComponentSettings(extra_inputs=(LossInputRef(arg="m", key="features.matrix"),))
    metric_specs = (
        MetricComponentSettings(extra_inputs=(MetricInputRef(arg="m", key="features.matrix"),)),
    )

    deps = collect_feature_dependencies(features, targets, loss_spec, metric_specs)

    assert "x" in deps
    assert "matrix" in deps
    assert "model_input" in deps["x"]
    assert "loss_input" in deps["matrix"]
    assert "loss_extra" in deps["matrix"]
    assert "metric_extra" in deps["matrix"]
    assert "target_feature_ref" in deps["matrix"]


def test_validate_feature_selection_raises_for_missing_dependency(tmp_path) -> None:
    matrix_path = tmp_path / "matrix.npy"
    y_path = tmp_path / "y.npy"
    np.save(matrix_path, np.zeros((2, 2, 2), dtype=np.float32))
    np.save(y_path, np.zeros((2, 1), dtype=np.float32))

    features = (Feature(name="matrix", path=matrix_path, model_input=False),)
    loss_spec = LossComponentSettings(
        extra_inputs=(LossInputRef(arg="missing", key="features.not_in_config"),)
    )
    deps = collect_feature_dependencies(features, (Target(name="y", path=y_path),), loss_spec, ())
    selected = select_required_features(features, deps)

    with pytest.raises(ValueError, match="not_in_config"):
        validate_feature_selection(selected, deps)
