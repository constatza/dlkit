"""Regression lock: template illustrative values must track real schema defaults.

Each `build_*_template_dict()` sources its illustrative values from the actual
Pydantic settings classes wherever those classes have a meaningful default
(`_template_helpers._default()`), instead of hardcoding a number that can
silently drift from the schema it's supposed to illustrate - see
`_template_helpers.py`'s module docstring / `_default()` for the mechanism.
This test locks the specific values that were found drifted (and fixed) when
that mechanism was introduced, so a future schema-default change is caught
here instead of silently reintroducing the same class of bug.
"""

from __future__ import annotations

from dlkit.infrastructure.config._template_helpers import build_training_template_dict
from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.optimizer_component import AdamWSettings
from dlkit.infrastructure.config.search_settings import SearchSettings


def test_template_batch_size_matches_real_schema_default() -> None:
    template = build_training_template_dict()
    assert template["data"]["batch_size"] == DataSettings.model_fields["batch_size"].default


def test_template_weight_decay_matches_real_schema_default() -> None:
    template = build_training_template_dict()
    expected = AdamWSettings.model_fields["weight_decay"].default
    assert template["training"]["optimizer"]["weight_decay"] == expected


def test_template_n_trials_matches_real_schema_default() -> None:
    from dlkit.infrastructure.config._template_helpers import build_search_template_dict

    template = build_search_template_dict()
    assert template["search"]["n_trials"] == SearchSettings.model_fields["n_trials"].default


def test_template_has_no_invalid_splits_train_key() -> None:
    """`data.splits.train` isn't a real IndexSplitSettings field - must stay absent."""
    template = build_training_template_dict()
    assert "train" not in template["data"]["splits"]


def test_template_omits_default_root_dir() -> None:
    """DirectoryPath requires the path to exist; the real default is None - omit it."""
    template = build_training_template_dict()
    assert "default_root_dir" not in template["training"]["trainer"]


def test_mlflow_template_omits_nonexistent_register_model_field() -> None:
    """`experiment.register_model` isn't a field on ExperimentSettings - must stay absent."""
    from dlkit.infrastructure.config._template_helpers import build_mlflow_template_dict

    template = build_mlflow_template_dict()
    assert "register_model" not in template["experiment"]
