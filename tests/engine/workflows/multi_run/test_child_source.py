"""Tests for expand_child_sources() and its ChildSource variants.

Covers:
- ExplicitFileSource: loads via load_job(), applies patches, builds a RunSpec
- ExplicitFileSource: merges multiple files left-to-right (later wins)
- GlobSource: expands every match in sorted order with id_prefix+stem ids
- GlobSource: raises ConfigValidationError on zero matches
- LoadedSettingsSource: wraps already-validated settings directly, no loading
- expand_child_sources: raises ConfigValidationError on duplicate ids
- expand_child_sources: raises ConfigValidationError on a nested multirun child
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from dlkit.common.errors import ConfigValidationError
from dlkit.engine.workflows.multi_run import (
    ExplicitFileSource,
    GlobSource,
    LoadedSettingsSource,
    RunSpec,
    expand_child_sources,
)
from dlkit.infrastructure.config.job_config import JobConfig, TrainingJobConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def override_toml_path(tmp_path: Path) -> Path:
    """A small override-only TOML patching simple_train's experiment name.

    Args:
        tmp_path: Pytest tmp_path fixture.

    Returns:
        Path to the written override TOML file.
    """
    path = tmp_path / "override.toml"
    path.write_text('[experiment]\nname = "overridden-name"\n')
    return path


@pytest.fixture
def variants_dir(tmp_path: Path, simple_train_path: Path) -> Path:
    """A directory of two TOML job files for GlobSource tests.

    Args:
        tmp_path: Pytest tmp_path fixture.
        simple_train_path: Shared minimal single-file train TOML fixture.

    Returns:
        Path to a directory containing "b_variant.toml" and "a_variant.toml"
        (named out of alphabetical order to exercise sorted() glob expansion).
    """
    source_text = simple_train_path.read_text()
    variants = tmp_path / "variants"
    variants.mkdir()
    (variants / "b_variant.toml").write_text(source_text)
    (variants / "a_variant.toml").write_text(source_text)
    return variants


# ---------------------------------------------------------------------------
# ExplicitFileSource
# ---------------------------------------------------------------------------


def test_explicit_file_source_builds_run_spec(simple_train_path: Path) -> None:
    """ExplicitFileSource loads the file and builds a matching RunSpec.

    Args:
        simple_train_path: Shared minimal single-file train TOML fixture.
    """
    source = ExplicitFileSource(id="child-1", label="Child One", files=(simple_train_path,))
    (spec,) = expand_child_sources([source])

    assert isinstance(spec, RunSpec)
    assert spec.id == "child-1"
    assert spec.label == "Child One"
    assert spec.run_name == "Child One"
    assert isinstance(spec.settings, TrainingJobConfig)


def test_explicit_file_source_merges_files_job_file_wins(
    simple_train_path: Path,
    override_toml_path: Path,
) -> None:
    """Multiple files are merged left-to-right; the later file wins.

    Args:
        simple_train_path: Base single-file train TOML fixture.
        override_toml_path: Override-only TOML patching experiment.name.
    """
    source = ExplicitFileSource(
        id="child-1", label="child-1", files=(simple_train_path, override_toml_path)
    )
    (spec,) = expand_child_sources([source])

    assert spec.settings.experiment is not None
    assert spec.settings.experiment.name == "overridden-name"


def test_explicit_file_source_applies_patches(simple_train_path: Path) -> None:
    """ExplicitFileSource.patches are applied via settings.patch() after loading.

    Args:
        simple_train_path: Shared minimal single-file train TOML fixture.
    """
    source = ExplicitFileSource(
        id="child-1",
        label="child-1",
        files=(simple_train_path,),
        patches={"experiment": {"name": "patched-name"}},
    )
    (spec,) = expand_child_sources([source])

    assert spec.settings.experiment is not None
    assert spec.settings.experiment.name == "patched-name"


def test_explicit_file_source_run_type_override(simple_train_path: Path) -> None:
    """ExplicitFileSource.run_type overrides run.type the same way load_job() does.

    Args:
        simple_train_path: TOML fixture with run.type="train" already set;
            overriding it to the same value is a no-op assertion that the
            parameter is actually threaded through to load_job().
    """
    source = ExplicitFileSource(
        id="child-1", label="child-1", files=(simple_train_path,), run_type="train"
    )
    (spec,) = expand_child_sources([source])
    assert isinstance(spec.settings, TrainingJobConfig)


# ---------------------------------------------------------------------------
# GlobSource
# ---------------------------------------------------------------------------


def test_glob_source_expands_every_match_in_sorted_order(variants_dir: Path) -> None:
    """GlobSource expands every matched file, in lexical (sorted path) order.

    Args:
        variants_dir: Directory with "b_variant.toml" and "a_variant.toml".
    """
    source = GlobSource(id_prefix="variant-", pattern="*.toml", base_dir=variants_dir)
    specs = expand_child_sources([source])

    assert [spec.id for spec in specs] == ["variant-a_variant", "variant-b_variant"]
    assert all(isinstance(spec.settings, TrainingJobConfig) for spec in specs)


def test_glob_source_zero_matches_raises(tmp_path: Path) -> None:
    """GlobSource raises ConfigValidationError when the pattern matches nothing.

    Args:
        tmp_path: Empty directory — no TOML files present.
    """
    source = GlobSource(id_prefix="variant-", pattern="*.toml", base_dir=tmp_path)
    with pytest.raises(ConfigValidationError):
        expand_child_sources([source])


# ---------------------------------------------------------------------------
# LoadedSettingsSource
# ---------------------------------------------------------------------------


def test_loaded_settings_source_wraps_settings_directly(
    job_config_settings: JobConfig,
) -> None:
    """LoadedSettingsSource wraps already-validated settings without loading.

    Args:
        job_config_settings: Real JobConfig fixture.
    """
    source = LoadedSettingsSource(id="loaded-1", label="Loaded One", settings=job_config_settings)
    (spec,) = expand_child_sources([source])

    assert spec.settings is job_config_settings
    assert spec.run_name == "Loaded One"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_duplicate_ids_raise(job_config_settings: JobConfig) -> None:
    """expand_child_sources raises ConfigValidationError on duplicate ids.

    Args:
        job_config_settings: Real JobConfig fixture, reused for both sources.
    """
    sources = [
        LoadedSettingsSource(id="dup", label="a", settings=job_config_settings),
        LoadedSettingsSource(id="dup", label="b", settings=job_config_settings),
    ]
    with pytest.raises(ConfigValidationError):
        expand_child_sources(sources)


def test_nested_multirun_child_raises() -> None:
    """A resolved child with run.type == "multirun" is rejected.

    RunSettings.type doesn't accept "multirun" as a real Literal value yet
    (that's follow-up work), so this stubs a duck-typed settings object with
    run.type="multirun" to exercise the guard directly.
    """

    class _FakeRun:
        type = "multirun"

    class _FakeSettings:
        run = _FakeRun()

    source = LoadedSettingsSource(id="nested", label="nested", settings=cast(Any, _FakeSettings()))
    with pytest.raises(ConfigValidationError):
        expand_child_sources([source])
