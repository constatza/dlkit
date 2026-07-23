"""Tests for load_job() and _deep_merge() in dlkit.infrastructure.config.factories."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.common.errors import ConfigValidationError
from dlkit.infrastructure.config.factories import _deep_merge, load_job
from dlkit.infrastructure.config.job_config import (
    ConvergenceJobConfig,
    MultiRunJobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "jobs"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_train_path() -> Path:
    """Path to the simple single-file train TOML fixture.

    Returns:
        Absolute path to tests/fixtures/jobs/simple_train.toml.
    """
    return FIXTURES / "simple_train.toml"


@pytest.fixture
def composed_train_path() -> Path:
    """Path to the composed training TOML that references profile files.

    Returns:
        Absolute path to tests/fixtures/jobs/composed_train.toml.
    """
    return FIXTURES / "composed_train.toml"


@pytest.fixture
def search_path() -> Path:
    """Path to the HPO search TOML fixture.

    Returns:
        Absolute path to tests/fixtures/jobs/search.toml.
    """
    return FIXTURES / "search.toml"


@pytest.fixture
def convergence_path() -> Path:
    """Path to the sample-size convergence TOML fixture.

    Returns:
        Absolute path to tests/fixtures/jobs/convergence.toml.
    """
    return FIXTURES / "convergence.toml"


@pytest.fixture
def multirun_path() -> Path:
    """Path to the multirun sweep TOML fixture.

    Returns:
        Absolute path to tests/fixtures/jobs/multirun.toml.
    """
    return FIXTURES / "multirun.toml"


@pytest.fixture
def data_profile_path() -> Path:
    """Path to the data profile TOML (used for wrong-section test).

    Returns:
        Absolute path to tests/fixtures/jobs/profiles/data.toml.
    """
    return FIXTURES / "profiles" / "data.toml"


# ---------------------------------------------------------------------------
# Tests: load_job — simple single file
# ---------------------------------------------------------------------------


def test_load_simple_train(simple_train_path: Path) -> None:
    """load_job() returns a TrainingJobConfig for a self-contained TOML."""
    cfg = load_job(simple_train_path)
    assert isinstance(cfg, TrainingJobConfig)
    assert cfg.run.type == "train"
    assert cfg.run.seed == 42
    assert cfg.model.name == "FFNN"
    assert cfg.model.hidden_size == 64
    assert cfg.data.batch_size == 8
    assert cfg.training.loss.name == "mse"
    assert cfg.training.stopping.patience == 5


def test_load_composed_train(composed_train_path: Path) -> None:
    """load_job() resolves profile references and returns TrainingJobConfig."""
    cfg = load_job(composed_train_path)
    assert isinstance(cfg, TrainingJobConfig)
    assert cfg.experiment is not None
    assert cfg.experiment.name == "test-composed"
    assert cfg.model.name == "FFNN"
    assert cfg.plots.enabled is True
    assert cfg.plots.loss_curve is True


def test_load_search(search_path: Path) -> None:
    """load_job() returns a SearchJobConfig with typed search space params."""
    from dlkit.infrastructure.config.search_settings import CategoricalParam, LogFloatParam

    cfg = load_job(search_path)
    assert isinstance(cfg, SearchJobConfig)
    assert isinstance(cfg.search.space["training.optimizer.lr"], LogFloatParam)
    assert isinstance(cfg.search.space["model.hidden_size"], CategoricalParam)


def test_load_convergence(convergence_path: Path) -> None:
    """load_job() returns a ConvergenceJobConfig for run.type = 'convergence'."""
    cfg = load_job(convergence_path)
    assert isinstance(cfg, ConvergenceJobConfig)
    assert cfg.run.type == "convergence"
    assert cfg.convergence.sizes == (100, 500, 1000)
    assert cfg.convergence.repeats == 2
    assert cfg.convergence.target == 0.05


def test_load_multirun(multirun_path: Path) -> None:
    """load_job() returns a MultiRunJobConfig for run.type = 'multirun'.

    The `[multirun]` table's keys are hoisted onto the top-level model, and
    each child entry's `files` is resolved to an absolute path/pattern
    relative to the multirun config file's own directory.
    """
    cfg = load_job(multirun_path)
    assert isinstance(cfg, MultiRunJobConfig)
    assert cfg.run.type == "multirun"
    assert cfg.experiment_name == "test-multirun"
    assert cfg.parent_run_name == "sweep-parent"
    assert cfg.failure_policy == "continue_mark_parent_failed"
    assert len(cfg.runs) == 2

    explicit_child, glob_child = cfg.runs
    assert explicit_child.id == "explicit"
    assert explicit_child.label == "Explicit Child"
    assert explicit_child.files == [str(multirun_path.parent / "simple_train.toml")]

    assert glob_child.id == "variants"
    assert glob_child.files == str(multirun_path.parent / "multirun_variants" / "*.toml")


# ---------------------------------------------------------------------------
# Tests: run_type override
# ---------------------------------------------------------------------------


def test_run_type_override(simple_train_path: Path) -> None:
    """CLI can supply run_type when not in TOML; existing type is preserved if kwarg matches."""
    cfg = load_job(simple_train_path, run_type="train")
    assert cfg.run.type == "train"


# ---------------------------------------------------------------------------
# Tests: error paths
# ---------------------------------------------------------------------------


def test_profile_wrong_section_raises(tmp_path: Path, data_profile_path: Path) -> None:
    """Profile referenced as run.model must have a [model] section.

    Creates a job TOML that references the data profile as the model profile.
    Expects ConfigValidationError mentioning the wrong section.
    """
    job_toml = tmp_path / "bad_job.toml"
    job_toml.write_text(
        f'[run]\ntype = "train"\nmodel = "{data_profile_path.as_posix()}"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match=r"must contain a \[model\] section"):
        load_job(job_toml)


def test_plots_profile_wrong_section_raises(tmp_path: Path, data_profile_path: Path) -> None:
    """Profile referenced as run.plots must have a [plots] section.

    Creates a job TOML that references the data profile as the plots profile.
    Expects ConfigValidationError mentioning the wrong section.
    """
    job_toml = tmp_path / "bad_job.toml"
    job_toml.write_text(
        f'[run]\ntype = "train"\nplots = "{data_profile_path.as_posix()}"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match=r"must contain a \[plots\] section"):
        load_job(job_toml)


def test_missing_run_type_raises(tmp_path: Path) -> None:
    """load_job() raises ConfigValidationError when run.type is absent and no kwarg given."""
    job_toml = tmp_path / "no_type.toml"
    job_toml.write_text("[run]\nseed = 1\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="run.type"):
        load_job(job_toml)


# ---------------------------------------------------------------------------
# Tests: _deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_job_wins_over_profile() -> None:
    """Inline job keys override profile keys; non-conflicting keys are preserved."""
    base = {"training": {"optimizer": {"lr": 1e-3, "name": "Adam"}}}
    override = {"training": {"optimizer": {"lr": 5e-4}}}
    merged = _deep_merge(base, override)
    training = merged.get("training")
    assert isinstance(training, dict)
    optimizer = training.get("optimizer")
    assert isinstance(optimizer, dict)
    assert optimizer.get("lr") == 5e-4
    assert optimizer.get("name") == "Adam"


def test_deep_merge_does_not_mutate_inputs() -> None:
    """_deep_merge must not mutate base or override."""
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    _deep_merge(base, override)
    assert base == {"a": {"x": 1}}
    assert override == {"a": {"y": 2}}


def test_deep_merge_leaf_override() -> None:
    """Non-dict values in override replace non-dict values in base."""
    base = {"key": "old"}
    override = {"key": "new"}
    merged = _deep_merge(base, override)
    assert merged["key"] == "new"


def test_deep_merge_adds_missing_keys() -> None:
    """Keys present only in override are added to result."""
    base = {"a": 1}
    override = {"b": 2}
    merged = _deep_merge(base, override)
    assert merged == {"a": 1, "b": 2}
