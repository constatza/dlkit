"""Tests for RunSpec/MultiRunSpec value objects.

Covers:
- RunSpec is frozen (attribute assignment raises FrozenInstanceError)
- RunSpec.tags/params/metadata default to empty dicts when not supplied
- MultiRunSpec is frozen and defaults failure_policy to "fail_fast"
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from dlkit.engine.workflows.multi_run import MultiRunSpec, RunSpec
from dlkit.infrastructure.config.job_config import JobConfig


def test_run_spec_is_frozen(spec_a: RunSpec) -> None:
    """RunSpec is a frozen dataclass — attribute assignment must raise.

    Args:
        spec_a: A RunSpec fixture.
    """
    mutable_spec = cast(Any, spec_a)
    with pytest.raises(FrozenInstanceError):
        mutable_spec.run_name = "other"


def test_run_spec_stores_id_and_run_name(spec_a: RunSpec) -> None:
    """RunSpec stores id/run_name correctly.

    Args:
        spec_a: RunSpec fixture with id="a", run_name="variant_a".
    """
    assert spec_a.id == "a"
    assert spec_a.run_name == "variant_a"


def test_run_spec_default_optional_fields_are_empty(job_config_settings: JobConfig) -> None:
    """RunSpec.tags/params/metadata default to empty dicts when not supplied.

    Args:
        job_config_settings: Real JobConfig fixture.
    """
    spec = RunSpec(id="x", label="x", settings=job_config_settings, run_name="x")
    assert spec.tags == {}
    assert spec.params == {}
    assert spec.metadata == {}


def test_multi_run_spec_is_frozen(spec_a: RunSpec) -> None:
    """MultiRunSpec is a frozen dataclass — attribute assignment must raise.

    Args:
        spec_a: A RunSpec fixture used as the sole child.
    """
    multi_run_spec = MultiRunSpec(
        experiment_name="exp",
        parent_run_name="parent",
        children=(spec_a,),
    )
    mutable_spec = cast(Any, multi_run_spec)
    with pytest.raises(FrozenInstanceError):
        mutable_spec.experiment_name = "other"


def test_multi_run_spec_defaults_to_fail_fast(spec_a: RunSpec) -> None:
    """MultiRunSpec.failure_policy defaults to 'fail_fast' when not supplied.

    Args:
        spec_a: A RunSpec fixture used as the sole child.
    """
    multi_run_spec = MultiRunSpec(
        experiment_name="exp",
        parent_run_name="parent",
        children=(spec_a,),
    )
    assert multi_run_spec.failure_policy == "fail_fast"
