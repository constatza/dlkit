"""Tests for the ComputeEnvironmentSettings discriminated union.

LSF/MPI/Kubeflow can't auto-derive devices/num_nodes (see
dlkit.infrastructure.compute.compute.md), so those three settings classes
declare devices/num_nodes as required — an under-configured job for one of
them must fail at config-load time, not silently train on Lightning's
single-node default.
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from dlkit.infrastructure.compute.resolver import ComputeEnvironmentName
from dlkit.infrastructure.config.compute_settings import (
    AutoComputeSettings,
    ComputeEnvironmentSettings,
    KubeflowComputeSettings,
    LocalComputeSettings,
    LSFComputeSettings,
    MPIComputeSettings,
    SlurmComputeSettings,
    TorchElasticComputeSettings,
)
from dlkit.infrastructure.config.run_settings import RunSettings


def test_run_settings_defaults_compute_to_auto():
    assert isinstance(RunSettings().compute, AutoComputeSettings)


@pytest.mark.parametrize(
    ("environment", "extra", "expected_type"),
    [
        ("local", {}, LocalComputeSettings),
        ("slurm", {}, SlurmComputeSettings),
        ("torchelastic", {}, TorchElasticComputeSettings),
        ("lsf", {"devices": 4, "num_nodes": 2}, LSFComputeSettings),
        ("mpi", {"devices": 4, "num_nodes": 2}, MPIComputeSettings),
        ("kubeflow", {"devices": 4, "num_nodes": 2}, KubeflowComputeSettings),
    ],
)
def test_discriminator_dispatches_to_the_matching_class(environment, extra, expected_type):
    run = RunSettings.model_validate({"compute": {"environment": environment, **extra}})

    assert isinstance(run.compute, expected_type)


@pytest.mark.parametrize("environment", ["lsf", "mpi", "kubeflow"])
def test_undetectable_environments_require_devices_and_num_nodes(environment):
    with pytest.raises(ValidationError):
        RunSettings.model_validate({"compute": {"environment": environment}})


@pytest.mark.parametrize("environment", ["lsf", "mpi", "kubeflow"])
def test_undetectable_environments_reject_missing_num_nodes_alone(environment):
    with pytest.raises(ValidationError):
        RunSettings.model_validate({"compute": {"environment": environment, "devices": 4}})


@pytest.mark.parametrize("environment", ["local", "slurm", "torchelastic"])
def test_auto_derivable_environments_have_no_devices_or_num_nodes_fields(environment):
    """Forcing a specific devices/num_nodes for these environments is
    TrainerSettings's own job (it mirrors Trainer's constructor directly) —
    ComputeEnvironmentSettings only says which environment to resolve against, so these
    classes carry no override fields, and reject any if given."""
    run = RunSettings.model_validate({"compute": {"environment": environment}})

    assert not hasattr(run.compute, "devices")
    assert not hasattr(run.compute, "num_nodes")

    with pytest.raises(ValidationError):
        RunSettings.model_validate({"compute": {"environment": environment, "devices": 4}})


def test_unknown_environment_name_is_rejected_by_the_discriminator():
    with pytest.raises(ValidationError):
        RunSettings.model_validate({"compute": {"environment": "pbs"}})


def test_compute_settings_environments_match_resolver_registered_names():
    """Every ComputeEnvironmentSettings variant's `environment` literal must
    have a matching entry in the resolver's registry, and vice versa.
    resolver.py's module docstring claims adding a scheduler only touches
    environments.py + resolver.py — that claim is only true if this
    enumeration and the resolver's stay in sync, and nothing else enforces
    that; this is the test that keeps it honest."""
    union_members = get_args(get_args(ComputeEnvironmentSettings)[0])
    settings_names = {cls.model_fields["environment"].default for cls in union_members}

    resolver_names = set(get_args(ComputeEnvironmentName))

    assert settings_names == resolver_names
