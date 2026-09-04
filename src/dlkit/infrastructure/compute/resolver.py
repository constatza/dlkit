"""Selects and resolves the active compute environment.

Adding a new scheduler touches this module (one new ``ComputeEnvironment``
subclass in ``environments.py`` plus one entry in ``_ENVIRONMENTS``, or
``_EXPLICIT_ONLY`` if it can never be auto-detected like Kubeflow) *and*
``infrastructure.config.compute_settings`` (one matching settings class in
the ``ComputeEnvironmentSettings`` union, using the same environment name) —
the environment name is duplicated across both files with no shared source
of truth, so
``tests/infrastructure/config/test_compute_settings.py::test_compute_settings_environments_match_resolver_registered_names``
asserts they stay in sync. Forgetting the settings-side entry doesn't fail
loudly on its own: the resolver would work if reached, but
``RunSettings.compute.environment`` could never select it, since the
discriminated union would reject the name.
"""

from __future__ import annotations

from typing import Literal

from .environments import (
    KubeflowComputeEnvironment,
    LocalComputeEnvironment,
    LSFComputeEnvironment,
    MPIComputeEnvironment,
    SlurmComputeEnvironment,
    TorchElasticComputeEnvironment,
)
from .topology import ComputeEnvironment, ComputeTopology

ComputeEnvironmentName = Literal["auto", "local", "slurm", "torchelastic", "lsf", "mpi", "kubeflow"]

# Priority order for "auto" — mirrors Lightning's own
# _choose_and_init_cluster_environment() order, with Local as the
# always-true fallback appended last.
_ENVIRONMENTS: tuple[type[ComputeEnvironment], ...] = (
    TorchElasticComputeEnvironment,
    SlurmComputeEnvironment,
    LSFComputeEnvironment,
    MPIComputeEnvironment,
    LocalComputeEnvironment,
)

# Never joins the "auto" cascade — must be selected explicitly.
_EXPLICIT_ONLY: dict[str, type[ComputeEnvironment]] = {
    "kubeflow": KubeflowComputeEnvironment,
}

_BY_NAME: dict[str, type[ComputeEnvironment]] = {
    "local": LocalComputeEnvironment,
    "slurm": SlurmComputeEnvironment,
    "torchelastic": TorchElasticComputeEnvironment,
    "lsf": LSFComputeEnvironment,
    "mpi": MPIComputeEnvironment,
    **_EXPLICIT_ONLY,
}


def resolve_compute_environment(override: ComputeEnvironmentName = "auto") -> ComputeTopology:
    """Resolve the active compute topology.

    Args:
        override: "auto" walks the detection-priority order and uses the
            first environment whose ``detect()`` returns True. Any other
            name forces that environment directly, bypassing detection —
            e.g. for testing SLURM-shaped values off-cluster, or selecting
            "kubeflow", which never auto-detects.

    Returns:
        The resolved ComputeTopology.

    Raises:
        ValueError: If override names an unknown environment.
    """
    if override == "auto":
        env_cls = next(env for env in _ENVIRONMENTS if env.detect())
        return env_cls().resolve()

    env_cls = _BY_NAME.get(override)
    if env_cls is None:
        raise ValueError(
            f"Unknown compute environment {override!r}. Valid values: "
            f"{', '.join(['auto', *_BY_NAME])}"
        )
    return env_cls().resolve()
