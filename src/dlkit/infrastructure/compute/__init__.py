"""Compute topology resolution: local process vs. any Lightning-supported scheduler.

Bridges to Lightning's own ``ClusterEnvironment`` plugins for rank/world-size
wiring; this package only derives node/device counts. See ``compute.md``.
"""

from .environments import (
    KubeflowComputeEnvironment,
    LocalComputeEnvironment,
    LSFComputeEnvironment,
    MPIComputeEnvironment,
    SlurmComputeEnvironment,
    TorchElasticComputeEnvironment,
)
from .resolver import ComputeEnvironmentName, resolve_compute_environment
from .topology import ComputeEnvironment, ComputeTopology

__all__ = [
    "ComputeEnvironment",
    "ComputeEnvironmentName",
    "ComputeTopology",
    "KubeflowComputeEnvironment",
    "LSFComputeEnvironment",
    "LocalComputeEnvironment",
    "MPIComputeEnvironment",
    "SlurmComputeEnvironment",
    "TorchElasticComputeEnvironment",
    "resolve_compute_environment",
]
