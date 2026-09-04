"""Compute topology contracts: what a cluster environment resolves to.

``ComputeEnvironment`` bridges to Lightning's own ``ClusterEnvironment``
plugins (``lightning.fabric.plugins.environments``) rather than
reimplementing scheduler detection or rank/world-size wiring — Lightning
already owns that. Each concrete environment answers exactly one question:
how many nodes, how many devices per node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from lightning.fabric.plugins.environments import ClusterEnvironment


@dataclass(frozen=True)
class ComputeTopology:
    """Node/device counts resolved for the current process, plus an optional
    pre-built cluster environment for schedulers Lightning cannot auto-detect.

    Attributes:
        num_nodes: Number of nodes in the job, or None if not derivable from
            the environment (caller must supply it explicitly).
        devices: Devices per node (count, explicit index list, or "auto"),
            or None if not derivable from the environment.
        cluster_environment: A pre-built Lightning ``ClusterEnvironment``
            instance to pass as ``Trainer(plugins=[...])``. Only set for
            environments Lightning never auto-detects (Kubeflow) — for every
            auto-detected environment (SLURM, TorchElastic, LSF, MPI) this is
            None, since Lightning already selects and wires it up on its own.
    """

    num_nodes: int | None = None
    devices: int | list[int] | str | None = None
    cluster_environment: ClusterEnvironment | None = None


class ComputeEnvironment(ABC):
    """Bridges a DLKit topology resolver to a Lightning ``ClusterEnvironment``.

    ``detect()`` delegates to the bridged Lightning class by default, so
    DLKit never reimplements scheduler-detection edge cases (e.g. SLURM's
    interactive-session exclusion) that Lightning already maintains.
    """

    cluster_environment: ClassVar[type[ClusterEnvironment]]

    @classmethod
    def detect(cls) -> bool:
        """Return True if the current process was launched under this environment."""
        return cls.cluster_environment.detect()

    @abstractmethod
    def resolve(self) -> ComputeTopology:
        """Derive the node/device topology for the current process."""
