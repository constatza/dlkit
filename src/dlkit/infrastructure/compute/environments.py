"""Concrete compute environments, each bridging to one Lightning ``ClusterEnvironment``.

Rank/world-size/``MASTER_ADDR``/``MASTER_PORT`` wiring is always Lightning's
job (via ``Trainer``'s own cluster-environment auto-detection). Every class
here only answers "how many nodes, how many devices per node" — and is
explicit about when it cannot.

Environment variable access for this concern is confined to this file.

What works out of the box vs. what the deployment must already provide:

- Local: always works, zero setup.
- SLURM: works fully automatically under ``srun`` — no extra dlkit config
  needed. Requires the job to actually be launched with ``srun`` (not a bare
  ``sbatch`` script that calls ``python`` directly) — Lightning's own
  ``SLURMEnvironment`` warns if ``srun`` is available but unused.
- TorchElastic (torchrun): works fully automatically when launched via
  ``torchrun`` — it sets both ``WORLD_SIZE`` and ``LOCAL_WORLD_SIZE``, which
  is all this needs.
- LSF (jsrun): rank wiring works automatically, but node/device *counts*
  cannot be derived from environment variables alone (LSF only exposes a
  host rank file, not a plain node/task count) — set
  ``compute.devices``/``compute.num_nodes`` explicitly in config.
- MPI: rank wiring works automatically once launched via ``mpirun``/``mpiexec``
  with ``mpi4py`` installed, but node/device counts require live MPI
  collective calls to determine (not something to do just to size a
  ``Trainer`) — set ``compute.devices``/``compute.num_nodes`` explicitly.
- Kubeflow: does **not** auto-activate. Requires ``compute.environment =
  "kubeflow"`` explicitly, requires ``MASTER_ADDR``/``MASTER_PORT``/
  ``WORLD_SIZE``/``RANK`` to already be set by the PyTorchJob operator (which
  it does), and node/device counts must still be set explicitly in config —
  Kubeflow's PyTorchJob CRD does not expose them as env vars either.
"""

from __future__ import annotations

import os

from lightning.fabric.plugins.environments import (
    KubeflowEnvironment,
    LightningEnvironment,
    LSFEnvironment,
    MPIEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)

from .topology import ComputeEnvironment, ComputeTopology


class LocalComputeEnvironment(ComputeEnvironment):
    """Single-process local execution. Always available; the final fallback."""

    cluster_environment = LightningEnvironment

    @classmethod
    def detect(cls) -> bool:
        return True

    def resolve(self) -> ComputeTopology:
        return ComputeTopology(num_nodes=1, devices="auto")


class SlurmComputeEnvironment(ComputeEnvironment):
    """SLURM allocation launched via ``srun``. Fully automatic — no config needed."""

    cluster_environment = SLURMEnvironment

    def resolve(self) -> ComputeTopology:
        num_nodes = _read_int("SLURM_NNODES")
        devices = _read_int("SLURM_NTASKS_PER_NODE")
        return ComputeTopology(num_nodes=num_nodes, devices=devices)


class TorchElasticComputeEnvironment(ComputeEnvironment):
    """``torchrun``-launched job. Fully automatic when ``LOCAL_WORLD_SIZE`` is set."""

    cluster_environment = TorchElasticEnvironment

    def resolve(self) -> ComputeTopology:
        world_size = _read_int("WORLD_SIZE")
        local_world_size = _read_int("LOCAL_WORLD_SIZE")
        if world_size is None or local_world_size is None or local_world_size == 0:
            return ComputeTopology()
        return ComputeTopology(num_nodes=world_size // local_world_size, devices=local_world_size)


class LSFComputeEnvironment(ComputeEnvironment):
    """LSF allocation launched via ``jsrun``. Rank wiring is automatic; topology is not.

    Node/device counts require parsing ``LSB_DJOB_RANKFILE`` (which Lightning
    does internally via a private method) — not depended on here. Set
    ``compute.devices``/``compute.num_nodes`` explicitly under LSF.
    """

    cluster_environment = LSFEnvironment

    def resolve(self) -> ComputeTopology:
        return ComputeTopology()


class MPIComputeEnvironment(ComputeEnvironment):
    """MPI-launched job (``mpi4py`` required). Rank wiring is automatic; topology is not.

    Node/device counts are only computable via live MPI collective calls
    (``comm.gather``/``comm.bcast``), which is too invasive to run just to
    size a ``Trainer``. Set ``compute.devices``/``compute.num_nodes``
    explicitly under MPI.
    """

    cluster_environment = MPIEnvironment

    def resolve(self) -> ComputeTopology:
        return ComputeTopology()


class KubeflowComputeEnvironment(ComputeEnvironment):
    """Kubeflow ``PyTorchJob``. Never auto-selected — Lightning's own
    ``KubeflowEnvironment.detect()`` raises ``NotImplementedError`` by design,
    so this must be chosen explicitly via ``compute.environment = "kubeflow"``.

    Node/device counts are not exposed by the PyTorchJob CRD as env vars
    either — set ``compute.devices``/``compute.num_nodes`` explicitly.
    """

    cluster_environment = KubeflowEnvironment

    @classmethod
    def detect(cls) -> bool:
        return False

    def resolve(self) -> ComputeTopology:
        return ComputeTopology(cluster_environment=KubeflowEnvironment())


def _read_int(name: str) -> int | None:
    """Read an environment variable as int, or None if absent/invalid."""
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
