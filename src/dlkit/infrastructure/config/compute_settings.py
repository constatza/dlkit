"""Compute topology settings — which runtime environment to resolve against.

One settings class per environment (a discriminated union). Most environments
carry no override fields at all: forcing a specific `devices`/`num_nodes` for
training is `TrainerSettings`'s own job (it mirrors `Trainer.__init__`
directly, the same way `TrainerSettings.strategy`/`.accelerator` already do)
— this model only says *which* environment to resolve against.

The exception is LSF/MPI/Kubeflow: those environments structurally cannot
auto-derive `devices`/`num_nodes` (see
`dlkit.infrastructure.compute.compute.md`), so leaving them unset isn't a
valid "let it auto-detect" state — it's a configuration error. Declaring them
as *required* fields on exactly those three classes makes an
under-configured LSF/MPI/Kubeflow job fail at config-load time instead of
silently training on Lightning's single-node default.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from dlkit.infrastructure.config.core.base_settings import BasicSettings

DeviceSpec = int | list[int] | Literal["auto"]


class AutoComputeSettings(BasicSettings):
    """Auto-detect among local/SLURM/torchelastic/LSF/MPI."""

    environment: Literal["auto"] = "auto"


class LocalComputeSettings(BasicSettings):
    """Single-process local execution."""

    environment: Literal["local"] = "local"


class SlurmComputeSettings(BasicSettings):
    """SLURM allocation launched via `srun`. devices/num_nodes auto-derive."""

    environment: Literal["slurm"] = "slurm"


class TorchElasticComputeSettings(BasicSettings):
    """`torchrun`-launched job. devices/num_nodes auto-derive from LOCAL_WORLD_SIZE."""

    environment: Literal["torchelastic"] = "torchelastic"


class LSFComputeSettings(BasicSettings):
    """LSF allocation launched via `jsrun`. devices/num_nodes cannot be auto-derived."""

    environment: Literal["lsf"] = "lsf"
    devices: DeviceSpec
    num_nodes: int


class MPIComputeSettings(BasicSettings):
    """MPI-launched job. devices/num_nodes cannot be auto-derived."""

    environment: Literal["mpi"] = "mpi"
    devices: DeviceSpec
    num_nodes: int


class KubeflowComputeSettings(BasicSettings):
    """Kubeflow PyTorchJob. Never auto-selected; devices/num_nodes are required."""

    environment: Literal["kubeflow"] = "kubeflow"
    devices: DeviceSpec
    num_nodes: int


ComputeEnvironmentSettings = Annotated[
    AutoComputeSettings
    | LocalComputeSettings
    | SlurmComputeSettings
    | TorchElasticComputeSettings
    | LSFComputeSettings
    | MPIComputeSettings
    | KubeflowComputeSettings,
    Field(discriminator="environment"),
]
