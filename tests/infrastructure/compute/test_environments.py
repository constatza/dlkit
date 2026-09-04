"""Unit tests for each ComputeEnvironment's detect()/resolve()."""

from __future__ import annotations

from lightning.fabric.plugins.environments import (
    KubeflowEnvironment,
    LSFEnvironment,
    MPIEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)

from dlkit.infrastructure.compute import (
    ComputeTopology,
    KubeflowComputeEnvironment,
    LocalComputeEnvironment,
    LSFComputeEnvironment,
    MPIComputeEnvironment,
    SlurmComputeEnvironment,
    TorchElasticComputeEnvironment,
)

from .conftest import apply_env


def test_local_always_detects_and_resolves_single_node_auto_devices():
    assert LocalComputeEnvironment.detect() is True
    assert LocalComputeEnvironment().resolve() == ComputeTopology(num_nodes=1, devices="auto")


def test_slurm_resolves_nodes_and_devices_from_env(monkeypatch, slurm_env):
    apply_env(monkeypatch, slurm_env)

    assert SlurmComputeEnvironment.detect() is True
    topology = SlurmComputeEnvironment().resolve()

    assert topology == ComputeTopology(num_nodes=2, devices=4)


def test_slurm_does_not_detect_without_slurm_env_vars(monkeypatch):
    monkeypatch.delenv("SLURM_NTASKS", raising=False)

    assert SlurmComputeEnvironment.detect() is False


def test_torchelastic_resolves_nodes_and_devices_from_env(monkeypatch, torchelastic_env):
    apply_env(monkeypatch, torchelastic_env)

    assert TorchElasticComputeEnvironment.detect() is True
    topology = TorchElasticComputeEnvironment().resolve()

    assert topology == ComputeTopology(num_nodes=2, devices=4)


def test_torchelastic_resolves_nothing_when_local_world_size_absent(monkeypatch):
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "test-run-id")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)

    topology = TorchElasticComputeEnvironment().resolve()

    assert topology == ComputeTopology()


def test_lsf_detects_via_env_but_cannot_derive_topology(monkeypatch, lsf_env):
    apply_env(monkeypatch, lsf_env)

    assert LSFComputeEnvironment.detect() is True
    assert LSFComputeEnvironment().resolve() == ComputeTopology()


def test_mpi_cannot_derive_topology_even_when_detected(monkeypatch):
    monkeypatch.setattr(MPIEnvironment, "detect", staticmethod(lambda: True))

    assert MPIComputeEnvironment.detect() is True
    assert MPIComputeEnvironment().resolve() == ComputeTopology()


def test_kubeflow_never_auto_detects(monkeypatch):
    # Lightning's own KubeflowEnvironment.detect() raises NotImplementedError;
    # KubeflowComputeEnvironment must never call it.
    assert KubeflowComputeEnvironment.detect() is False


def test_kubeflow_resolve_carries_a_lightning_cluster_environment_for_plugins():
    topology = KubeflowComputeEnvironment().resolve()

    assert topology.num_nodes is None
    assert topology.devices is None
    assert isinstance(topology.cluster_environment, KubeflowEnvironment)


def test_each_environment_detect_delegates_to_its_bridged_lightning_class(monkeypatch):
    """detect() must reflect whatever the bridged Lightning class reports —
    DLKit must never reimplement scheduler-detection logic itself."""
    for compute_env, lightning_env in (
        (SlurmComputeEnvironment, SLURMEnvironment),
        (TorchElasticComputeEnvironment, TorchElasticEnvironment),
        (LSFComputeEnvironment, LSFEnvironment),
        (MPIComputeEnvironment, MPIEnvironment),
    ):
        monkeypatch.setattr(lightning_env, "detect", staticmethod(lambda: True))
        assert compute_env.detect() is True

        monkeypatch.setattr(lightning_env, "detect", staticmethod(lambda: False))
        assert compute_env.detect() is False
