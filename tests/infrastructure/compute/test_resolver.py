"""Tests for resolve_compute_environment: detection priority and override forcing."""

from __future__ import annotations

import pytest

from dlkit.infrastructure.compute import ComputeTopology, resolve_compute_environment

from .conftest import apply_env


def test_auto_falls_back_to_local_with_no_environment_detected():
    assert resolve_compute_environment("auto") == ComputeTopology(num_nodes=1, devices="auto")


def test_auto_selects_slurm_when_slurm_env_vars_present(monkeypatch, slurm_env):
    apply_env(monkeypatch, slurm_env)

    assert resolve_compute_environment("auto") == ComputeTopology(num_nodes=2, devices=4)


def test_auto_prefers_torchelastic_over_slurm_when_both_present(
    monkeypatch, slurm_env, torchelastic_env
):
    """Mirrors Lightning's own _choose_and_init_cluster_environment() priority:
    TorchElastic is checked before SLURM, since torchrun can itself run inside
    a SLURM allocation."""
    apply_env(monkeypatch, slurm_env)
    apply_env(monkeypatch, torchelastic_env)

    assert resolve_compute_environment("auto") == ComputeTopology(num_nodes=2, devices=4)


def test_override_forces_local_even_under_slurm(monkeypatch, slurm_env):
    apply_env(monkeypatch, slurm_env)

    assert resolve_compute_environment("local") == ComputeTopology(num_nodes=1, devices="auto")


def test_override_forces_slurm_derivation_off_cluster(monkeypatch, slurm_env):
    """Forcing bypasses detection entirely — useful for testing SLURM-shaped
    values without actually running under srun."""
    apply_env(monkeypatch, slurm_env)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)  # detect() would say False

    assert resolve_compute_environment("slurm") == ComputeTopology(num_nodes=2, devices=4)


def test_kubeflow_only_reachable_via_explicit_override():
    topology = resolve_compute_environment("kubeflow")

    assert topology.cluster_environment is not None


def test_unknown_override_raises_value_error():
    with pytest.raises(ValueError, match="Unknown compute environment"):
        resolve_compute_environment("pbs")  # type: ignore[arg-type]
