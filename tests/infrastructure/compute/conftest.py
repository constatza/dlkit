"""Fixtures for infrastructure.compute tests: env vars each launcher sets."""

from __future__ import annotations

import pytest


@pytest.fixture
def slurm_env() -> dict[str, str]:
    """Env vars `srun` sets for a 2-node, 4-tasks-per-node SLURM job."""
    return {
        "SLURM_NTASKS": "8",
        "SLURM_NNODES": "2",
        "SLURM_NTASKS_PER_NODE": "4",
    }


@pytest.fixture
def torchelastic_env() -> dict[str, str]:
    """Env vars `torchrun` sets for a 2-node, 4-devices-per-node job."""
    return {
        "TORCHELASTIC_RUN_ID": "test-run-id",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "4",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "GROUP_RANK": "0",
    }


@pytest.fixture
def lsf_env() -> dict[str, str]:
    """Minimal env vars LSFEnvironment.detect() requires (jsrun launch)."""
    return {
        "LSB_JOBID": "12345",
        "LSB_DJOB_RANKFILE": "/tmp/dlkit-test-rankfile-does-not-need-to-exist",
        "JSM_NAMESPACE_LOCAL_RANK": "0",
        "JSM_NAMESPACE_SIZE": "4",
    }


def apply_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    """Set every key/value in `env` via monkeypatch.setenv."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
