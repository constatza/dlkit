"""Fixtures for engine.data.sources tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from dlkit.common.sources import ArraySource
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import NpyEntry, NpzEntry, ValueEntry

# ---------------------------------------------------------------------------
# Raw array fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def n_samples() -> int:
    """Canonical sample count used across source tests.

    Returns:
        Integer sample count (20).
    """
    return 20


@pytest.fixture
def sample_shape() -> tuple[int, ...]:
    """Shape of a single sample (feature vector).

    Returns:
        Tuple ``(8,)``.
    """
    return (8,)


@pytest.fixture
def feature_tensor(n_samples: int, sample_shape: tuple[int, ...]) -> torch.Tensor:
    """Float32 feature tensor of shape ``(n_samples, *sample_shape)``.

    Args:
        n_samples: Number of samples fixture.
        sample_shape: Sample shape fixture.

    Returns:
        Deterministic float32 tensor.
    """
    rng = torch.Generator()
    rng.manual_seed(0)
    return torch.rand(n_samples, *sample_shape, generator=rng, dtype=torch.float32)


@pytest.fixture
def target_tensor(n_samples: int) -> torch.Tensor:
    """Float32 target tensor of shape ``(n_samples, 1)``.

    Args:
        n_samples: Number of samples fixture.

    Returns:
        Deterministic float32 tensor.
    """
    rng = torch.Generator()
    rng.manual_seed(1)
    return torch.rand(n_samples, 1, generator=rng, dtype=torch.float32)


@pytest.fixture
def npy_feature_path(tmp_path: Path, feature_tensor: torch.Tensor) -> Path:
    """Save ``feature_tensor`` as a ``.npy`` file and return its path.

    Args:
        tmp_path: Pytest temporary directory fixture.
        feature_tensor: Feature tensor fixture.

    Returns:
        Path to the saved ``.npy`` file.
    """
    path = tmp_path / "x.npy"
    np.save(path, feature_tensor.numpy())
    return path


@pytest.fixture
def npz_feature_path(tmp_path: Path, feature_tensor: torch.Tensor) -> dict[str, Any]:
    """Save ``feature_tensor`` as a ``.npz`` file and return path + key.

    Args:
        tmp_path: Pytest temporary directory fixture.
        feature_tensor: Feature tensor fixture.

    Returns:
        Dict with ``path`` (Path) and ``key`` (str).
    """
    path = tmp_path / "data.npz"
    np.savez(path, features=feature_tensor.numpy())
    return {"path": path, "key": "features"}


# ---------------------------------------------------------------------------
# Entry fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def npy_feature_entry(npy_feature_path: Path) -> NpyEntry:
    """``NpyEntry`` pointing at the ``.npy`` feature file.

    Args:
        npy_feature_path: Path fixture for the ``.npy`` feature file.

    Returns:
        Configured ``NpyEntry`` with ``data_role=FEATURE``.
    """
    return NpyEntry(name="x", path=npy_feature_path, data_role=DataRole.FEATURE, mmap=False)


@pytest.fixture
def npz_feature_entry(npz_feature_path: dict[str, Any]) -> NpzEntry:
    """``NpzEntry`` pointing at the ``.npz`` feature archive.

    Args:
        npz_feature_path: Path-and-key fixture for the ``.npz`` archive.

    Returns:
        Configured ``NpzEntry`` with ``data_role=FEATURE``.
    """
    return NpzEntry(
        name="features",
        path=npz_feature_path["path"],
        key=npz_feature_path["key"],
        data_role=DataRole.FEATURE,
        mmap=False,
    )


@pytest.fixture
def value_feature_entry(feature_tensor: torch.Tensor) -> ValueEntry:
    """``ValueEntry`` wrapping ``feature_tensor`` as a feature.

    Args:
        feature_tensor: In-memory feature tensor fixture.

    Returns:
        Configured ``ValueEntry`` with ``data_role=FEATURE``.
    """
    return ValueEntry(name="x", value=feature_tensor, data_role=DataRole.FEATURE)


@pytest.fixture
def value_target_entry(target_tensor: torch.Tensor) -> ValueEntry:
    """``ValueEntry`` wrapping ``target_tensor`` as a target.

    Args:
        target_tensor: In-memory target tensor fixture.

    Returns:
        Configured ``ValueEntry`` with ``data_role=TARGET``.
    """
    return ValueEntry(name="y", value=target_tensor, data_role=DataRole.TARGET)


@pytest.fixture
def single_sample_value_entry() -> ValueEntry:
    """``ValueEntry`` with exactly 1 sample — triggers broadcast wrapping.

    Returns:
        ``ValueEntry`` with shape ``(1, 4)`` and ``data_role=FEATURE``.
    """
    return ValueEntry(
        name="bias",
        value=torch.ones(1, 4, dtype=torch.float32),
        data_role=DataRole.FEATURE,
    )


@pytest.fixture
def none_value_entry() -> ValueEntry:
    """``ValueEntry`` with ``value=None`` — placeholder that must raise on dispatch.

    Returns:
        ``ValueEntry`` with no concrete value and ``data_role=FEATURE``.
    """
    return ValueEntry(name="x", value=None, data_role=DataRole.FEATURE)


@pytest.fixture
def conflicting_feature_entry(feature_tensor: torch.Tensor) -> ValueEntry:
    """``ValueEntry`` for use in conflicting-n_samples tests.

    Args:
        feature_tensor: Feature tensor fixture (canonical n_samples).

    Returns:
        ``ValueEntry`` wrapping ``feature_tensor`` with ``data_role=FEATURE``.
    """
    return ValueEntry(name="a", value=feature_tensor, data_role=DataRole.FEATURE)


@pytest.fixture
def conflicting_target_tensor(n_samples: int) -> torch.Tensor:
    """Tensor with ``n_samples + 5`` rows — creates a sample-count conflict.

    Args:
        n_samples: Canonical sample count fixture.

    Returns:
        Float32 tensor of shape ``(n_samples + 5, 4)``.
    """
    return torch.zeros(n_samples + 5, 4, dtype=torch.float32)


@pytest.fixture
def conflicting_target_entry(conflicting_target_tensor: torch.Tensor) -> ValueEntry:
    """``ValueEntry`` wrapping the conflicting target tensor.

    Args:
        conflicting_target_tensor: Tensor with a mismatched sample count.

    Returns:
        ``ValueEntry`` with ``data_role=TARGET``.
    """
    return ValueEntry(name="b", value=conflicting_target_tensor, data_role=DataRole.TARGET)


# ---------------------------------------------------------------------------
# Zarr stub fixtures (for source_from_entry case 2 dispatch test)
# ---------------------------------------------------------------------------


class _StubArraySource:
    """Minimal ``ArraySource`` implementation backed by a tensor.

    Used as a stand-in for ``ZarrLazyReader`` in unit tests — open_reader()
    returns this instance directly (case 2 of ``source_from_entry``).
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    @property
    def n_samples(self) -> int:
        """Number of samples (leading dimension).

        Returns:
            Leading dimension of the backing tensor.
        """
        return self._tensor.shape[0]

    def get_item(self, idx: int) -> torch.Tensor:
        """Return the sample at position ``idx``.

        Args:
            idx: Sample index.

        Returns:
            Tensor of shape ``(*sample_shape)``.
        """
        return self._tensor[idx]

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        """Return a batch of samples.

        Args:
            indices: List of sample indices.

        Returns:
            Tensor of shape ``(B, *sample_shape)``.
        """
        return self._tensor[indices]


class _StubArrayBasedEntry:
    """Minimal path-based entry whose ``open_reader()`` returns an ``ArraySource``.

    Simulates entries such as ``ZarrEntry`` where ``open_reader()`` returns a
    live ``ArraySource`` rather than a ``Path``.  ``isinstance(entry,
    IValueBased)`` must be ``False`` for the dispatch to reach case 2.
    """

    def __init__(self, name: str, source: ArraySource) -> None:
        self.name = name
        self._source = source

    def open_reader(self) -> ArraySource:
        """Return the pre-built ``ArraySource`` directly.

        Returns:
            The ``ArraySource`` passed at construction.
        """
        return self._source


@pytest.fixture
def zarr_stub_source(feature_tensor: torch.Tensor) -> _StubArraySource:
    """``_StubArraySource`` backed by ``feature_tensor``.

    Args:
        feature_tensor: Feature tensor fixture.

    Returns:
        ``_StubArraySource`` with ``n_samples`` matching the fixture.
    """
    return _StubArraySource(feature_tensor)


@pytest.fixture
def zarr_stub_entry(zarr_stub_source: _StubArraySource) -> _StubArrayBasedEntry:
    """``_StubArrayBasedEntry`` whose ``open_reader()`` returns ``zarr_stub_source``.

    Args:
        zarr_stub_source: Stub ``ArraySource`` fixture.

    Returns:
        ``_StubArrayBasedEntry`` named ``"x"``.
    """
    return _StubArrayBasedEntry(name="x", source=zarr_stub_source)
