"""Tests for engine.data.sources implementations.

Covers:
- ``EagerFileSource``: load from ``.npy`` / ``.npz`` tmp files
- ``TensorSource``: wrap an in-memory tensor
- ``BroadcastSource``: singleton broadcasting semantics
- ``RoleSourceMap.n_samples``: canonical-N resolution
- ``source_from_entry``: the three dispatch cases
- ``build_role_source_map``: end-to-end with real entry objects
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from dlkit.common.sources import ArraySource
from dlkit.engine.data.sources import (
    BroadcastSource,
    EagerFileSource,
    RoleSourceMap,
    TensorSource,
    build_role_source_map,
    source_from_entry,
)
from dlkit.infrastructure.config.entry_types import NpyEntry, NpzEntry, ValueEntry

# ═══════════════════════════════════════════════════════════════════════════
# EagerFileSource
# ═══════════════════════════════════════════════════════════════════════════


class TestEagerFileSource:
    """Unit tests for ``EagerFileSource``."""

    def test_n_samples_from_npy(
        self,
        npy_feature_path: Path,
        n_samples: int,
    ) -> None:
        """n_samples matches the leading dimension of the .npy file.

        Args:
            npy_feature_path: Path to a saved ``.npy`` feature file.
            n_samples: Expected sample count fixture.
        """
        src = EagerFileSource(npy_feature_path)
        assert src.n_samples == n_samples

    def test_get_item_shape(
        self,
        npy_feature_path: Path,
        sample_shape: tuple[int, ...],
    ) -> None:
        """get_item returns a tensor with the expected sample shape.

        Args:
            npy_feature_path: Path to a saved ``.npy`` feature file.
            sample_shape: Expected per-sample shape fixture.
        """
        src = EagerFileSource(npy_feature_path)
        item = src.get_item(0)
        assert item.shape == torch.Size(sample_shape)

    def test_get_batch_shape(
        self,
        npy_feature_path: Path,
        sample_shape: tuple[int, ...],
    ) -> None:
        """get_batch returns a tensor of shape (B, *sample_shape).

        Args:
            npy_feature_path: Path to a saved ``.npy`` feature file.
            sample_shape: Expected per-sample shape fixture.
        """
        src = EagerFileSource(npy_feature_path)
        batch = src.get_batch([0, 1, 2])
        assert batch.shape == torch.Size((3, *sample_shape))

    def test_get_item_values_match_saved_array(
        self,
        npy_feature_path: Path,
        feature_tensor: torch.Tensor,
    ) -> None:
        """get_item values match the originally saved array.

        Args:
            npy_feature_path: Path to a saved ``.npy`` feature file.
            feature_tensor: The original tensor used to create the file.
        """
        src = EagerFileSource(npy_feature_path)
        # dtype may differ due to precision service; compare after casting
        assert torch.allclose(src.get_item(0).float(), feature_tensor[0].float(), atol=1e-5)

    def test_get_batch_values_match_saved_array(
        self,
        npy_feature_path: Path,
        feature_tensor: torch.Tensor,
    ) -> None:
        """get_batch values match the originally saved array.

        Args:
            npy_feature_path: Path to a saved ``.npy`` feature file.
            feature_tensor: The original tensor used to create the file.
        """
        src = EagerFileSource(npy_feature_path)
        indices = [2, 5, 7]
        batch = src.get_batch(indices)
        expected = feature_tensor[indices].float()
        assert torch.allclose(batch.float(), expected, atol=1e-5)

    def test_load_from_npz_with_array_key(
        self,
        npz_feature_path: dict[str, Any],
        feature_tensor: torch.Tensor,
        n_samples: int,
        sample_shape: tuple[int, ...],
    ) -> None:
        """EagerFileSource loads the correct array from a .npz archive.

        Args:
            npz_feature_path: Path-and-key fixture for the ``.npz`` archive.
            feature_tensor: The original tensor used to create the archive.
            n_samples: Expected sample count fixture.
            sample_shape: Expected per-sample shape fixture.
        """
        src = EagerFileSource(npz_feature_path["path"], array_key=npz_feature_path["key"])
        assert src.n_samples == n_samples
        assert src.get_item(0).shape == torch.Size(sample_shape)

    def test_satisfies_array_source_protocol(self, npy_feature_path: Path) -> None:
        """EagerFileSource satisfies the ArraySource runtime-checkable protocol.

        Args:
            npy_feature_path: Path to a saved ``.npy`` feature file.
        """
        src = EagerFileSource(npy_feature_path)
        assert isinstance(src, ArraySource)


# ═══════════════════════════════════════════════════════════════════════════
# TensorSource
# ═══════════════════════════════════════════════════════════════════════════


class TestTensorSource:
    """Unit tests for ``TensorSource``."""

    def test_n_samples(self, feature_tensor: torch.Tensor, n_samples: int) -> None:
        """n_samples equals the leading dimension of the wrapped tensor.

        Args:
            feature_tensor: In-memory feature tensor fixture.
            n_samples: Expected sample count fixture.
        """
        src = TensorSource(feature_tensor)
        assert src.n_samples == n_samples

    def test_get_item_shape(
        self,
        feature_tensor: torch.Tensor,
        sample_shape: tuple[int, ...],
    ) -> None:
        """get_item returns a tensor with the expected sample shape.

        Args:
            feature_tensor: In-memory feature tensor fixture.
            sample_shape: Expected per-sample shape fixture.
        """
        src = TensorSource(feature_tensor)
        assert src.get_item(0).shape == torch.Size(sample_shape)

    def test_get_item_is_same_data(self, feature_tensor: torch.Tensor) -> None:
        """get_item returns a view into the original tensor data.

        Args:
            feature_tensor: In-memory feature tensor fixture.
        """
        src = TensorSource(feature_tensor)
        assert torch.equal(src.get_item(3), feature_tensor[3])

    def test_get_batch_shape(
        self,
        feature_tensor: torch.Tensor,
        sample_shape: tuple[int, ...],
    ) -> None:
        """get_batch returns a tensor of shape (B, *sample_shape).

        Args:
            feature_tensor: In-memory feature tensor fixture.
            sample_shape: Expected per-sample shape fixture.
        """
        src = TensorSource(feature_tensor)
        batch = src.get_batch([0, 1, 2])
        assert batch.shape == torch.Size((3, *sample_shape))

    def test_get_batch_values(self, feature_tensor: torch.Tensor) -> None:
        """get_batch returns the correct rows from the wrapped tensor.

        Args:
            feature_tensor: In-memory feature tensor fixture.
        """
        src = TensorSource(feature_tensor)
        indices = [0, 4, 9]
        assert torch.equal(src.get_batch(indices), feature_tensor[indices])

    def test_satisfies_array_source_protocol(self, feature_tensor: torch.Tensor) -> None:
        """TensorSource satisfies the ArraySource runtime-checkable protocol.

        Args:
            feature_tensor: In-memory feature tensor fixture.
        """
        src = TensorSource(feature_tensor)
        assert isinstance(src, ArraySource)


# ═══════════════════════════════════════════════════════════════════════════
# BroadcastSource
# ═══════════════════════════════════════════════════════════════════════════


class TestBroadcastSource:
    """Unit tests for ``BroadcastSource``."""

    @pytest.fixture
    def single_sample_src(self) -> TensorSource:
        """TensorSource wrapping a single 4-element sample.

        Returns:
            ``TensorSource`` with shape ``(1, 4)``.
        """
        return TensorSource(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    @pytest.fixture
    def broadcast_src(self, single_sample_src: TensorSource) -> BroadcastSource:
        """BroadcastSource wrapping ``single_sample_src``.

        Args:
            single_sample_src: Single-sample ``TensorSource`` fixture.

        Returns:
            ``BroadcastSource`` wrapping the fixture.
        """
        return BroadcastSource(single_sample_src)

    def test_n_samples_always_one(self, broadcast_src: BroadcastSource) -> None:
        """n_samples is always 1, regardless of the inner source.

        Args:
            broadcast_src: ``BroadcastSource`` fixture.
        """
        assert broadcast_src.n_samples == 1

    def test_get_item_any_index_returns_sample_zero(self, broadcast_src: BroadcastSource) -> None:
        """get_item with any index returns the single inner sample.

        Args:
            broadcast_src: ``BroadcastSource`` fixture.
        """
        item_at_99 = broadcast_src.get_item(99)
        item_at_0 = broadcast_src.get_item(0)
        assert torch.equal(item_at_99, item_at_0)

    def test_get_item_shape(self, broadcast_src: BroadcastSource) -> None:
        """get_item returns a tensor of the inner sample shape.

        Args:
            broadcast_src: ``BroadcastSource`` fixture.
        """
        assert broadcast_src.get_item(5).shape == torch.Size((4,))

    def test_get_batch_shape(self, broadcast_src: BroadcastSource) -> None:
        """get_batch returns (B, *sample_shape) for any list of indices.

        Args:
            broadcast_src: ``BroadcastSource`` fixture.
        """
        batch = broadcast_src.get_batch([0, 1, 2])
        assert batch.shape == torch.Size((3, 4))

    def test_get_batch_all_rows_equal(self, broadcast_src: BroadcastSource) -> None:
        """All rows in the batch are identical (broadcast semantics).

        Args:
            broadcast_src: ``BroadcastSource`` fixture.
        """
        batch = broadcast_src.get_batch([0, 1, 2])
        assert torch.equal(batch[0], batch[1])
        assert torch.equal(batch[0], batch[2])

    def test_get_batch_large_request(self, broadcast_src: BroadcastSource) -> None:
        """get_batch with B=100 returns a (100, 4) tensor.

        Args:
            broadcast_src: ``BroadcastSource`` fixture.
        """
        batch = broadcast_src.get_batch(list(range(100)))
        assert batch.shape == torch.Size((100, 4))

    def test_satisfies_array_source_protocol(self, broadcast_src: BroadcastSource) -> None:
        """BroadcastSource satisfies the ArraySource runtime-checkable protocol.

        Args:
            broadcast_src: ``BroadcastSource`` fixture.
        """
        assert isinstance(broadcast_src, ArraySource)


# ═══════════════════════════════════════════════════════════════════════════
# RoleSourceMap
# ═══════════════════════════════════════════════════════════════════════════


class TestRoleSourceMap:
    """Unit tests for ``RoleSourceMap``."""

    def test_n_samples_from_single_feature(
        self, feature_tensor: torch.Tensor, n_samples: int
    ) -> None:
        """n_samples equals the feature source's sample count.

        Args:
            feature_tensor: In-memory feature tensor fixture.
            n_samples: Expected sample count fixture.
        """
        src = TensorSource(feature_tensor)
        rsm = RoleSourceMap(features=(("x", src),), targets=())
        assert rsm.n_samples == n_samples

    def test_n_samples_skips_broadcast_sources(
        self,
        feature_tensor: torch.Tensor,
        n_samples: int,
    ) -> None:
        """n_samples ignores BroadcastSource entries.

        Args:
            feature_tensor: In-memory feature tensor fixture.
            n_samples: Expected sample count fixture.
        """
        main_src = TensorSource(feature_tensor)
        bias_inner = TensorSource(torch.zeros(1, 4))
        bias_src = BroadcastSource(bias_inner)
        rsm = RoleSourceMap(
            features=(("x", main_src), ("bias", bias_src)),
            targets=(),
        )
        assert rsm.n_samples == n_samples

    def test_n_samples_raises_on_conflicting_sources(self) -> None:
        """n_samples raises ValueError when sources disagree on sample count."""
        src_a = TensorSource(torch.zeros(10, 4))
        src_b = TensorSource(torch.zeros(20, 4))
        rsm = RoleSourceMap(features=(("x", src_a),), targets=(("y", src_b),))
        with pytest.raises(ValueError, match="Conflicting n_samples"):
            _ = rsm.n_samples

    def test_n_samples_raises_when_only_broadcasts(self) -> None:
        """n_samples raises ValueError when all sources are BroadcastSource."""
        inner = TensorSource(torch.zeros(1, 4))
        broadcast = BroadcastSource(inner)
        rsm = RoleSourceMap(features=(("x", broadcast),), targets=())
        with pytest.raises(ValueError, match="no non-broadcast sources"):
            _ = rsm.n_samples

    def test_features_dict_roundtrip(self, feature_tensor: torch.Tensor) -> None:
        """features_dict returns a regular dict mirroring the features tuple.

        Args:
            feature_tensor: In-memory feature tensor fixture.
        """
        src = TensorSource(feature_tensor)
        rsm = RoleSourceMap(features=(("x", src),), targets=())
        d = rsm.features_dict()
        assert isinstance(d, dict)
        assert d["x"] is src

    def test_targets_dict_roundtrip(self, target_tensor: torch.Tensor) -> None:
        """targets_dict returns a regular dict mirroring the targets tuple.

        Args:
            target_tensor: In-memory target tensor fixture.
        """
        src = TensorSource(target_tensor)
        rsm = RoleSourceMap(features=(), targets=(("y", src),))
        d = rsm.targets_dict()
        assert isinstance(d, dict)
        assert d["y"] is src

    def test_is_frozen(self, feature_tensor: torch.Tensor) -> None:
        """RoleSourceMap is immutable — attribute assignment raises FrozenInstanceError.

        Args:
            feature_tensor: In-memory feature tensor fixture.
        """
        from dataclasses import FrozenInstanceError

        src = TensorSource(feature_tensor)
        rsm = RoleSourceMap(features=(("x", src),), targets=())
        with pytest.raises(FrozenInstanceError):
            rsm.features = ()  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# source_from_entry
# ═══════════════════════════════════════════════════════════════════════════


class TestSourceFromEntry:
    """Tests for the ``source_from_entry`` dispatch function."""

    def test_value_entry_returns_tensor_source(
        self,
        value_feature_entry: ValueEntry,
    ) -> None:
        """ValueEntry dispatches to a TensorSource.

        Args:
            value_feature_entry: ``ValueEntry`` fixture wrapping a feature tensor.
        """
        src = source_from_entry(value_feature_entry)
        assert isinstance(src, TensorSource)

    def test_value_entry_n_samples(
        self,
        value_feature_entry: ValueEntry,
        n_samples: int,
    ) -> None:
        """TensorSource derived from ValueEntry has the correct n_samples.

        Args:
            value_feature_entry: ``ValueEntry`` fixture wrapping a feature tensor.
            n_samples: Expected sample count fixture.
        """
        src = source_from_entry(value_feature_entry)
        assert src.n_samples == n_samples

    def test_npy_entry_returns_eager_file_source(
        self,
        npy_feature_entry: NpyEntry,
    ) -> None:
        """NpyEntry (path-based) dispatches to an EagerFileSource.

        Args:
            npy_feature_entry: ``NpyEntry`` fixture.
        """
        src = source_from_entry(npy_feature_entry)
        assert isinstance(src, EagerFileSource)

    def test_npy_entry_n_samples(
        self,
        npy_feature_entry: NpyEntry,
        n_samples: int,
    ) -> None:
        """EagerFileSource derived from NpyEntry has the correct n_samples.

        Args:
            npy_feature_entry: ``NpyEntry`` fixture.
            n_samples: Expected sample count fixture.
        """
        src = source_from_entry(npy_feature_entry)
        assert src.n_samples == n_samples

    def test_npz_entry_returns_eager_file_source(
        self,
        npz_feature_entry: NpzEntry,
    ) -> None:
        """NpzEntry (path-based) dispatches to an EagerFileSource.

        Args:
            npz_feature_entry: ``NpzEntry`` fixture.
        """
        src = source_from_entry(npz_feature_entry)
        assert isinstance(src, EagerFileSource)

    def test_value_entry_none_value_raises(self, none_value_entry: ValueEntry) -> None:
        """Placeholder ValueEntry (value=None) raises TypeError.

        Args:
            none_value_entry: ``ValueEntry`` fixture with ``value=None``.
        """
        with pytest.raises(TypeError, match="returned None"):
            source_from_entry(none_value_entry)

    def test_array_source_entry_returns_source_directly(
        self,
        zarr_stub_entry: Any,
        zarr_stub_source: Any,
    ) -> None:
        """Entry whose open_reader() returns an ArraySource is used directly (case 2).

        Args:
            zarr_stub_entry: Stub entry whose ``open_reader()`` returns an ``ArraySource``.
            zarr_stub_source: The ``ArraySource`` instance returned by ``open_reader()``.
        """
        src = source_from_entry(zarr_stub_entry)
        assert src is zarr_stub_source

    def test_all_sources_satisfy_protocol(
        self,
        value_feature_entry: ValueEntry,
        npy_feature_entry: NpyEntry,
    ) -> None:
        """All dispatched sources satisfy the ArraySource protocol.

        Args:
            value_feature_entry: ``ValueEntry`` fixture.
            npy_feature_entry: ``NpyEntry`` fixture.
        """
        for entry in (value_feature_entry, npy_feature_entry):
            src = source_from_entry(entry)
            assert isinstance(src, ArraySource)


# ═══════════════════════════════════════════════════════════════════════════
# build_role_source_map
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildRoleSourceMap:
    """End-to-end tests for ``build_role_source_map``."""

    def test_partitions_features_and_targets(
        self,
        value_feature_entry: ValueEntry,
        value_target_entry: ValueEntry,
    ) -> None:
        """Features and targets are correctly partitioned in the map.

        Args:
            value_feature_entry: Feature ``ValueEntry`` fixture.
            value_target_entry: Target ``ValueEntry`` fixture.
        """
        rsm = build_role_source_map([value_feature_entry, value_target_entry])
        assert len(rsm.features) == 1
        assert len(rsm.targets) == 1
        assert rsm.features[0][0] == value_feature_entry.name
        assert rsm.targets[0][0] == value_target_entry.name

    def test_n_samples_consistent(
        self,
        value_feature_entry: ValueEntry,
        value_target_entry: ValueEntry,
        n_samples: int,
    ) -> None:
        """build_role_source_map produces a map with the correct n_samples.

        Args:
            value_feature_entry: Feature ``ValueEntry`` fixture.
            value_target_entry: Target ``ValueEntry`` fixture.
            n_samples: Expected sample count fixture.
        """
        rsm = build_role_source_map([value_feature_entry, value_target_entry])
        assert rsm.n_samples == n_samples

    def test_singleton_source_wrapped_in_broadcast(
        self,
        value_feature_entry: ValueEntry,
        value_target_entry: ValueEntry,
        single_sample_value_entry: ValueEntry,
    ) -> None:
        """A 1-sample source is automatically wrapped in BroadcastSource.

        Args:
            value_feature_entry: Multi-sample feature ``ValueEntry`` fixture.
            value_target_entry: Multi-sample target ``ValueEntry`` fixture.
            single_sample_value_entry: Singleton feature ``ValueEntry`` fixture.
        """
        rsm = build_role_source_map(
            [value_feature_entry, single_sample_value_entry, value_target_entry]
        )
        sources_dict = rsm.features_dict()
        assert isinstance(sources_dict[single_sample_value_entry.name], BroadcastSource)
        assert not isinstance(sources_dict[value_feature_entry.name], BroadcastSource)

    def test_conflicting_n_samples_raises(
        self,
        conflicting_feature_entry: ValueEntry,
        conflicting_target_entry: ValueEntry,
    ) -> None:
        """Entries with conflicting n_samples raise ValueError.

        Args:
            conflicting_feature_entry: Feature entry with canonical sample count.
            conflicting_target_entry: Target entry with a mismatched sample count.
        """
        with pytest.raises(ValueError, match="Conflicting n_samples"):
            build_role_source_map([conflicting_feature_entry, conflicting_target_entry])

    def test_npy_entry_integration(
        self,
        npy_feature_entry: NpyEntry,
        value_target_entry: ValueEntry,
        n_samples: int,
    ) -> None:
        """build_role_source_map works end-to-end with real NpyEntry + ValueEntry.

        Args:
            npy_feature_entry: ``NpyEntry`` fixture backed by a ``.npy`` file.
            value_target_entry: Target ``ValueEntry`` fixture.
            n_samples: Expected sample count fixture.
        """
        rsm = build_role_source_map([npy_feature_entry, value_target_entry])
        assert rsm.n_samples == n_samples
        assert rsm.features[0][0] == npy_feature_entry.name
        assert rsm.targets[0][0] == value_target_entry.name

    def test_result_is_frozen(
        self,
        value_feature_entry: ValueEntry,
    ) -> None:
        """build_role_source_map returns a frozen RoleSourceMap.

        Args:
            value_feature_entry: Feature ``ValueEntry`` fixture.
        """
        from dataclasses import FrozenInstanceError

        rsm = build_role_source_map([value_feature_entry])
        with pytest.raises(FrozenInstanceError):
            rsm.features = ()  # type: ignore[misc]

    def test_empty_entries_produces_empty_map(self) -> None:
        """An empty entry list produces a RoleSourceMap with no sources."""
        rsm = build_role_source_map([])
        assert rsm.features == ()
        assert rsm.targets == ()
