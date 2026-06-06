"""Tests for dlkit.engine.data.geometry.infer_geometry."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from dlkit.common.geometry import FieldRole, GeometryKind, GeometrySpec, TopologyKind
from dlkit.engine.data.geometry import infer_geometry
from dlkit.infrastructure.config.data_entries import DataEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    name: str,
    role: FieldRole = FieldRole.FEATURE,
    geometry_kind: GeometryKind = GeometryKind.TABULAR,
    transforms: list | None = None,
) -> DataEntry:
    """Create a minimal DataEntry stub using MagicMock.

    Args:
        name: Entry name.
        role: Physics-domain role.
        geometry_kind: Spatial structure kind.
        transforms: Optional transform settings list.

    Returns:
        MagicMock configured to look like a DataEntry.
    """
    entry = MagicMock(spec=DataEntry)
    entry.name = name
    entry.field_role = role
    entry.geometry_kind = geometry_kind
    entry.transforms = transforms or []
    return entry


def _make_dataset(feature_dict: dict[str, torch.Tensor]) -> MagicMock:
    """Create a mock dataset whose __getitem__(0) returns a nested TensorDict.

    Args:
        feature_dict: Mapping from feature name to sample tensor (no batch dim).

    Returns:
        MagicMock with __getitem__ returning the expected TensorDict structure.
    """
    features_td = TensorDict(cast("Any", feature_dict), batch_size=[])
    sample = TensorDict({"features": features_td}, batch_size=[])
    dataset = MagicMock()
    dataset.__getitem__ = MagicMock(return_value=sample)
    return dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tabular_entry() -> DataEntry:
    """Single TABULAR FEATURE entry with shape (8,).

    Returns:
        Mock DataEntry for tabular feature.
    """
    return _make_entry("x", FieldRole.FEATURE, GeometryKind.TABULAR)


@pytest.fixture
def tabular_dataset() -> MagicMock:
    """Dataset returning a single (8,) tabular feature tensor.

    Returns:
        Mock dataset with tabular feature sample.
    """
    return _make_dataset({"x": torch.zeros(8)})


@pytest.fixture
def graph_entry() -> DataEntry:
    """Single GRAPH FEATURE entry with shape (16,).

    Returns:
        Mock DataEntry for graph node features.
    """
    return _make_entry("node_feats", FieldRole.FEATURE, GeometryKind.GRAPH)


@pytest.fixture
def graph_dataset() -> MagicMock:
    """Dataset returning a (16,) node feature tensor.

    Returns:
        Mock dataset with graph feature sample.
    """
    return _make_dataset({"node_feats": torch.zeros(16)})


@pytest.fixture
def multi_entry_tabular() -> tuple[tuple[DataEntry, ...], MagicMock]:
    """Two tabular FEATURE entries with shapes (8,) and (4,).

    Returns:
        Tuple of (feature_entries, dataset).
    """
    entries = (
        _make_entry("a", FieldRole.FEATURE, GeometryKind.TABULAR),
        _make_entry("b", FieldRole.FEATURE, GeometryKind.TABULAR),
    )
    dataset = _make_dataset({"a": torch.zeros(8), "b": torch.zeros(4)})
    return entries, dataset


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInferGeometryBasic:
    """Basic shape and field spec correctness."""

    @staticmethod
    def _mutate_attr(obj: object, name: str, value: object) -> None:
        setattr(obj, name, value)

    def test_single_tabular_field_shape(
        self, tabular_entry: DataEntry, tabular_dataset: MagicMock
    ) -> None:
        """FieldSpec shape matches the raw tensor shape.

        Args:
            tabular_entry: Single tabular FEATURE entry fixture.
            tabular_dataset: Dataset returning (8,) tensor fixture.
        """
        result = infer_geometry((tabular_entry,), tabular_dataset)

        assert len(result.fields) == 1
        assert result.fields[0].shape == (8,)

    def test_field_name_matches_entry(
        self, tabular_entry: DataEntry, tabular_dataset: MagicMock
    ) -> None:
        """FieldSpec name is taken from the entry's name attribute.

        Args:
            tabular_entry: Single tabular FEATURE entry fixture.
            tabular_dataset: Dataset returning (8,) tensor fixture.
        """
        result = infer_geometry((tabular_entry,), tabular_dataset)

        assert result.fields[0].name == tabular_entry.name

    def test_field_role_matches_entry(
        self, tabular_entry: DataEntry, tabular_dataset: MagicMock
    ) -> None:
        """FieldSpec role equals the entry's field_role.

        Args:
            tabular_entry: Single tabular FEATURE entry fixture.
            tabular_dataset: Dataset returning (8,) tensor fixture.
        """
        result = infer_geometry((tabular_entry,), tabular_dataset)

        assert result.fields[0].role == FieldRole.FEATURE

    def test_field_geometry_kind_matches_entry(
        self, tabular_entry: DataEntry, tabular_dataset: MagicMock
    ) -> None:
        """FieldSpec geometry_kind equals the entry's geometry_kind.

        Args:
            tabular_entry: Single tabular FEATURE entry fixture.
            tabular_dataset: Dataset returning (8,) tensor fixture.
        """
        result = infer_geometry((tabular_entry,), tabular_dataset)

        assert result.fields[0].geometry_kind == GeometryKind.TABULAR

    def test_multiple_fields_in_order(
        self, multi_entry_tabular: tuple[tuple[DataEntry, ...], MagicMock]
    ) -> None:
        """Multiple FieldSpecs are built in entry_configs order.

        Args:
            multi_entry_tabular: Two-entry (entries, dataset) fixture.
        """
        entries, dataset = multi_entry_tabular
        result = infer_geometry(entries, dataset)

        assert len(result.fields) == 2
        assert result.fields[0].name == "a"
        assert result.fields[0].shape == (8,)
        assert result.fields[1].name == "b"
        assert result.fields[1].shape == (4,)

    def test_returns_frozen_geometry_spec(
        self, tabular_entry: DataEntry, tabular_dataset: MagicMock
    ) -> None:
        """infer_geometry returns a frozen GeometrySpec dataclass.

        Args:
            tabular_entry: Single tabular FEATURE entry fixture.
            tabular_dataset: Dataset returning (8,) tensor fixture.
        """
        result = infer_geometry((tabular_entry,), tabular_dataset)

        assert isinstance(result, GeometrySpec)
        with pytest.raises((TypeError, AttributeError)):
            self._mutate_attr(result, "fields", ())


class TestTopologyKindInference:
    """Tests for topology_kind detection."""

    def test_tabular_entries_have_no_topology(
        self, tabular_entry: DataEntry, tabular_dataset: MagicMock
    ) -> None:
        """Non-graph entries produce topology_kind = None.

        Args:
            tabular_entry: Single tabular FEATURE entry fixture.
            tabular_dataset: Dataset returning (8,) tensor fixture.
        """
        result = infer_geometry((tabular_entry,), tabular_dataset)

        assert result.topology_kind is None

    def test_graph_entry_sets_edge_index_topology(
        self, graph_entry: DataEntry, graph_dataset: MagicMock
    ) -> None:
        """A GRAPH entry triggers TopologyKind.EDGE_INDEX.

        Args:
            graph_entry: GRAPH FEATURE entry fixture.
            graph_dataset: Dataset returning graph node tensor fixture.
        """
        result = infer_geometry((graph_entry,), graph_dataset)

        assert result.topology_kind == TopologyKind.EDGE_INDEX

    def test_graph_entry_edge_feature_dim_is_none(
        self, graph_entry: DataEntry, graph_dataset: MagicMock
    ) -> None:
        """edge_feature_dim is always None (graph edge features extracted by PyG).

        Args:
            graph_entry: GRAPH FEATURE entry fixture.
            graph_dataset: Dataset returning graph node tensor fixture.
        """
        result = infer_geometry((graph_entry,), graph_dataset)

        assert result.edge_feature_dim is None


class TestTransformPropagation:
    """Tests that transform chains are propagated through shape inference."""

    def test_shape_preserving_transform_leaves_shape_unchanged(self) -> None:
        """An identity (shape-preserving) transform does not change the shape."""

        class _DummyPreserving:
            def infer_output_shape(self, in_shape: tuple[int, ...]) -> tuple[int, ...]:
                return in_shape

        transform_settings = MagicMock()
        transform_settings.name = _DummyPreserving
        transform_settings.module_path = None
        transform_settings.model_dump.return_value = {}

        entry = _make_entry(
            "z", FieldRole.FEATURE, GeometryKind.TABULAR, transforms=[transform_settings]
        )
        dataset = _make_dataset({"z": torch.zeros(12)})

        result = infer_geometry((entry,), dataset)

        assert result.fields[0].shape == (12,)

    def test_transform_without_infer_output_shape_raises_value_error(self) -> None:
        """A transform without ``infer_output_shape()`` raises ValueError."""

        class _NoShapeInference:
            pass

        transform_settings = MagicMock()
        transform_settings.name = _NoShapeInference
        transform_settings.module_path = None
        transform_settings.model_dump.return_value = {}

        entry = _make_entry(
            "z", FieldRole.FEATURE, GeometryKind.TABULAR, transforms=[transform_settings]
        )
        dataset = _make_dataset({"z": torch.zeros(12)})

        with pytest.raises(ValueError, match="infer_output_shape"):
            infer_geometry((entry,), dataset)


class TestInferGeometryErrors:
    """Error path coverage."""

    def test_non_tensordict_dataset_raises_value_error(self) -> None:
        """A dataset returning a plain tensor raises ValueError.

        Tests that the TensorDict type check fires on non-TensorDict samples.
        """
        entry = _make_entry("x", FieldRole.FEATURE, GeometryKind.TABULAR)

        dataset = MagicMock()
        dataset.__getitem__ = MagicMock(return_value=torch.zeros(8))

        with pytest.raises(ValueError, match="nested TensorDict"):
            infer_geometry((entry,), dataset)

    def test_mismatched_entry_count_raises_value_error(self) -> None:
        """More entries than features keys raises ValueError.

        Tests that positional alignment is enforced.
        """
        entries = (
            _make_entry("a", FieldRole.FEATURE),
            _make_entry("b", FieldRole.FEATURE),
        )
        # Only one key in the dataset
        dataset = _make_dataset({"a": torch.zeros(8)})

        with pytest.raises(ValueError, match="entries has 2 entries"):
            infer_geometry(entries, dataset)
