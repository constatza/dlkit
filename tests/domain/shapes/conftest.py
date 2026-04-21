"""Shared fixtures for domain/shapes test suite.

Provides composable fixtures for ShapeEntry, ShapeData, and related test data
used across test_strategies.py and other shape test modules.
"""

from __future__ import annotations

import pytest

from dlkit.domain.shapes.value_objects import ModelFamily, ShapeData, ShapeEntry, ShapeSource

# ---------------------------------------------------------------------------
# Named constants for dimension tuples
# ---------------------------------------------------------------------------

DIMS_X: tuple[int, ...] = (10, 5)
DIMS_Y: tuple[int, ...] = (5,)
DIMS_FEATURES: tuple[int, ...] = (20, 8)
DIMS_FIRST: tuple[int, ...] = (32, 4)
DIMS_SECOND: tuple[int, ...] = (4,)


# ---------------------------------------------------------------------------
# ShapeEntry fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def entry_x() -> ShapeEntry:
    """A ShapeEntry named 'x' with dimensions (10, 5).

    Returns:
        ShapeEntry: Input shape entry.
    """
    return ShapeEntry(name="x", dimensions=DIMS_X)


@pytest.fixture
def entry_y() -> ShapeEntry:
    """A ShapeEntry named 'y' with dimensions (5,).

    Returns:
        ShapeEntry: Output shape entry.
    """
    return ShapeEntry(name="y", dimensions=DIMS_Y)


@pytest.fixture
def entry_features() -> ShapeEntry:
    """A ShapeEntry named 'features' with dimensions (20, 8).

    Returns:
        ShapeEntry: Single generic feature entry.
    """
    return ShapeEntry(name="features", dimensions=DIMS_FEATURES)


@pytest.fixture
def entry_first() -> ShapeEntry:
    """A ShapeEntry named 'first' with dimensions (32, 4).

    Returns:
        ShapeEntry: First of two unnamed entries.
    """
    return ShapeEntry(name="first", dimensions=DIMS_FIRST)


@pytest.fixture
def entry_second() -> ShapeEntry:
    """A ShapeEntry named 'second' with dimensions (4,).

    Returns:
        ShapeEntry: Second of two unnamed entries.
    """
    return ShapeEntry(name="second", dimensions=DIMS_SECOND)


# ---------------------------------------------------------------------------
# ShapeData fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data(entry_x: ShapeEntry, entry_y: ShapeEntry) -> ShapeData:
    """ShapeData with x=(10,5) and y=(5,) entries, DLKIT_NN family.

    Args:
        entry_x: Input shape entry fixture.
        entry_y: Output shape entry fixture.

    Returns:
        ShapeData: Standard two-entry shape collection.
    """
    return ShapeData(
        entries={"x": entry_x, "y": entry_y},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.TRAINING_DATASET,
    )


@pytest.fixture
def single_entry_data(entry_features: ShapeEntry) -> ShapeData:
    """ShapeData with a single entry named 'features', DLKIT_NN family.

    Args:
        entry_features: Feature shape entry fixture.

    Returns:
        ShapeData: Single-entry shape collection.
    """
    return ShapeData(
        entries={"features": entry_features},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.TRAINING_DATASET,
    )


@pytest.fixture
def two_entry_data(entry_first: ShapeEntry, entry_second: ShapeEntry) -> ShapeData:
    """ShapeData with two entries named 'first'/'second', DLKIT_NN family.

    Args:
        entry_first: First shape entry fixture.
        entry_second: Second shape entry fixture.

    Returns:
        ShapeData: Two-entry collection without x/y aliases.
    """
    return ShapeData(
        entries={"first": entry_first, "second": entry_second},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.TRAINING_DATASET,
    )


@pytest.fixture
def empty_data() -> ShapeData:
    """ShapeData with no entries, DLKIT_NN family.

    Returns:
        ShapeData: Empty shape collection.
    """
    return ShapeData(
        entries={},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.DEFAULT_FALLBACK,
    )


@pytest.fixture
def data_with_defaults(entry_x: ShapeEntry, entry_y: ShapeEntry) -> ShapeData:
    """ShapeData with explicit default_input and default_output set.

    Args:
        entry_x: Input shape entry fixture.
        entry_y: Output shape entry fixture.

    Returns:
        ShapeData: Shape collection with explicit defaults.
    """
    return ShapeData(
        entries={"x": entry_x, "y": entry_y},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.TRAINING_DATASET,
        default_input="x",
        default_output="y",
    )


@pytest.fixture
def external_empty_data() -> ShapeData:
    """ShapeData with no entries and EXTERNAL family.

    Returns:
        ShapeData: Empty external-family shape collection.
    """
    return ShapeData(
        entries={},
        model_family=ModelFamily.EXTERNAL,
        source=ShapeSource.DEFAULT_FALLBACK,
    )
