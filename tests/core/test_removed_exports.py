from __future__ import annotations

import pytest


def test_batch_no_longer_exported_from_core_datatypes() -> None:
    from dlkit.core import datatypes

    assert not hasattr(datatypes, "Batch")


@pytest.mark.parametrize(
    "symbol",
    (
        "ShapeCache",
        "LRUShapeCache",
        "CachingShapeInferencer",
        "BatchShapeProcessor",
        "CacheStats",
        "PerformanceMonitor",
        "timed_operation",
    ),
)
def test_shape_specs_performance_symbols_are_not_exported(symbol: str) -> None:
    from dlkit.core import shape_specs

    assert not hasattr(shape_specs, symbol)
