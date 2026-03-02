"""Performance optimization components for shape handling.

This module provides caching, batch processing, and other performance
optimizations for the shape handling system.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from threading import RLock

from .value_objects import ShapeData, ShapeEntry, ModelFamily, ShapeSource
from .core import IShapeSpec, create_shape_spec
from .inference import InferenceContext, ShapeInferenceChain
from .strategies import ValidationResult, ShapeValidator, ShapeSerializer


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100.0

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate as percentage."""
        return 100.0 - self.hit_rate


class ShapeCache(ABC):
    """Abstract base class for shape caching implementations."""

    @abstractmethod
    def get(self, key: str) -> Optional[ShapeData]:
        """Get cached shape data by key.

        Args:
            key: Cache key

        Returns:
            Cached ShapeData or None if not found
        """
        ...

    @abstractmethod
    def put(self, key: str, shape_data: ShapeData) -> None:
        """Store shape data in cache.

        Args:
            key: Cache key
            shape_data: Shape data to cache
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached data."""
        ...

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache performance statistics.

        Returns:
            CacheStats object with performance metrics
        """
        ...


@dataclass
class CacheEntry:
    """Internal cache entry with metadata."""

    data: ShapeData
    timestamp: float
    access_count: int = 0

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp == 0:
            self.timestamp = time.time()


class LRUShapeCache(ShapeCache):
    """Least Recently Used (LRU) cache implementation for shape data.

    This cache implementation evicts least recently used entries when
    the cache reaches its maximum capacity.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._stats = CacheStats()
        self._lock = RLock()

    def get(self, key: str) -> Optional[ShapeData]:
        """Get cached shape data by key.

        Args:
            key: Cache key

        Returns:
            Cached ShapeData or None if not found or expired
        """
        with self._lock:
            self._stats.total_requests += 1

            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            # Check if entry has expired
            if time.time() - entry.timestamp > self._ttl_seconds:
                self._remove_entry(key)
                self._stats.misses += 1
                return None

            # Update access order and count
            self._update_access(key)
            entry.access_count += 1
            self._stats.hits += 1

            return entry.data

    def put(self, key: str, shape_data: ShapeData) -> None:
        """Store shape data in cache.

        Args:
            key: Cache key
            shape_data: Shape data to cache
        """
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Evict entries if cache is full
            while len(self._cache) >= self._max_size:
                self._evict_lru()

            # Add new entry
            entry = CacheEntry(data=shape_data, timestamp=time.time())
            self._cache[key] = entry
            self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                total_requests=self._stats.total_requests,
            )

    def _update_access(self, key: str) -> None:
        """Update access order for LRU tracking."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and access order."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order[0]
            self._remove_entry(lru_key)
            self._stats.evictions += 1


class CachingShapeInferencer:
    """Shape inferencer with caching capability.

    This wrapper adds caching to any shape inference chain to avoid
    repeated expensive computations.
    """

    def __init__(
        self,
        base_chain: ShapeInferenceChain,
        cache: Optional[ShapeCache] = None,
        cache_ttl: float = 3600,
    ):
        """Initialize caching inferencer.

        Args:
            base_chain: Base inference chain to wrap
            cache: Optional cache implementation (creates LRU if None)
            cache_ttl: Cache time-to-live in seconds
        """
        self._base_chain = base_chain
        self._cache = cache or LRUShapeCache(max_size=1000, ttl_seconds=cache_ttl)

    def infer_shape_spec(self, context: InferenceContext) -> IShapeSpec:
        """Infer shape spec with caching.

        Args:
            context: Inference context

        Returns:
            IShapeSpec with caching applied
        """
        cache_key = self._generate_cache_key(context)

        # Try cache first
        cached_data = self._cache.get(cache_key)
        if cached_data is not None:
            # Create shape spec from cached data
            shapes = {name: entry.dimensions for name, entry in cached_data.entries.items()}
            return create_shape_spec(
                shapes=shapes, model_family=cached_data.model_family, source=cached_data.source
            )

        # Cache miss - use base chain
        shape_spec = self._base_chain.infer_shape_spec(context)

        # Cache the result if it's not empty
        if not shape_spec.is_empty():
            # Extract shape data for caching
            entries = {
                name: ShapeEntry(name=name, dimensions=shape)
                for name, shape in shape_spec.get_all_shapes().items()
            }

            model_family = ModelFamily(shape_spec.model_family())
            source = ShapeSource.TRAINING_DATASET  # Default source for cached items

            cached_data = ShapeData(entries=entries, model_family=model_family, source=source)

            self._cache.put(cache_key, cached_data)

        return shape_spec

    def get_cache_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        return self._cache.get_stats()

    def clear_cache(self) -> None:
        """Clear the shape cache."""
        self._cache.clear()

    def _generate_cache_key(self, context: InferenceContext) -> str:
        """Generate cache key from inference context.

        Args:
            context: Inference context

        Returns:
            Cache key string
        """
        # Create a hash from relevant context components
        key_components = []

        # Dataset hash (if available and reasonable size)
        if context.dataset is not None:
            try:
                # Try to get dataset identifier or hash small datasets
                if hasattr(context.dataset, "__len__") and len(context.dataset) < 10000:
                    # Small dataset - can hash first few samples
                    sample_data = str(context.dataset[0] if len(context.dataset) > 0 else "")
                    key_components.append(
                        f"dataset:{hashlib.md5(sample_data.encode()).hexdigest()[:8]}"
                    )
                elif hasattr(context.dataset, "name"):
                    key_components.append(f"dataset:{context.dataset.name}")
                else:
                    key_components.append(f"dataset:{id(context.dataset)}")
            except Exception:
                key_components.append("dataset:unknown")

        # Checkpoint path
        if context.checkpoint_path is not None:
            key_components.append(f"checkpoint:{context.checkpoint_path}")

        # Model settings hash
        if context.model_settings is not None:
            try:
                settings_str = str(context.model_settings)
                settings_hash = hashlib.md5(settings_str.encode()).hexdigest()[:8]
                key_components.append(f"settings:{settings_hash}")
            except Exception:
                key_components.append(f"settings:{id(context.model_settings)}")

        # Entry configs hash
        if context.entry_configs is not None:
            try:
                configs_str = str(context.entry_configs)
                configs_hash = hashlib.md5(configs_str.encode()).hexdigest()[:8]
                key_components.append(f"configs:{configs_hash}")
            except Exception:
                key_components.append("configs:unknown")

        # Model family
        if context.model_family is not None:
            key_components.append(f"family:{context.model_family.value}")

        # Combine all components
        cache_key = "|".join(key_components)
        return hashlib.md5(cache_key.encode()).hexdigest()


class BatchShapeProcessor:
    """Processor for batch shape operations to improve performance."""

    def __init__(
        self,
        validator: Optional[ShapeValidator] = None,
        serializer: Optional[ShapeSerializer] = None,
    ):
        """Initialize batch processor.

        Args:
            validator: Optional shape validator
            serializer: Optional shape serializer
        """
        self._validator = validator or ShapeValidator()
        self._serializer = serializer or ShapeSerializer()

    def validate_batch(self, shape_data_list: List[ShapeData]) -> List[ValidationResult]:
        """Validate multiple shape data objects efficiently.

        Args:
            shape_data_list: List of shape data to validate

        Returns:
            List of validation results in same order as input
        """
        results = []

        # Group by model family for optimized validation
        family_groups = {}
        for i, shape_data in enumerate(shape_data_list):
            family = shape_data.model_family
            if family not in family_groups:
                family_groups[family] = []
            family_groups[family].append((i, shape_data))

        # Process each family group
        indexed_results = {}
        for family, items in family_groups.items():
            for index, shape_data in items:
                # Use cached validation engine for same family
                result = self._validator.validate_collection(shape_data)
                indexed_results[index] = result

        # Reconstruct results in original order
        for i in range(len(shape_data_list)):
            results.append(indexed_results[i])

        return results

    def serialize_batch(self, shape_data_list: List[ShapeData]) -> List[Dict[str, Any]]:
        """Serialize multiple shape data objects efficiently.

        Args:
            shape_data_list: List of shape data to serialize

        Returns:
            List of serialized dictionaries in same order as input
        """
        return [self._serializer.serialize(shape_data) for shape_data in shape_data_list]

    def deserialize_batch(self, serialized_list: List[Dict[str, Any]]) -> List[ShapeData]:
        """Deserialize multiple shape data objects efficiently.

        Args:
            serialized_list: List of serialized shape data

        Returns:
            List of ShapeData objects in same order as input

        Raises:
            ValueError: If any deserialization fails
        """
        results = []
        for i, serialized_data in enumerate(serialized_list):
            try:
                shape_data = self._serializer.deserialize(serialized_data)
                results.append(shape_data)
            except Exception as e:
                raise ValueError(f"Failed to deserialize item {i}: {e}") from e
        return results

    def process_validation_batch(
        self, shape_data_list: List[ShapeData], fail_fast: bool = False
    ) -> Tuple[List[ValidationResult], List[int]]:
        """Process validation batch with optional fail-fast behavior.

        Args:
            shape_data_list: List of shape data to validate
            fail_fast: If True, stop on first validation failure

        Returns:
            Tuple of (validation_results, failed_indices)
        """
        results = []
        failed_indices = []

        for i, shape_data in enumerate(shape_data_list):
            result = self._validator.validate_collection(shape_data)
            results.append(result)

            if not result.is_valid:
                failed_indices.append(i)
                if fail_fast:
                    break

        return results, failed_indices


class PerformanceMonitor:
    """Monitor performance metrics for shape operations."""

    def __init__(self):
        """Initialize performance monitor."""
        self._metrics: Dict[str, List[float]] = {}
        self._lock = RLock()

    def record_operation(self, operation: str, duration: float) -> None:
        """Record operation duration.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
        """
        with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = []
            self._metrics[operation].append(duration)

    def get_statistics(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation.

        Args:
            operation: Operation name

        Returns:
            Dictionary with min, max, avg, total times
        """
        with self._lock:
            if operation not in self._metrics or not self._metrics[operation]:
                return {}

            durations = self._metrics[operation]
            return {
                "count": len(durations),
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "total": sum(durations),
            }

    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations.

        Returns:
            Dictionary mapping operation names to their statistics
        """
        with self._lock:
            return {op: self.get_statistics(op) for op in self._metrics.keys()}

    def clear_metrics(self) -> None:
        """Clear all recorded metrics."""
        with self._lock:
            self._metrics.clear()


class TimedOperation:
    """Context manager for timing operations."""

    def __init__(self, monitor: PerformanceMonitor, operation: str):
        """Initialize timed operation.

        Args:
            monitor: Performance monitor to record to
            operation: Operation name
        """
        self._monitor = monitor
        self._operation = operation
        self._start_time: float = 0

    def __enter__(self):
        """Start timing."""
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and record result."""
        duration = time.time() - self._start_time
        self._monitor.record_operation(self._operation, duration)


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


def timed_operation(operation: str) -> TimedOperation:
    """Create a timed operation context manager.

    Args:
        operation: Operation name

    Returns:
        TimedOperation context manager
    """
    return TimedOperation(_performance_monitor, operation)
