"""Tests for PrecisionContext."""

import pytest
import threading
from concurrent.futures import ThreadPoolExecutor

from dlkit.tools.config.precision import PrecisionStrategy
from dlkit.interfaces.api.domain.precision import (
    PrecisionContext,
    get_global_precision_context,
    current_precision_override,
    precision_override,
)


class TestPrecisionContext:
    """Test suite for PrecisionContext."""

    @pytest.fixture
    def context(self):
        """Create a fresh PrecisionContext for testing."""
        context = PrecisionContext()
        # Clear any existing override to start fresh
        context.clear_override()
        return context

    def test_initial_state(self, context):
        """Test initial context state."""
        assert context.get_override() is None
        assert not context.has_override()

    def test_set_and_get_override(self, context):
        """Test setting and getting precision override."""
        context.set_override(PrecisionStrategy.MIXED_16)

        assert context.get_override() == PrecisionStrategy.MIXED_16
        assert context.has_override()

    def test_clear_override(self, context):
        """Test clearing precision override."""
        context.set_override(PrecisionStrategy.MIXED_16)
        context.clear_override()

        assert context.get_override() is None
        assert not context.has_override()

    def test_resolve_precision_with_override(self, context):
        """Test precision resolution with override."""
        default = PrecisionStrategy.FULL_32
        override = PrecisionStrategy.TRUE_16

        # Without override, should return default
        assert context.resolve_precision(default) == default

        # With override, should return override
        context.set_override(override)
        assert context.resolve_precision(default) == override

    def test_context_manager(self, context):
        """Test precision override context manager."""
        default = PrecisionStrategy.FULL_32
        override = PrecisionStrategy.MIXED_16

        # Before context manager
        assert context.resolve_precision(default) == default

        with context.precision_override(override):
            # Inside context manager
            assert context.resolve_precision(default) == override

        # After context manager
        assert context.resolve_precision(default) == default

    def test_nested_context_managers(self, context):
        """Test nested precision override context managers."""
        default = PrecisionStrategy.FULL_32
        first_override = PrecisionStrategy.MIXED_16
        second_override = PrecisionStrategy.TRUE_BF16

        with context.precision_override(first_override):
            assert context.resolve_precision(default) == first_override

            with context.precision_override(second_override):
                assert context.resolve_precision(default) == second_override

            # Should restore first override
            assert context.resolve_precision(default) == first_override

        # Should restore to no override
        assert context.resolve_precision(default) == default

    def test_class_level_context_manager(self):
        """Test class-level precision override context manager."""
        override = PrecisionStrategy.MIXED_BF16
        default = PrecisionStrategy.FULL_32

        with PrecisionContext.override(override) as ctx:
            assert ctx.get_override() == override
            assert ctx.resolve_precision(default) == override

    def test_thread_isolation(self):
        """Test that precision context is isolated between threads."""
        results = {}
        barrier = threading.Barrier(2)

        def worker(thread_id, precision):
            context = PrecisionContext()
            context.set_override(precision)

            # Wait for both threads to set their precision
            barrier.wait()

            # Check that each thread sees its own precision
            results[thread_id] = context.get_override()

        thread1_precision = PrecisionStrategy.MIXED_16
        thread2_precision = PrecisionStrategy.TRUE_BF16

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(worker, 1, thread1_precision),
                executor.submit(worker, 2, thread2_precision),
            ]
            for future in futures:
                future.result()

        # Each thread should see its own precision
        assert results[1] == thread1_precision
        assert results[2] == thread2_precision

    def test_global_context_functions(self):
        """Test global context convenience functions."""
        # Test global context access
        global_context = get_global_precision_context()
        assert isinstance(global_context, PrecisionContext)

        # Clear any existing override from other tests
        global_context.clear_override()

        # Test current override function
        assert current_precision_override() is None

        global_context.set_override(PrecisionStrategy.MIXED_16)
        assert current_precision_override() == PrecisionStrategy.MIXED_16

        global_context.clear_override()
        assert current_precision_override() is None

        # Cleanup for other tests
        global_context.clear_override()

    def test_global_precision_override_context_manager(self):
        """Test global precision override context manager."""
        override = PrecisionStrategy.TRUE_16

        # Clear any existing override from other tests
        get_global_precision_context().clear_override()

        # Before override
        assert current_precision_override() is None

        with precision_override(override):
            # Inside override
            assert current_precision_override() == override

        # After override
        assert current_precision_override() is None

        # Cleanup for other tests
        get_global_precision_context().clear_override()

    def test_string_representation(self, context):
        """Test string representation."""
        # Without override
        repr_str = repr(context)
        assert "PrecisionContext(no_override)" == repr_str

        # With override
        context.set_override(PrecisionStrategy.MIXED_16)
        repr_str = repr(context)
        assert "PrecisionContext(override=MIXED_16)" == repr_str

    def test_thread_local_independence(self):
        """Test that thread-local storage is truly independent."""
        main_context = PrecisionContext()
        main_context.set_override(PrecisionStrategy.FULL_32)

        def worker():
            worker_context = PrecisionContext()
            # Worker should not see main thread's override
            assert worker_context.get_override() is None

            worker_context.set_override(PrecisionStrategy.MIXED_16)
            return worker_context.get_override()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker)
            worker_override = future.result()

        # Worker set its own override
        assert worker_override == PrecisionStrategy.MIXED_16

        # Main thread should still have its override
        assert main_context.get_override() == PrecisionStrategy.FULL_32
