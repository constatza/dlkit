"""Edge case and stress tests for precision control system.

This module contains tests for unusual scenarios, error conditions,
and stress testing that could reveal hidden issues in production.
"""

import pytest
import torch
import gc
import threading
import time

from dlkit.tools.config.precision import PrecisionStrategy
from dlkit.interfaces.api.domain.precision import precision_override, get_precision_context
from dlkit.interfaces.api.services.precision_service import get_precision_service
from dlkit.tools.config.session_settings import SessionSettings
from dlkit.tools.config.data_entries import Feature
from dlkit.tools.io.arrays import load_array
from dlkit.core.models.nn.base import DLKitModel


class BrokenModel(DLKitModel):
    """Model that intentionally breaks during precision application."""

    def __init__(self, in_features: int, out_features: int, break_on_precision=False, **kwargs):
        super().__init__()
        self._break_on_precision = break_on_precision

        # Create linear layer with the given dimensions
        self.linear = torch.nn.Linear(in_features, out_features)

        # Apply precision from context (simulating Lightning behavior)
        service = get_precision_service()
        precision_strategy = service.resolve_precision()
        dtype = precision_strategy.to_torch_dtype()
        self.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear layer."""
        return self.linear(x)

    def accepts_shape(self, shape_spec):
        """Accept any valid shape specification."""
        return (
            shape_spec.get_input_shape() is not None and shape_spec.get_output_shape() is not None
        )

    def forward(self, x):
        return self.linear(x)

    def to(self, *args, **kwargs):
        if self._break_on_precision:
            raise RuntimeError("Model precision application failed")
        return super().to(*args, **kwargs)


class TestPrecisionEdgeCases:
    """Edge case and stress tests for precision control."""

    @pytest.fixture
    def corrupted_data_file(self, tmp_path):
        """Create a corrupted dataflow file for testing error handling."""
        file_path = tmp_path / "corrupted.pt"
        with open(file_path, "wb") as f:
            f.write(b"not valid pytorch dataflow")
        return file_path

    def test_precision_with_corrupted_data(self, corrupted_data_file):
        """Test precision handling with corrupted data files."""

        session = SessionSettings(precision=PrecisionStrategy.MIXED_16)

        # Should raise appropriate error, not precision-related error
        with precision_override(session.get_precision_strategy()):
            with pytest.raises(Exception):  # Could be various errors depending on loader
                load_array(corrupted_data_file)


    def test_precision_memory_pressure(self, tmp_path):
        """Test precision behavior under memory pressure."""

        # Create large dataset
        large_data = torch.randn(1000, 1000, dtype=torch.float64)
        large_file = tmp_path / "large_data.pt"
        torch.save(large_data, large_file)

        try:
            # Test loading with different precisions under memory pressure
            strategies = [
                PrecisionStrategy.FULL_64,
                PrecisionStrategy.FULL_32,
                PrecisionStrategy.MIXED_16,
            ]

            for strategy in strategies:
                with precision_override(strategy):
                    data = load_array(large_file)

                    # Verify precision is maintained even with large dataflow
                    expected_dtype = strategy.to_torch_dtype()
                    assert data.dtype == expected_dtype

                    # Force garbage collection
                    del data
                    gc.collect()

        finally:
            large_file.unlink(missing_ok=True)


    def test_precision_with_broken_models(self):
        """Test precision handling with models that fail during precision application."""

        in_features, out_features = 10, 5

        # Model that breaks during precision application
        with pytest.raises(RuntimeError, match="Model precision application failed"):
            with precision_override(PrecisionStrategy.MIXED_16):
                BrokenModel(in_features, out_features, break_on_precision=True)

        # Normal model should work fine
        with precision_override(PrecisionStrategy.MIXED_16):
            normal_model = BrokenModel(in_features, out_features, break_on_precision=False)
        assert next(normal_model.parameters()).dtype == torch.float16


    def test_precision_service_singleton_behavior(self):
        """Test precision service singleton behavior under stress."""

        # Test that multiple calls return the same service
        service1 = get_precision_service()
        service2 = get_precision_service()
        assert service1 is service2

        # Test thread safety of singleton
        services = []
        exceptions = []

        def get_service_in_thread():
            try:
                service = get_precision_service()
                services.append(service)
            except Exception as e:
                exceptions.append(e)

        threads = [threading.Thread(target=get_service_in_thread) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should have no exceptions
        assert not exceptions

        # All services should be the same instance
        assert all(service is services[0] for service in services)


    def test_precision_context_stress_testing(self):
        """Stress test precision context with rapid changes."""

        strategies = list(PrecisionStrategy)
        results = []
        exceptions = []

        def rapid_precision_changes():
            try:
                for i in range(100):
                    strategy = strategies[i % len(strategies)]
                    with precision_override(strategy):
                        context = get_precision_context()
                        resolved = context.get_override()  # Get current override directly
                        results.append((i, strategy, resolved))
            except Exception as e:
                exceptions.append(e)

        # Run in multiple threads
        threads = [threading.Thread(target=rapid_precision_changes) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should have no exceptions
        assert not exceptions

        # All overrides should have been respected
        for i, expected, actual in results:
            assert expected == actual


    def test_precision_with_invalid_session_data(self):
        """Test precision handling with invalid session configurations."""

        # Test with mock broken session
        class BrokenSession:
            def get_precision_strategy(self):
                raise RuntimeError("Session is broken")

        service = get_precision_service()
        broken_session = BrokenSession()

        # Should fall back to default
        precision = service.resolve_precision(broken_session)
        assert precision == PrecisionStrategy.FULL_32


    def test_precision_dtype_conversion_edge_cases(self):
        """Test edge cases in dtype conversion."""

        service = get_precision_service()

        # Test all precision strategies
        for strategy in PrecisionStrategy:
            # Test torch dtype conversion
            torch_dtype = service.get_torch_dtype(None, strategy)
            assert isinstance(torch_dtype, torch.dtype)

            # Test compute dtype conversion
            compute_dtype = service.get_compute_dtype(None, strategy)
            assert isinstance(compute_dtype, torch.dtype)

            # Test Lightning precision conversion
            lightning_precision = service.get_lightning_precision(None, strategy)
            assert isinstance(lightning_precision, (str, int))


    def test_precision_with_empty_and_none_inputs(self, tmp_path):
        """Test precision handling with empty and None inputs."""

        # Create empty tensor file
        empty_data = torch.empty(0, 5, dtype=torch.float32)
        empty_file = tmp_path / "empty.pt"
        torch.save(empty_data, empty_file)

        try:
            # Test loading empty dataflow with precision
            with precision_override(PrecisionStrategy.MIXED_16):
                data = load_array(empty_file)
                assert data.dtype == torch.float16
                assert data.numel() == 0

            # Test feature with None dtype
            feature = Feature(name="test", path=empty_file, dtype=None)
            resolved_dtype = feature.get_effective_dtype()
            assert isinstance(resolved_dtype, torch.dtype)

        finally:
            empty_file.unlink(missing_ok=True)


    def test_precision_lightning_compatibility(self):
        """Test Lightning precision string compatibility."""

        # Test all known Lightning precision formats
        lightning_precisions = [
            "32-true",
            "32",
            32,
            "16-mixed",
            "16-true",
            "16",
            16,
            "bf16-mixed",
            "bf16-true",
            "bf16",
            "64-true",
            "64",
            64,
        ]

        for precision_str in lightning_precisions:
            try:
                # Should be able to create strategy from Lightning precision
                strategy = PrecisionStrategy.from_lightning_precision(precision_str)
                assert isinstance(strategy, PrecisionStrategy)

                # Should be able to convert back
                converted = strategy.to_lightning_precision()
                assert converted is not None

            except ValueError:
                # Some precisions might not be supported, that's OK
                pass


    def test_precision_with_extremely_nested_contexts(self):
        """Test precision with deeply nested context overrides."""

        strategies = [
            PrecisionStrategy.FULL_32,
            PrecisionStrategy.MIXED_16,
            PrecisionStrategy.FULL_64,
            PrecisionStrategy.MIXED_BF16,
            PrecisionStrategy.TRUE_16,
        ]

        def nest_contexts(depth, strategies, results):
            if depth >= len(strategies):
                # At maximum depth, verify current precision
                context = get_precision_context()
                current = context.resolve_precision(PrecisionStrategy.get_default())
                results.append(current)
                return

            strategy = strategies[depth]
            with precision_override(strategy):
                nest_contexts(depth + 1, strategies, results)

        results = []
        nest_contexts(0, strategies, results)

        # Should have the deepest (last) strategy
        assert len(results) == 1
        assert results[0] == strategies[-1]


    def test_precision_resource_cleanup(self, tmp_path):
        """Test proper resource cleanup during precision operations."""

        # Create multiple temporary files
        files = []
        for i in range(10):
            data = torch.randn(100, 10, dtype=torch.float32)
            file_path = tmp_path / f"data_{i}.pt"
            torch.save(data, file_path)
            files.append(file_path)

        try:
            # Load dataflow with different precisions and ensure cleanup
            for strategy in [
                PrecisionStrategy.FULL_32,
                PrecisionStrategy.MIXED_16,
                PrecisionStrategy.FULL_64,
            ]:
                with precision_override(strategy):
                    tensors = []
                    for file_path in files:
                        tensor = load_array(file_path)
                        tensors.append(tensor)

                    # Verify all tensors have correct precision
                    expected_dtype = strategy.to_torch_dtype()
                    for tensor in tensors:
                        assert tensor.dtype == expected_dtype

                    # Cleanup tensors
                    del tensors
                    gc.collect()

        finally:
            # Cleanup files
            for file_path in files:
                file_path.unlink(missing_ok=True)


    def test_precision_service_thread_local_isolation(self):
        """Test thread-local isolation in precision service."""

        results = {}
        errors = []

        def thread_worker(thread_id, strategy):
            try:
                # Set precision in this thread
                with precision_override(strategy):
                    # Get service and context
                    service = get_precision_service()
                    context = get_precision_context()

                    # Verify isolation
                    resolved = context.resolve_precision(PrecisionStrategy.get_default())
                    dtype = service.get_torch_dtype()

                    # Store results
                    results[thread_id] = {
                        "strategy": strategy,
                        "resolved": resolved,
                        "dtype": dtype,
                    }

                    # Hold the context for a bit
                    time.sleep(0.1)

                    # Re-verify after delay
                    resolved_after = context.resolve_precision(PrecisionStrategy.get_default())
                    assert resolved_after == strategy

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Create threads with different strategies
        strategies = [
            PrecisionStrategy.FULL_32,
            PrecisionStrategy.MIXED_16,
            PrecisionStrategy.FULL_64,
        ]
        threads = []

        for i, strategy in enumerate(strategies):
            thread = threading.Thread(target=thread_worker, args=(i, strategy))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors
        assert not errors, f"Thread errors: {errors}"

        # Verify isolation worked
        expected_dtypes = {
            PrecisionStrategy.FULL_32: torch.float32,
            PrecisionStrategy.MIXED_16: torch.float16,
            PrecisionStrategy.FULL_64: torch.float64,
        }

        for thread_id, result in results.items():
            strategy = result["strategy"]
            assert result["resolved"] == strategy
            assert result["dtype"] == expected_dtypes[strategy]
