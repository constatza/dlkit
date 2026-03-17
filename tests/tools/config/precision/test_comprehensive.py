"""Comprehensive end-to-end tests for precision control system.

This module contains comprehensive tests for production scenarios,
edge cases, and complete workflow integration that might not be
covered in individual component tests.
"""

import pytest
import torch
import threading
import time
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor, as_completed

from dlkit.tools.config.precision import PrecisionStrategy
from dlkit.interfaces.api.domain.precision import precision_override
from dlkit.interfaces.api.services.precision_service import get_precision_service
from dlkit.tools.config.session_settings import SessionSettings
from dlkit.tools.config.data_entries import Feature, Target
from dlkit.tools.config.trainer_settings import TrainerSettings
from dlkit.tools.io.arrays import load_array
from dlkit.core.models.nn.base import DLKitModel
from dlkit.core.shape_specs import create_shape_spec


class ProductionTestModel(DLKitModel):
    """Realistic model for production testing."""

    def __init__(self, shape, **kwargs):
        super().__init__()

        # Convert shape dict to unified_shape
        if isinstance(shape, dict):
            unified_shape = create_shape_spec(shape)
        else:
            # Assume it's already a shape spec
            unified_shape = shape
        self._unified_shape = unified_shape

        # Extract shapes from unified_shape for building layers
        input_shape = unified_shape.get_input_shape()
        output_shape = unified_shape.get_output_shape()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Linear(64, output_shape[0])

        # Apply precision from context (simulating Lightning behavior)
        service = get_precision_service()
        precision_strategy = service.resolve_precision()
        dtype = precision_strategy.to_torch_dtype()
        self.to(dtype)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


class TestComprehensivePrecision:
    """Comprehensive precision control tests."""

    @pytest.fixture
    def sample_datasets(self, tmp_path):
        """Create multiple sample datasets for testing."""
        datasets = {}

        # Create different precision datasets
        for name, dtype in [
            ("float32", torch.float32),
            ("float64", torch.float64),
            ("float16", torch.float16),
        ]:
            data = torch.randn(100, 20, dtype=dtype)
            file_path = tmp_path / f"data_{name}.pt"
            torch.save(data, file_path)
            datasets[name] = file_path

        return datasets

    def test_end_to_end_training_workflow_precision(self, sample_datasets):
        """Test complete training workflow with precision control."""

        # Setup session with mixed precision
        session = SessionSettings(precision=PrecisionStrategy.MIXED_16)

        # Create features and targets with session precision
        feature = Feature(name="input", path=sample_datasets["float32"])
        target = Target(name="output", path=sample_datasets["float32"])

        # Load data with session precision using context
        with precision_override(session.get_precision_strategy()):
            input_data = load_array(feature.path)
            target_data = load_array(target.path)

        # Verify data precision
        assert input_data.dtype == torch.float16
        assert target_data.dtype == torch.float16

        # Create model with session precision (using precision override context)
        shape = {"x": (20,), "y": (20,)}
        with precision_override(PrecisionStrategy.MIXED_16):
            model = ProductionTestModel(shape)

        # Verify model precision matches session
        model_dtype = next(model.parameters()).dtype
        assert model_dtype == torch.float16

        # Test model forward pass with precision consistency
        output = model(input_data[:10])
        assert output.dtype == torch.float16


    def test_precision_override_workflow(self, sample_datasets):
        """Test precision override scenarios in realistic workflows."""

        # Default session
        session = SessionSettings(precision=PrecisionStrategy.FULL_32)

        # Normal operation with context
        with precision_override(session.get_precision_strategy()):
            data_normal = load_array(sample_datasets["float32"])
        assert data_normal.dtype == torch.float32

        # Override for memory-constrained operation
        with precision_override(PrecisionStrategy.MIXED_16):
            data_memory_save = load_array(sample_datasets["float32"])
            assert data_memory_save.dtype == torch.float16

            # Model creation in override context
            shape = {"x": (20,), "y": (20,)}
            model_memory = ProductionTestModel(shape)
            model_dtype = next(model_memory.parameters()).dtype
            assert model_dtype == torch.float16

        # Override for high-precision computation
        with precision_override(PrecisionStrategy.FULL_64):
            data_high_precision = load_array(sample_datasets["float32"])
            assert data_high_precision.dtype == torch.float64

            model_high_precision = ProductionTestModel(shape)
            model_dtype = next(model_high_precision.parameters()).dtype
            assert model_dtype == torch.float64


    def test_multi_threaded_precision_isolation(self, sample_datasets):
        """Test precision isolation across multiple threads."""

        results = {}
        errors = []

        def worker_thread(thread_id, strategy, datasets):
            try:
                with precision_override(strategy):
                    # Load dataflow in thread-specific precision
                    data = load_array(datasets["float32"])

                    # Create model in thread-specific precision
                    shape = {"x": (20,), "y": (5,)}
                    model = ProductionTestModel(shape)

                    # Verify precision isolation
                    data_dtype = data.dtype
                    model_dtype = next(model.parameters()).dtype

                    # Store results
                    results[thread_id] = {
                        "strategy": strategy,
                        "data_dtype": data_dtype,
                        "model_dtype": model_dtype,
                    }

                    # Simulate some work
                    time.sleep(0.1)

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Create multiple threads with different precision strategies
        threads = []
        strategies = [
            PrecisionStrategy.FULL_32,
            PrecisionStrategy.MIXED_16,
            PrecisionStrategy.FULL_64,
            PrecisionStrategy.MIXED_BF16,
        ]

        for i, strategy in enumerate(strategies):
            thread = threading.Thread(target=worker_thread, args=(i, strategy, sample_datasets))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors
        assert not errors, f"Thread errors: {errors}"

        # Verify each thread got its expected precision
        assert len(results) == 4

        expected_dtypes = {
            PrecisionStrategy.FULL_32: torch.float32,
            PrecisionStrategy.MIXED_16: torch.float16,
            PrecisionStrategy.FULL_64: torch.float64,
            PrecisionStrategy.MIXED_BF16: torch.bfloat16,
        }

        for thread_id, result in results.items():
            strategy = result["strategy"]
            expected_dtype = expected_dtypes[strategy]
            assert result["data_dtype"] == expected_dtype
            assert result["model_dtype"] == expected_dtype


    def test_trainer_settings_lightning_integration(self):
        """Test complete TrainerSettings integration with Lightning precision."""

        # Test with session precision
        _session = SessionSettings(precision=PrecisionStrategy.MIXED_BF16)
        trainer_settings = TrainerSettings()

        # Mock the service to test integration
        with patch(
            "dlkit.interfaces.api.services.precision_service.get_precision_service"
        ) as mock_service:
            mock_service.return_value.get_lightning_precision.return_value = "bf16-mixed"

            # Build trainer (would normally create actual Trainer)
            try:
                _trainer = trainer_settings.build()
                # This might fail due to missing dependencies, but we test the precision integration
            except Exception:
                pass  # Expected - we're testing precision logic, not actual trainer creation

            # Verify precision service was called
            mock_service.return_value.get_lightning_precision.assert_called_once()

        # Test with explicit precision override
        trainer_settings_explicit = TrainerSettings(precision="32")
        assert trainer_settings_explicit.precision == "32"


    def test_error_recovery_and_fallbacks(self, sample_datasets):
        """Test error recovery and fallback mechanisms."""

        # Test DataEntry fallback when precision service fails
        feature = Feature(name="test", path=sample_datasets["float32"])

        # Mock service failure
        with patch(
            "dlkit.interfaces.api.services.precision_service.get_precision_service"
        ) as mock_service:
            mock_service.side_effect = RuntimeError("Service unavailable")

            # Should fall back to provided default
            dtype = feature.resolve_dtype_with_fallback(torch.float64)
            assert dtype == torch.float64

        # Test load_array with service failure
        with patch("dlkit.tools.io.arrays.get_precision_service") as mock_service:
            mock_service.side_effect = RuntimeError("Service unavailable")

            # Should work with explicit dtype
            data = load_array(sample_datasets["float32"], dtype=torch.float32)
            assert data.dtype == torch.float32


    def test_memory_optimization_scenarios(self, sample_datasets):
        """Test memory optimization scenarios with different precision strategies."""

        shape = {"x": (20,), "y": (10,)}

        # Test memory factors are correctly calculated
        _service = get_precision_service()

        memory_32 = PrecisionStrategy.FULL_32.get_memory_factor()
        memory_16 = PrecisionStrategy.MIXED_16.get_memory_factor()
        memory_64 = PrecisionStrategy.FULL_64.get_memory_factor()

        # Verify memory relationships
        assert memory_16 < memory_32 < memory_64
        assert memory_16 == 0.7  # Mixed precision ~30% savings
        assert memory_32 == 1.0  # Full precision baseline
        assert memory_64 == 2.0  # Double precision

        # Test actual memory usage with different precisions
        models = {}
        for strategy in [
            PrecisionStrategy.FULL_32,
            PrecisionStrategy.MIXED_16,
            PrecisionStrategy.FULL_64,
        ]:
            with precision_override(strategy):
                model = ProductionTestModel(shape)
            models[strategy] = model

        # Verify models have expected precision
        assert next(models[PrecisionStrategy.FULL_32].parameters()).dtype == torch.float32
        assert next(models[PrecisionStrategy.MIXED_16].parameters()).dtype == torch.float16
        assert next(models[PrecisionStrategy.FULL_64].parameters()).dtype == torch.float64


    def test_numerical_stability_edge_cases(self, sample_datasets):
        """Test numerical stability with different precision strategies."""

        # Test with very small and very large numbers
        small_data = torch.tensor([1e-8, 1e-10, 1e-12], dtype=torch.float64)
        large_data = torch.tensor([1e8, 1e10, 1e12], dtype=torch.float64)

        # Save test dataflow
        small_file = sample_datasets["float64"].parent / "small_data.pt"
        large_file = sample_datasets["float64"].parent / "large_data.pt"
        torch.save(small_data, small_file)
        torch.save(large_data, large_file)

        try:
            # Test different precision strategies with extreme values
            strategies = [
                PrecisionStrategy.FULL_64,
                PrecisionStrategy.FULL_32,
                PrecisionStrategy.MIXED_16,
            ]

            for strategy in strategies:
                with precision_override(strategy):
                    small_loaded = load_array(small_file)
                    large_loaded = load_array(large_file)

                    # Verify dataflow is loaded (precision loss expected for lower precision)
                    assert small_loaded.numel() == 3
                    assert large_loaded.numel() == 3

                    # Check that 64-bit preserves precision better
                    if strategy == PrecisionStrategy.FULL_64:
                        assert torch.allclose(
                            small_loaded, small_data.to(small_loaded.dtype), rtol=1e-10
                        )

        finally:
            # Cleanup
            small_file.unlink(missing_ok=True)
            large_file.unlink(missing_ok=True)


    def test_production_pipeline_integration(self, sample_datasets):
        """Test complete production pipeline integration."""

        # Simulate production pipeline with multiple precision requirements
        session = SessionSettings(precision=PrecisionStrategy.MIXED_16)

        # Data loading phase with precision context
        features = []
        with precision_override(session.get_precision_strategy()):
            for i, (name, path) in enumerate(sample_datasets.items()):
                feature = Feature(name=f"feature_{i}", path=path)
                data = load_array(feature.path)
                features.append(data)
                assert data.dtype == torch.float16

        # Model creation phase
        shape = {"x": (20,), "y": (5,)}
        model = ProductionTestModel(shape)

        # Apply precision to model using precision service
        _service = get_precision_service()
        model = _service.apply_precision_to_model(model, provider=session)
        assert next(model.parameters()).dtype == torch.float16

        # Training simulation phase
        sample_input = features[0][:10]  # Batch of 10

        # Forward pass
        output = model(sample_input)
        assert output.dtype == torch.float16

        # Backward pass simulation (would normally involve loss calculation)
        loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
        assert loss.dtype == torch.float16

        # High precision validation phase
        with precision_override(PrecisionStrategy.FULL_64):
            validation_data = load_array(sample_datasets["float32"])
            validation_model = ProductionTestModel(shape)

            # Cast input to validation model precision (manual cast for validation)
            validation_input = sample_input.to(torch.float64)
            validation_output = validation_model(validation_input)

            assert validation_data.dtype == torch.float64
            assert validation_input.dtype == torch.float64
            assert validation_output.dtype == torch.float64


    def test_concurrent_precision_operations(self, sample_datasets):
        """Test concurrent precision operations under load."""

        def precision_worker(worker_id, datasets):
            """Worker function for concurrent testing."""
            results = []
            strategies = [
                PrecisionStrategy.FULL_32,
                PrecisionStrategy.MIXED_16,
                PrecisionStrategy.FULL_64,
            ]

            for strategy in strategies:
                with precision_override(strategy):
                    # Load dataflow
                    data = load_array(datasets["float32"])

                    # Create model
                    shape = {"x": (20,), "y": (5,)}
                    model = ProductionTestModel(shape)

                    # Forward pass
                    output = model(data[:5])

                    results.append({
                        "worker_id": worker_id,
                        "strategy": strategy,
                        "data_dtype": data.dtype,
                        "model_dtype": next(model.parameters()).dtype,
                        "output_dtype": output.dtype,
                    })

            return results

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(precision_worker, i, sample_datasets) for i in range(4)]

            all_results = []
            for future in as_completed(futures):
                worker_results = future.result()
                all_results.extend(worker_results)

        # Verify all operations completed correctly
        assert len(all_results) == 12  # 4 workers * 3 strategies

        # Verify precision consistency
        expected_dtypes = {
            PrecisionStrategy.FULL_32: torch.float32,
            PrecisionStrategy.MIXED_16: torch.float16,
            PrecisionStrategy.FULL_64: torch.float64,
        }

        for result in all_results:
            strategy = result["strategy"]
            expected_dtype = expected_dtypes[strategy]

            assert result["data_dtype"] == expected_dtype
            assert result["model_dtype"] == expected_dtype
            assert result["output_dtype"] == expected_dtype


    def test_precision_configuration_validation(self):
        """Test precision configuration validation and compatibility."""

        _service = get_precision_service()

        # Test all strategy combinations are valid
        for strategy in PrecisionStrategy:
            lightning_precision = strategy.to_lightning_precision()
            torch_dtype = strategy.to_torch_dtype()
            compute_dtype = strategy.get_compute_dtype()

            # Verify Lightning precision is valid
            assert isinstance(lightning_precision, (str, int))

            # Verify torch dtype is valid
            assert isinstance(torch_dtype, torch.dtype)

            # Verify compute dtype is valid
            assert isinstance(compute_dtype, torch.dtype)

            # Test service can handle all strategies
            session = SessionSettings(precision=strategy)
            resolved_precision = _service.resolve_precision(session)
            assert resolved_precision == strategy

            resolved_dtype = _service.get_torch_dtype(session)
            assert resolved_dtype == torch_dtype
