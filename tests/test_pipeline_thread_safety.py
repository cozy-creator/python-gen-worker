"""
Tests for thread-safe pipeline access in PipelineLoader.

These tests verify that get_for_inference() properly creates thread-safe
pipeline copies with fresh schedulers to avoid concurrent access issues.
"""
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Check if torch is available for skip markers
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MockSchedulerConfig:
    """Mock scheduler config for testing."""
    def __init__(self) -> None:
        self.num_train_timesteps = 1000
        self.beta_start = 0.0001
        self.beta_end = 0.02


class MockScheduler:
    """Mock scheduler that tracks instance creation."""
    _instance_count = 0

    def __init__(self) -> None:
        MockScheduler._instance_count += 1
        self.instance_id = MockScheduler._instance_count
        self.config = MockSchedulerConfig()
        # Simulate internal state that would cause issues if shared
        self._timesteps: List[int] = []
        self._step_index: Optional[int] = None

    @classmethod
    def from_config(cls, config: MockSchedulerConfig) -> "MockScheduler":
        """Create new scheduler from config (like diffusers does)."""
        scheduler = cls()
        scheduler.config = config
        return scheduler

    @classmethod
    def reset_instance_count(cls) -> None:
        cls._instance_count = 0


class MockPipeline:
    """Mock pipeline for testing thread-safe access."""
    _from_pipe_calls = 0

    def __init__(self, scheduler: Optional[MockScheduler] = None) -> None:
        self.scheduler = scheduler or MockScheduler()
        self.unet = MagicMock()  # Simulates heavy component
        self.vae = MagicMock()   # Simulates heavy component

    @classmethod
    def from_pipe(cls, base_pipeline: "MockPipeline", scheduler: MockScheduler) -> "MockPipeline":
        """Create new pipeline sharing components but with fresh scheduler."""
        MockPipeline._from_pipe_calls += 1
        new_pipeline = cls(scheduler=scheduler)
        # Share heavy components
        new_pipeline.unet = base_pipeline.unet
        new_pipeline.vae = base_pipeline.vae
        return new_pipeline

    @classmethod
    def reset_call_count(cls) -> None:
        cls._from_pipe_calls = 0


class MockLoadedPipeline:
    """Mock LoadedPipeline container."""
    def __init__(self, pipeline: MockPipeline, model_id: str) -> None:
        self.pipeline = pipeline
        self.model_id = model_id
        self.pipeline_class = "MockPipeline"
        self.dtype = "float16"
        self.size_gb = 10.0
        self.load_format = "safetensors"


class TestGetForInferenceLogic:
    """
    Tests for get_for_inference() logic without requiring torch.
    Uses direct method testing with mocked dependencies.
    """

    def setup_method(self) -> None:
        """Reset mocks before each test."""
        MockScheduler.reset_instance_count()
        MockPipeline.reset_call_count()

    def test_creates_fresh_scheduler_logic(self) -> None:
        """Test that get_for_inference creates a fresh scheduler."""
        # Test the logic directly without PipelineLoader instantiation
        base_scheduler = MockScheduler()
        base_pipeline = MockPipeline(scheduler=base_scheduler)

        initial_count = MockScheduler._instance_count

        # Simulate what get_for_inference does
        fresh_scheduler = base_pipeline.scheduler.from_config(
            base_pipeline.scheduler.config
        )
        task_pipeline = MockPipeline.from_pipe(base_pipeline, scheduler=fresh_scheduler)

        # Verify new scheduler was created
        assert MockScheduler._instance_count == initial_count + 1
        assert task_pipeline.scheduler.instance_id != base_scheduler.instance_id

    def test_shares_heavy_components_logic(self) -> None:
        """Test that heavy components are shared."""
        base_pipeline = MockPipeline()
        original_unet = base_pipeline.unet
        original_vae = base_pipeline.vae

        # Simulate what get_for_inference does
        fresh_scheduler = base_pipeline.scheduler.from_config(
            base_pipeline.scheduler.config
        )
        task_pipeline = MockPipeline.from_pipe(base_pipeline, scheduler=fresh_scheduler)

        # Heavy components should be shared
        assert task_pipeline.unet is original_unet
        assert task_pipeline.vae is original_vae
        # Scheduler should be different
        assert task_pipeline.scheduler is not base_pipeline.scheduler


class TestConcurrentAccess:
    """Tests for concurrent pipeline access patterns."""

    def setup_method(self) -> None:
        """Reset mocks before each test."""
        MockScheduler.reset_instance_count()
        MockPipeline.reset_call_count()

    def test_concurrent_scheduler_creation_is_safe(self) -> None:
        """Multiple concurrent scheduler creations should be independent."""
        base_pipeline = MockPipeline()
        results: List[MockPipeline] = []
        errors: List[Exception] = []
        lock = threading.Lock()

        def create_task_pipeline() -> None:
            try:
                # Simulate what get_for_inference does
                fresh_scheduler = base_pipeline.scheduler.from_config(
                    base_pipeline.scheduler.config
                )
                task_pipeline = MockPipeline.from_pipe(
                    base_pipeline, scheduler=fresh_scheduler
                )
                with lock:
                    results.append(task_pipeline)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Spawn multiple threads
        threads = [threading.Thread(target=create_task_pipeline) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0, f"Got errors: {errors}"
        assert len(results) == 10

        # Each result should have a unique scheduler
        scheduler_ids = [r.scheduler.instance_id for r in results]
        assert len(set(scheduler_ids)) == 10  # All unique

    def test_concurrent_access_simulated_inference(self) -> None:
        """
        Simulate concurrent inference to verify no state corruption.

        This test verifies that separate schedulers prevent the
        'IndexError: index N is out of bounds' that occurs when
        multiple threads share a scheduler's internal state.
        """
        base_pipeline = MockPipeline()
        errors: List[Exception] = []
        completed = 0
        lock = threading.Lock()

        def simulate_inference() -> None:
            nonlocal completed
            try:
                # Create thread-safe pipeline (what get_for_inference does)
                fresh_scheduler = base_pipeline.scheduler.from_config(
                    base_pipeline.scheduler.config
                )
                task_pipeline = MockPipeline.from_pipe(
                    base_pipeline, scheduler=fresh_scheduler
                )

                # Simulate scheduler state modification during inference
                task_pipeline.scheduler._timesteps = list(range(1000))
                task_pipeline.scheduler._step_index = 0

                # Simulate stepping through inference
                for i in range(50):
                    idx = task_pipeline.scheduler._step_index
                    if idx is not None and idx < len(task_pipeline.scheduler._timesteps):
                        # This would cause IndexError if scheduler is shared
                        _ = task_pipeline.scheduler._timesteps[idx]
                    task_pipeline.scheduler._step_index = i + 1
                    time.sleep(0.001)  # Simulate work

                with lock:
                    completed += 1

            except Exception as e:
                with lock:
                    errors.append(e)

        # Spawn concurrent inference threads
        threads = [threading.Thread(target=simulate_inference) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should complete without errors
        assert len(errors) == 0, f"Got errors: {errors}"
        assert completed == 5


class TestModelManagementInterface:
    """Tests for ModelManagementInterface.get_for_inference()"""

    def test_default_implementation_calls_get_active_pipeline(self) -> None:
        """Default get_for_inference should fall back to get_active_pipeline."""
        from gen_worker.model_interface import ModelManagementInterface

        class TestManager(ModelManagementInterface):
            def __init__(self) -> None:
                self.get_active_pipeline_called = False

            async def process_supported_models_config(
                self, supported_model_ids: List[str], downloader_instance: Any
            ) -> None:
                pass

            async def load_model_into_vram(self, model_id: str) -> bool:
                return True

            async def get_active_pipeline(self, model_id: str) -> Any:
                self.get_active_pipeline_called = True
                return MockPipeline()

            async def get_active_model_bundle(self, model_id: str) -> Any:
                return None

            def get_vram_loaded_models(self) -> List[str]:
                return []

        manager = TestManager()
        result = manager.get_for_inference("test-model")

        assert manager.get_active_pipeline_called
        assert result is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestPipelineLoaderIntegration:
    """Integration tests that require torch."""

    def setup_method(self) -> None:
        """Reset mocks before each test."""
        MockScheduler.reset_instance_count()
        MockPipeline.reset_call_count()

    def test_get_for_inference_with_real_loader(self) -> None:
        """Test get_for_inference with actual PipelineLoader."""
        from gen_worker.pipeline_loader import PipelineLoader

        loader = PipelineLoader()

        # Create mock loaded pipeline
        base_pipeline = MockPipeline()
        loaded = MockLoadedPipeline(base_pipeline, "test-model")
        loader._loaded_pipelines = {"test-model": loaded}

        result = loader.get_for_inference("test-model")

        assert result is not None
        # Should have different scheduler
        assert result.scheduler is not base_pipeline.scheduler
        # Should share heavy components
        assert result.unet is base_pipeline.unet

    def test_get_for_inference_returns_none_for_missing(self) -> None:
        """Test get_for_inference returns None for unloaded models."""
        from gen_worker.pipeline_loader import PipelineLoader

        loader = PipelineLoader()
        loader._loaded_pipelines = {}

        result = loader.get_for_inference("nonexistent")

        assert result is None

    def test_concurrent_get_for_inference_with_loader(self) -> None:
        """Test concurrent get_for_inference calls with real PipelineLoader."""
        from gen_worker.pipeline_loader import PipelineLoader

        loader = PipelineLoader()

        # Create mock loaded pipeline
        base_pipeline = MockPipeline()
        loaded = MockLoadedPipeline(base_pipeline, "test-model")
        loader._loaded_pipelines = {"test-model": loaded}

        results: List[Any] = []
        errors: List[Exception] = []
        lock = threading.Lock()

        def get_pipeline() -> None:
            try:
                result = loader.get_for_inference("test-model")
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Spawn concurrent threads
        threads = [threading.Thread(target=get_pipeline) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Got errors: {errors}"
        assert len(results) == 10

        # Each should have unique scheduler but share heavy components
        scheduler_ids = set()
        for r in results:
            assert r is not None
            scheduler_ids.add(r.scheduler.instance_id)
            assert r.unet is base_pipeline.unet  # Shared

        # All schedulers should be unique
        assert len(scheduler_ids) == 10
