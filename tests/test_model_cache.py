"""Tests for the ModelCache class."""

import unittest
from unittest.mock import patch, MagicMock

from gen_worker.model_cache import (
    ModelCache,
    ModelCacheStats,
    ModelLocation,
    CachedModel,
)


class TestModelCache(unittest.TestCase):
    """Tests for ModelCache LRU eviction and stats."""

    def setUp(self) -> None:
        """Create a fresh ModelCache for each test."""
        # Patch torch to avoid CUDA detection
        with patch("gen_worker.model_cache.ModelCache._detect_total_vram", return_value=24.0):
            self.cache = ModelCache(
                max_vram_gb=20.0,
                vram_safety_margin_gb=4.0,
            )

    def test_register_model_vram(self) -> None:
        """Test registering a model in VRAM."""
        self.cache.register_model(
            model_id="model-a",
            location=ModelLocation.VRAM,
            size_gb=5.0,
            pipeline=MagicMock(),
        )

        self.assertTrue(self.cache.has_model("model-a"))
        self.assertTrue(self.cache.is_in_vram("model-a"))
        self.assertFalse(self.cache.is_on_disk("model-a"))

    def test_register_model_disk(self) -> None:
        """Test registering a model on disk."""
        from pathlib import Path

        self.cache.register_model(
            model_id="model-b",
            location=ModelLocation.DISK,
            size_gb=10.0,
            disk_path=Path("/tmp/model-b"),
        )

        self.assertTrue(self.cache.has_model("model-b"))
        self.assertFalse(self.cache.is_in_vram("model-b"))
        self.assertTrue(self.cache.is_on_disk("model-b"))

    def test_lru_ordering(self) -> None:
        """Test that LRU ordering works correctly."""
        # Register three models
        self.cache.register_model("model-a", ModelLocation.VRAM, 5.0, MagicMock())
        self.cache.register_model("model-b", ModelLocation.VRAM, 5.0, MagicMock())
        self.cache.register_model("model-c", ModelLocation.VRAM, 5.0, MagicMock())

        # model-a is LRU (first registered)
        lru = self.cache._get_lru_vram_models()
        self.assertEqual(lru[0], "model-a")

        # Access model-a, now model-b should be LRU
        self.cache._touch("model-a")
        lru = self.cache._get_lru_vram_models()
        self.assertEqual(lru[0], "model-b")

    def test_lru_eviction(self) -> None:
        """Test that LRU eviction frees space."""
        # Fill cache: 5 + 5 + 5 = 15GB used out of 20GB max
        self.cache.register_model("model-a", ModelLocation.VRAM, 5.0, MagicMock())
        self.cache.register_model("model-b", ModelLocation.VRAM, 5.0, MagicMock())
        self.cache.register_model("model-c", ModelLocation.VRAM, 5.0, MagicMock())

        self.assertEqual(self.cache._vram_used_gb, 15.0)

        # Try to add 10GB model - should evict LRU models
        # Need to evict 5GB (15 + 10 - 20 = 5)
        freed = self.cache._evict_lru_for_space(10.0)

        # model-a (5GB) should be evicted
        self.assertEqual(freed, 5.0)
        self.assertEqual(self.cache._vram_used_gb, 10.0)
        self.assertFalse(self.cache.is_in_vram("model-a"))

    def test_get_stats(self) -> None:
        """Test stats generation for heartbeat."""
        self.cache.register_model("model-vram-1", ModelLocation.VRAM, 5.0, MagicMock())
        self.cache.register_model("model-vram-2", ModelLocation.VRAM, 3.0, MagicMock())
        from pathlib import Path
        self.cache.register_model("model-disk-1", ModelLocation.DISK, 10.0, disk_path=Path("/tmp/m"))
        self.cache.mark_downloading("model-dl-1", 0.5)

        stats = self.cache.get_stats()

        self.assertIsInstance(stats, ModelCacheStats)
        self.assertEqual(len(stats.vram_models), 2)
        self.assertIn("model-vram-1", stats.vram_models)
        self.assertIn("model-vram-2", stats.vram_models)
        self.assertEqual(len(stats.disk_models), 1)
        self.assertIn("model-disk-1", stats.disk_models)
        self.assertEqual(len(stats.downloading_models), 1)
        self.assertIn("model-dl-1", stats.downloading_models)
        self.assertEqual(stats.vram_used_gb, 8.0)
        self.assertEqual(stats.vram_model_count, 2)
        self.assertEqual(stats.disk_model_count, 1)
        self.assertEqual(stats.total_models, 4)

    def test_stats_to_dict(self) -> None:
        """Test stats serialization to dict."""
        self.cache.register_model("model-a", ModelLocation.VRAM, 5.5, MagicMock())
        stats = self.cache.get_stats()
        d = stats.to_dict()

        self.assertIsInstance(d, dict)
        self.assertIn("vram_models", d)
        self.assertIn("vram_used_gb", d)
        self.assertEqual(d["vram_used_gb"], 5.5)

    def test_unload_model(self) -> None:
        """Test unloading a model from cache."""
        self.cache.register_model("model-a", ModelLocation.VRAM, 5.0, MagicMock())
        self.assertTrue(self.cache.has_model("model-a"))
        self.assertEqual(self.cache._vram_used_gb, 5.0)

        result = self.cache.unload_model("model-a")
        self.assertTrue(result)
        self.assertFalse(self.cache.has_model("model-a"))
        self.assertEqual(self.cache._vram_used_gb, 0.0)

    def test_unload_nonexistent(self) -> None:
        """Test unloading a model that doesn't exist."""
        result = self.cache.unload_model("nonexistent")
        self.assertFalse(result)

    def test_mark_loaded_to_vram(self) -> None:
        """Test marking a model as loaded to VRAM."""
        self.cache.mark_loaded_to_vram("model-a", MagicMock(), 8.0)

        self.assertTrue(self.cache.is_in_vram("model-a"))
        self.assertEqual(self.cache._vram_used_gb, 8.0)

    def test_mark_cached_to_disk(self) -> None:
        """Test marking a model as cached to disk."""
        from pathlib import Path

        # First load to VRAM
        self.cache.mark_loaded_to_vram("model-a", MagicMock(), 8.0)
        self.assertTrue(self.cache.is_in_vram("model-a"))

        # Then mark as disk-cached (offloaded)
        self.cache.mark_cached_to_disk("model-a", Path("/tmp/model-a"), 8.0)

        self.assertFalse(self.cache.is_in_vram("model-a"))
        self.assertTrue(self.cache.is_on_disk("model-a"))
        self.assertEqual(self.cache._vram_used_gb, 0.0)

    def test_can_fit_in_vram(self) -> None:
        """Test checking if model can fit in VRAM."""
        # Empty cache with 20GB max
        self.assertTrue(self.cache.can_fit_in_vram(10.0))
        self.assertTrue(self.cache.can_fit_in_vram(20.0))
        self.assertFalse(self.cache.can_fit_in_vram(25.0))

        # With some models loaded
        self.cache.register_model("model-a", ModelLocation.VRAM, 15.0, MagicMock())
        self.assertTrue(self.cache.can_fit_in_vram(5.0))  # 15 + 5 = 20
        self.assertTrue(self.cache.can_fit_in_vram(20.0))  # Can evict model-a
        self.assertFalse(self.cache.can_fit_in_vram(25.0))  # Too big even after eviction

    def test_get_pipeline(self) -> None:
        """Test getting a pipeline from cache."""
        pipeline = MagicMock()
        self.cache.register_model("model-a", ModelLocation.VRAM, 5.0, pipeline)

        retrieved = self.cache.get_pipeline("model-a")
        self.assertIs(retrieved, pipeline)

        # Getting pipeline should update LRU
        self.cache.register_model("model-b", ModelLocation.VRAM, 5.0, MagicMock())
        self.cache.get_pipeline("model-a")
        lru = self.cache._get_lru_vram_models()
        self.assertEqual(lru[0], "model-b")  # model-b is now LRU

    def test_get_pipeline_disk_model(self) -> None:
        """Test getting pipeline for disk-cached model returns None."""
        from pathlib import Path
        self.cache.register_model("model-a", ModelLocation.DISK, 5.0, disk_path=Path("/tmp"))

        retrieved = self.cache.get_pipeline("model-a")
        self.assertIsNone(retrieved)

    def test_download_progress(self) -> None:
        """Test download progress tracking."""
        self.cache.mark_downloading("model-a", 0.0)
        model = self.cache._models.get("model-a")
        self.assertIsNotNone(model)
        self.assertEqual(model.location, ModelLocation.DOWNLOADING)
        self.assertEqual(model.download_progress, 0.0)

        self.cache.update_download_progress("model-a", 0.5)
        self.assertEqual(model.download_progress, 0.5)

        self.cache.update_download_progress("model-a", 1.0)
        self.assertEqual(model.download_progress, 1.0)


class TestModelCacheEnvironment(unittest.TestCase):
    """Test ModelCache configuration from environment."""

    def test_env_config(self) -> None:
        """Test that environment variables configure the cache."""
        import os

        with patch.dict(os.environ, {
            "WORKER_MAX_VRAM_GB": "16",
            "WORKER_VRAM_SAFETY_MARGIN_GB": "2.5",
        }):
            with patch("gen_worker.model_cache.ModelCache._detect_total_vram", return_value=24.0):
                cache = ModelCache()
                self.assertEqual(cache._max_vram_gb, 16.0)
                self.assertEqual(cache._vram_safety_margin, 2.5)


class TestProgressiveAvailability(unittest.TestCase):
    """Tests for progressive model availability."""

    def setUp(self) -> None:
        """Create a fresh ModelCache for each test."""
        with patch("gen_worker.model_cache.ModelCache._detect_total_vram", return_value=24.0):
            self.cache = ModelCache(max_vram_gb=20.0)

    def test_are_models_available_all_ready(self) -> None:
        """Test that available check passes when all models are ready."""
        self.cache.register_model("model-a", ModelLocation.VRAM, 5.0, MagicMock())
        from pathlib import Path
        self.cache.register_model("model-b", ModelLocation.DISK, 5.0, disk_path=Path("/tmp"))

        # Both VRAM and disk models should be considered available
        self.assertTrue(self.cache.are_models_available(["model-a"]))
        self.assertTrue(self.cache.are_models_available(["model-b"]))
        self.assertTrue(self.cache.are_models_available(["model-a", "model-b"]))

    def test_are_models_available_downloading(self) -> None:
        """Test that available check fails when model is downloading."""
        self.cache.register_model("model-a", ModelLocation.VRAM, 5.0, MagicMock())
        self.cache.mark_downloading("model-b", 0.5)

        # model-a is available, model-b is not
        self.assertTrue(self.cache.are_models_available(["model-a"]))
        self.assertFalse(self.cache.are_models_available(["model-b"]))
        self.assertFalse(self.cache.are_models_available(["model-a", "model-b"]))

    def test_are_models_available_missing(self) -> None:
        """Test that available check fails for unknown models."""
        self.cache.register_model("model-a", ModelLocation.VRAM, 5.0, MagicMock())

        self.assertTrue(self.cache.are_models_available(["model-a"]))
        self.assertFalse(self.cache.are_models_available(["model-a", "model-unknown"]))
        self.assertFalse(self.cache.are_models_available(["model-unknown"]))

    def test_get_available_models(self) -> None:
        """Test getting list of available models."""
        self.cache.register_model("model-vram", ModelLocation.VRAM, 5.0, MagicMock())
        from pathlib import Path
        self.cache.register_model("model-disk", ModelLocation.DISK, 5.0, disk_path=Path("/tmp"))
        self.cache.mark_downloading("model-dl", 0.5)

        available = self.cache.get_available_models()
        self.assertEqual(len(available), 2)
        self.assertIn("model-vram", available)
        self.assertIn("model-disk", available)
        self.assertNotIn("model-dl", available)

    def test_get_downloading_models(self) -> None:
        """Test getting list of downloading models."""
        self.cache.register_model("model-a", ModelLocation.VRAM, 5.0, MagicMock())
        self.cache.mark_downloading("model-b", 0.3)
        self.cache.mark_downloading("model-c", 0.7)

        downloading = self.cache.get_downloading_models()
        self.assertEqual(len(downloading), 2)
        self.assertIn("model-b", downloading)
        self.assertIn("model-c", downloading)

    def test_get_download_progress(self) -> None:
        """Test getting download progress for a model."""
        self.cache.mark_downloading("model-a", 0.5)
        self.cache.register_model("model-b", ModelLocation.VRAM, 5.0, MagicMock())

        self.assertEqual(self.cache.get_download_progress("model-a"), 0.5)
        self.assertIsNone(self.cache.get_download_progress("model-b"))
        self.assertIsNone(self.cache.get_download_progress("unknown"))

    def test_max_concurrent_downloads_config(self) -> None:
        """Test max concurrent downloads configuration."""
        import os

        # Default value
        self.assertEqual(self.cache.get_max_concurrent_downloads(), 2)

        # From environment
        with patch.dict(os.environ, {"WORKER_MAX_CONCURRENT_DOWNLOADS": "4"}):
            with patch("gen_worker.model_cache.ModelCache._detect_total_vram", return_value=24.0):
                cache = ModelCache()
                self.assertEqual(cache.get_max_concurrent_downloads(), 4)


if __name__ == "__main__":
    unittest.main()
