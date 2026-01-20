"""
LRU Model Cache with VRAM tracking for gen-worker.

This module provides a model cache that:
- Tracks models loaded in VRAM vs cached on disk
- Implements LRU eviction when VRAM is exhausted
- Reports cache stats for orchestrator heartbeats
- Supports orchestrator-commanded load/unload operations
- Supports progressive model availability (accept jobs as models become ready)

Configuration via environment variables:
- WORKER_MAX_VRAM_GB: Maximum VRAM to use (default: auto-detect - safety margin)
- WORKER_VRAM_SAFETY_MARGIN_GB: Reserved VRAM for working memory (default: 3.5)
- WORKER_MODEL_CACHE_DIR: Directory for disk-cached models (default: /tmp/cozy/models)
- WORKER_MAX_CONCURRENT_DOWNLOADS: Maximum concurrent model downloads (default: 2)
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Constants - can be overridden via environment variables
DEFAULT_VRAM_SAFETY_MARGIN_GB = 3.5
DEFAULT_WORKING_MEMORY_GB = 2.0
DEFAULT_RAM_SAFETY_MARGIN_GB = 10.0
DEFAULT_MAX_CONCURRENT_DOWNLOADS = 2


class ModelLocation(str, Enum):
    """Where a model is currently stored."""
    VRAM = "vram"       # Loaded in GPU VRAM, ready for inference
    DISK = "disk"       # Cached on disk, needs loading to use
    DOWNLOADING = "downloading"  # Currently being downloaded


@dataclass
class CachedModel:
    """Metadata about a cached model."""
    model_id: str
    location: ModelLocation
    size_gb: float = 0.0
    last_accessed: float = field(default_factory=time.time)
    pipeline: Any = None  # The actual pipeline object (when in VRAM)
    disk_path: Optional[Path] = None  # Path on disk (when cached)
    download_progress: float = 0.0  # 0.0-1.0 progress for downloading models


@dataclass
class ModelCacheStats:
    """Stats for heartbeat reporting to orchestrator."""
    # Models currently loaded in VRAM
    vram_models: List[str]
    # Models cached on disk (can be loaded quickly)
    disk_models: List[str]
    # Models currently being downloaded
    downloading_models: List[str]
    # VRAM usage
    vram_used_gb: float
    vram_total_gb: float
    vram_available_gb: float
    # Counts
    total_models: int
    vram_model_count: int
    disk_model_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for protobuf/JSON serialization."""
        return {
            "vram_models": self.vram_models,
            "disk_models": self.disk_models,
            "downloading_models": self.downloading_models,
            "vram_used_gb": round(self.vram_used_gb, 2),
            "vram_total_gb": round(self.vram_total_gb, 2),
            "vram_available_gb": round(self.vram_available_gb, 2),
            "total_models": self.total_models,
            "vram_model_count": self.vram_model_count,
            "disk_model_count": self.disk_model_count,
        }


class ModelCache:
    """
    LRU Model Cache with VRAM tracking.

    Tracks models in three states:
    - VRAM: Loaded and ready for inference (most expensive, limited)
    - Disk: Cached locally, fast to load (cheaper, larger capacity)
    - Downloading: Being fetched from remote storage

    Uses LRU eviction to manage VRAM when loading new models.
    """

    def __init__(
        self,
        max_vram_gb: Optional[float] = None,
        vram_safety_margin_gb: Optional[float] = None,
        model_cache_dir: Optional[str] = None,
    ):
        """
        Initialize the model cache.

        Args:
            max_vram_gb: Maximum VRAM to use. If None, auto-detects.
            vram_safety_margin_gb: VRAM to reserve for working memory.
            model_cache_dir: Directory for disk-cached models.
        """
        # Configuration from args or environment
        self._vram_safety_margin = vram_safety_margin_gb or float(
            os.getenv("WORKER_VRAM_SAFETY_MARGIN_GB", str(DEFAULT_VRAM_SAFETY_MARGIN_GB))
        )
        cache_dir = model_cache_dir or os.getenv("WORKER_MODEL_CACHE_DIR") or "/tmp/cozy/models"
        self._model_cache_dir = Path(cache_dir)
        self._model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Detect or configure VRAM
        self._total_vram_gb = self._detect_total_vram()
        if max_vram_gb is not None:
            self._max_vram_gb = max_vram_gb
        else:
            env_max = os.getenv("WORKER_MAX_VRAM_GB", "").strip()
            if env_max:
                self._max_vram_gb = float(env_max)
            else:
                # Auto: total VRAM minus safety margin
                self._max_vram_gb = max(0.0, self._total_vram_gb - self._vram_safety_margin)

        # Model tracking with LRU ordering
        # OrderedDict maintains insertion order; we move items to end on access
        self._models: OrderedDict[str, CachedModel] = OrderedDict()
        self._lock = threading.RLock()

        # Track current VRAM usage (sum of sizes of VRAM-loaded models)
        self._vram_used_gb = 0.0

        logger.info(
            f"ModelCache initialized: total_vram={self._total_vram_gb:.1f}GB, "
            f"max_usable={self._max_vram_gb:.1f}GB, safety_margin={self._vram_safety_margin:.1f}GB"
        )

    def _detect_total_vram(self) -> float:
        """Detect total GPU VRAM in GB."""
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # Get VRAM of first GPU (index 0)
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024 ** 3)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to detect VRAM: {e}")
        return 0.0

    def _get_current_vram_used(self) -> float:
        """Get current VRAM usage from PyTorch."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 3)
        except ImportError:
            pass
        except Exception:
            pass
        return self._vram_used_gb

    def _flush_memory(self) -> None:
        """Clear unused memory from GPU and run garbage collection."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error flushing GPU memory: {e}")

    # -------------------------------------------------------------------------
    # LRU Operations
    # -------------------------------------------------------------------------

    def _touch(self, model_id: str) -> None:
        """Mark a model as recently used (move to end of LRU)."""
        with self._lock:
            if model_id in self._models:
                self._models.move_to_end(model_id)
                self._models[model_id].last_accessed = time.time()

    def _get_lru_vram_models(self) -> List[str]:
        """Get VRAM-loaded models ordered by least recently used first."""
        with self._lock:
            return [
                m.model_id for m in self._models.values()
                if m.location == ModelLocation.VRAM
            ]

    def _evict_lru_for_space(self, needed_gb: float) -> float:
        """
        Evict least recently used models from VRAM until we have enough space.

        Args:
            needed_gb: Amount of VRAM space needed.

        Returns:
            Amount of space freed in GB.
        """
        freed = 0.0
        available = self._max_vram_gb - self._vram_used_gb

        if available >= needed_gb:
            return 0.0  # Already have enough space

        # Get LRU-ordered VRAM models
        lru_models = self._get_lru_vram_models()

        for model_id in lru_models:
            if available + freed >= needed_gb:
                break

            model = self._models.get(model_id)
            if model and model.location == ModelLocation.VRAM:
                evicted_size = self._unload_from_vram(model_id, keep_on_disk=True)
                freed += evicted_size
                logger.info(f"LRU evicted {model_id} from VRAM ({evicted_size:.1f}GB freed)")

        return freed

    # -------------------------------------------------------------------------
    # Model Operations
    # -------------------------------------------------------------------------

    def register_model(
        self,
        model_id: str,
        location: ModelLocation,
        size_gb: float = 0.0,
        pipeline: Any = None,
        disk_path: Optional[Path] = None,
    ) -> None:
        """
        Register a model in the cache.

        Args:
            model_id: Unique model identifier.
            location: Where the model is stored.
            size_gb: Size of the model in GB.
            pipeline: The pipeline object (if in VRAM).
            disk_path: Path on disk (if cached).
        """
        with self._lock:
            model = CachedModel(
                model_id=model_id,
                location=location,
                size_gb=size_gb,
                pipeline=pipeline,
                disk_path=disk_path,
            )
            self._models[model_id] = model
            self._models.move_to_end(model_id)  # Mark as most recently used

            if location == ModelLocation.VRAM:
                self._vram_used_gb += size_gb

            logger.info(f"Registered model {model_id} in {location.value} ({size_gb:.1f}GB)")

    def get_pipeline(self, model_id: str) -> Optional[Any]:
        """
        Get a pipeline for inference, loading from disk if needed.

        Args:
            model_id: Model to get pipeline for.

        Returns:
            The pipeline object, or None if not available.
        """
        with self._lock:
            model = self._models.get(model_id)
            if not model:
                return None

            self._touch(model_id)

            if model.location == ModelLocation.VRAM and model.pipeline:
                return model.pipeline

            # Model is on disk or downloading - caller needs to load it
            return None

    def has_model(self, model_id: str) -> bool:
        """Check if a model is in the cache (any location)."""
        with self._lock:
            return model_id in self._models

    def is_in_vram(self, model_id: str) -> bool:
        """Check if a model is loaded in VRAM."""
        with self._lock:
            model = self._models.get(model_id)
            return model is not None and model.location == ModelLocation.VRAM

    def is_on_disk(self, model_id: str) -> bool:
        """Check if a model is cached on disk."""
        with self._lock:
            model = self._models.get(model_id)
            return model is not None and model.location == ModelLocation.DISK

    def mark_loaded_to_vram(
        self,
        model_id: str,
        pipeline: Any,
        size_gb: float,
    ) -> None:
        """
        Mark a model as loaded into VRAM.

        Call this after successfully loading a model's pipeline.
        Will evict LRU models if needed to make space.

        Args:
            model_id: Model identifier.
            pipeline: The loaded pipeline object.
            size_gb: Size of the model in VRAM.
        """
        with self._lock:
            # Evict if needed
            self._evict_lru_for_space(size_gb)

            model = self._models.get(model_id)
            if model:
                # Update existing entry
                if model.location != ModelLocation.VRAM:
                    self._vram_used_gb += size_gb
                model.location = ModelLocation.VRAM
                model.pipeline = pipeline
                model.size_gb = size_gb
            else:
                # New entry
                model = CachedModel(
                    model_id=model_id,
                    location=ModelLocation.VRAM,
                    size_gb=size_gb,
                    pipeline=pipeline,
                )
                self._models[model_id] = model
                self._vram_used_gb += size_gb

            self._touch(model_id)
            logger.info(f"Model {model_id} loaded to VRAM ({size_gb:.1f}GB)")

    def mark_cached_to_disk(
        self,
        model_id: str,
        disk_path: Path,
        size_gb: float = 0.0,
    ) -> None:
        """
        Mark a model as cached on disk.

        Args:
            model_id: Model identifier.
            disk_path: Path where model is cached.
            size_gb: Size of the model on disk.
        """
        with self._lock:
            model = self._models.get(model_id)
            if model:
                if model.location == ModelLocation.VRAM:
                    self._vram_used_gb -= model.size_gb
                model.location = ModelLocation.DISK
                model.disk_path = disk_path
                model.size_gb = size_gb
                model.pipeline = None
            else:
                model = CachedModel(
                    model_id=model_id,
                    location=ModelLocation.DISK,
                    size_gb=size_gb,
                    disk_path=disk_path,
                )
                self._models[model_id] = model

            self._touch(model_id)
            logger.info(f"Model {model_id} cached to disk at {disk_path}")

    def mark_downloading(
        self,
        model_id: str,
        progress: float = 0.0,
    ) -> None:
        """
        Mark a model as currently being downloaded.

        Args:
            model_id: Model identifier.
            progress: Download progress (0.0-1.0).
        """
        with self._lock:
            model = self._models.get(model_id)
            if model:
                model.location = ModelLocation.DOWNLOADING
                model.download_progress = progress
            else:
                model = CachedModel(
                    model_id=model_id,
                    location=ModelLocation.DOWNLOADING,
                    download_progress=progress,
                )
                self._models[model_id] = model

    def update_download_progress(self, model_id: str, progress: float) -> None:
        """Update download progress for a model."""
        with self._lock:
            model = self._models.get(model_id)
            if model and model.location == ModelLocation.DOWNLOADING:
                model.download_progress = progress

    def _unload_from_vram(self, model_id: str, keep_on_disk: bool = True) -> float:
        """
        Unload a model from VRAM.

        Args:
            model_id: Model to unload.
            keep_on_disk: If True, mark as disk-cached; if False, remove entirely.

        Returns:
            Amount of VRAM freed in GB.
        """
        with self._lock:
            model = self._models.get(model_id)
            if not model or model.location != ModelLocation.VRAM:
                return 0.0

            freed = model.size_gb

            # Clean up pipeline
            if model.pipeline:
                try:
                    del model.pipeline
                except Exception as e:
                    logger.warning(f"Error deleting pipeline for {model_id}: {e}")
                model.pipeline = None

            self._vram_used_gb -= freed

            if keep_on_disk and model.disk_path:
                model.location = ModelLocation.DISK
            else:
                del self._models[model_id]

            self._flush_memory()
            logger.info(f"Unloaded {model_id} from VRAM ({freed:.1f}GB freed)")
            return freed

    def unload_model(self, model_id: str) -> bool:
        """
        Completely unload a model from the cache.

        Args:
            model_id: Model to unload.

        Returns:
            True if model was found and unloaded.
        """
        with self._lock:
            model = self._models.get(model_id)
            if not model:
                return False

            if model.location == ModelLocation.VRAM:
                self._unload_from_vram(model_id, keep_on_disk=False)
            else:
                del self._models[model_id]

            logger.info(f"Completely unloaded model {model_id}")
            return True

    def can_fit_in_vram(self, size_gb: float) -> bool:
        """Check if a model of given size can fit in VRAM (with potential eviction)."""
        with self._lock:
            available = self._max_vram_gb - self._vram_used_gb
            if available >= size_gb:
                return True

            # Check if eviction could free enough space
            evictable = sum(
                m.size_gb for m in self._models.values()
                if m.location == ModelLocation.VRAM
            )
            return available + evictable >= size_gb

    # -------------------------------------------------------------------------
    # Stats for Heartbeat
    # -------------------------------------------------------------------------

    def get_stats(self) -> ModelCacheStats:
        """
        Get cache statistics for heartbeat reporting.

        Returns:
            ModelCacheStats with current cache state.
        """
        with self._lock:
            vram_models = []
            disk_models = []
            downloading_models = []

            for model in self._models.values():
                if model.location == ModelLocation.VRAM:
                    vram_models.append(model.model_id)
                elif model.location == ModelLocation.DISK:
                    disk_models.append(model.model_id)
                elif model.location == ModelLocation.DOWNLOADING:
                    downloading_models.append(model.model_id)

            return ModelCacheStats(
                vram_models=vram_models,
                disk_models=disk_models,
                downloading_models=downloading_models,
                vram_used_gb=self._vram_used_gb,
                vram_total_gb=self._total_vram_gb,
                vram_available_gb=self._max_vram_gb - self._vram_used_gb,
                total_models=len(self._models),
                vram_model_count=len(vram_models),
                disk_model_count=len(disk_models),
            )

    def get_vram_models(self) -> List[str]:
        """Get list of models currently in VRAM."""
        with self._lock:
            return [
                m.model_id for m in self._models.values()
                if m.location == ModelLocation.VRAM
            ]

    def get_disk_models(self) -> List[str]:
        """Get list of models cached on disk."""
        with self._lock:
            return [
                m.model_id for m in self._models.values()
                if m.location == ModelLocation.DISK
            ]

    def get_all_models(self) -> List[str]:
        """Get list of all tracked models."""
        with self._lock:
            return list(self._models.keys())

    def are_models_available(self, model_ids: List[str]) -> bool:
        """
        Check if all specified models are available (VRAM or disk).

        This is used for progressive model availability - a worker can
        accept jobs as soon as the required models are downloaded,
        even if other models are still downloading.

        Args:
            model_ids: List of model IDs to check.

        Returns:
            True if all models are in VRAM or on disk (not downloading).
        """
        with self._lock:
            for model_id in model_ids:
                model = self._models.get(model_id)
                if model is None:
                    return False  # Model not tracked at all
                if model.location == ModelLocation.DOWNLOADING:
                    return False  # Still downloading
            return True

    def get_available_models(self) -> List[str]:
        """
        Get list of models that are available for inference.

        Returns models in VRAM or on disk (not downloading).
        """
        with self._lock:
            return [
                m.model_id for m in self._models.values()
                if m.location in (ModelLocation.VRAM, ModelLocation.DISK)
            ]

    def get_downloading_models(self) -> List[str]:
        """Get list of models currently being downloaded."""
        with self._lock:
            return [
                m.model_id for m in self._models.values()
                if m.location == ModelLocation.DOWNLOADING
            ]

    def get_download_progress(self, model_id: str) -> Optional[float]:
        """
        Get download progress for a model.

        Args:
            model_id: Model to check.

        Returns:
            Progress (0.0-1.0) if downloading, None otherwise.
        """
        with self._lock:
            model = self._models.get(model_id)
            if model and model.location == ModelLocation.DOWNLOADING:
                return model.download_progress
            return None

    def get_max_concurrent_downloads(self) -> int:
        """Get the maximum number of concurrent downloads allowed."""
        return int(os.getenv(
            "WORKER_MAX_CONCURRENT_DOWNLOADS",
            str(DEFAULT_MAX_CONCURRENT_DOWNLOADS)
        ))
