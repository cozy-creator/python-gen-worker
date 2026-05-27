"""
LRU Model Cache with VRAM tracking for gen-worker.

This module provides a model cache that:
- Tracks models loaded in VRAM vs cached on disk
- Implements LRU eviction when VRAM is exhausted
- Reports cache stats for orchestrator heartbeats
- Supports orchestrator-commanded load/unload operations
- Supports progressive model availability (accept jobs as models become ready)
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

from .cache_paths import tensorhub_cas_dir

logger = logging.getLogger(__name__)

DEFAULT_VRAM_SAFETY_MARGIN_GB = 3.5
# #337: leave headroom for the host so warm-tier variants don't push it to swap.
DEFAULT_RAM_SAFETY_MARGIN_GB = 8.0
DEFAULT_MAX_CONCURRENT_DOWNLOADS = 2


class ModelLocation(str, Enum):
    """Where a model is currently stored.

    Issue #337 inserts a CPU-RAM warm tier between VRAM and disk so a swappable
    variant (e.g. an SDXL UNet) can be demoted off the GPU but kept in host RAM
    for a fast PCIe swap-in (~0.1-0.5s for ~2.5GB) instead of re-reading +
    re-loading from disk. The residency ladder is:

        VRAM  (GPU, active + hot)
          ^  demote .to("cpu")  |  promote .to("cuda")
        CPU   (host RAM, warm)
          ^  drop bytes         |  re-load from disk
        DISK  (cached, re-loadable)
          ^  evict file         |  re-download from source
        DOWNLOADING / (absent)
    """
    VRAM = "vram"       # Loaded in GPU VRAM, ready for inference
    CPU = "cpu"         # Warm in host RAM, fast PCIe swap-in to VRAM (#337)
    DISK = "disk"       # Cached on disk, needs loading to use
    DOWNLOADING = "downloading"  # Currently being downloaded


@dataclass
class CachedModel:
    """Metadata about a cached model."""
    model_id: str
    location: ModelLocation
    size_gb: float = 0.0
    last_accessed: float = field(default_factory=time.time)
    pipeline: Any = None  # The actual pipeline object (when in VRAM or CPU)
    disk_path: Optional[Path] = None  # Path on disk (when cached)
    download_progress: float = 0.0  # 0.0-1.0 progress for downloading models
    # #337: pinned models are never evicted from VRAM (the shared base — text
    # encoders + VAE — that every variant pipeline references by pointer).
    pinned: bool = False


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
        self._vram_safety_margin = vram_safety_margin_gb or DEFAULT_VRAM_SAFETY_MARGIN_GB
        cache_dir = model_cache_dir or str(tensorhub_cas_dir())
        self._model_cache_dir = Path(os.path.expanduser(cache_dir))
        self._model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Detect or configure VRAM
        self._total_vram_gb = self._detect_total_vram()
        if max_vram_gb is not None:
            self._max_vram_gb = max_vram_gb
        else:
            # Auto: total VRAM minus safety margin
            self._max_vram_gb = max(0.0, self._total_vram_gb - self._vram_safety_margin)

        # #337 CPU-RAM warm tier auto-sizing: budget = host RAM minus a safety
        # margin so demoted variants stay warm for a fast swap-in without
        # pushing the host into swap. Auto-detected from psutil when available.
        self._ram_safety_margin_gb = DEFAULT_RAM_SAFETY_MARGIN_GB
        self._total_ram_gb = self._detect_total_ram()
        self._max_ram_gb = max(0.0, self._total_ram_gb - self._ram_safety_margin_gb)
        self._ram_used_gb = 0.0

        # Model tracking with LRU ordering
        # OrderedDict maintains insertion order; we move items to end on access
        self._models: OrderedDict[str, CachedModel] = OrderedDict()
        self._lock = threading.RLock()

        # Track current VRAM usage (sum of sizes of VRAM-loaded models)
        self._vram_used_gb = 0.0

        logger.info(
            f"ModelCache initialized: total_vram={self._total_vram_gb:.1f}GB, "
            f"max_usable={self._max_vram_gb:.1f}GB, safety_margin={self._vram_safety_margin:.1f}GB, "
            f"ram_tier_budget={self._max_ram_gb:.1f}GB (total_ram={self._total_ram_gb:.1f}GB)"
        )

    def _detect_total_ram(self) -> float:
        """Detect total host RAM in GB (psutil; 0.0 when unavailable)."""
        try:
            import psutil
            return float(psutil.virtual_memory().total) / (1024 ** 3)
        except Exception:
            return 0.0

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
        """VRAM-loaded models, LRU first. #337: PINNED models (the shared base)
        are excluded — they are never eviction candidates."""
        with self._lock:
            return [
                m.model_id for m in self._models.values()
                if m.location == ModelLocation.VRAM and not m.pinned
            ]

    def _get_lru_cpu_models(self) -> List[str]:
        """Warm CPU-RAM-tier models, LRU first (#337). Pinned excluded."""
        with self._lock:
            return [
                m.model_id for m in self._models.values()
                if m.location == ModelLocation.CPU and not m.pinned
            ]

    def _evict_lru_for_space(self, needed_gb: float) -> float:
        """
        Evict least recently used models from VRAM until we have enough space.

        #337: eviction now DEMOTES the LRU variant to the CPU-RAM warm tier
        (``.to("cpu")``) rather than dropping straight to disk, so a re-selected
        variant swaps back over PCIe instead of re-loading from disk. PINNED
        models (the shared base) are never evicted.

        Args:
            needed_gb: Amount of VRAM space needed.

        Returns:
            Amount of VRAM freed in GB.
        """
        freed = 0.0
        available = self._max_vram_gb - max(self._vram_used_gb, self._get_current_vram_used())

        if available >= needed_gb:
            return 0.0  # Already have enough space

        # Get LRU-ordered VRAM models (pinned excluded)
        lru_models = self._get_lru_vram_models()

        for model_id in lru_models:
            if available + freed >= needed_gb:
                break

            model = self._models.get(model_id)
            if model and model.location == ModelLocation.VRAM and not model.pinned:
                evicted_size = self._demote_to_cpu(model_id)
                freed += evicted_size
                logger.info(f"LRU demoted {model_id} VRAM->CPU ({evicted_size:.1f}GB freed)")

        return freed

    # -------------------------------------------------------------------------
    # Model Operations
    # -------------------------------------------------------------------------

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

    def is_in_vram(self, model_id: str) -> bool:
        """Check if a model is loaded in VRAM."""
        with self._lock:
            model = self._models.get(model_id)
            return model is not None and model.location == ModelLocation.VRAM

    def mark_loaded_to_vram(
        self,
        model_id: str,
        pipeline: Any,
        size_gb: float,
        *,
        pinned: bool = False,
    ) -> None:
        """
        Mark a model as loaded into VRAM.

        Call this after successfully loading a model's pipeline.
        Will evict LRU models if needed to make space.

        Args:
            model_id: Model identifier.
            pipeline: The loaded pipeline object.
            size_gb: Size of the model in VRAM.
            pinned: #337 — when True the model is NEVER evicted (the shared
                base: text encoders + VAE that every variant references).
        """
        with self._lock:
            # Evict if needed
            self._evict_lru_for_space(size_gb)

            model = self._models.get(model_id)
            if model:
                # Update existing entry
                if model.location == ModelLocation.CPU:
                    self._ram_used_gb = max(0.0, self._ram_used_gb - model.size_gb)
                if model.location != ModelLocation.VRAM:
                    self._vram_used_gb += size_gb
                model.location = ModelLocation.VRAM
                model.pipeline = pipeline
                model.size_gb = size_gb
                if pinned:
                    model.pinned = True
            else:
                # New entry
                model = CachedModel(
                    model_id=model_id,
                    location=ModelLocation.VRAM,
                    size_gb=size_gb,
                    pipeline=pipeline,
                    pinned=pinned,
                )
                self._models[model_id] = model
                self._vram_used_gb += size_gb

            self._touch(model_id)
            logger.info(
                f"Model {model_id} loaded to VRAM ({size_gb:.1f}GB)"
                + (" [PINNED]" if model.pinned else "")
            )

    def pin(self, model_id: str) -> None:
        """Mark a model as pinned — never evicted from VRAM (#337 shared base)."""
        with self._lock:
            model = self._models.get(model_id)
            if model is not None:
                model.pinned = True

    def is_pinned(self, model_id: str) -> bool:
        with self._lock:
            model = self._models.get(model_id)
            return bool(model is not None and model.pinned)

    # -------------------------------------------------------------------------
    # #337 CPU-RAM warm tier transitions (VRAM <-> CPU). All callers hold the
    # GPU mutex (worker `_gpu_semaphore`), so demote/promote never race a load
    # or an in-flight inference on the same device.
    # -------------------------------------------------------------------------

    def _move_pipeline_to_device(self, pipeline: Any, device: str) -> None:
        """Best-effort ``pipeline.to(device)`` for a diffusers pipeline / module."""
        if pipeline is None:
            return
        try:
            to = getattr(pipeline, "to", None)
            if callable(to):
                to(device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ModelCache: pipeline.to(%s) failed: %s", device, exc)

    def _demote_to_cpu(self, model_id: str) -> float:
        """Demote a VRAM model to the CPU-RAM warm tier (``.to("cpu")``).

        Frees VRAM but keeps the object in host RAM for a fast PCIe swap-in.
        Evicts LRU CPU-tier models to disk first if the RAM budget is tight.
        Returns the VRAM (GB) freed.
        """
        with self._lock:
            model = self._models.get(model_id)
            if not model or model.location != ModelLocation.VRAM or model.pinned:
                return 0.0
            freed = model.size_gb
            # Make room in the RAM tier before parking the object there.
            self._evict_lru_cpu_for_space(model.size_gb, exclude=model_id)
            self._move_pipeline_to_device(model.pipeline, "cpu")
            self._vram_used_gb = max(0.0, self._vram_used_gb - freed)
            model.location = ModelLocation.CPU
            self._ram_used_gb += model.size_gb
            self._flush_memory()
            self._touch(model_id)
            return freed

    def _promote_from_cpu(self, model_id: str, device: str = "cuda") -> bool:
        """Promote a warm CPU-tier model back into VRAM (``.to("cuda")``).

        Evicts LRU VRAM models (to CPU) first to make room. Returns True when
        the model is resident in VRAM afterward.
        """
        with self._lock:
            model = self._models.get(model_id)
            if not model or model.pipeline is None:
                return False
            if model.location == ModelLocation.VRAM:
                self._touch(model_id)
                return True
            if model.location != ModelLocation.CPU:
                return False
            self._evict_lru_for_space(model.size_gb)
            self._move_pipeline_to_device(model.pipeline, device)
            self._ram_used_gb = max(0.0, self._ram_used_gb - model.size_gb)
            model.location = ModelLocation.VRAM
            self._vram_used_gb += model.size_gb
            self._touch(model_id)
            return True

    def _evict_lru_cpu_for_space(self, needed_gb: float, *, exclude: str = "") -> float:
        """Drop LRU CPU-tier models (keep disk file) until the RAM budget fits."""
        if self._max_ram_gb <= 0.0:
            return 0.0
        freed = 0.0
        available = self._max_ram_gb - self._ram_used_gb
        if available >= needed_gb:
            return 0.0
        for model_id in self._get_lru_cpu_models():
            if model_id == exclude:
                continue
            if available + freed >= needed_gb:
                break
            model = self._models.get(model_id)
            if model and model.location == ModelLocation.CPU:
                freed += model.size_gb
                self._ram_used_gb = max(0.0, self._ram_used_gb - model.size_gb)
                if model.pipeline is not None:
                    try:
                        del model.pipeline
                    except Exception:
                        pass
                    model.pipeline = None
                if model.disk_path:
                    model.location = ModelLocation.DISK
                else:
                    del self._models[model_id]
                logger.info(f"ModelCache: dropped warm CPU model {model_id} ({model.size_gb:.1f}GB)")
        self._flush_memory()
        return freed

    def residency_tier(self, model_id: str) -> str:
        """Return the residency tier string for a model (#337 availability).

        One of ``"VRAM"``, ``"RAM"``, ``"DISK"``, ``"DOWNLOADING"``,
        ``"ABSENT"`` — the vocabulary the orchestrator routes on.
        """
        with self._lock:
            model = self._models.get(model_id)
            if model is None:
                return "ABSENT"
            return {
                ModelLocation.VRAM: "VRAM",
                ModelLocation.CPU: "RAM",
                ModelLocation.DISK: "DISK",
                ModelLocation.DOWNLOADING: "DOWNLOADING",
            }.get(model.location, "ABSENT")

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
                elif model.location == ModelLocation.CPU:
                    self._ram_used_gb = max(0.0, self._ram_used_gb - model.size_gb)
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

    def evict_lru_vram_until_count(self, max_vram_models: int) -> int:
        """
        Evict least-recently-used VRAM models until we have <= max_vram_models.

        Returns the number of models evicted.
        """
        try:
            max_vram_models = int(max_vram_models)
        except Exception:
            max_vram_models = 0
        if max_vram_models < 0:
            max_vram_models = 0

        evicted = 0
        # Evict outside the lock (flush can be slow).
        while True:
            with self._lock:
                vram = [m.model_id for m in self._models.values() if m.location == ModelLocation.VRAM]
                if len(vram) <= max_vram_models:
                    break
                victim = vram[0]
            freed = self._unload_from_vram(victim, keep_on_disk=True)
            if freed <= 0:
                # Avoid infinite loops if unload fails for some reason.
                break
            evicted += 1

        return evicted

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
                elif model.location == ModelLocation.CPU:
                    # #337 warm CPU-RAM tier: reported as disk-resident to the
                    # legacy stats consumer (it can be served quickly). The
                    # fine-grained tier is carried separately by residency_tier().
                    disk_models.append(model.model_id)
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
        """Get list of models cached on disk (incl. the warm CPU tier — both
        are fast to serve relative to a cold download)."""
        with self._lock:
            return [
                m.model_id for m in self._models.values()
                if m.location in (ModelLocation.DISK, ModelLocation.CPU)
            ]

    def get_cpu_models(self) -> List[str]:
        """Get list of models warm in the CPU-RAM tier (#337)."""
        with self._lock:
            return [
                m.model_id for m in self._models.values()
                if m.location == ModelLocation.CPU
            ]

    def get_residency_map(self) -> Dict[str, str]:
        """Snapshot of ``{model_id: tier}`` for tier-aware availability (#337)."""
        with self._lock:
            return {m.model_id: self.residency_tier(m.model_id) for m in self._models.values()}

    def get_max_concurrent_downloads(self) -> int:
        """Get the maximum number of concurrent downloads allowed."""
        return DEFAULT_MAX_CONCURRENT_DOWNLOADS

    @property
    def max_vram_gb(self) -> float:
        """Maximum VRAM (in GB) the cache is willing to occupy.

        Public accessor for the auto-detected (or operator-configured) VRAM
        budget after subtracting the safety margin. Used by the prefetch
        path to decide whether the next ref fits eagerly into VRAM
        (gen-worker #21).
        """
        with self._lock:
            return float(self._max_vram_gb)

    @property
    def vram_used_gb(self) -> float:
        """Sum of declared sizes for models currently in VRAM (GB)."""
        with self._lock:
            return float(self._vram_used_gb)

    @property
    def vram_free_gb(self) -> float:
        """Remaining VRAM (in GB) inside the safety-margin budget.

        Computed as ``max_vram_gb - vram_used_gb``. Used by the prefetch
        eager-load gate (gen-worker #21): callers compare this to the
        estimated VRAM footprint of the next ref and only call
        ``load_model_into_vram`` if there's headroom.
        """
        with self._lock:
            return max(0.0, float(self._max_vram_gb) - float(self._vram_used_gb))
