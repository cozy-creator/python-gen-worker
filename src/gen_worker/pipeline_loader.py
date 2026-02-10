"""
Diffusers Pipeline Loader

Loads diffusers pipelines with support for:
- FlashPack format (fastest, 2-4s)
- Safetensors format (fast, 8-12s)
- Single-file checkpoints (slower)
- Component reference resolution (_cozy_ref)
- Automatic dtype selection (bfloat16 for Flux, float16 for others)
- VAE tiling/slicing for memory efficiency
- Conditional optimizations based on VRAM availability
- Warm-up inference to pre-compile kernels
- Model downloading from Cozy Hub
- Local NVMe cache for NFS models
- Thread-safe concurrent inference via get_for_inference()

Thread Safety
-------------
Diffusers schedulers maintain internal state (timesteps, sigmas, step_index)
that gets corrupted when multiple threads use the same scheduler simultaneously,
causing 'IndexError: index N is out of bounds for dimension 0 with size N'.

The solution is to create a fresh scheduler instance for each concurrent request
while sharing the heavy pipeline components (UNet ~10GB, VAE ~300MB, encoders ~1GB).
Only the scheduler (~few KB) is recreated per-request.

Use get_for_inference() instead of get() for concurrent workloads:

    pipeline = loader.get_for_inference(model_id)  # Thread-safe
    result = pipeline(prompt=..., ...)

References:
- https://huggingface.co/docs/diffusers/using-diffusers/create_a_server
- https://github.com/huggingface/diffusers/issues/3672
"""

import asyncio
import gc
import hashlib
import importlib
import inspect
import json
import logging
import os
import random
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Error Classes
# =============================================================================


class PipelineLoaderError(Exception):
    """Base exception for pipeline loader errors."""

    pass


class ModelNotFoundError(PipelineLoaderError):
    """Model not found in local cache or remote hub."""

    def __init__(self, model_id: str, path: Optional[Path] = None):
        self.model_id = model_id
        self.path = path
        msg = f"Model not found: {model_id}"
        if path:
            msg += f" (checked {path})"
        super().__init__(msg)


class ModelDownloadError(PipelineLoaderError):
    """Failed to download model from Cozy Hub."""

    def __init__(self, model_id: str, reason: str, retryable: bool = True):
        self.model_id = model_id
        self.reason = reason
        self.retryable = retryable
        super().__init__(f"Failed to download {model_id}: {reason}")


class IncompatibleFormatError(PipelineLoaderError):
    """Model format is incompatible with requested pipeline class."""

    def __init__(self, model_id: str, expected_format: str, actual_format: str):
        self.model_id = model_id
        self.expected_format = expected_format
        self.actual_format = actual_format
        super().__init__(
            f"Incompatible format for {model_id}: expected {expected_format}, got {actual_format}"
        )


class CudaOutOfMemoryError(PipelineLoaderError):
    """CUDA out of memory during model loading."""

    def __init__(self, model_id: str, required_gb: float, available_gb: float):
        self.model_id = model_id
        self.required_gb = required_gb
        self.available_gb = available_gb
        super().__init__(
            f"CUDA OOM loading {model_id}: requires {required_gb:.1f}GB, only {available_gb:.1f}GB available"
        )


class ComponentMissingError(PipelineLoaderError):
    """Required component is missing from model."""

    def __init__(self, model_id: str, component: str):
        self.model_id = model_id
        self.component = component
        super().__init__(f"Missing component '{component}' in model {model_id}")

# Constants
VRAM_SAFETY_MARGIN_GB = 3.5
DEFAULT_MAX_VRAM_BUFFER_GB = 2.0

# FlashPack constants
FLASHPACK_SUFFIX = ".flashpack"
FLASHPACK_COMPONENTS = ["unet", "vae", "text_encoder", "text_encoder_2", "transformer"]

# Pipeline component definitions by pipeline type
MODEL_COMPONENTS: Dict[str, List[str]] = {
    "FluxPipeline": [
        "vae", "transformer", "text_encoder", "text_encoder_2",
        "scheduler", "tokenizer", "tokenizer_2",
    ],
    "FluxInpaintPipeline": [
        "vae", "transformer", "text_encoder", "text_encoder_2",
        "scheduler", "tokenizer", "tokenizer_2",
    ],
    "StableDiffusionXLPipeline": [
        "vae", "unet", "text_encoder", "text_encoder_2",
        "scheduler", "tokenizer", "tokenizer_2",
    ],
    "StableDiffusionXLImg2ImgPipeline": [
        "vae", "unet", "text_encoder", "text_encoder_2",
        "scheduler", "tokenizer", "tokenizer_2",
    ],
    "StableDiffusionPipeline": [
        "vae", "unet", "text_encoder", "scheduler", "tokenizer",
    ],
    "StableDiffusion3Pipeline": [
        "vae", "transformer", "text_encoder", "text_encoder_2", "text_encoder_3",
        "scheduler", "tokenizer", "tokenizer_2", "tokenizer_3",
    ],
}


# =============================================================================
# LocalModelCache - NFS to NVMe optimization
# =============================================================================


class LocalModelCache:
    """
    Caches models from NFS/shared storage to local NVMe for faster loading.

    When loading from NFS, this copies model files to local NVMe first.
    FlashPack files are prioritized for copying as they give the most benefit.
    Supports background prefetching of models that might be needed soon.
    """

    def __init__(
        self,
        local_cache_dir: str,
        max_cache_size_gb: float = 100.0,
    ):
        """
        Initialize local model cache.

        Args:
            local_cache_dir: Local NVMe directory for caching
            max_cache_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(local_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_gb = max_cache_size_gb
        # Multi-GB directory copies are blocking; guard to prevent duplicate work.
        # Async callers will run copy operations via asyncio.to_thread.
        self._cache_lock = threading.Lock()
        self._prefetch_tasks: Dict[str, asyncio.Task] = {}

    def _get_cache_path(self, model_id: str) -> Path:
        """Get the local cache path for a model."""
        # Use hash to avoid path issues with slashes in model IDs
        safe_name = hashlib.sha256(model_id.encode()).hexdigest()[:16]
        # Keep the model name readable
        readable_name = model_id.replace("/", "--")[:64]
        return self.cache_dir / f"{readable_name}_{safe_name}"

    def _get_cache_size_gb(self) -> float:
        """Get current cache size in GB."""
        total = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total / (1024**3)

    def _evict_lru(self, needed_gb: float) -> None:
        """Evict least recently used models to make space."""
        if needed_gb <= 0:
            return

        # Sort by access time (oldest first)
        cached = []
        for path in self.cache_dir.iterdir():
            if path.is_dir():
                try:
                    stat = path.stat()
                    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    cached.append((path, stat.st_atime, size / (1024**3)))
                except OSError:
                    continue

        cached.sort(key=lambda x: x[1])  # Sort by access time

        freed = 0.0
        for path, _, size_gb in cached:
            if freed >= needed_gb:
                break
            logger.info(f"Evicting {path.name} ({size_gb:.1f}GB) from local cache")
            shutil.rmtree(path, ignore_errors=True)
            freed += size_gb

    def is_cached(self, model_id: str) -> bool:
        """Check if model is in local cache."""
        cache_path = self._get_cache_path(model_id)
        return cache_path.exists()

    def get_cached_path(self, model_id: str) -> Optional[Path]:
        """Get local cache path if model is cached."""
        cache_path = self._get_cache_path(model_id)
        if cache_path.exists():
            # Update access time for LRU
            cache_path.touch()
            return cache_path
        return None

    def cache_model_blocking(
        self,
        model_id: str,
        source_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        """
        Blocking copy model from source to local cache.

        - This is safe for sync callers (e.g. worker injection paths).
        - Async callers should use cache_model(), which delegates to a thread.
        """
        cache_path = self._get_cache_path(model_id)
        if cache_path.exists():
            try:
                cache_path.touch()
            except Exception:
                pass
            return cache_path

        with self._cache_lock:
            if cache_path.exists():
                try:
                    cache_path.touch()
                except Exception:
                    pass
                return cache_path

            # Estimate size.
            source_size_gb = sum(
                f.stat().st_size for f in source_path.rglob("*") if f.is_file()
            ) / (1024**3)

            # Evict if needed.
            current_size = self._get_cache_size_gb()
            if current_size + source_size_gb > self.max_cache_size_gb:
                needed = current_size + source_size_gb - self.max_cache_size_gb + 1.0
                self._evict_lru(needed)

            logger.info(f"Caching {model_id} ({source_size_gb:.1f}GB) to local cache")

            # Create temp directory for atomic copy.
            temp_path = cache_path.with_suffix(".tmp")
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)

            _ = self._copy_model_prioritized_blocking(source_path, temp_path, progress_callback)

            # Atomic rename.
            temp_path.rename(cache_path)
            logger.info(f"Cached {model_id} to {cache_path}")
            return cache_path

    def cache_model_blocking_with_stats(
        self,
        model_id: str,
        source_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> tuple[Path, Optional[int]]:
        """
        Like cache_model_blocking(), but also returns best-effort bytes_copied.

        If the model was already cached, bytes_copied is None.
        """
        cache_path = self._get_cache_path(model_id)
        if cache_path.exists():
            try:
                cache_path.touch()
            except Exception:
                pass
            return cache_path, None

        with self._cache_lock:
            if cache_path.exists():
                try:
                    cache_path.touch()
                except Exception:
                    pass
                return cache_path, None

            source_size_gb = sum(
                f.stat().st_size for f in source_path.rglob("*") if f.is_file()
            ) / (1024**3)
            current_size = self._get_cache_size_gb()
            if current_size + source_size_gb > self.max_cache_size_gb:
                needed = current_size + source_size_gb - self.max_cache_size_gb + 1.0
                self._evict_lru(needed)

            logger.info(f"Caching {model_id} ({source_size_gb:.1f}GB) to local cache")

            temp_path = cache_path.with_suffix(".tmp")
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)

            bytes_copied = self._copy_model_prioritized_blocking(source_path, temp_path, progress_callback)
            temp_path.rename(cache_path)
            logger.info(f"Cached {model_id} to {cache_path}")
            return cache_path, bytes_copied

    async def cache_model(
        self,
        model_id: str,
        source_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        return await asyncio.to_thread(self.cache_model_blocking, model_id, source_path, progress_callback)

    def _copy_model_prioritized_blocking(
        self,
        source: Path,
        dest: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> int:
        """Copy model files with FlashPack files first (blocking). Returns bytes copied."""
        dest.mkdir(parents=True, exist_ok=True)

        # Collect all files and sort by priority.
        all_files = list(source.rglob("*"))
        files_to_copy = [f for f in all_files if f.is_file()]

        # Sort: .flashpack files first, then safetensors, then rest.
        def priority(f: Path) -> int:
            if f.suffix == ".flashpack":
                return 0
            if f.suffix == ".safetensors":
                return 1
            return 2

        files_to_copy.sort(key=priority)

        total_size = sum(f.stat().st_size for f in files_to_copy)
        copied = 0

        for file in files_to_copy:
            rel = file.relative_to(source)
            dst = dest / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dst)
            copied += file.stat().st_size

            if progress_callback and total_size > 0:
                progress_callback("caching", copied / total_size * 100)

        # Copy empty directories.
        for d in all_files:
            if d.is_dir():
                rel = d.relative_to(source)
                (dest / rel).mkdir(parents=True, exist_ok=True)
        return int(total_size)

    def start_prefetch(self, model_id: str, source_path: Path) -> None:
        """Start background prefetch of a model."""
        if model_id in self._prefetch_tasks:
            return
        if self.is_cached(model_id):
            return

        async def prefetch() -> None:
            try:
                await self.cache_model(model_id, source_path)
            except Exception as e:
                logger.warning(f"Prefetch failed for {model_id}: {e}")
            finally:
                self._prefetch_tasks.pop(model_id, None)

        self._prefetch_tasks[model_id] = asyncio.create_task(prefetch())
        logger.debug(f"Started prefetch for {model_id}")

    async def wait_for_prefetch(self, model_id: str, timeout: float = 60.0) -> bool:
        """Wait for a prefetch to complete."""
        task = self._prefetch_tasks.get(model_id)
        if task is None:
            return self.is_cached(model_id)

        try:
            await asyncio.wait_for(task, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Prefetch timeout for {model_id}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cached_models = []
        total_size = 0
        for path in self.cache_dir.iterdir():
            if path.is_dir() and not path.name.endswith(".tmp"):
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                cached_models.append(path.name)
                total_size += size

        return {
            "cached_models": cached_models,
            "cache_size_gb": total_size / (1024**3),
            "max_cache_size_gb": self.max_cache_size_gb,
            "prefetching": list(self._prefetch_tasks.keys()),
        }


@dataclass
class LoadedPipeline:
    """Container for a loaded pipeline with metadata."""
    pipeline: Any
    model_id: str
    pipeline_class: str
    dtype: str
    size_gb: float
    load_format: str  # "flashpack", "safetensors", "single_file"
    components: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Configuration for loading a pipeline."""
    model_path: str
    pipeline_class: Optional[str] = None
    custom_pipeline: Optional[str] = None
    dtype: Optional[str] = None  # "float16", "bfloat16", "float32"
    device: str = "cuda"
    enable_vae_tiling: bool = True
    enable_vae_slicing: bool = True
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    scheduler_class: Optional[str] = None
    warmup_steps: int = 4
    variant: Optional[str] = None  # "fp16", etc.


def _check_torch_available() -> bool:
    """Check if torch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _check_diffusers_available() -> bool:
    """Check if diffusers is available."""
    try:
        import diffusers
        return True
    except ImportError:
        return False


def _check_flashpack_available() -> bool:
    """Check if flashpack is available."""
    try:
        from flashpack import assign_from_file
        return True
    except ImportError:
        return False


def get_torch_dtype(dtype_str: Optional[str], model_id: str) -> Any:
    """
    Get torch dtype based on string or model type.

    Args:
        dtype_str: Explicit dtype string ("float16", "bfloat16", "float32")
        model_id: Model identifier for automatic selection

    Returns:
        torch.dtype
    """
    import torch

    if dtype_str:
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return dtype_map.get(dtype_str.lower(), torch.float16)

    # Automatic selection based on model type
    model_lower = model_id.lower()
    if "flux" in model_lower:
        return torch.bfloat16
    elif "sd3" in model_lower or "stable-diffusion-3" in model_lower:
        return torch.bfloat16
    else:
        return torch.float16


def detect_diffusers_variant(model_path: Path) -> Optional[str]:
    """
    Detect a diffusers `variant=` value ("bf16", "fp8", "fp16") from files on disk.

    Many diffusers repos store weights as:
      - `unet/diffusion_pytorch_model.fp16.safetensors`
      - `text_encoder/model.fp16.safetensors`
      - sharded: `*.fp16.safetensors.index.json` + `*.fp16-00001-of-0000N.safetensors`
    """
    candidates = ["bf16", "fp8", "fp16"]
    for p in model_path.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        for v in candidates:
            if f".{v}." in name and name.endswith((".safetensors", ".json")):
                return v
    return None


def get_pipeline_class(
    class_name: Union[str, Tuple[str, str], None],
    model_path: str,
) -> Tuple[Type, Optional[str]]:
    """
    Get the appropriate pipeline class.

    Args:
        class_name: Pipeline class name, tuple of (package, class), or None for auto-detect
        model_path: Path to model for auto-detection

    Returns:
        Tuple of (Pipeline class, custom_pipeline name or None)
    """
    from diffusers import DiffusionPipeline

    custom_pipeline: Optional[str] = None

    # Cozy pipeline YAML is the authoritative equivalent of model_index.json.
    # Prefer cozy.pipeline.lock.yaml when present; fall back to cozy.pipeline.yaml.
    if class_name is None:
        try:
            from .cozy_pipeline_spec import (
                cozy_custom_pipeline_arg,
                ensure_diffusers_model_index_json,
                load_cozy_pipeline_spec,
            )

            root = Path(model_path)
            spec = load_cozy_pipeline_spec(root)
            if spec is not None:
                _ = ensure_diffusers_model_index_json(root)
                if spec.pipe_class:
                    class_name = spec.pipe_class
                try:
                    custom_pipeline = cozy_custom_pipeline_arg(root, spec)
                except Exception:
                    custom_pipeline = None
        except Exception:
            pass

    # Auto-detect from model_index.json if not specified
    if class_name is None:
        model_index_path = Path(model_path) / "model_index.json"
        if model_index_path.exists():
            with open(model_index_path) as f:
                model_index = json.load(f)
                class_name = model_index.get("_class_name")

    if class_name is None:
        return (DiffusionPipeline, custom_pipeline)

    # Handle tuple format (package, class)
    if isinstance(class_name, (list, tuple)):
        package, cls = class_name
        module = importlib.import_module(package)
        return (getattr(module, cls), custom_pipeline)

    # Try loading as a diffusers class
    try:
        pipeline_class = getattr(importlib.import_module("diffusers"), class_name)
        if not issubclass(pipeline_class, DiffusionPipeline):
            raise TypeError(f"{class_name} does not inherit from DiffusionPipeline")
        return (pipeline_class, custom_pipeline)
    except (ImportError, AttributeError):
        # Assume it's a custom pipeline name
        return (DiffusionPipeline, custom_pipeline or class_name)


def get_scheduler_class(scheduler_name: str) -> Type:
    """
    Dynamically import a scheduler class from diffusers.

    Args:
        scheduler_name: Name of the scheduler class

    Returns:
        Scheduler class
    """
    try:
        return getattr(importlib.import_module("diffusers"), scheduler_name)
    except AttributeError:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def resolve_cozy_refs(model_path: Path, base_models_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Resolve _cozy_ref entries in model_index.json.

    Some models reference components from other models using _cozy_ref.
    This resolves those references to actual paths.

    Args:
        model_path: Path to the model directory
        base_models_dir: Base directory for resolving references

    Returns:
        Dict mapping component names to resolved paths
    """
    model_index_path = model_path / "model_index.json"
    if not model_index_path.exists():
        return {}

    with open(model_index_path) as f:
        model_index = json.load(f)

    resolved = {}
    for key, value in model_index.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list) and len(value) == 2:
            # Standard diffusers format: [module_path, class_name]
            continue
        if isinstance(value, dict) and "_cozy_ref" in value:
            ref = value["_cozy_ref"]
            if base_models_dir:
                ref_path = base_models_dir / ref
                if ref_path.exists():
                    resolved[key] = ref_path
                    logger.info(f"Resolved _cozy_ref for {key}: {ref_path}")

    return resolved


def missing_component_overrides_for_from_pretrained(pipeline_class: Type, model_path: Path) -> Dict[str, Any]:
    """
    If a pipeline's `model_index.json` references optional components that are not
    present on disk (common when we intentionally skip safety_checker /
    feature_extractor), diffusers will fail unless we pass explicit overrides.

    We only pass overrides for parameters that exist on the pipeline __init__.
    """
    kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(pipeline_class.__init__)
        params = set(sig.parameters.keys())
    except Exception:
        return kwargs

    if "safety_checker" in params and not (model_path / "safety_checker").exists():
        kwargs["safety_checker"] = None

    if "feature_extractor" in params and not (model_path / "feature_extractor").exists():
        kwargs["feature_extractor"] = None

    # Some pipelines expose a boolean flag.
    if "requires_safety_checker" in params and "safety_checker" in kwargs:
        kwargs["requires_safety_checker"] = False

    return kwargs


def estimate_model_size_gb(model_path: Path) -> float:
    """
    Estimate model size in GB based on file sizes.

    Args:
        model_path: Path to model directory

    Returns:
        Estimated size in GB
    """
    total_bytes = 0
    weight_extensions = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".flashpack"}

    for file in model_path.rglob("*"):
        if file.is_file() and file.suffix in weight_extensions:
            total_bytes += file.stat().st_size

    return total_bytes / (1024 ** 3)


def get_available_vram_gb() -> float:
    """Get available VRAM in GB."""
    try:
        import torch
    except Exception:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0

    try:
        free_mem = torch.cuda.mem_get_info()[0]
        return free_mem / (1024 ** 3)
    except Exception:
        return 0.0


def get_total_vram_gb() -> float:
    """Get total VRAM in GB."""
    try:
        import torch
    except Exception:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0

    try:
        total_mem = torch.cuda.get_device_properties(0).total_memory
        return total_mem / (1024 ** 3)
    except Exception:
        return 0.0


def flush_memory() -> None:
    """Flush GPU memory and run garbage collection."""
    gc.collect()
    if _check_torch_available():
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class PipelineLoader:
    """
    Loads diffusers pipelines with optimizations and format priority.

    Loading priority:
    1. FlashPack (fastest, ~2-4s)
    2. Safetensors (fast, ~8-12s)
    3. Single-file checkpoint (slower)
    """

    def __init__(
        self,
        models_dir: Optional[str] = None,
        local_cache_dir: Optional[str] = None,
        max_vram_gb: Optional[float] = None,
        vram_safety_margin_gb: float = VRAM_SAFETY_MARGIN_GB,
        cozy_hub_url: Optional[str] = None,
        cozy_hub_token: Optional[str] = None,
        downloader: Optional[Any] = None,
    ):
        """
        Initialize the pipeline loader.

        Args:
            models_dir: Base directory for models
            local_cache_dir: Local cache directory for NFS->NVMe optimization
            max_vram_gb: Maximum VRAM to use (auto-detect if None)
            vram_safety_margin_gb: VRAM reserved for working memory
            cozy_hub_url: Base URL for Cozy Hub API
            cozy_hub_token: Authentication token for Cozy Hub
        """
        if models_dir is None:
            # Standard shared disk cache root (Runpod: /workspace/..., local dev: /tmp/...).
            models_dir = os.environ.get("WORKER_MODEL_CACHE_DIR", "/tmp/cozy/models")
        self.models_dir = Path(models_dir) if models_dir else None
        self.local_cache_dir: Optional[Path] = None
        self.vram_safety_margin_gb = vram_safety_margin_gb

        # Cozy Hub configuration
        self._cozy_hub_url = cozy_hub_url or os.environ.get("COZY_HUB_URL", "")
        self._cozy_hub_token = cozy_hub_token or os.environ.get("COZY_HUB_TOKEN", "")
        self._downloader = downloader

        # Auto-detect VRAM
        if max_vram_gb is not None:
            self._max_vram_gb = max_vram_gb
        else:
            self._max_vram_gb = get_total_vram_gb() - vram_safety_margin_gb

        self._flashpack_available = _check_flashpack_available()
        if self._flashpack_available:
            logger.info("FlashPack loading enabled")

        # Track loaded pipelines for memory management
        self._loaded_pipelines: Dict[str, LoadedPipeline] = {}

        # Local NVMe cache for NFS optimization
        self._local_cache: Optional[LocalModelCache] = None
        if local_cache_dir is None:
            # Standard env for local (non-NFS) cache. Empty disables.
            local_cache_dir = os.environ.get("WORKER_LOCAL_MODEL_CACHE_DIR", "/tmp/cozy/local-model-cache").strip() or None
        if local_cache_dir:
            # Validate local cache isn't itself on NFS; otherwise localization is pointless.
            try:
                from .mount_backend import mount_backend_for_path

                mb = mount_backend_for_path(local_cache_dir)
                if mb is not None and mb.is_nfs:
                    logger.warning(
                        "WORKER_LOCAL_MODEL_CACHE_DIR appears to be on NFS (%s, %s); disabling localization cache",
                        mb.fstype,
                        mb.mountpoint,
                    )
                    local_cache_dir = None
            except Exception:
                # best-effort; if mount detection fails, still allow cache usage
                pass
        if local_cache_dir:
            max_cache_gb = float(os.environ.get("WORKER_LOCAL_CACHE_GB", "100"))
            self._local_cache = LocalModelCache(local_cache_dir, max_cache_gb)
            self.local_cache_dir = Path(local_cache_dir)
            logger.info(f"Local cache enabled: {local_cache_dir} ({max_cache_gb}GB max)")

        # Download semaphore for concurrent downloads
        max_concurrent = int(os.environ.get("WORKER_MAX_CONCURRENT_DOWNLOADS", "2"))
        self._download_semaphore = asyncio.Semaphore(max_concurrent)

    def _find_flashpack_path(self, model_path: Path) -> Optional[Path]:
        """Find FlashPack version of a model if it exists."""
        if not self._flashpack_available:
            return None

        # Check for .flashpack sibling directory
        flashpack_path = model_path.parent / (model_path.name + FLASHPACK_SUFFIX)
        if flashpack_path.exists() and (flashpack_path / "pipeline").exists():
            return flashpack_path

        # Check for pipeline/ subdirectory with flashpack files
        if (model_path / "pipeline").exists():
            has_flashpack = any(
                (model_path / f"{comp}.flashpack").exists()
                for comp in FLASHPACK_COMPONENTS
            )
            if has_flashpack:
                return model_path

        return None

    def _detect_load_format(self, model_path: Path) -> str:
        """
        Detect the best loading format for a model.

        Returns: "flashpack", "safetensors", or "single_file"
        """
        # Check FlashPack first
        if self._find_flashpack_path(model_path):
            return "flashpack"

        # Check for safetensors files
        safetensor_files = list(model_path.glob("**/*.safetensors"))
        if safetensor_files:
            return "safetensors"

        # Check for single-file checkpoint
        single_file_exts = [".safetensors", ".ckpt", ".pt", ".bin"]
        for ext in single_file_exts:
            if model_path.suffix == ext:
                return "single_file"
            single_files = list(model_path.glob(f"*{ext}"))
            if len(single_files) == 1:
                return "single_file"

        # Default to safetensors format (from_pretrained will handle it)
        return "safetensors"

    # =========================================================================
    # Model Downloading
    # =========================================================================

    async def ensure_model_available(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        """
        Ensure a model is available locally, downloading if necessary.

        Args:
            model_id: Cozy Hub model ID
            progress_callback: Optional callback(stage, progress_pct)

        Returns:
            Path to the local model directory

        Raises:
            ModelNotFoundError: If model cannot be found locally or remotely
            ModelDownloadError: If download fails
        """
        # Check local models dir first
        if self.models_dir:
            local_path = self.models_dir / model_id
            if local_path.exists():
                logger.debug(f"Model {model_id} found locally: {local_path}")

                # If the snapshot is on NFS/shared storage, prefer on-demand localization
                # to local disk for faster model load.
                if self._local_cache:
                    try:
                        from .mount_backend import mount_backend_for_path

                        mb = mount_backend_for_path(local_path)
                        if mb is not None and mb.is_nfs:
                            cached = self._local_cache.get_cached_path(model_id)
                            if cached:
                                return cached
                            return await self._local_cache.cache_model(model_id, local_path)
                    except Exception:
                        # best-effort; fall back to original path
                        pass

                return local_path

        # Try to download from Cozy Hub
        if self._downloader is not None:
            # Prefer the unified model-ref downloader (cozy snapshots, hf repos, etc.)
            try:
                local = self._downloader.download(model_id, str(self.models_dir))
                return Path(local)
            except Exception as e:
                raise ModelDownloadError(model_id, str(e), retryable=False)

        # Legacy Cozy Hub "model manifest" API is no longer supported. Use the
        # unified downloader instead (ModelRefDownloader / Cozy snapshot APIs).
        if self._cozy_hub_url:
            raise ModelDownloadError(
                model_id,
                "legacy cozy hub manifest downloads are not supported; configure a model ref downloader",
                retryable=False,
            )

        if not self._cozy_hub_url:
            raise ModelNotFoundError(
                model_id,
                self.models_dir / model_id if self.models_dir else None,
            )

        # Unreachable: we always either have a downloader or fail above.
        raise ModelNotFoundError(model_id, self.models_dir / model_id if self.models_dir else None)

    async def _download_from_cozy_hub(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        """
        Download a model from Cozy Hub.

        Cozy Hub provides a model manifest API that lists all files in a model.
        We download each file and reconstruct the directory structure.
        """
        import aiohttp
        import backoff

        if not self.models_dir:
            raise ModelDownloadError(model_id, "No models_dir configured")

        dest_dir = self.models_dir / model_id
        temp_dir = dest_dir.with_suffix(".downloading")

        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)

        try:
            headers = {}
            if self._cozy_hub_token:
                headers["Authorization"] = f"Bearer {self._cozy_hub_token}"

            # Get model manifest from Cozy Hub
            manifest_url = f"{self._cozy_hub_url}/models/{model_id}/manifest"
            timeout = aiohttp.ClientTimeout(total=30)

            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                # Fetch manifest
                async with session.get(manifest_url) as resp:
                    if resp.status == 404:
                        raise ModelDownloadError(
                            model_id, "Model not found in Cozy Hub", retryable=False
                        )
                    resp.raise_for_status()
                    manifest = await resp.json()

                files = manifest.get("files", [])
                if not files:
                    raise ModelDownloadError(model_id, "Empty model manifest")

                total_size = sum(f.get("size", 0) for f in files)
                downloaded = 0

                logger.info(
                    f"Downloading {model_id}: {len(files)} files, "
                    f"{total_size / (1024**3):.1f}GB"
                )

                # Download each file
                for file_info in files:
                    file_path = file_info["path"]
                    file_url = file_info.get("url") or f"{self._cozy_hub_url}/models/{model_id}/files/{file_path}"
                    file_size = file_info.get("size", 0)

                    dest_path = temp_dir / file_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    # Download with retries
                    @backoff.on_exception(
                        backoff.expo,
                        (aiohttp.ClientError, asyncio.TimeoutError),
                        max_tries=3,
                    )
                    async def download_file() -> None:
                        download_timeout = aiohttp.ClientTimeout(total=600)
                        async with aiohttp.ClientSession(
                            timeout=download_timeout, headers=headers
                        ) as dl_session:
                            async with dl_session.get(file_url) as dl_resp:
                                dl_resp.raise_for_status()
                                with open(dest_path, "wb") as f:
                                    async for chunk in dl_resp.content.iter_chunked(
                                        1 << 20
                                    ):
                                        f.write(chunk)

                    await download_file()
                    downloaded += file_size

                    if progress_callback and total_size > 0:
                        progress_callback("downloading", downloaded / total_size * 100)

            # Atomic rename
            temp_dir.rename(dest_dir)
            logger.info(f"Downloaded {model_id} to {dest_dir}")

            return dest_dir

        except aiohttp.ClientError as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ModelDownloadError(model_id, str(e), retryable=True)
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ModelDownloadError(model_id, str(e), retryable=False)

    async def download_models(
        self,
        model_ids: List[str],
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
        randomize_order: bool = True,
    ) -> Dict[str, Path]:
        """
        Download multiple models, optionally with randomized order.

        Randomization helps distribute load across workers when they all
        start simultaneously.

        Args:
            model_ids: List of model IDs to download
            progress_callback: Optional callback(model_id, stage, progress_pct)
            randomize_order: Randomize download order (default True)

        Returns:
            Dict mapping model_id to local path
        """
        ids = list(model_ids)
        if randomize_order:
            random.shuffle(ids)

        results = {}
        for model_id in ids:

            def make_callback(mid: str) -> Optional[Callable[[str, float], None]]:
                if progress_callback:
                    return lambda stage, pct: progress_callback(mid, stage, pct)
                return None

            try:
                path = await self.ensure_model_available(
                    model_id, make_callback(model_id)
                )
                results[model_id] = path
            except ModelDownloadError as e:
                logger.error(f"Failed to download {model_id}: {e}")
                if not e.retryable:
                    raise

        return results

    # =========================================================================
    # Startup Initialization
    # =========================================================================

    async def initialize_startup_models(
        self,
        model_ids: List[str],
        preload_first: Optional[str] = None,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize models at worker startup.

        Downloads models with randomized order (to distribute load),
        then optionally preloads one model into VRAM.

        Args:
            model_ids: List of models to initialize
            preload_first: Optional model to preload into VRAM first
            progress_callback: Optional callback(model_id, stage, progress_pct)
        """
        if not model_ids:
            return

        logger.info(f"Initializing {len(model_ids)} models at startup")
        start_time = time.monotonic()

        # Download all models (randomized order)
        paths = await self.download_models(
            model_ids,
            progress_callback=progress_callback,
            randomize_order=True,
        )

        download_time = time.monotonic() - start_time
        logger.info(f"Downloaded {len(paths)} models in {download_time:.1f}s")

        # If local cache is enabled, prefetch to NVMe
        if self._local_cache:
            for model_id, path in paths.items():
                self._local_cache.start_prefetch(model_id, path)

        # Optionally preload first model into VRAM
        if preload_first and preload_first in paths:
            logger.info(f"Preloading {preload_first} into VRAM")
            try:
                await self.load(preload_first, str(paths[preload_first]))
            except Exception as e:
                logger.warning(f"Failed to preload {preload_first}: {e}")

        total_time = time.monotonic() - start_time
        logger.info(f"Startup initialization complete in {total_time:.1f}s")

    # =========================================================================
    # Pipeline Loading
    # =========================================================================

    async def _load_from_flashpack(
        self,
        model_path: Path,
        pipeline_class: Type,
        torch_dtype: Any,
    ) -> Any:
        """Load a pipeline from FlashPack format."""
        from flashpack import assign_from_file

        flashpack_path = self._find_flashpack_path(model_path)
        if not flashpack_path:
            raise ValueError(f"FlashPack not found for {model_path}")

        logger.info(f"Loading from FlashPack: {flashpack_path}")

        # Load pipeline config
        pipeline_config_dir = flashpack_path / "pipeline"

        # Load base pipeline structure
        pipeline = await asyncio.to_thread(
            pipeline_class.from_pretrained,
            str(pipeline_config_dir),
            torch_dtype=torch_dtype,
        )

        # Assign FlashPack weights to each component
        for component_name in FLASHPACK_COMPONENTS:
            fp_file = flashpack_path / f"{component_name}.flashpack"
            if fp_file.exists() and hasattr(pipeline, component_name):
                component = getattr(pipeline, component_name)
                if component is not None:
                    logger.info(f"  Assigning {component_name} from FlashPack...")
                    await asyncio.to_thread(assign_from_file, component, str(fp_file))

        return pipeline

    async def _load_from_pretrained(
        self,
        model_path: Path,
        pipeline_class: Type,
        custom_pipeline: Optional[str],
        torch_dtype: Any,
        variant: Optional[str],
    ) -> Any:
        """Load a pipeline using from_pretrained."""
        logger.info(f"Loading from pretrained: {model_path}")

        kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
        }

        kwargs.update(missing_component_overrides_for_from_pretrained(pipeline_class, model_path))

        if custom_pipeline:
            kwargs["custom_pipeline"] = custom_pipeline

        if variant:
            kwargs["variant"] = variant

        # Resolve _cozy_ref components
        cozy_refs = resolve_cozy_refs(model_path, self.models_dir)
        for component_name, ref_path in cozy_refs.items():
            # Load referenced component and pass it
            logger.info(f"  Loading referenced component {component_name} from {ref_path}")
            kwargs[component_name] = ref_path

        pipeline = await asyncio.to_thread(
            pipeline_class.from_pretrained,
            str(model_path),
            **kwargs,
        )

        return pipeline

    async def _load_from_single_file(
        self,
        model_path: Path,
        pipeline_class: Type,
        torch_dtype: Any,
    ) -> Any:
        """Load a pipeline from a single checkpoint file."""
        from diffusers.loaders import FromSingleFileMixin

        logger.info(f"Loading from single file: {model_path}")

        # Find the checkpoint file
        if model_path.is_file():
            checkpoint_path = model_path
        else:
            single_file_exts = [".safetensors", ".ckpt", ".pt", ".bin"]
            checkpoint_path = None
            for ext in single_file_exts:
                files = list(model_path.glob(f"*{ext}"))
                if files:
                    checkpoint_path = files[0]
                    break

        if not checkpoint_path:
            raise ValueError(f"No checkpoint file found in {model_path}")

        # Check if pipeline supports single-file loading
        if not issubclass(pipeline_class, FromSingleFileMixin):
            raise ValueError(f"{pipeline_class} does not support single-file loading")

        pipeline = await asyncio.to_thread(
            pipeline_class.from_single_file,
            str(checkpoint_path),
            torch_dtype=torch_dtype,
        )

        return pipeline

    def _apply_vae_optimizations(self, pipeline: Any) -> None:
        """Apply VAE tiling and slicing for memory efficiency."""
        if hasattr(pipeline, "vae") and pipeline.vae is not None:
            if hasattr(pipeline.vae, "enable_tiling"):
                pipeline.vae.enable_tiling()
                logger.info("  VAE tiling enabled")
            if hasattr(pipeline.vae, "enable_slicing"):
                pipeline.vae.enable_slicing()
                logger.info("  VAE slicing enabled")

    def _apply_memory_optimizations(
        self,
        pipeline: Any,
        model_size_gb: float,
        enable_cpu_offload: bool = False,
        enable_sequential_offload: bool = False,
    ) -> None:
        """
        Apply memory optimizations if model is larger than available VRAM.

        Optimizations are CONDITIONAL - only applied when needed.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                logger.info("  CPU mode: skipping VRAM/offload optimizations")
                return
        except Exception:
            logger.info("  CPU mode: skipping VRAM/offload optimizations")
            return

        available_vram = get_available_vram_gb()
        needs_optimization = model_size_gb > (available_vram - self.vram_safety_margin_gb)

        if not needs_optimization and not enable_cpu_offload and not enable_sequential_offload:
            logger.info(f"  Model fits in VRAM ({model_size_gb:.1f}GB < {available_vram:.1f}GB), no optimizations needed")
            return

        if enable_sequential_offload or (needs_optimization and model_size_gb > self._max_vram_gb):
            # Most aggressive - sequential offload
            if hasattr(pipeline, "enable_sequential_cpu_offload"):
                pipeline.enable_sequential_cpu_offload()
                logger.info("  Sequential CPU offload enabled (most memory efficient)")
        elif enable_cpu_offload or needs_optimization:
            # Moderate - model CPU offload
            if hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload()
                logger.info("  Model CPU offload enabled")

    def _configure_scheduler(
        self,
        pipeline: Any,
        scheduler_class_name: Optional[str],
    ) -> None:
        """Configure the pipeline's scheduler."""
        if not scheduler_class_name:
            return

        try:
            scheduler_class = get_scheduler_class(scheduler_class_name)
            if hasattr(pipeline, "scheduler"):
                pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)
                logger.info(f"  Scheduler set to {scheduler_class_name}")
        except Exception as e:
            logger.warning(f"  Failed to set scheduler {scheduler_class_name}: {e}")

    async def _warmup_pipeline(
        self,
        pipeline: Any,
        steps: int = 4,
    ) -> None:
        """
        Run warm-up inference to pre-compile kernels and optimize memory.

        Args:
            pipeline: The loaded pipeline
            steps: Number of inference steps for warmup
        """
        import torch

        logger.info(f"  Running warm-up inference ({steps} steps)...")

        try:
            # Determine pipeline type and run appropriate warmup
            pipeline_name = pipeline.__class__.__name__

            # Common parameters
            warmup_kwargs = {
                "num_inference_steps": steps,
                "output_type": "pil",
            }

            if "Flux" in pipeline_name or "SD3" in pipeline_name or "StableDiffusion3" in pipeline_name:
                warmup_kwargs["prompt"] = "warmup"
                warmup_kwargs["height"] = 256
                warmup_kwargs["width"] = 256
            elif "StableDiffusion" in pipeline_name:
                warmup_kwargs["prompt"] = "warmup"
                warmup_kwargs["height"] = 256
                warmup_kwargs["width"] = 256
            else:
                # Generic warmup
                warmup_kwargs["prompt"] = "warmup"

            # Run inference
            with torch.no_grad():
                await asyncio.to_thread(pipeline, **warmup_kwargs)

            logger.info("  Warm-up complete")

        except Exception as e:
            logger.warning(f"  Warm-up failed (non-fatal): {e}")

        finally:
            # Always flush memory after warmup
            flush_memory()

    async def load(
        self,
        model_id: str,
        model_path: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
    ) -> LoadedPipeline:
        """
        Load a diffusers pipeline.

        Args:
            model_id: Identifier for the model
            model_path: Path to the model (uses models_dir/model_id if not specified)
            config: Pipeline configuration

        Returns:
            LoadedPipeline with the loaded pipeline and metadata

        Raises:
            ModelNotFoundError: Model not found at specified path
            IncompatibleFormatError: Model format doesn't match expected pipeline
            CudaOutOfMemoryError: Not enough VRAM to load model
            ComponentMissingError: Required component missing from model
            PipelineLoaderError: General loading error
        """
        if not _check_torch_available() or not _check_diffusers_available():
            raise ImportError("torch and diffusers are required for pipeline loading")

        import torch

        # Resolve model path
        if model_path:
            path = Path(model_path)
        elif self.models_dir:
            path = self.models_dir / model_id
        else:
            raise PipelineLoaderError("model_path or models_dir must be specified")

        if not path.exists():
            raise ModelNotFoundError(model_id, path)

        config = config or PipelineConfig(model_path=str(path))
        if config.variant is None:
            config.variant = detect_diffusers_variant(path)

        # Determine dtype
        torch_dtype = get_torch_dtype(config.dtype, model_id)
        dtype_str = str(torch_dtype).replace("torch.", "")
        logger.info(f"Loading {model_id} with dtype={dtype_str}")

        # Check VRAM availability before loading
        model_size_gb = estimate_model_size_gb(path)
        available_vram = get_available_vram_gb()

        if (
            config.device == "cuda"
            and model_size_gb > available_vram
            and not config.enable_model_cpu_offload
            and not config.enable_sequential_cpu_offload
        ):
            # Check if we'll definitely OOM (no offload enabled)
            if model_size_gb > self._max_vram_gb:
                logger.warning(
                    f"Model {model_id} ({model_size_gb:.1f}GB) exceeds max VRAM "
                    f"({self._max_vram_gb:.1f}GB), enabling CPU offload"
                )
                config.enable_model_cpu_offload = True

        try:
            # Get pipeline class
            pipeline_class, custom_pipeline = get_pipeline_class(
                config.pipeline_class, str(path)
            )
            pipeline_class_name = pipeline_class.__name__
            logger.info(f"  Pipeline class: {pipeline_class_name}")

            # Detect and use best loading format
            load_format = self._detect_load_format(path)
            logger.info(f"  Load format: {load_format}")

            # Load the pipeline
            try:
                if load_format == "flashpack":
                    pipeline = await self._load_from_flashpack(
                        path, pipeline_class, torch_dtype
                    )
                elif load_format == "single_file":
                    pipeline = await self._load_from_single_file(
                        path, pipeline_class, torch_dtype
                    )
                else:
                    pipeline = await self._load_from_pretrained(
                        path,
                        pipeline_class,
                        custom_pipeline,
                        torch_dtype,
                        config.variant,
                    )
            except FileNotFoundError as e:
                # Missing component file
                raise ComponentMissingError(model_id, str(e))
            except Exception as e:
                error_msg = str(e).lower()
                if "safetensor" in error_msg or "checkpoint" in error_msg:
                    raise IncompatibleFormatError(
                        model_id, load_format, "unknown"
                    ) from e
                raise PipelineLoaderError(f"Failed to load {model_id}: {e}") from e

            # Move to device with OOM handling
            try:
                if config.device == "cuda" and torch.cuda.is_available():
                    pipeline = pipeline.to("cuda")
            except torch.cuda.OutOfMemoryError as e:
                flush_memory()
                raise CudaOutOfMemoryError(
                    model_id, model_size_gb, get_available_vram_gb()
                ) from e

            # Apply VAE optimizations (always enabled)
            if config.enable_vae_tiling or config.enable_vae_slicing:
                self._apply_vae_optimizations(pipeline)

            logger.info(f"  Estimated model size: {model_size_gb:.1f} GB")

            # Apply memory optimizations (conditional)
            self._apply_memory_optimizations(
                pipeline,
                model_size_gb,
                config.enable_model_cpu_offload,
                config.enable_sequential_cpu_offload,
            )

            # Configure scheduler
            if config.scheduler_class:
                self._configure_scheduler(pipeline, config.scheduler_class)

            # Run warm-up inference with OOM handling
            if config.warmup_steps > 0:
                try:
                    await self._warmup_pipeline(pipeline, config.warmup_steps)
                except torch.cuda.OutOfMemoryError as e:
                    flush_memory()
                    logger.warning(
                        f"Warm-up OOM for {model_id}, continuing without warm-up"
                    )

            # Get component list
            components = MODEL_COMPONENTS.get(pipeline_class_name, [])

            # Create loaded pipeline container
            loaded = LoadedPipeline(
                pipeline=pipeline,
                model_id=model_id,
                pipeline_class=pipeline_class_name,
                dtype=dtype_str,
                size_gb=model_size_gb,
                load_format=load_format,
                components=components,
            )

            # Track loaded pipeline
            self._loaded_pipelines[model_id] = loaded

            logger.info(f"Successfully loaded {model_id}")
            return loaded

        except (
            ModelNotFoundError,
            IncompatibleFormatError,
            CudaOutOfMemoryError,
            ComponentMissingError,
            PipelineLoaderError,
        ):
            # Re-raise our custom exceptions
            raise
        except torch.cuda.OutOfMemoryError as e:
            flush_memory()
            raise CudaOutOfMemoryError(
                model_id, model_size_gb, get_available_vram_gb()
            ) from e
        except Exception as e:
            raise PipelineLoaderError(f"Failed to load {model_id}: {e}") from e

    def unload(self, model_id: str) -> bool:
        """
        Unload a pipeline and free memory.

        Args:
            model_id: Model identifier to unload

        Returns:
            True if unloaded, False if not found
        """
        loaded = self._loaded_pipelines.pop(model_id, None)
        if not loaded:
            return False

        logger.info(f"Unloading {model_id}")

        pipeline = loaded.pipeline

        # Remove hooks if using CPU offload
        if hasattr(pipeline, "remove_all_hooks"):
            pipeline.remove_all_hooks()

        # Delete components explicitly
        for component_name in loaded.components:
            if hasattr(pipeline, component_name):
                component = getattr(pipeline, component_name)
                if component is not None:
                    delattr(pipeline, component_name)
                    del component

        # Delete pipeline
        del pipeline
        del loaded

        # Flush memory
        flush_memory()

        logger.info(f"Unloaded {model_id}")
        return True

    def get(self, model_id: str) -> Optional[Any]:
        """Get a loaded pipeline by model ID."""
        loaded = self._loaded_pipelines.get(model_id)
        return loaded.pipeline if loaded else None

    def get_for_inference(self, model_id: str) -> Optional[Any]:
        """
        Get a thread-safe pipeline copy for concurrent inference.

        The diffusers scheduler maintains internal state (timesteps, sigmas) that
        gets corrupted when multiple threads use it simultaneously, causing:
        'IndexError: index N is out of bounds for dimension 0 with size N'

        This method creates a fresh scheduler instance while sharing the heavy
        pipeline components (UNet, VAE, text encoders). Only the scheduler (~few KB)
        is recreated; the model weights (~10+ GB) remain shared.

        References:
        - https://huggingface.co/docs/diffusers/using-diffusers/create_a_server
        - https://github.com/huggingface/diffusers/issues/3672

        Args:
            model_id: The model ID to get a pipeline for

        Returns:
            A thread-safe pipeline copy, or None if model not loaded
        """
        loaded = self._loaded_pipelines.get(model_id)
        if not loaded:
            return None

        base_pipeline = loaded.pipeline

        # Check if pipeline has a scheduler (some pipelines might not)
        if not hasattr(base_pipeline, 'scheduler') or base_pipeline.scheduler is None:
            # No scheduler to worry about - return base pipeline
            return base_pipeline

        try:
            # Create fresh scheduler from config
            fresh_scheduler = base_pipeline.scheduler.from_config(
                base_pipeline.scheduler.config
            )

            # Create new pipeline instance with shared components but fresh scheduler
            # from_pipe() shares all components except those explicitly overridden
            pipeline_class = type(base_pipeline)
            if hasattr(pipeline_class, 'from_pipe'):
                task_pipeline = pipeline_class.from_pipe(
                    base_pipeline,
                    scheduler=fresh_scheduler
                )
                logger.debug(f"Created thread-safe pipeline for {model_id}")
                return task_pipeline
            else:
                # Fallback for older diffusers without from_pipe
                # Just set the scheduler directly (less safe but better than nothing)
                logger.warning(
                    f"Pipeline {pipeline_class.__name__} lacks from_pipe(); "
                    "falling back to direct scheduler assignment"
                )
                base_pipeline.scheduler = fresh_scheduler
                return base_pipeline

        except Exception as e:
            logger.error(f"Failed to create thread-safe pipeline for {model_id}: {e}")
            # Fall back to base pipeline - concurrent access may cause issues
            return base_pipeline

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model IDs."""
        return list(self._loaded_pipelines.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            "loaded_models": list(self._loaded_pipelines.keys()),
            "total_size_gb": sum(p.size_gb for p in self._loaded_pipelines.values()),
            "max_vram_gb": self._max_vram_gb,
            "available_vram_gb": get_available_vram_gb(),
            "flashpack_available": self._flashpack_available,
        }
