"""NFS-to-NVMe local model cache used by PipelineLoader."""

import asyncio
import hashlib
import logging
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


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

        # Sort by modification time (oldest first — .touch() updates mtime, not atime)
        cached = []
        for path in self.cache_dir.iterdir():
            if path.is_dir():
                try:
                    stat = path.stat()
                    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    cached.append((path, stat.st_mtime, size / (1024**3)))
                except OSError:
                    continue

        cached.sort(key=lambda x: x[1])  # Sort by modification time

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

        try:
            loop = asyncio.get_running_loop()
            self._prefetch_tasks[model_id] = loop.create_task(prefetch())
            logger.debug("Started prefetch for %s", model_id)
        except RuntimeError:
            logger.debug("start_prefetch called outside event loop, skipping for %s", model_id)

    async def wait_for_prefetch(self, model_id: str, timeout: float = 60.0) -> bool:
        """Wait for a prefetch to complete."""
        prefetch_future = self._prefetch_tasks.get(model_id)
        if prefetch_future is None:
            return self.is_cached(model_id)

        try:
            await asyncio.wait_for(prefetch_future, timeout=timeout)
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
