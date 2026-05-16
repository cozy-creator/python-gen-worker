"""
Auto-wired ModelManagementInterface for diffusers pipelines.

When no explicit MODEL_MANAGER_CLASS is set, the Worker auto-creates this
adapter so that LoadModelCommand / heartbeat VramModels work out of the box
for any endpoint that uses diffusers pipeline injection.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional

from gen_worker.models.interface import ModelManagementInterface

logger = logging.getLogger(__name__)


# Worker injects these via attribute assignment after construction. We do
# NOT extend the interface signature — existing third-party managers keep
# their no-op shape — but the auto-wired manager picks them up and runs the
# real prefetch + cache-update flow.
OnModelDownloaded = Callable[[str, Path], None]  # (canonical_ref, local_path)
OnModelDownloadFailed = Callable[[str, BaseException], None]


class DiffusersModelManager(ModelManagementInterface):
    """Wraps PipelineLoader to satisfy the scheduler's model lifecycle protocol."""

    def __init__(self) -> None:
        from .loader import PipelineLoader

        self._loader = PipelineLoader()
        self._downloader: Any = None
        # Injected by Worker.__init__ when this manager is auto-wired. None
        # for third-party managers; the download loop short-circuits to a
        # log line in that case, preserving the pre-fix no-op semantics.
        self._on_model_downloaded: Optional[OnModelDownloaded] = None
        self._on_model_download_failed: Optional[OnModelDownloadFailed] = None

    async def process_supported_models_config(
        self,
        supported_model_ids: List[str],
        downloader_instance: Optional[Any],
    ) -> None:
        """Pre-download every required model so the worker's startup-phase
        gate can flip to ``ready`` once the bytes are on disk.

        The old body was a no-op (it just stashed the downloader and logged).
        Combined with the worker never marking refs as "downloading" before
        registration, the orchestrator saw an empty ``loading_functions``
        list and dispatched requests to a cold worker (gen-worker #341).

        Each completed download fires the worker-injected
        ``_on_model_downloaded`` callback, which updates the model cache,
        emits the typed ``WorkerModelReadySignal``, forces an out-of-band
        heartbeat, and flips ``startup_phase=ready`` if this was the last
        required ref.
        """
        self._downloader = downloader_instance

        refs = [str(m or "").strip() for m in (supported_model_ids or [])]
        refs = [r for r in refs if r]
        logger.info("DiffusersModelManager: %d supported models configured", len(refs))

        if not refs:
            return
        if downloader_instance is None:
            # Without a downloader we cannot fetch bytes; the worker
            # registration's `loading_functions` will stay populated and
            # `ready` will never fire — exactly the right behaviour.
            logger.warning(
                "DiffusersModelManager: no downloader configured; cannot prefetch %d models",
                len(refs),
            )
            return

        from gen_worker.models.cache_paths import tensorhub_cas_dir

        cache_dir = tensorhub_cas_dir()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Download serially. The previous (no-op) implementation downloaded
        # nothing at all so any positive throughput is a win; per-ref
        # parallelism can be reintroduced if needed but adds little when
        # most HF / civitai endpoints throttle the source side anyway.
        for ref in refs:
            try:
                local_path_str: str = await asyncio.to_thread(
                    downloader_instance.download, ref, str(cache_dir)
                )
                local_path = Path(local_path_str) if local_path_str else None
                if local_path is None or not local_path.exists():
                    raise RuntimeError(
                        f"downloader returned missing path for {ref!r}: {local_path_str!r}"
                    )
                cb = self._on_model_downloaded
                if cb is not None:
                    try:
                        cb(ref, local_path)
                    except Exception:
                        logger.exception(
                            "DiffusersModelManager: on_model_downloaded callback raised for %s",
                            ref,
                        )
            except BaseException as e:
                logger.exception("DiffusersModelManager: download failed for %s", ref)
                fail_cb = self._on_model_download_failed
                if fail_cb is not None:
                    try:
                        fail_cb(ref, e)
                    except Exception:
                        logger.exception(
                            "DiffusersModelManager: on_model_download_failed callback raised for %s",
                            ref,
                        )

    async def load_model_into_vram(self, model_id: str) -> bool:
        try:
            if self._loader.get(model_id) is not None:
                return True

            local_path: Optional[str] = None
            if self._downloader is not None:
                from gen_worker.models.cache_paths import tensorhub_cas_dir

                cache_dir = str(tensorhub_cas_dir())
                try:
                    local_path = await asyncio.to_thread(
                        self._downloader.download, model_id, cache_dir
                    )
                except Exception as e:
                    logger.warning("DiffusersModelManager: download failed for %s: %s", model_id, e)

            loaded = await self._loader.load(model_id, model_path=local_path)
            logger.info(
                "DiffusersModelManager: loaded %s into VRAM (%.1f GB)",
                model_id,
                loaded.size_gb,
            )
            return True
        except Exception:
            logger.exception("DiffusersModelManager: load_model_into_vram failed for %s", model_id)
            return False

    async def get_active_pipeline(self, model_id: str) -> Optional[Any]:
        return self._loader.get(model_id)

    def get_for_inference(self, model_id: str) -> Optional[Any]:
        return self._loader.get_for_inference(model_id)

    async def get_active_model_bundle(self, model_id: str) -> Optional[Any]:
        return self._loader.get(model_id)

    def get_vram_loaded_models(self) -> List[str]:
        return list(self._loader._loaded_pipelines.keys())
