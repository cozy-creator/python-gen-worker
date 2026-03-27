"""
Auto-wired ModelManagementInterface for diffusers pipelines.

When no explicit MODEL_MANAGER_CLASS is set, the Worker auto-creates this
adapter so that LoadModelCommand / heartbeat VramModels work out of the box
for any endpoint that uses diffusers pipeline injection.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from gen_worker.models.interface import ModelManagementInterface

logger = logging.getLogger(__name__)


class DiffusersModelManager(ModelManagementInterface):
    """Wraps PipelineLoader to satisfy the scheduler's model lifecycle protocol."""

    def __init__(self) -> None:
        from .loader import PipelineLoader

        self._loader = PipelineLoader()
        self._downloader: Any = None

    async def process_supported_models_config(
        self,
        supported_model_ids: List[str],
        downloader_instance: Optional[Any],
    ) -> None:
        self._downloader = downloader_instance
        logger.info("DiffusersModelManager: %d supported models configured", len(supported_model_ids))

    async def load_model_into_vram(self, model_id: str) -> bool:
        try:
            if self._loader.get(model_id) is not None:
                return True

            local_path: Optional[str] = None
            if self._downloader is not None:
                from gen_worker.models.cache_paths import worker_model_cache_dir
                from gen_worker.models.refs import parse_model_ref
                from pathlib import Path

                cache_dir = str(worker_model_cache_dir())
                try:
                    # Use async download path directly to avoid nested event loop issues.
                    if hasattr(self._downloader, '_download_async'):
                        parsed = parse_model_ref(model_id)
                        result = await self._downloader._download_async(parsed, Path(cache_dir))
                        local_path = result.as_posix()
                    else:
                        local_path = self._downloader.download(model_id, cache_dir)
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
