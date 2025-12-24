from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from gen_worker.model_interface import DownloaderType, ModelManagementInterface


class StubModelManager(ModelManagementInterface):
    """
    Minimal model manager for E2E testing.

    Downloads model artifacts via the provided downloader and returns a
    lightweight "pipeline" dict containing the local path.
    """

    def __init__(self) -> None:
        self._cache_dir = os.getenv("MODEL_CACHE_DIR", "/models")
        self._downloader: Optional[DownloaderType] = None
        self._models: Dict[str, str] = {}
        self._vram_loaded: List[str] = []
        self._lock = asyncio.Lock()

    async def process_supported_models_config(
        self,
        supported_model_ids: List[str],
        downloader_instance: Optional[DownloaderType],
    ) -> None:
        self._downloader = downloader_instance
        if not supported_model_ids:
            return
        for model_id in supported_model_ids:
            await self._ensure_download(model_id)
            if model_id not in self._vram_loaded:
                self._vram_loaded.append(model_id)

    async def load_model_into_vram(self, model_id: str) -> bool:
        if not model_id:
            return False
        await self._ensure_download(model_id)
        if model_id not in self._vram_loaded:
            self._vram_loaded.append(model_id)
        return True

    async def get_active_pipeline(self, model_id: str) -> Optional[Any]:
        if not model_id:
            return None
        path = await self._ensure_download(model_id)
        return {
            "model_id": model_id,
            "model_path": path,
        }

    async def get_active_model_bundle(self, model_id: str) -> Optional[Any]:
        # For the stub manager, the "bundle" is the same lightweight dict we
        # return as the active pipeline.
        return await self.get_active_pipeline(model_id)

    def get_vram_loaded_models(self) -> List[str]:
        return list(self._vram_loaded)

    async def _ensure_download(self, model_id: str) -> str:
        if model_id in self._models:
            return self._models[model_id]
        if not self._downloader:
            raise RuntimeError("Model downloader is not configured")
        os.makedirs(self._cache_dir, exist_ok=True)
        async with self._lock:
            if model_id in self._models:
                return self._models[model_id]
            local_path = await self._downloader.download(model_id, self._cache_dir)
            self._models[model_id] = local_path
            return local_path
