from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict

DownloaderType = Any 

class ModelManagementInterface(ABC):
    @abstractmethod
    async def process_supported_models_config(
        self, 
        supported_model_ids: List[str], 
        downloader_instance: Optional[DownloaderType] 
    ) -> None:
        pass

    @abstractmethod
    async def load_model_into_vram(self, model_id: str) -> bool:
        pass
    
    @abstractmethod
    async def get_active_pipeline(self, model_id: str) -> Optional[Any]:
        pass

    def get_for_inference(self, model_id: str) -> Optional[Any]:
        """
        Get a thread-safe pipeline copy for concurrent inference.

        The diffusers scheduler maintains internal state that gets corrupted
        when multiple threads use it simultaneously. This method should return
        a pipeline with a fresh scheduler instance while sharing heavy components.

        Default implementation falls back to get_active_pipeline() via asyncio.run().
        Implementations should override this for proper thread-safety.

        Args:
            model_id: The model ID to get a pipeline for

        Returns:
            A thread-safe pipeline, or None if not loaded
        """
        import asyncio
        try:
            return asyncio.run(self.get_active_pipeline(model_id))
        except Exception:
            return None

    @abstractmethod
    async def get_active_model_bundle(self, model_id: str) -> Optional[Any]:
        pass

    @abstractmethod
    def get_vram_loaded_models(self) -> List[str]:
        pass


class ModelManager(ABC):
    """
    Core model manager interface (no torch imports).
    Implementations are responsible for loading/unloading models into memory.
    """

    @abstractmethod
    def load(self, model_ref: str, local_path: Optional[str] = None, **opts: Any) -> Any:
        pass

    @abstractmethod
    def get(self, model_ref: str) -> Optional[Any]:
        pass

    @abstractmethod
    def unload(self, model_ref: str) -> None:
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        pass
