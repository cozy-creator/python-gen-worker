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
