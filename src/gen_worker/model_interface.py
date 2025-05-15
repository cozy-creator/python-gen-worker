from abc import ABC, abstractmethod
from typing import List, Any, Optional

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
    def get_vram_loaded_models(self) -> List[str]:
        pass
