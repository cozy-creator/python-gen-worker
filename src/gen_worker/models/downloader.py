from abc import ABC, abstractmethod
from typing import Optional


class ModelDownloader(ABC):
    @abstractmethod
    def download(self, model_ref: str, dest_dir: str, filename: Optional[str] = None) -> str:
        """Download a model artifact and return the local file path."""
        raise NotImplementedError
