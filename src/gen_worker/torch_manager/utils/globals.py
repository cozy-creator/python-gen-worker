from typing import Type, Any

from .base_types.architecture import Architecture
from .base_types.common import TorchDevice
from .device import get_torch_device


_available_torch_device: TorchDevice = get_torch_device()

# Model Memory Manager
_MODEL_MEMORY_MANAGER = None

# Model Downloader
_MODEL_DOWNLOADER = None


_ARCHITECTURES: dict[str, type[Architecture[Any]]] = {}
"""
Global class containing all architecture definitions
"""


def get_model_downloader():
    """Get or create the global ModelManager instance"""
    global _MODEL_DOWNLOADER
    if _MODEL_DOWNLOADER is None:
        from .model_downloader import ModelManager

        _MODEL_DOWNLOADER = ModelManager()
    return _MODEL_DOWNLOADER


def get_model_memory_manager():
    global _MODEL_MEMORY_MANAGER
    if _MODEL_MEMORY_MANAGER is None:
        from ..manager import ModelMemoryManager

        _MODEL_MEMORY_MANAGER = ModelMemoryManager()
    return _MODEL_MEMORY_MANAGER


def update_architectures(architectures: dict[str, Type["Architecture"]]):
    global _ARCHITECTURES
    _ARCHITECTURES.update(architectures)


def get_architectures() -> dict[str, Type["Architecture"]]:
    return _ARCHITECTURES


def get_available_torch_device():
    global _available_torch_device
    return _available_torch_device


def set_available_torch_device(device: TorchDevice):
    print("Setting device", device)
    global _available_torch_device
    _available_torch_device = device
