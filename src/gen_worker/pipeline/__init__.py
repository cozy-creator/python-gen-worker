"""Diffusers pipeline loading (torch-optional)."""

from .mount_backend import MountBackend, mount_backend_for_path
from .spec import CozyPipelineSpec, load_cozy_pipeline_spec

try:
    from .loader import (
        CudaOutOfMemoryError,
        LoadedPipeline,
        ModelNotFoundError,
        PipelineConfig,
        PipelineLoader,
        PipelineLoaderError,
    )
    from .model_manager import DiffusersModelManager
except ImportError:
    pass

__all__ = [
    "MountBackend",
    "mount_backend_for_path",
    "CozyPipelineSpec",
    "load_cozy_pipeline_spec",
    "CudaOutOfMemoryError",
    "LoadedPipeline",
    "ModelNotFoundError",
    "PipelineConfig",
    "PipelineLoader",
    "PipelineLoaderError",
    "DiffusersModelManager",
]
