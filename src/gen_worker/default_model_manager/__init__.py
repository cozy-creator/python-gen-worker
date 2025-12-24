"""Compatibility shim for legacy imports.

Use gen_worker.torch_manager instead.
"""
from ..torch_manager import DefaultModelManager, load_config, set_config  # noqa: F401
