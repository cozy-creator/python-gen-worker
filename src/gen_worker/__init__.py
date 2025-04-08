# Make src/gen_worker_sdk a Python package
from .decorators import worker_function, ResourceRequirements

__all__ = ["worker_function", "ResourceRequirements"] 