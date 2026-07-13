"""Build-time function and endpoint discovery."""

from .heavy_deps import DEFAULT_HEAVY_ROOTS, HeavyDepStubError, stub_missing_heavy_deps
from .names import slugify_name
from .project import ProjectConfig, load_project_config
from .validation import (
    EndpointLockValidationResult,
    EndpointValidationResult,
    validate_endpoint,
    validate_endpoint_lock,
)
from .walk import EndpointImportError

__all__ = [
    "DEFAULT_HEAVY_ROOTS",
    "EndpointImportError",
    "HeavyDepStubError",
    "stub_missing_heavy_deps",
    "slugify_name",
    "ProjectConfig",
    "load_project_config",
    "EndpointValidationResult",
    "EndpointLockValidationResult",
    "validate_endpoint",
    "validate_endpoint_lock",
]
