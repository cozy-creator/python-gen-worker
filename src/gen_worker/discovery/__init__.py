"""Build-time function and endpoint discovery."""

from .heavy_deps import DEFAULT_HEAVY_ROOTS, HeavyDepStubError, stub_missing_heavy_deps
from .names import slugify_name
from .project import ProjectConfig, load_project_config
from .validation import (
    EndpointLockValidationResult,
    validate_endpoint_lock,
)
from .entrypoints_v2 import EntrypointDiscoveryError

__all__ = [
    "DEFAULT_HEAVY_ROOTS",
    "EntrypointDiscoveryError",
    "HeavyDepStubError",
    "stub_missing_heavy_deps",
    "slugify_name",
    "ProjectConfig",
    "load_project_config",
    "EndpointLockValidationResult",
    "validate_endpoint_lock",
]
