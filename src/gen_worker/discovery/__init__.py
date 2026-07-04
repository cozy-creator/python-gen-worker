"""Build-time function and endpoint discovery."""

from .names import slugify_name
from .project import ProjectConfig, load_project_config
from .validation import (
    EndpointLockValidationResult,
    EndpointValidationResult,
    validate_endpoint,
    validate_endpoint_lock,
)

__all__ = [
    "slugify_name",
    "ProjectConfig",
    "load_project_config",
    "EndpointValidationResult",
    "EndpointLockValidationResult",
    "validate_endpoint",
    "validate_endpoint_lock",
]
