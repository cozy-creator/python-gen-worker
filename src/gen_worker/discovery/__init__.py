"""Build-time function and endpoint discovery."""

from .names import slugify_name
from .toml_manifest import EndpointResources, EndpointToml, load_endpoint_toml
from .validation import (
    EndpointLockValidationResult,
    EndpointValidationResult,
    validate_endpoint,
    validate_endpoint_lock,
)

__all__ = [
    "slugify_name",
    "EndpointResources",
    "EndpointToml",
    "load_endpoint_toml",
    "EndpointValidationResult",
    "EndpointLockValidationResult",
    "validate_endpoint",
    "validate_endpoint_lock",
]
