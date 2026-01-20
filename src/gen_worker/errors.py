class WorkerError(Exception):
    """Base class for worker execution errors."""


class ValidationError(WorkerError):
    """Bad user input; do not retry."""


class RetryableError(WorkerError):
    """Indicates the job can be retried safely."""


class ResourceError(WorkerError):
    """Predictable resource exhaustion (e.g., OOM); do not retry."""


class CanceledError(WorkerError):
    """Job was canceled; do not retry."""


class FatalError(WorkerError):
    """Indicates the job should not be retried."""


class AuthError(WorkerError):
    """Authentication/authorization failure; do not retry (token expired or invalid)."""
