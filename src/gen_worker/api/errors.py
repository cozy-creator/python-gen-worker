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


class OutputTooLargeError(ValidationError):
    """Output artifact exceeds the configured worker-side size limit."""

    def __init__(self, *, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = int(max(0, size_bytes))
        self.max_bytes = int(max_bytes)
        super().__init__(f"output file too large (size_bytes={self.size_bytes}, max_bytes={self.max_bytes})")


class RefCompatibilitySurprise(ValidationError):
    """Post-download runtime mismatch on a caller-supplied PAYLOAD_REF.

    Raised by the worker (or tenant code) when a ref passed the
    orchestrator's pre-dispatch compat gates — so file_layout / pipeline-
    class / architectures / components all looked correct based on the
    ref's metadata — but the actual `from_pretrained` or
    `state_dict.load_state_dict` call failed in a way that points at
    content the pre-dispatch check couldn't see:

    - merged LoRAs that added tensor names the base pipeline doesn't expect
    - fine-tunes that renamed a tensor block
    - VAE swaps (compatible pipeline class, different latent channels)
    - subtle scheduler config drift

    Surfaces as ``error_type=ref_compatibility_surprise`` in the job
    result so callers can distinguish "incompatible ref" from "infra
    flake." Not retryable on the same worker (same ref → same failure),
    and not retryable on a different worker (not a hardware issue).

    Tenant code can raise this directly when it detects a known
    compatibility pattern. Otherwise the worker's `_map_exception`
    auto-classifies common heuristics (state_dict key mismatch, missing
    diffusers components) as this type on any request whose payload
    included a caller ref.
    """

    def __init__(
        self,
        message: str = "",
        *,
        ref: str = "",
        axis: str = "",
    ) -> None:
        # Axis distinguishes which post-download check failed:
        # 'state_dict' / 'pipeline_load' / 'component_missing' / 'shape' / ''
        self.ref = ref
        self.axis = axis
        detail = message or "caller-supplied ref failed post-download compatibility check"
        if ref:
            detail = f"{detail} (ref={ref})"
        super().__init__(detail)
