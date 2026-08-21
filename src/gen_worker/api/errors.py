from typing import Any


class WorkerError(Exception):
    """Base class for worker execution errors."""


class ValidationError(WorkerError):
    """Bad user input; do not retry."""


class IllegalCombination(ValidationError, ValueError):
    """This combination of payload field VALUES is outside the endpoint's contract, even though each field is individually legal."""


class RetryableError(WorkerError):
    """Indicates the job can be retried safely."""


class ResourceError(WorkerError):
    """Predictable resource exhaustion (e.g., OOM); do not retry."""


class CanceledError(WorkerError):
    """Job was canceled; do not retry."""


class GpuSlotUnreachable(RetryableError):
    """A mid-handler GPU-permit re-acquire can never be satisfied."""


class FatalError(WorkerError):
    """Indicates the job should not be retried."""


class AuthError(WorkerError):
    """Authentication/authorization failure; do not retry (token expired or invalid)."""


class SnapshotBuildFailedError(WorkerError):
    """Dataset snapshot build failed hub-side (typed ``snapshot_build_failed``)."""

    def __init__(self, message: str, *, error_code: str = "") -> None:
        self.error_code = str(error_code or "")
        super().__init__(message)


class ArtifactTransferError(WorkerError):
    """Model/artifact upload or download failed in a provider transfer path."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        phase: str = "",
        retryable: bool = False,
        status_code: int | None = None,
        cause_type: str = "",
    ) -> None:
        self.provider = str(provider or "")
        self.phase = str(phase or "")
        self.retryable = bool(retryable)
        self.status_code = status_code
        self.cause_type = str(cause_type or "")

        detail = str(message or "").strip() or "artifact transfer failed"
        context: list[str] = []
        if self.provider:
            context.append(f"provider={self.provider}")
        if self.phase:
            context.append(f"phase={self.phase}")
        if self.status_code is not None:
            context.append(f"status={int(self.status_code)}")
        if self.cause_type:
            context.append(f"cause={self.cause_type}")
        if context:
            detail = f"{detail} ({', '.join(context)})"
        super().__init__(detail)


class OutputTooLargeError(ValidationError):
    """Output artifact exceeds the configured worker-side size limit."""

    def __init__(self, *, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = int(max(0, size_bytes))
        self.max_bytes = int(max_bytes)
        super().__init__(f"output file too large (size_bytes={self.size_bytes}, max_bytes={self.max_bytes})")


class PayloadRefError(ValidationError):
    """A ref the REQUEST PAYLOAD named could not be resolved."""

    def __init__(self, detail: str, *, code: str, ref: str = "") -> None:
        self.code = code
        self.ref = ref
        super().__init__(f"{code}: {detail}")


class BlobNotFoundError(PayloadRefError):
    """No CAS blob exists at a caller-supplied content digest."""

    def __init__(self, digest: str) -> None:
        super().__init__(
            f"no blob exists at digest {digest} supplied by the request payload — "
            "the address must be a CONTENT DIGEST, not an object-key stem",
            code="blob_not_found", ref=digest,
        )


class BlobDigestMalformedError(PayloadRefError):
    """A caller-supplied blob address is not an algorithm-tagged digest."""

    def __init__(self, digest: str, detail: str) -> None:
        super().__init__(
            f"blob address {digest!r} supplied by the request payload is not a "
            f"valid content digest ({detail}) — write it algorithm-tagged, as "
            '"sha256:<64 hex>" or "blake3:<64 hex>"',
            code="blob_digest_malformed", ref=digest,
        )


class BlobForbiddenError(PayloadRefError):
    """A caller-supplied digest is not readable under this request's grant."""

    def __init__(self, digest: str, status: int) -> None:
        super().__init__(
            f"digest {digest} supplied by the request payload is not readable "
            f"by this request ({status})",
            code="blob_forbidden", ref=digest,
        )


class DatasetNotFoundError(PayloadRefError):
    """A caller-supplied dataset ref does not resolve to a dataset."""

    def __init__(self, ref: str, detail: str = "") -> None:
        super().__init__(
            f"no dataset exists at ref {ref!r} supplied by the request payload"
            + (f" ({detail})" if detail else ""),
            code="dataset_not_found", ref=ref,
        )


class EndpointSetupFailed(WorkerError):
    """The pod's OWN warm / compile pass failed."""

    def __init__(self, function: str, phase: str, cause: BaseException) -> None:
        self.function = str(function or "")
        self.phase = str(phase or "")
        self.cause = cause
        detail = str(cause).splitlines()[0] if str(cause) else ""
        super().__init__(
            f"phase={self.phase} function={self.function}: "
            f"{type(cause).__name__}: {detail}".rstrip(": ")
        )


class ModelSlotIdentityError(WorkerError):
    """Dispatched model slot's repo differs from the function's declared ref."""

    def __init__(
        self, function: str, slot: str, *, declared_ref: str, dispatched_ref: str,
    ) -> None:
        self.function = str(function or "")
        self.slot = str(slot or "")
        self.declared_ref = str(declared_ref or "")
        self.dispatched_ref = str(dispatched_ref or "")
        super().__init__(
            f"{self.function!r} slot {self.slot!r}: dispatched repo "
            f"{self.dispatched_ref!r} != declared {self.declared_ref!r}"
        )


class DeclaredSlotResolutionError(WorkerError, ValueError):
    """A FIXED release-declared model slot failed to resolve."""


class RefCompatibilitySurprise(ValidationError):
    """Post-download runtime mismatch on a caller-supplied PAYLOAD_REF."""

    def __init__(
        self,
        message: str = "",
        *,
        ref: str = "",
        axis: str = "",
    ) -> None:
        self.ref = ref
        self.axis = axis
        detail = message or "caller-supplied ref failed post-download compatibility check"
        if ref:
            detail = f"{detail} (ref={ref})"
        super().__init__(detail)


class AdapterFidelityRefused(RefCompatibilitySurprise):
    """The adapter's delta does not SURVIVE the grid it would serve through, so attaching or fusing it would render something that looks adapted and is not."""

    def __init__(
        self,
        message: str = "",
        *,
        ref: str = "",
        survival: Any = None,
    ) -> None:
        self.survival = survival
        super().__init__(message, ref=ref, axis="state_dict")


class ChildCallError(WorkerError):
    """Base class for call-out primitive failures (ctx.call_endpoint)."""


class ChildCallRefusedError(ChildCallError):
    """The hub refused the child-call admission (typed, deterministic)."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = str(code or "").strip()
        super().__init__(message or self.code)


class ChildRequestFailedError(ChildCallError):
    """The child request reached ``failed``."""

    def __init__(self, request_id: str, error_type: str = "", error_message: str = "") -> None:
        self.request_id = str(request_id or "")
        self.error_type = str(error_type or "")
        self.error_message = str(error_message or "")
        detail = f"child request {self.request_id} failed"
        if self.error_type:
            detail += f" ({self.error_type})"
        if self.error_message:
            detail += f": {self.error_message}"
        super().__init__(detail)


class ChildRequestCanceledError(ChildCallError):
    """The child request reached ``canceled`` (e.g."""

    def __init__(self, request_id: str) -> None:
        self.request_id = str(request_id or "")
        super().__init__(f"child request {self.request_id} was canceled")


class ChildCallTimeoutError(ChildCallError):
    """The caller's wait budget ran out."""

    def __init__(self, request_id: str, timeout_s: float) -> None:
        self.request_id = str(request_id or "")
        self.timeout_s = float(timeout_s)
        super().__init__(
            f"child request {self.request_id} did not finish within {self.timeout_s:.0f}s"
        )


class HostRamMoveRefusedError(WorkerError):
    """A ``module.to("cpu")`` that cannot fit the container's RAM budget, refused BEFORE allocating."""

    def __init__(
        self, *, incoming_bytes: int, available_bytes: int,
        floor_bytes: int, limit_bytes: int | None,
    ) -> None:
        self.incoming_bytes = int(incoming_bytes)
        self.available_bytes = int(available_bytes)
        self.floor_bytes = int(floor_bytes)
        self.limit_bytes = limit_bytes
        gib = float(1 << 30)
        limit = f"{limit_bytes / gib:.1f}GiB" if limit_bytes else "uncapped"
        super().__init__(
            f"host-RAM move refused: ~{incoming_bytes / gib:.1f}GiB of weights "
            f"into {available_bytes / gib:.1f}GiB available (floor "
            f"{floor_bytes / gib:.1f}GiB, cgroup limit {limit}); completing it "
            f"would OOM-kill the worker. Free the model instead of copying it "
            f"to CPU (del + cleanup), or run on a pod with more host RAM"
        )


class OutputIntegrityError(WorkerError):
    """The decoded output failed the NOISE/BLANK floor, so the request must NOT bank as a clean success."""

    def __init__(
        self, verdict: str, *, ref: str = "", kind: str = "", summary: str = "",
    ) -> None:
        self.verdict = str(verdict or "")
        self.ref = str(ref or "")
        self.kind = str(kind or "")
        self.summary = str(summary or "")
        where = f"{self.kind} {self.ref}".strip() or "output"
        super().__init__(
            f"{self.verdict}: {where} failed the output-integrity floor "
            f"({self.summary}) — refused before upload so it cannot bank as a "
            f"successful render"
        )


class PublishNotDeclaredError(ValidationError):

    def __init__(self, surface: str) -> None:
        self.surface = str(surface or "publish")
        super().__init__(
            f"{self.surface} refused: this function did not declare "
            "publishes=True. Add it to the decorator "
            "(@job(publishes=True) / @endpoint(publishes=True)) and republish "
            "— the declaration is what justifies the hub minting the write "
            "grant, so without it no destination exists to write to."
        )


class MediaNotDeclaredError(ValidationError):

    def __init__(self, surface: str) -> None:
        self.surface = str(surface or "media")
        super().__init__(
            f"{self.surface} refused: this job did not declare "
            "emits_media=True. Add it to the decorator "
            "(@job(emits_media=True)) and republish — the declaration is what "
            "justifies the hub minting the upload_media grant, so without it "
            "there is nothing to upload against. It is independent of "
            "publishes=: a job may emit media and write no repo."
        )


class LaneNotDeclaredError(ValidationError):

    def __init__(self, surface: str = "ctx.execution_lane") -> None:
        self.surface = str(surface or "ctx.execution_lane")
        super().__init__(
            f"{self.surface} refused: this function did not declare "
            "handles=[...]. Add the concrete lane BODIES this code branches on "
            "to the decorator (@entrypoint(handles=(\"fp8-w8a8-dynamic\",))) "
            "and republish — the declaration is what tells the platform the "
            "body diverges per lane, and reading the lane without it is a "
            "divergence nothing downstream can see."
        )


class NonMonotonicProgressError(ValidationError):
    """``ctx.progress`` was handed a position BELOW the last one for its phase."""

    def __init__(self, phase: str, last: float, attempted: float) -> None:
        self.phase = str(phase or "")
        self.last = float(last)
        self.attempted = float(attempted)
        where = f"phase {self.phase!r}" if self.phase else "the unnamed phase"
        super().__init__(
            f"ctx.progress: position went BACKWARDS in {where} "
            f"({self.last} -> {self.attempted}). Position is monotonic and "
            "load-bearing — name a new phase to restart the count."
        )


class JobProgressStalledError(RetryableError):
    """A job phase reported no position advance within its declared budget."""

    def __init__(self, phase: str, budget_s: float, position: float | None) -> None:
        self.phase = str(phase or "")
        self.budget_s = float(budget_s)
        self.position = position
        where = f"phase {self.phase!r}" if self.phase else "the unnamed phase"
        seen = "never reported a position" if position is None else (
            f"has been stuck at position {position}")
        super().__init__(
            f"job progress stalled: {where} {seen} for its whole "
            f"{self.budget_s:g}s budget. Liveness is position advance, so a "
            "loop that reports nothing is indistinguishable from a wedged one "
            "— report ctx.progress(position=..., phase=...) as the work moves."
        )
