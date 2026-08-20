from typing import Any


class WorkerError(Exception):
    """Base class for worker execution errors."""


class ValidationError(WorkerError):
    """Bad user input; do not retry."""


class IllegalCombination(ValidationError, ValueError):
    """This combination of payload field VALUES is outside the endpoint's
    contract, even though each field is individually legal.

    Raise it from a payload struct's ``__post_init__`` for inter-field
    constraints — ltx-video-2.3's 4K is 16:9 and B200-only, its ultrawide is
    1080p-class and frame-capped, its 48 fps caps at 10 s: 29 of 120 axis
    combinations are legal, and the rest are not requests the endpoint can
    serve.

    Why the type matters. ``CompileAxis`` partitions fields INDEPENDENTLY, so
    the derived warm plan (:mod:`gen_worker.warmup`) is the cross-product of
    those partitions — for a sparse legal set most of that product is
    unbuildable, and a plain ``ValueError`` there is indistinguishable from a
    genuine payload-synthesis BUG. This type is the endpoint DECLARING "not
    legal, by contract": the plan skips those combinations and states the
    coverage it actually achieved, while every other exception still fails the
    load loudly. Declaring axes therefore stops being a boot hazard, and the
    computable "fully warmed?" property survives.

    It subclasses BOTH ``ValidationError`` (so the worker classifies a
    request carrying such a combination as bad input, never infra) and
    ``ValueError`` (so msgspec still converts it to a wire
    ``ValidationError`` when raised inside ``__post_init__`` during decode —
    the wire behaviour is unchanged).
    """


class RetryableError(WorkerError):
    """Indicates the job can be retried safely."""


class ResourceError(WorkerError):
    """Predictable resource exhaustion (e.g., OOM); do not retry."""


class CanceledError(WorkerError):
    """Job was canceled; do not retry."""


class GpuSlotUnreachable(RetryableError):
    """A mid-handler GPU-permit re-acquire can never be satisfied.

    The lease yields the permit across a blob upload and takes it back before
    returning to tenant code. That wait has NO honest progress
    signal of its own — FIFO position does not move while a healthy holder
    computes for four hours, and a holder's silence is not the waiter's fault
    (the publish phase is log-silent and GPU-idle by construction). So the
    wait is bounded by REACHABILITY, never by a clock: it is refused
    only when the permit ledger proves an outstanding permit belongs to no
    live holder — a raw acquirer outside the ledger, or a hold whose owning
    task already finished. Neither state resolves itself.

    RETRYABLE: nothing about the request is wrong and another worker (or this
    one after a recycle) serves it fine.
    """


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
    """A ref the REQUEST PAYLOAD named could not be resolved.

    PROVENANCE decides the class, not the status code or the message: an
    address the caller supplied is the caller's to get right, so this is a
    request error (JOB_STATUS_INVALID, 4xx) and never model-health evidence.
    The identical 404 on a ref the RELEASE declares stays fatal — that one
    IS a statement about the release.

    Raise it only at a resolve boundary that KNOWS the ref came from the
    payload. ``code`` is the machine-readable class the hub surfaces to the
    caller; ``str(exc)`` is ``"<code>: <detail>"`` so the code survives the
    ``safe_message`` wire hop into the request's ``error.code``.
    """

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
    """A caller-supplied blob address is not an algorithm-tagged digest.

    The hub's `storage.ParseDigest` refuses bare hex outright, so guessing an
    algorithm here would only turn a local contract error into a remote 400 —
    or, worse, silently address the wrong one of the two live CAS namespaces
    (repo-CAS is sha256; dataset-CAS is blake3).
    """

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
    """The pod's OWN warm / compile pass failed. No request participates.

    A warm forward runs a payload the WORKER synthesized from the endpoint's
    own schema and calls the endpoint's own handler with it; the same is true
    of the trace and inductor passes. Nothing about any caller reaches those
    boundaries. Untyped, such a failure maps through the generic FATAL tail and
    the hub books it against whichever paying request happened to wake the pod
    — a payload verdict on a payload that was perfectly formed.

    The label is the whole point: it is the worker's ORIGIN claim, made at the
    boundary that knows it, so the hub can route the blame to the RELEASE at
    any hub version instead of guessing from an exception class it cannot
    attribute. The ``phase=`` token is the same one on the activity row, so
    ``worker_activity_events`` and the request row read one vocabulary.

    Deliberately NOT raised for the LOAD phase (a caller-routed slot can fail
    there), for anything already typed (``WorkerError`` and its subclasses
    carry their own origin), or for anything that maps to a non-FATAL status
    (a warm-phase OOM is still an OOM).
    """

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
    """Dispatched model slot's repo differs from the function's declared ref.

    Fail-closed, named-axis refusal: every refusal names the exact axis plus
    both conflicting values. ``@endpoint(models={...})``
    fixes a slot's repo identity unless the slot declares
    ``Slot(selected_by=...)`` (a hub-owned catalog the endpoint explicitly
    opted into) — for a fixed slot, a dispatch naming a DIFFERENT repo is
    silent drift, not a legitimate pick. Hub-resolved tag/flavor picks for
    the SAME repo are never a mismatch; this checks repo identity only.
    """

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
    """A FIXED release-declared model slot failed to resolve.

    The failing ref is the RELEASE'S OWN declaration — no payload field
    participates — so the label is the worker's origin claim and the hub
    classifies it as `declared_ref_load` evidence at ANY worker version
    (`declaredFaultLabels`, blame_ladder.go). A `Slot(selected_by=...)` slot
    stays a bare `ValueError`: the payload picks there, so its failure is not
    version-independent evidence about the release.

    Subclasses `ValueError` because it replaces one at `ctx.slots[name]` —
    handler code that already catches `ValueError` keeps working.
    """


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


class AdapterFidelityRefused(RefCompatibilitySurprise):
    """The adapter's delta does not SURVIVE the grid it would serve through,
    so attaching or fusing it would render something that looks adapted and is
    not.

    Distinct from the zero-delta refusal, which catches an EMPTY
    adapter. This one catches an INERT (or actively corrupting) adapter: fully
    populated, arithmetically real, and destroyed by the target dtype. The
    measured shape is qwen Lightning fused into fp8-E4M3 — surviving-delta
    cosine 0.074 with 15x the true delta's norm, i.e. the served weights carry
    a perturbation LARGER than the adapter and 99.7% orthogonal to it.

    A ``ref_compatibility_surprise`` subtype deliberately: it is a property of
    THIS adapter against THIS release's serving grid, deterministic, and
    retrying anywhere reproduces it. ``survival``
    (:class:`~gen_worker.models.adapter_fidelity.AdapterSurvival`) carries the
    per-module evidence and the grid that was judged.
    """

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
    """The hub refused the child-call admission (typed, deterministic).

    ``code`` is the platform refusal code: ``call_depth_exceeded``,
    ``call_cycle_detected``, ``tree_budget_exceeded``,
    ``tier_escalation_denied``, ``parent_not_running``, ``budget_not_root``,
    or ``child_calls_not_declared`` (this invocation's function did not
    declare ``@entrypoint(child_calls=True)``, so it holds no child-call
    credential — pgw#1579).
    """

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
    """The child request reached ``canceled`` (e.g. a tree cancel)."""

    def __init__(self, request_id: str) -> None:
        self.request_id = str(request_id or "")
        super().__init__(f"child request {self.request_id} was canceled")


class ChildCallTimeoutError(ChildCallError):
    """The caller's wait budget ran out. The child keeps running — the caller
    may ``cancel()`` it or keep polling via the handle."""

    def __init__(self, request_id: str, timeout_s: float) -> None:
        self.request_id = str(request_id or "")
        self.timeout_s = float(timeout_s)
        super().__init__(
            f"child request {self.request_id} did not finish within {self.timeout_s:.0f}s"
        )


class HostRamMoveRefusedError(WorkerError):
    """A ``module.to("cpu")`` that cannot fit the container's RAM budget,
    refused BEFORE allocating. Host-RAM exhaustion is an uncatchable cgroup
    SIGKILL, so the only honest failure is this typed error before the copy
    starts. Maps FATAL on the wire under this class label."""

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
    """The decoded output failed the NOISE/BLANK floor, so the request must NOT
    bank as a clean success.

    Container metadata plus a billing row is not proof that pixels were
    rendered. Raised from the shared save path
    (:mod:`gen_worker.output_integrity`) BEFORE any encode or upload, so the
    garbage never reaches storage and the request reaches the hub as a
    deterministic worker-side failure carrying the measured statistic.

    BLAME. A rendered output is produced by the
    release's code, the model's runtime state AND the request's payload
    together, so this label is NOT version-independent evidence about the
    release and must never be added to the hub's `declaredFaultLabels`: that
    map is explicitly "never a label a payload field can participate in
    producing". As a plain FATAL it classifies under the existing
    `jobResultEvidence` FATAL arm as `EvidenceExecutionFatal` — "the release's
    CODE answering a REQUEST", the strongest honest class a job result can
    carry — which is exactly right and needs no hub change.

    NOT retryable, deliberately. The measured cause is persistent model state
    on one pod, and a retry there re-renders noise at full GPU cost. The blame
    ladder's distinct-worker machinery is what moves a request off a bad pod; a
    self-declared RETRYABLE would only spend the attempt budget rendering the
    same garbage.
    """

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
    """This function reached the publisher surface without declaring
    ``publishes=True`` (pgw#1294 / th#2049).

    The hub-write declaration is ONE surface for ``@endpoint`` and ``@job``
    alike: the decorator says the function MAY write tensors/repos, the request
    still says WHERE. It is also what justifies the hub MINTING the
    worker-capability write grant at launch — grants stay the whole write
    authority, with no org-wide widening.

    So this refusal is not a second gate: the hub never minted a grant for an
    undeclared function, and the write would have failed there anyway, after
    an upload. Raising here makes it a designed, typed refusal at the call
    site, before a byte moves.

    FATAL, not retryable: nothing about the request is wrong and no retry can
    add a declaration to an already-published release.
    """

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
    """This JOB reached a media-writing surface without declaring
    ``emits_media=True`` (th#2069).

    The sibling of :class:`PublishNotDeclaredError`, on the other grant: the
    hub mints ``upload_media`` for a job only when its release declares it, so
    an undeclared job's token cannot upload media and the write would fail at
    the hub after the bytes moved. Raising here makes it a typed refusal at the
    call site instead.

    Endpoints never see this: media IS an endpoint's product, so it declares
    nothing and the authority comes with the kind.

    FATAL, not retryable: no retry adds a declaration to a published release.
    """

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
    """This function read ``ctx.execution_lane`` without declaring
    ``handles=[...]`` (pgw#1580).

    The third sibling of :class:`PublishNotDeclaredError` /
    :class:`MediaNotDeclaredError`, on the ruling that BEHAVIORAL DIVERGENCE is
    declared in the manifest and never inferred from code shape. Reading the
    executing lane is what "this body branches on the lane" MEANS, and the hub
    validates the declared tokens against its concrete lane-body table and
    hydrates them onto ``FunctionMetadata.Handles`` for selection — so an
    undeclared branch diverges invisibly to everything that plans around it.

    The refusal is what makes the declaration load-bearing rather than
    decorative: without it, ``ctx.execution_lane`` handed back a plausible
    default (``bf16-w16a16+eager``) and an undeclared body branched on a value
    nobody had checked.

    FATAL, not retryable: no retry adds a declaration to a published release.
    """

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
    """``ctx.progress`` was handed a position BELOW the last one for its phase.

    Hub liveness for a job is position ADVANCE within a phase budget — never
    pulse, never duration (th#1908) — so the position is load-bearing: a stall
    detector reads it to decide whether a pod is working or wedged. A position
    that moves backwards manufactures apparent liveness out of a loop going
    nowhere, and a lying instrument is worse than no instrument. Restart a
    phase by naming a new phase.
    """

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
    """A job phase reported no position advance within its declared budget.

    pgw#1287's class, made impossible to ship silently: a transfer loop that
    is silent for its whole duration used to look identical to a healthy one,
    and the hub condemned pods that were downloading correctly. Under the job
    harness a phase that declares a budget must ADVANCE inside it; not
    advancing is the fault, and it is booked here rather than inferred from a
    clock.

    RETRYABLE: an infra stall is exactly what ``max_retries`` counts.
    """

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
