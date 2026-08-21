from __future__ import annotations

import hashlib
import logging
import msgspec
import os
import base64
import re
import shutil
import tempfile
import threading
import time
import urllib.parse
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from ..hostfacts import cuda_ready
from .. import scratchrepo
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
)

if TYPE_CHECKING:  # heavy deps stay import-time-free; methods import lazily
    import numpy as np
    import torch
    from PIL import Image

    from ._concurrent_upload import BudgetGate
    from ..callout import CalloutClient


class LoraOverlay(TypedDict):
    """One per-request LoRA overlay riding a model slot."""

    ref: str
    weight: float


LogLevel = Literal["debug", "info", "warning", "error"]
"""Severity for :meth:`RequestContext.log` (pgw#508's operator stream)."""

from ..api.errors import (
    AuthError,
    BlobDigestMalformedError,
    BlobForbiddenError,
    BlobNotFoundError,
    DatasetNotFoundError,
    DeclaredSlotResolutionError,
)

# Ref provenance: a resolve boundary must say WHERE the address came from —
# that, not the HTTP status or message text, decides whether a terminal miss is
# the CALLER's error (typed 4xx, no health signal) or the RELEASE's (fatal,
# model-health evidence).
REF_ORIGIN_PAYLOAD = "payload"
"""The address came from this request's payload. The caller owns it."""
REF_ORIGIN_PLATFORM = "platform"
"""The address was produced by the platform (hub manifest, release
declaration). A miss is the platform's fault and stays fatal."""
from ..families.base import GenerationDefaults
from ..deferred_outputs import (
    DeferredImageAsset,
    DeferredTail,
    PendingOutput,
    fill_from,
)
from ..io import (
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_IMAGE_QUALITY,
    ImageFormat,
    encode_image,
    image_format,
)
from ..output_integrity import guard_image, judged
from ..stage_timing import StageTimer
from ..api.types import (
    Asset,
    AudioAsset,
    ImageAsset,
    Tensors,
    VideoAsset,
)


def _copy_context_metadata(value: _MetaT) -> _MetaT:
    # Structure-preserving deep copy. mypy cannot prove the rebuilt
    # dict/list/tuple is the SAME type it was handed, so the narrowing is a
    # single `_cast` here rather than an `Any` escaping into every caller.
    if isinstance(value, dict):
        return _cast(_MetaT, {str(k): _copy_context_metadata(v) for k, v in value.items()})
    if isinstance(value, list):
        return _cast(_MetaT, [_copy_context_metadata(v) for v in value])
    if isinstance(value, tuple):
        return _cast(_MetaT, tuple(_copy_context_metadata(v) for v in value))
    return value


_CAS_DIGEST_WIDTHS = {"blake3": 64, "sha256": 64}
"""The algorithms the hub's CAS keys on, and their hex widths — the exact
content of tensorhub `storage.casSupportedAlgos`. Two live namespaces with
different algorithms is why a bare hex string cannot be promoted to a digest
by guessing."""


def _parse_cas_digest(digest: str, *, origin: str) -> str:
    """Return the canonical ``<algo>:<hex>`` form, or refuse.

    Byte-for-byte the same contract as the hub's `storage.ParseDigest`
    (`internal/storage/cas_paths.go`), which `validateDigestParam` fronts on
    every by-digest route: algorithm-tagged, a supported algorithm, hex of
    that algorithm's width, lowercased. Bare hex is refused rather than
    tagged, because the two CAS namespaces disagree on the algorithm and
    guessing addresses the wrong one silently.
    """
    raw = (digest or "").strip()
    caller_supplied = origin == REF_ORIGIN_PAYLOAD

    def _refuse(detail: str) -> NoReturn:
        if caller_supplied:
            raise BlobDigestMalformedError(raw, detail)
        raise RuntimeError(f"malformed platform blob digest {raw!r}: {detail}")

    if not raw:
        _refuse("empty")
    algo, sep, hexpart = raw.partition(":")
    if not sep:
        _refuse("not algorithm-tagged")
    algo = algo.strip().lower()
    hexpart = hexpart.strip()
    width = _CAS_DIGEST_WIDTHS.get(algo)
    if width is None:
        _refuse(f"unsupported algorithm {algo!r}")
    if not re.fullmatch(r"[0-9a-fA-F]*", hexpart):
        _refuse("non-hex character")
    if len(hexpart) != width:
        _refuse(f"{algo} hex must be {width} chars")
    return f"{algo}:{hexpart.lower()}"


def _cas_hasher(algo: str) -> Any:
    """A hasher for a CAS algorithm ``_parse_cas_digest`` already accepted.

    Every entry in :data:`_CAS_DIGEST_WIDTHS` must be constructible here: a
    digest this repo can address but not hash is a download it would have to
    take on trust.
    """
    if algo == "sha256":
        return hashlib.sha256()
    if algo == "blake3":
        import blake3 as _blake3

        return _blake3.blake3()
    raise RuntimeError(f"no hasher for CAS algorithm {algo!r} — cannot verify the bytes")


def _refuse_without_disk_room(root: Path, declared_bytes: int, what: str) -> None:
    """Refuse a transfer the destination filesystem cannot hold.

    The in-loop cap stops a source that LIES about its size; it does nothing
    about one that truthfully declares more bytes than the pod has, which
    ENOSPCs and takes every other writer on the pod with it. Deciding before
    the first byte turns that into a typed refusal.
    """
    from ..bounded_stream import DISK_RESERVE_BYTES
    from ..capability import InsufficientDiskError

    try:
        free = int(shutil.disk_usage(root).free)
    except OSError:
        return  # unmeasurable: the in-loop cap is still enforced
    required = declared_bytes + DISK_RESERVE_BYTES
    if required > free:
        raise InsufficientDiskError(
            f"insufficient disk for {what}: needs {declared_bytes} bytes "
            f"(+{DISK_RESERVE_BYTES} reserve), {free} free at {root}",
            available_bytes=free,
            required_bytes=required,
            path=str(root),
        )


def _as_asset(asset: Asset, cls: Type[_AssetT]) -> _AssetT:
    """Re-type a plain Asset as a media Asset subclass (same fields)."""
    kw = {f: getattr(asset, f) for f in asset.__struct_fields__}
    return cls(**kw)


logger = logging.getLogger(__name__)


# Helpers, constants, and JWT/SSRF utilities live in _helpers.py. They are
# re-exported here so existing `from gen_worker.request_context import _foo`
# call sites (worker.py, tests) keep working.
from ._helpers import (
    _MAX_OUTPUT_FILE_BYTES,
    _decode_unverified_jwt_claims,
    _enforce_output_file_size_limit,
    _infer_mime_type,
    _infer_tensors_format,
    _is_private_ip_str,
    _normalize_output_ref,
    _parse_owner_repo,
    _require_worker_capability_token,
    _sha256_file,
    _url_is_blocked,
)


from ._stream import _RequestOutputStream

from typing import Generic, TypeVar
from typing import cast as _cast
from ..view import for_request as _view_for_request
from ..callout import CalloutClient
from ..callout import ChildRequest

D = TypeVar("D", bound=GenerationDefaults)
#: These two helpers are shape-preserving; saying so removes an Any from
#: every caller rather than at each call site.
_MetaT = TypeVar("_MetaT")
_AssetT = TypeVar("_AssetT", bound=Asset)


class RequestContext(Generic[D]):
    """Context object passed to request handlers.

    SDK v2: ``ctx`` carries only REQUEST-SCOPED facts — resolved
    refs/digests (honest output metadata), the typed per-model config
    (``ctx.defaults``), progress/cancel, request id, timing. Model
    instances live on the endpoint object (``self.pipeline``, stored by
    ``setup()``); the runtime routed this request here because the
    instance's bindings match.

    The type parameter is the handler's derived config schema:
    ``ctx: RequestContext[SdxlDefaults]`` — the registry derives the schema
    from this annotation and catalog values decode against it.

    Output contract: RETURN the output object (``ImageOutput`` etc.); the
    SDK owns encode + upload. A handler that hand-uploads inside its body
    opts out of the encode/upload tail overlap.
    """

    def __init__(
        self,
        request_id: str,
        job_id: Optional[str] = None,
        emitter: Optional[Callable[[Dict[str, Any]], None]] = None,
        owner: Optional[str] = None,
        invoker_id: Optional[str] = None,
        file_api_base_url: Optional[str] = None,
        worker_capability_token: Optional[str] = None,
        local_output_dir: Optional[str] = None,
        execution_hints: Optional[Dict[str, Any]] = None,
        models: Optional[Dict[str, Any]] = None,
        loras: Optional[Dict[str, Any]] = None,
        boot_warmup: bool = False,
        publishes: bool = False,
        emits_media: Optional[bool] = None,
        child_calls: bool = False,
        handles: Optional[Sequence[str]] = None,
    ) -> None:
        self._request_id = str(request_id or "").strip()
        self._job_id = str(job_id or "").strip() or None
        self._owner = owner
        self._invoker_id = invoker_id
        self._file_api_base_url = (file_api_base_url or "").strip() or None
        self._worker_capability_token = (worker_capability_token or "").strip() or None
        self._local_output_dir = (local_output_dir or "").strip() or None
        self._execution_hints = dict(execution_hints or {})
        self._started_at = time.time()
        self._canceled = False
        self._boot_warmup = bool(boot_warmup)
        self._execution_lane = ""  # executing lane, set by the executor
        # th#2049/pgw#1294: the executing function DECLARED `publishes=True`,
        # so it may write tensors/repos to the hub. Kind stops implying write
        # authority; this declaration is the one justification. Stamped from
        # the spec at dispatch.
        self._publishes = bool(publishes)
        # th#2069: the media sibling, and JOBS ONLY. None = not job-scoped (an
        # endpoint, whose product IS media); False = a job that declared none,
        # for which the hub minted no `upload_media` grant. Stamped from the
        # spec at dispatch, like `publishes`.
        self._emits_media = emits_media if emits_media is None else bool(emits_media)
        # pgw#1579/pgw#1580: the two declarations the v1 hardcut dropped while
        # the hub kept reading them. Same shape as `publishes` above — the
        # manifest asks the hub for the capability, the spec stamps the same
        # fact here, and the SDK surface refuses undeclared code at the call
        # site instead of letting the hub decline a credential mid-request.
        self._child_calls = bool(child_calls)
        self._handles: Tuple[str, ...] = tuple(handles or ())
        # Monotonic progress POSITION per phase (pgw#1294). Liveness for a job
        # is position ADVANCE within a phase budget — never pulse, never
        # duration — so a position that goes backwards is a lying instrument
        # and raises rather than reports.
        self._progress_positions: Dict[str, float] = {}
        self._config: Dict[str, Any] = {}  # effective config params
        self._config_snapshot: Optional[bytes] = None
        self._cancel_event = threading.Event()
        self._emitter = emitter
        self._cached_repo_job_scope: Optional[tuple[str, str, str]] = None
        # One confession per context for an unparseable destination_repo
        # (see `_repo_job_upload_scope`); the getter runs once per file.
        self._repo_scope_parse_reported = False
        self._models = _copy_context_metadata(models or {})
        self._loras = _copy_context_metadata(loras or {})
        # Caller-visible adjustment warnings: rows the merge/clamp layer emits
        # whenever a requested value is modified. They ride the RESULT ENVELOPE
        # (JobResult.adjustments) + the hub's request record/events stream; pod
        # logs alone never reach a caller.
        self._adjustments: List[Dict[str, str]] = []

        # Capability-budget gate, lazy-built from the capability token's
        # max_total_bytes + max_bytes_per_file claims on first upload. Lives on
        # the base (not the producer mixin) because the base save_file path
        # reserves against it too. Blocks new reservations until in-flight
        # bytes fit the aggregate budget, since the pool's per-file fan-out can
        # otherwise over-commit on parallel multi-GiB shards.
        self._upload_budget_gate = None  # type: Optional["BudgetGate"]
        self._upload_budget_gate_lock = threading.Lock()

        # GPU-slot lease, set by the executor for GPU jobs; lets blocking
        # uploads release the GPU slot while they wait on the network. None for
        # CPU jobs and local (CLI) runs.
        self._gpu_slot_lease: Optional[Any] = None
        # May a CHILD-CALL wait yield the GPU slot? Stamped per job by the
        # executor: True only when this job holds no instance gate and no
        # per-request adapters — see _child_call_wait for why both matter.
        self._child_call_slot_yieldable: bool = False
        # Executor callback fired on the TERMINAL slot release at the
        # decode->finalize handoff, so the worker's finalizing-job count (and
        # the hub's StateDelta view of it) tracks the encode/upload tail.
        self._on_finalize_release: Optional[Callable[[], None]] = None

        # Outputs whose encode+upload run in the finalize tail, after the
        # executor releases the GPU permit. Disarmed by default — CLI runs,
        # endpoint unit tests and streaming handlers stay eager.
        self._deferred = DeferredTail()

        # Per-stage timing. Framework hooks (permit wait, input fetch, encode,
        # stamp, upload, denoise steps) record unconditionally; endpoints
        # refine with ctx.stage().
        self._stages = StageTimer()

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def boot_warmup(self) -> bool:
        """True when this call is the worker's boot-time synthetic warmup
: the output is discarded, so a handler MAY cheapen the run
        (e.g. ``steps = 1 if ctx.boot_warmup else steps``) — the allocator
        peak is shape-driven, not step-driven."""
        return self._boot_warmup

    @property
    def handles(self) -> Tuple[str, ...]:
        """The lane BODIES this function declared it branches on
        (``@entrypoint(handles=…)``), stamped from the spec at dispatch."""
        return self._handles

    @property
    def child_calls(self) -> bool:
        """Did this function declare ``@entrypoint(child_calls=True)``? The
        SDK mirror of the hub's ``invoke_child`` grant decision."""
        return self._child_calls

    @property
    def execution_lane(self) -> str:
        """The EXECUTING precision lane of this call, a full descriptor id like
        ``"fp8-w8a8-dynamic+compiled"`` — post-degrade truth, the same value
        JobMetrics.lane reports.

        DECLARED READERS ONLY (pgw#1580). Reading the executing lane IS the
        behavioral divergence ``handles=`` declares, so a function that did not
        declare one is refused typed rather than handed
        ``"bf16-w16a16+eager"`` — a plausible default that an undeclared body
        would branch on and nothing downstream could see. Same
        one-fact-two-enforcers shape as ``publishes=``: the manifest tells the
        hub, this refuses the author.
        """
        self._require_lane_declaration("ctx.execution_lane")
        return self._execution_lane or "bf16-w16a16+eager"

    def _require_lane_declaration(self, surface: str) -> None:
        if not self._handles:
            from ..api.errors import LaneNotDeclaredError

            raise LaneNotDeclaredError(surface)

    def _declare_from_spec(self, spec: Any) -> None:
        """Worker-internal: stamp an ``EntrypointSpec``'s declarations onto a
        context built before the function was known.

        The serve loop passes them at construction; the LOCAL host
        (``EndpointHost.dispatch``, the CLI and the daemon) builds one context
        per request and only then routes it to a function, so it stamps here.
        Both ends read the SAME spec fields, which is the point — a declaration
        that only works under the serverless dispatcher is a declaration an
        author cannot test."""
        if spec is None:
            return
        self._child_calls = bool(getattr(spec, "child_calls", False))
        self._handles = tuple(getattr(spec, "handles", ()) or ())

    def _set_execution_lane(self, execution_lane: str) -> None:
        self._execution_lane = str(execution_lane or "").strip()

    @property
    def config(self) -> Dict[str, Any]:
        """Effective values for this endpoint's declared config parameters
        (``@endpoint(config=[ConfigParam(...)])``) at dispatch time: declared
        defaults overlaid with the deployer-set values at the worker's observed
        config generation. Read-only; a returned copy."""
        return dict(self._config)

    # writerless: pgw#1475 -> owed to the config-plane cutover, expiry 2026-09-15 — the THIRD sibling the hardcut orphaned, but unlike source_path and execution_lane it cannot simply be re-wired: `@endpoint(config=[ConfigParam(...)])` is v1's declaration and `@entrypoint` declares no config at all, so there is no schema to decode `RunJob.config_params` against. Whether v2 gains a config declaration is a design call, not a repair; `subproc.py:194` still reads `_config_snapshot`, so it is not dead either.
    def _set_config(
        self,
        values: Optional[Mapping[str, Any]],
        *,
        snapshot: Optional[bytes] = None,
    ) -> None:
        self._config = dict(values or {})
        self._config_snapshot = bytes(snapshot) if snapshot is not None else None

    @property
    def models(self) -> Dict[str, str]:
        """Resolved model refs for this invocation, keyed by slot name."""
        return _copy_context_metadata(self._models)

    @property
    def loras(self) -> Dict[str, Tuple[LoraOverlay, ...]]:
        """Per-request LoRA overlays riding each model slot:
        slot name -> tuple of ``{"ref", "weight"}``. Empty for adapter-free
        requests. The worker applies/removes the adapters around the handler
        call; this surface is read-only metadata."""
        return _copy_context_metadata(self._loras)

    def adjusted(
        self, field: str, requested: Any, applied: Any, reason: str,
    ) -> None:
        """Record a caller-visible ADJUSTMENT: the serve path
        modified a requested value (clamp, substitution, injection). Rows
        ride the result envelope (``JobResult.adjustments``) and the hub's
        request record + events stream, so API consumers and UIs can show
        e.g. "guidance clamped 15 -> 10 (model maximum)".

        Boundary: adjustments WARN-AND-SERVE. A catalog-LOCKED recipe field
        is the opposite contract — override attempts get a typed refusal
        upstream, never a warning-carrying garbage render."""
        self._adjustments.append({
            "field": str(field),
            "requested": "" if requested is None else str(requested),
            "applied": "" if applied is None else str(applied),
            "reason": str(reason or ""),
        })

    @property
    def adjustments(self) -> Tuple[Dict[str, str], ...]:
        """Immutable view of the adjustment ledger (:meth:`adjusted` /
        :meth:`clamp` rows, in emission order). The PUBLIC read side —
        endpoint tests assert against this, never ``_adjustments``."""
        return tuple(dict(row) for row in self._adjustments)

    def clamp(
        self,
        field: str,
        requested: float,
        *,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        reason: str = "",
    ) -> float:
        """Clamp ``requested`` into [lo, hi] and, when that CHANGES the
        value, record the adjustment (:meth:`adjusted`) — the one merge/
        clamp helper endpoints use so caller-visible coverage cannot drift
        per endpoint."""
        applied = float(requested)
        if lo is not None and applied < lo:
            applied = float(lo)
        if hi is not None and applied > hi:
            applied = float(hi)
        if applied != float(requested):
            self.adjusted(field, requested, applied, reason or "outside the model's supported range")
        return applied

    def for_request(
        self,
        pipeline: Any,
        *,
        sampler: str = "",
        seed: Optional[int] = None,
        generator: Optional["torch.Generator"] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        schedulers: Optional[Sequence[str]] = None,
    ) -> Any:
        """A per-request VIEW of ``pipeline``: same module objects (shared
        weights; the compiled graph stays bound), OWN scheduler — cloned from
        the instance scheduler's config, with ``sampler`` selecting the
        scheduler class from the SDK table (``gen_worker.view.SAMPLERS``).

        **THIS VIEW CARRIES NO OBJECTIVE, and that is now stated rather than
        promised (pgw#1583).** Until this docstring was corrected it claimed
        twice that the resolved checkpoint's objective was applied here and
        that ambiguity raised; the body passed ``objective=""`` unconditionally
        and ``slot=`` was accepted and discarded, so the raise was unreachable
        and every view built through this method denoised under the platform
        default. That is worse than an error: v-prediction and flow change what
        the scheduler DOES, so a wrong objective returns plausible output.

        The reason is plumbing, not policy: ``ModelBinding.objective`` (proto
        field 6) has no reader anywhere in the SDK — no ``_Pick`` field, no
        ``DeployBinding`` field, nothing on this context — so there is no
        checkpoint fact here to apply. **An author who needs one passes it
        explicitly to the module-level gate**, which is the only surface that
        honours it::

            from gen_worker.view import for_request

            view = for_request(model.pipe, objective="flow", sampler="ddim")

        EVERY sampler-shaped attribute is cloned, not just ``scheduler``: a
        second stateful sampler such as an ltx ``audio_scheduler`` would
        otherwise stay shared across concurrent requests. ``schedulers=`` pins
        the set explicitly when discovery is not wanted.

        Never assign ``self.pipeline.scheduler`` per request — that is an
        instance mutation two concurrent requests corrupt each other
        through, and a module swap the compiled graph guards against.
        """

        gen = generator
        if gen is None and seed is not None:
            gen = self.generator(seed)
        return _view_for_request(
            pipeline, sampler=sampler, objective="", generator=gen,
            scheduler_config=scheduler_config, schedulers=schedulers,
        )

    @property
    def device(self) -> "torch.device":
        """Torch device for this worker runtime (e.g. cuda:0 or cpu)."""
        try:
            import torch
        except Exception:
            raise RuntimeError("torch is not available in this runtime") from None
        if cuda_ready():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")

    def generator(self, seed: Optional[int] = None) -> "torch.Generator":
        """A ``torch.Generator`` on ``ctx.device``, seeded when ``seed`` is set."""
        import torch

        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        return gen

    def _get_file_api_base_url(self) -> str:
        if not self._file_api_base_url:
            raise RuntimeError(
                "file API base URL is not configured for this request — "
                "neither this dispatch's `RunJob.file_base_url` nor any "
                "HelloAck has carried one (executor.file_base_url is empty)"
            )
        return self._file_api_base_url.rstrip("/")

    def _get_upload_budget_gate(self) -> "BudgetGate":
        """Lazy-construct the capability-budget gate from the JWT claims.

        Pure pass-through when the token has no budget claims (dev/test
        paths). See ``_concurrent_upload.BudgetGate`` for semantics.
        """
        if self._upload_budget_gate is None:
            with self._upload_budget_gate_lock:
                if self._upload_budget_gate is None:
                    from ._concurrent_upload import budget_gate_from_capability_jwt
                    token = self._get_worker_capability_token() or ""
                    self._upload_budget_gate = budget_gate_from_capability_jwt(token)
        return self._upload_budget_gate

    def _get_worker_capability_token(self) -> str:
        if self._worker_capability_token:
            return self._worker_capability_token
        return _require_worker_capability_token()

    def _resolve_local_output_path(self, ref: str) -> Optional[str]:
        """
        Dev-only local output backend.

        When local_output_dir is set, RequestContext.save_* will write outputs to disk
        instead of using Cozy Hub's file API.
        """
        base = (self._local_output_dir or "").strip()
        if not base:
            return None

        # Normalize and prevent path traversal.
        ref = (ref or "").strip().replace("\\", "/").lstrip("/")
        if not ref:
            raise ValueError("invalid ref")
        out = (Path(base).expanduser() / ref).resolve()
        root = Path(base).expanduser().resolve()
        if root not in out.parents and out != root:
            raise ValueError("path traversal")
        return str(out)

    def _should_stream_output_to_file_api(self, ref: str) -> bool:
        try:
            if self._resolve_local_output_path(ref):
                return False
        except Exception:
            logger.debug("_should_stream_output_to_file_api: local path resolve failed for ref=%r", ref, exc_info=True)
            return False
        try:
            _ = self._get_file_api_base_url()
            _ = self._get_worker_capability_token()
        except Exception:
            logger.debug("_should_stream_output_to_file_api: file_api base or capability token unavailable", exc_info=True)
            return False
        return True

    def _repo_job_upload_scope(self) -> Optional[tuple[str, str, str]]:
        """Return (owner, repo, job_id) for repo-CAS uploads, or None.

        Pure getter — no HTTP calls or side effects. TensorHub auto-creates
        the repo and lineage record on first upload when the capability token
        is valid.
        """
        if self._cached_repo_job_scope is not None:
            return self._cached_repo_job_scope

        # Scope resolves whenever destination_repo + job_id are present — not
        # gated on kind, because @inference clone jobs also emit checkpoints.
        hints = dict(self._execution_hints or {})
        destination_repo = str(hints.get("destination_repo") or "").strip()
        if destination_repo == "":
            return None
        job_id = str(self._job_id or "").strip()
        if job_id == "":
            return None
        try:
            owner, repo = _parse_owner_repo(destination_repo)
        except Exception:
            logger.debug("_repo_job_upload_scope: destination_repo=%r did not parse as owner/repo", destination_repo, exc_info=True)
            # Returning None here does not mean "no repo was asked for" — a
            # destination_repo WAS supplied and did not parse, so this job's
            # outputs stop being repo-CAS writes and land on the
            # user-files/media path instead. Emitted ONCE per context: an
            # upload loop must not turn one config defect into a per-file
            # flood.
            if not self._repo_scope_parse_reported:
                self._repo_scope_parse_reported = True
                from .. import activity as activity_mod

                activity_mod.emit_event(
                    activity_mod.KIND_SERVE_DEGRADE,
                    f"job={self._job_id} destination_repo="
                    f"{destination_repo!r}: not parseable as owner/repo, so "
                    f"this job's outputs are NOT written as repo-CAS and fall "
                    f"back to the user-files/media path — a release-config "
                    f"defect, not a transient",
                    phase="repo_scope_unparseable",
                )
            return None

        result = (owner, repo, job_id)
        self._cached_repo_job_scope = result
        return result

    def _repo_job_release(self) -> str:
        """The release a repo-CAS checkpoint publish attaches to, or "".

        It is the caller's `destination.release`, carried through the execution
        hints. Empty means the invoke named none — which is th#2202's ORDINARY
        case, not a defect: see :meth:`_checkpoint_release`."""
        return str((self._execution_hints or {}).get("destination_release") or "").strip()

    def _checkpoint_release(self, repo: str) -> str:
        """THE release `ctx.save_checkpoint` publishes under, or "" to let the
        hub derive one. Raises when neither is available.

        th#2202. This is a NAMED decision rather than an inline `if` because
        the `if` was wrong for a year of cost and nothing could see it at $0:
        the only carrier for `destination.release` is the reserved
        `destination:{ref,release}` object, and an endpoint whose typed input
        declares the scalar `destination_repo` with `forbid_unknown_fields`
        can never be handed that key. So this raised at step 200 of a paid-for
        training run, on the SCRATCH repo the hub itself named — a repo no
        author ever names a release for, and one the hub cuts a release for on
        every publish. The refusal survives for a destination that HAS an
        author, which is where th#1987's deliberation rule lives.
        """
        release = self._repo_job_release()
        if release or scratchrepo.is_scratch_name(repo):
            return release
        raise RuntimeError(
            f"cannot publish into {repo!r}: the request named no "
            "`destination.release`, and th#1987 made it mandatory for a repo "
            "with an author — the hub refuses the declare with "
            "`release_required`. Cut a release and invoke with "
            "destination={ref, release}.")

    def _tensor_upload_execution_kind(self) -> str:
        hints = dict(self._execution_hints or {})
        return str(hints.get("kind", "") or "").strip().lower()

    def _require_repo_job_scope_for_tensors(self, ref: str) -> None:
        """
        for training/conversion checkpoints, remote tensor uploads must be job-scoped
        repo-cas writes. This prevents silent fallback to user-files/media uploads.
        """
        kind = self._tensor_upload_execution_kind()
        if kind != "training":
            return
        try:
            if self._resolve_local_output_path(ref):
                return
        except Exception:
            pass
        if self._repo_job_upload_scope() is None:
            raise RuntimeError(
                "tensor upload requires repo job scope (execution_hints.kind with destination_repo and job_id)"
            )

    @property
    def cancelled(self) -> bool:
        """True once the request has been cancelled."""
        return self._canceled

    def raise_if_cancelled(self, message: str = "request cancelled") -> None:
        """Raise ``CanceledError(message)`` if cancelled. No-op otherwise.

        The one cancellation idiom — call inside long-running loops.
        """
        if self._canceled:
            from ..api.errors import CanceledError
            raise CanceledError(message)

    def _cancel(self) -> None:
        """Worker-internal: mark the request as cancelled."""
        if not self._canceled:
            self._canceled = True
            self._cancel_event.set()
            logger.info("request %s marked for cancellation.", self.request_id)

    # -- call-out primitive --------------------------------------------------

    def _callout_client(self) -> "CalloutClient":
        # pgw#1579, and it must come FIRST: the hub mints `invoke_child` only
        # for a function whose manifest row declared it, so an undeclared body
        # would otherwise reach the wire and come back 403 —
        # `child_calls_not_declared` after the request was already running.
        # Same refusal CODE as the hub's, so both ends say one word.
        if not self._child_calls:
            from ..api.errors import ChildCallRefusedError

            raise ChildCallRefusedError(
                "child_calls_not_declared",
                "this function did not declare @entrypoint(child_calls=True), "
                "so the platform minted no invoke_child grant for it. Add the "
                "declaration and republish — it is a capability the manifest "
                "asks for, never one inferred from the body.",
            )

        if not self._file_api_base_url:
            from ..api.errors import ChildCallError

            raise ChildCallError(
                "no platform base URL in this invocation context; child calls "
                "require running under the platform (or cozy-local)"
            )
        return CalloutClient(
            base_url=self._file_api_base_url,
            parent_request_id=self._request_id,
            get_token=lambda: self._worker_capability_token or "",
            cancel_event=self._cancel_event,
        )

    def call_endpoint(
        self,
        endpoint: str,
        function: str,
        payload: Dict[str, Any],
        *,
        semver_major: int,
        wait: bool = True,
        timeout_s: Optional[float] = 3600.0,
        tier: Optional[str] = None,
        poll_interval_s: float = 2.0,
    ) -> Any:
        """Call another endpoint's function as a CHILD request.

        ``semver_major`` addresses the callee's serving pointer for that
        semver-major (``POST /{owner}/{name}/v{semver_major}/{function}``).
        It is REQUIRED and has no default: endpoint tags are dead (th#2044),
        an unassigned semver-major is a typed refusal naming the assigned
        ones, and there is no ``latest``.

        The function must be declared ``@entrypoint(child_calls=True)`` — the
        platform then scopes this invocation's credential for child calls, and
        an undeclared function is refused HERE (``child_calls_not_declared``)
        rather than by the hub mid-request.
        Children bill the parent request's payer, inherit its availability
        tier (``tier=`` may name a CHEAPER class, never escalate), count
        against the tree's depth/budget ceilings, and die with the parent
        when the tree is cancelled.

        ``wait=True`` (default) blocks to a terminal state and returns the
        child's output items (asset refs stay refs — pass them straight into
        the next call's payload). ``wait=False`` returns a
        :class:`~gen_worker.callout.ChildRequest` handle
        (``.status()`` / ``.result()`` / ``.cancel()``).

        While parked on the child (``wait=True`` here, or the handle's
        ``.result()``), the job's GPU slot is YIELDED when that is safe (see
        :meth:`_child_call_wait`) so another request — or a background mint
        turn — runs on the permit instead of the accelerator idling on a
        network round trip; the wait re-acquires before returning.

        Raises ``ChildCallRefusedError`` (typed admission refusals),
        ``ChildRequestFailedError`` / ``ChildRequestCanceledError``,
        ``ChildCallTimeoutError``, and ``CanceledError`` when this invocation
        itself is cancelled mid-wait.
        """
        self.raise_if_cancelled()
        client = self._callout_client()
        request_id = client.submit(
            endpoint, function, payload, semver_major=semver_major, tier=tier
        )

        handle = ChildRequest(client, request_id, wait_guard=self._child_call_wait)
        if not wait:
            return handle
        return handle.result(timeout_s, poll_interval_s=poll_interval_s)

    def workflow_checkpoint(self, key: str, fn: Callable[[], Any]) -> Any:
        """Memoize one workflow step's result under this request.

        Durability-by-memoization (WORKFLOW-DESIGN.md §4): the first call
        computes ``fn()`` and stores its JSON-serializable result under
        ``key``; a re-run of this invocation (worker death, retry attempt)
        returns the stored value without recomputing. Values are small JSON
        (step output refs, not media; 64KB cap).
        """
        client = self._callout_client()
        value, found = client.checkpoint_get(key)
        if found:
            return value
        value = fn()
        client.checkpoint_put(key, value)
        return value

    @contextmanager
    def _gpu_slot_yielded(self) -> "Iterator[None]":
        """Worker-internal: release the job's GPU slot for the duration of
        blocking non-GPU I/O (blob upload), re-acquiring before returning to
        tenant code. No-op when there is no lease (CPU jobs, local runs) or
        the slot is already yielded (executor freed it post-handler).

        If the job was cancelled while yielded (deadline / CancelJob), the
        re-acquired slot is released again immediately: the executor's final
        release already saw ``held == False`` and skipped, so the balance
        stays exact and the freed slot isn't captured by a dying job.

        ``reacquire`` raises ``GpuSlotUnreachable`` when the permit
        provably cannot come back. That refusal only ever REPLACES a clean
        exit — a failure raised by the block owns the outcome and is never
        masked by it.
        """
        lease = self._gpu_slot_lease
        if lease is None or not lease.yield_slot():
            yield
            return
        try:
            yield
        except BaseException:
            try:
                lease.reacquire()
            except Exception:
                logger.exception(
                    "request %s: GPU permit unreachable while unwinding; the "
                    "block's own failure stands", self.request_id)
            if self._canceled:
                lease.yield_slot()
            raise
        lease.reacquire()
        if self._canceled:
            lease.yield_slot()

    @contextmanager
    def _child_call_wait(self) -> "Iterator[None]":
        """Worker-internal: park for a child request's result with the GPU
        slot YIELDED — a parent waiting on a network round trip must not rent
        the accelerator. Same lease discipline as ``save_bytes``; the wait
        stays on the handler thread, so heartbeats and StateDeltas ride the
        free event loop.

        Gate-holding class endpoints yield too: every permit acquirer takes
        the instance gate FIRST, so a same-instance follower queues on
        ``run_lock`` holding no permit and the re-acquire cannot wedge. SCOPED
        OUT is a job with per-request adapters active — a follower on the
        shared pipeline would deactivate/replace this request's adapter state
        mid-handler, a data race no lock order fixes; those jobs keep the
        permit across the wait. Bracketed as its own stage so the child park
        shows up as GPU-idle time rather than compute.
        """
        with self._stages.stage("child_call_wait"):
            if not self._child_call_slot_yieldable:
                yield
                return
            with self._gpu_slot_yielded():
                yield

    def _release_gpu_slot_for_finalize(self) -> None:
        """Worker-internal: TERMINAL GPU-slot release at the decode->finalize
        handoff. The handler is done with GPU compute; the encode + upload
        tail proceeds slotless so a request on ANOTHER live instance can take
        the card instead of idling it.

        READ THIS BEFORE SIZING ANY OVERLAP AGAINST THIS RELEASE. It does NOT
        hand the card to a SAME-instance follower: the lock order is instance
        gate -> GPU permit, so the follower is queued on ``run_lock`` and this
        releases only the permit; the gate stays held until the handler
        RETURNS. Releasing the gate here too would let a follower mutate the
        instance graph under a running handler, so it is deliberately not
        done. Real fleet overlap comes from deferred outputs instead, which
        move encode+upload out of the handler so gate + permit fall together.
        Endpoints excluded from that arming (``output_mode == "stream"``,
        async-gen handlers) still serialize their whole tail against a
        same-instance follower.

        Unlike :meth:`_gpu_slot_yielded` there is no reacquire — a finishing
        request must never block behind the next request's denoise just to
        return. The executor's post-handler release no-ops (lease transitions
        are once-only), so the semaphore balance stays exact. Tenant GPU work
        after this call runs unscheduled — finalize helpers call it only once
        frames are on the host. No-op without a lease (CPU jobs, local runs)
        or when already yielded."""
        lease = self._gpu_slot_lease
        if lease is not None and lease.yield_slot():
            logger.info(
                "request %s: GPU slot released for finalize; encode/upload "
                "overlaps the next request's compute", self.request_id)
            notify = self._on_finalize_release
            if notify is not None:
                try:
                    notify()
                except Exception:
                    logger.exception(
                        "finalize-release notification failed (non-fatal)")

    def _emit_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Worker-internal: emit a progress/event payload (best-effort)."""
        if not self._emitter:
            logger.debug("emit(%s) dropped: no emitter configured", event_type)
            return
        self._emitter({
            "request_id": self._request_id,
            "type": event_type,
            "payload": payload or {},
            "timestamp": time.time(),
        })

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Bracket one stage of this request for :data:`JobMetrics.stage_ms`
::

            with ctx.stage("text_encode"):
                embeds = encode(prompt)

        The framework already times the permit wait, input fetch, denoise
        steps, image/video encode, credential stamp and upload; brackets add
        what only the endpoint knows (text encode, scheduler setup, an
        explicit VAE decode). Nested stages are charged exclusively, so the
        map always reconciles with ``runtime_ms``. Known names are classified
        GPU-busy / small-GPU / GPU-idle (see ``stage_timing``); unknown names
        are reported but left unclassified rather than guessed.
        """
        with self._stages.stage(name):
            yield

    def progress(
        self,
        progress: Optional[float] = None,
        stage: Optional[str] = None,
        *,
        step: Optional[int] = None,
        total: Optional[int] = None,
        position: Optional[float] = None,
        phase: Optional[str] = None,
    ) -> None:
        """Report progress (best-effort, rides ``request.progress``).

        This is the USER-facing stream — the cozy-art job feed renders it
        directly. For platform/operator-only diagnostics use :meth:`log`.

        Two spellings of ONE surface, so a body is portable between
        ``@endpoint`` and ``@job`` unchanged (pgw#1294):

        * the request spelling — ``ctx.progress(0.5, "denoise", step=5,
          total=20)``: ``progress`` is a 0..1 fraction, ``step``/``total`` the
          exact counter, ``stage`` the label.
        * the job spelling — ``ctx.progress(position=4096, total=31_000_000,
          phase="download")``: ``position`` is a MONOTONIC position within
          ``phase`` and ``total`` its extent. The fraction is derived.

        ``position`` and ``step`` are the same quantity under two names, as are
        ``phase`` and ``stage``; passing both spellings of one quantity raises.
        The positional job form ``ctx.progress(4096, 31_000_000, "download")``
        is NOT accepted — the second positional argument has always been the
        stage label and silently reinterpreting it by type is exactly the
        lying instrument this method refuses to be.

        **Position is MONOTONIC and load-bearing.** Hub liveness for a job is
        position ADVANCE within a phase budget — never pulse, never duration
        (th#1908) — so a position that goes BACKWARDS within a phase raises
        :class:`~gen_worker.api.errors.NonMonotonicProgressError` here rather
        than reporting a number nothing can trust. Restart a phase by naming a
        new phase.

        A call carrying a position is ALSO a stage-timing mark, so endpoints
        driving their own step loop get a denoise window without any endpoint
        change.
        """
        if position is not None and step is not None and float(position) != float(step):
            raise ValueError(
                "ctx.progress: position= and step= are the same quantity under "
                f"two names and disagree ({position!r} vs {step!r}) — pass one"
            )
        if phase is not None and stage is not None and str(phase) != str(stage):
            raise ValueError(
                "ctx.progress: phase= and stage= are the same quantity under "
                f"two names and disagree ({phase!r} vs {stage!r}) — pass one"
            )
        if stage is not None and not isinstance(stage, str):
            raise TypeError(
                f"ctx.progress: the second positional argument is the stage "
                f"label (a str), got {stage!r}. For the job spelling pass "
                "keywords: ctx.progress(position=..., total=..., phase=...)"
            )
        label = phase if phase is not None else stage
        pos = float(position) if position is not None else (
            float(step) if step is not None else None
        )
        if pos is not None:
            self._advance_position(str(label or ""), pos)
            timer = getattr(self, "_stages", None)
            if timer is not None and float(pos).is_integer():
                timer.mark_step(str(label or "denoise"), int(pos))
        fraction = progress
        if fraction is None and pos is not None and total:
            fraction = pos / float(total)
        payload: Dict[str, Any] = {"progress": 0.0 if fraction is None else fraction}
        if label is not None:
            payload["stage"] = label
        if pos is not None:
            # `step` is the key the HUB parses off this envelope
            # (`runtimestore.ParseRequestProgressPayload`, confirmed against
            # th#2050's landed `forkJobProgress`), so it is emitted for EVERY
            # position — a fractional position that only landed in `position`
            # would be invisible to the liveness sweep that condemns the pod.
            # `position` carries the exact value beside it.
            payload["position"] = pos
            payload["step"] = int(pos)
        if total is not None:
            payload["total"] = int(total)
        self._emit_event("request.progress", payload)

    def _advance_position(self, phase: str, position: float) -> None:
        """Record a monotonic position within ``phase``; refuse a regression.

        The counter this feeds is what a stall detector reads, so a position
        that moves backwards is not a cosmetic defect: it manufactures
        apparent liveness out of a loop that is going nowhere.
        """
        last = self._progress_positions.get(phase)
        if last is not None and position < last:
            from ..api.errors import NonMonotonicProgressError

            raise NonMonotonicProgressError(phase, last, position)
        self._progress_positions[phase] = position
        # Lazy: activity pulls in pb/psutil, and request_context is on the
        # `import gen_worker` path the discovery build step keeps thin.
        from .. import activity as activity_mod

        activity_mod.note_progress()

    def position(self, phase: str = "") -> Optional[float]:
        """The last position reported for ``phase``, or None. Read by the job
        runner's progress-liveness watch; also useful in tests."""
        return self._progress_positions.get(phase)

    # -- the hub-write declaration ------------------------------------------

    @property
    def publishes(self) -> bool:
        """The executing function DECLARED ``publishes=True``.

        The declaration says MAY write; the request still says WHERE. It is
        also what justifies the hub minting the worker-capability write grant
        at launch — grants remain the WHOLE write authority."""
        return self._publishes

    @property
    def emits_media(self) -> bool:
        """This function MAY write media.

        True for every endpoint (media is the product) and for a job that
        declared ``emits_media=True``; False only for a job that declared
        none, whose token carries no ``upload_media`` grant."""
        return self._emits_media is not False

    def _require_media_declaration(self, surface: str) -> None:
        """Refuse a media write to a job that declared none.

        Typed, and BEFORE a byte moves — the hub minted no `upload_media`
        grant, so this is the same refusal arriving at the call site.
        """
        if self._emits_media is not False:
            return
        from ..api.errors import MediaNotDeclaredError

        raise MediaNotDeclaredError(surface)

    #: Producer KINDS that still imply write authority. TRANSITIONAL: th#2052
    #: cuts it, at which point the declaration is the only justification and
    #: this tuple (and the branch reading it) is deleted whole. Until then a
    #: fleet endpoint that has not yet added ``publishes=True`` keeps working
    #: and is told so, loudly, once per call.
    _KIND_IMPLIES_PUBLISH = ("conversion", "training", "dataset")

    def _require_publish_declaration(self, surface: str) -> None:
        """Refuse the publisher surface to code that never declared it.

        Typed, and BEFORE a byte moves — the hub never minted the grant for an
        undeclared function, so this is the same refusal arriving at the call
        site instead of after an upload.
        """
        if self._publishes:
            return
        kind = self._tensor_upload_execution_kind()
        if kind in self._KIND_IMPLIES_PUBLISH:
            self.log(
                f"{surface} admitted by kind={kind!r} and NOT by declaration — "
                "add publishes=True to the decorator. Kind stops implying "
                "write authority at th#2052 and this call will then refuse.",
                level="warning",
                surface=surface,
                kind=kind,
            )
            return
        from ..api.errors import PublishNotDeclaredError

        raise PublishNotDeclaredError(surface)

    def _emit_checkpoint_saved(
        self,
        ref: str,
        *,
        step_number: Optional[int] = None,
        epoch_number: Optional[int] = None,
        output_kind: Optional[str] = None,
        size_bytes: Optional[int] = None,
    ) -> None:
        """Emit a checkpoint-saved event (best-effort; rides JobProgress)."""
        payload: Dict[str, Any] = {"ref": ref}
        if step_number is not None:
            payload["step_number"] = int(step_number)
        if epoch_number is not None:
            payload["epoch_number"] = int(epoch_number)
        if output_kind:
            payload["output_kind"] = str(output_kind)
        if size_bytes is not None:
            payload["size_bytes"] = int(size_bytes)
        self._emit_event("request.checkpoint", payload)

    def log(self, message: str, level: LogLevel = "info", **fields: Any) -> None:
        """Emit a request-scoped OPERATOR diagnostic (rides ``request.log``).

        This is the PLATFORM/OPERATOR debug stream, never user-facing:
        tensorhub persists it under an operator-only event kind and never
        serves it on a tenant-facing surface (SSE job feed, events.bin, poll).
        See proto/CONTRACT.md § "The ctx event lane" for the wire-level
        routing contract.

        One-line rule for authors: module-level ``logging.getLogger(__name__)``
        for boot-time/cross-request logging; ``ctx.log`` for anything scoped
        to THIS request you'd want when debugging it (resolved model/
        scheduler choice, retry/degradation detail, malformed-input detail);
        ``ctx.progress`` for what the human watching the job should see.
        There is deliberately no user-visible counterpart to ``ctx.log``.

        ``**fields`` rides the payload as structured JSON extras (e.g.
        ``ctx.log("OOM retry", level="warning", free_gb=2.1, rung="offload")``)
        so operators can filter/grep without parsing the message string.
        Best-effort like every ctx event: dropped silently if unencodable or
        no emitter is configured.
        """
        payload: Dict[str, Any] = {"message": message, "level": level}
        if fields:
            payload["fields"] = fields
        self._emit_event("request.log", payload)

    def _c2pa_manifest_kwargs(self) -> Dict[str, Any]:
        model_refs = [str(v) for v in (self._models or {}).values()]
        model_refs += [
            str(ov.get("ref", ""))
            for overlays in (self._loras or {}).values()
            for ov in overlays
        ]
        return {"request_id": self._request_id, "models": model_refs}

    def _c2pa_sign_bytes(self, ref: str, data: bytes) -> bytes:
        """C2PA-sign media payloads at the finalize seam.

        Returns ``data`` unchanged when signing is unconfigured or the
        payload is not a signable media format; raises when signing is
        configured but fails (an unlabeled asset must not ship silently).
        """
        # Deferred: content_credentials (c2pa) is +132 modules on the
        # `import gen_worker` path.
        from .. import content_credentials

        with self._stages.stage("credential_stamp"):
            return content_credentials.sign_media_bytes(
                data, ref=ref, **self._c2pa_manifest_kwargs())

    def _c2pa_sign_file(self, ref: str, src: str) -> Optional[str]:
        """File variant of :meth:`_c2pa_sign_bytes` — returns a signed temp
        path (caller unlinks) or None when signing doesn't apply."""
        # Deferred: content_credentials (c2pa) is +132 modules on the
        # `import gen_worker` path.
        from .. import content_credentials

        with self._stages.stage("credential_stamp"):
            return content_credentials.sign_media_file(
                src, ref=ref, **self._c2pa_manifest_kwargs())

    # Inline-bytes threshold: when the client requested
    # `Prefer: bytes=inline` AND the payload is at or below this many
    # bytes, skip the tensorhub upload and return the bytes directly
    # on the Asset (see Asset.bytes docstring). Default ~1 MiB matches
    # the orchestrator-side default ORCHESTRATOR_OUTPUT_INLINE_MAX_BYTES.
    _SAVE_BYTES_INLINE_THRESHOLD = 4 * 1024 * 1024

    def save_bytes(self, ref: str, data: bytes) -> Asset:
        return self._save_bytes(ref, data, allow_inline=True, media=True)

    def _save_result_envelope(self, ref: str, data: bytes) -> Asset:
        """Store the RESULT ENVELOPE blob, always as a real upload.

        The envelope is worker->orchestrator transport, not a client-visible
        media output: `JobResult` owns its own inline-vs-blob_ref choice at
        its own ceiling (`executor.INLINE_RESULT_MAX_BYTES`). The client's
        `Prefer: bytes=inline` hint is about MEDIA and must not reach here, or
        a `blob_ref` names a blob that was never uploaded.
        """
        return self._save_bytes(ref, data, allow_inline=False, media=False)

    def _save_bytes(
        self, ref: str, data: bytes, *, allow_inline: bool, media: bool = True,
    ) -> Asset:
        # `media=False` is the result envelope: worker->orchestrator transport,
        # not a client-visible output, so it rides no media grant.
        if media:
            self._require_media_declaration("save_bytes")
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("save_bytes expects bytes")
        data = bytes(data)
        ref = _normalize_output_ref(ref)
        data = self._c2pa_sign_bytes(ref, data)
        _enforce_output_file_size_limit(len(data))

        local_path = self._resolve_local_output_path(ref)
        if local_path:
            p = Path(local_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            return Asset(
                ref=ref,
                owner=self._owner,
                local_path=str(p),
                mime_type=None,
                size_bytes=len(data),
                sha256=sha,
            )

        # Inline path: client signaled `Prefer: bytes=inline` and the
        # payload fits under the inline threshold. Skip the tensorhub
        # upload entirely — return raw bytes on the Asset and let the
        # orchestrator pass them through to the client. msgpack on the
        # wire keeps the bytes raw (no base64 inflation); JSON clients
        # get them base64-encoded by Go's encoding/json on the way out.
        output_format = str(
            (self._execution_hints or {}).get("output_format", "")
        ).strip().lower()
        if (
            allow_inline
            and output_format == "inline"
            and len(data) <= self._SAVE_BYTES_INLINE_THRESHOLD
        ):
            return Asset(
                ref=ref,
                owner=self._owner,
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                inline_bytes=data,
            )

        stream = self._open_output_stream(ref, create=False, expected_size_bytes=len(data))
        with self._gpu_slot_yielded():
            stream.write(data)
            out = stream.finalize()
        if isinstance(out, Asset):
            return out
        raise RuntimeError("file save failed (invalid_asset_response)")

    def save_image(
        self,
        image: "Image.Image",
        ref: Optional[str] = None,
        *,
        format: ImageFormat = DEFAULT_IMAGE_FORMAT,
        quality: int = DEFAULT_IMAGE_QUALITY,
        lossless: bool = False,
        **encode_kwargs: Any,
    ) -> ImageAsset:
        """Encode + save an image; returns a typed :class:`ImageAsset`.

        ``format`` is ``webp`` (the platform default), ``png``, or ``jpg``.
        ``quality`` applies to webp/jpg; ``lossless`` is webp-only. The
        extension is derived from the format when ``ref`` has no suffix.

        On the serve path the encode + C2PA stamp + upload are DEFERRED to the
        finalize tail: the returned handle carries its ``ref`` immediately and
        its bytes fields fill in after the handler returns and the GPU permit
        is released, so the encode overlaps the next request's denoise instead
        of holding the card. Reading a bytes field (``size_bytes``,
        ``sha256``, ``inline_bytes``, ...) inside the handler is still
        correct; it just encodes inline and loses the overlap.
        """
        _pil_format, ext = image_format(format)
        if ref is None or str(ref).strip() == "":
            ref = f"outputs/{self.request_id}/image{ext}"
        else:
            ref = _normalize_output_ref(str(ref))
            if Path(ref).suffix == "":
                ref += ext
        # Output-integrity floor, on the pixels, before anything is encoded or
        # uploaded. Judged EAGERLY even on the deferred path so a rejected
        # render raises where the request can see it rather than inside the
        # post-handler finalize drain. Charged to its OWN stage so its wall is
        # attributed without borrowing `image_encode`.
        if judged(self):
            with self._stages.stage("output_integrity"):
                guard_image(image, ref=ref)
        if not self._deferred.armed:
            with self._stages.stage("image_encode"):
                payload, _ext = encode_image(
                    image, format=format, quality=quality, lossless=lossless,
                    **encode_kwargs,
                )
            return _as_asset(self.save_bytes(ref, payload), ImageAsset)

        # Snapshot the pixels: a handler that keeps mutating its PIL object
        # must not change what gets uploaded. A few ms of memcpy against a
        # ~1.1s encode.
        copier = getattr(image, "copy", None)
        snapshot = copier() if callable(copier) else image
        handle = DeferredImageAsset(ref=ref, owner=self._owner)
        target = ref

        def _materialize() -> None:
            with self._stages.stage("image_encode"):
                payload, _ext = encode_image(
                    snapshot, format=format, quality=quality,
                    lossless=lossless, **encode_kwargs,
                )
            fill_from(handle, self.save_bytes(target, payload))

        pending = PendingOutput(ref, _materialize)
        handle.__dict__["_gw_pending"] = pending
        self._deferred.defer(pending)
        return handle

    # -- deferred finalize tail ---------------------------------

    def _arm_deferred_outputs(self) -> None:
        """Worker-internal: let ``save_image`` defer its encode+upload to the
        finalize tail. Armed by the executor only where a post-handler tail
        exists (never for streaming handlers, which serialize mid-handler)."""
        self._deferred.armed = True

    def _drain_deferred_outputs(self) -> int:
        """Worker-internal: materialize the deferred outputs. Called by the
        executor AFTER the GPU permit is released, so this is the work that
        overlaps the next request's compute. Returns how many ran here."""
        return self._deferred.drain()

    def save_audio(
        self,
        audio: "np.ndarray[Any, Any] | torch.Tensor | bytes",
        ref: Optional[str] = None,
        *,
        sample_rate: int = 44100,
        format: str = "wav",
    ) -> AudioAsset:
        """Encode + save audio; returns a typed :class:`AudioAsset`.

        ``audio`` is a numpy array (frames[, channels]) or a torch tensor;
        raw ``bytes`` are stored as-is (assumed already encoded).
        """
        fmt = str(format or "wav").strip().lower()
        if ref is None or str(ref).strip() == "":
            ref = f"outputs/{self.request_id}/audio.{fmt}"
        else:
            ref = _normalize_output_ref(str(ref))
            if Path(ref).suffix == "":
                ref += f".{fmt}"
        if isinstance(audio, (bytes, bytearray)):
            data = bytes(audio)
        else:
            try:
                import numpy as np
                import soundfile as sf
            except ImportError as exc:
                from ..api.errors import ValidationError

                raise ValidationError(
                    "save_audio needs the audio extra: pip install 'gen-worker[audio]'"
                ) from exc
            arr: Any = audio
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T  # (channels, frames) -> (frames, channels)
            buf = BytesIO()
            sf.write(buf, arr, int(sample_rate), format=fmt.upper())
            data = buf.getvalue()
        return _as_asset(self.save_bytes(ref, data), AudioAsset)

    def save_video(
        self,
        video: "bytes | str | os.PathLike[str]",
        ref: Optional[str] = None,
        *,
        format: str = "mp4",
    ) -> VideoAsset:
        """Save an encoded video (bytes or a local file path); returns a
        typed :class:`VideoAsset` with probed container metadata
        (duration_s/fps/width/height/has_audio/sample_rate, best-effort)."""
        fmt = str(format or "mp4").strip().lower()
        if ref is None or str(ref).strip() == "":
            ref = f"outputs/{self.request_id}/video.{fmt}"
        else:
            ref = _normalize_output_ref(str(ref))
            if Path(ref).suffix == "":
                ref += f".{fmt}"
        asset: VideoAsset
        if isinstance(video, (bytes, bytearray)):
            asset = _as_asset(self.save_bytes(ref, bytes(video)), VideoAsset)
        else:
            asset = _as_asset(self.save_file(ref, video), VideoAsset)
        try:
            from ..io import probe_video

            for key, value in probe_video(
                bytes(video) if isinstance(video, (bytes, bytearray))
                else os.fspath(video)
            ).items():
                setattr(asset, key, value)
        except Exception:
            logger.debug("save_video: metadata probe failed", exc_info=True)
        return asset

    def save_file(
        self,
        ref: str,
        local_path: str | os.PathLike[str],
        *,
        create: bool = False,
    ) -> Asset:
        """Upload a local file as an output Asset.

        ``create=True`` requires the ref to be new (local backend: the
        destination path must not exist; remote: the upload session is
        opened in create mode).
        """
        ref = _normalize_output_ref(ref)
        src = str(os.fspath(local_path) if local_path else "").strip()
        if not src:
            raise ValueError("local_path is required")
        if not os.path.exists(src):
            raise FileNotFoundError(src)

        # C2PA signing: media files upload as a signed temp copy;
        # the caller's file is never mutated. No-op unless signing is
        # configured and the file is a signable media format.
        signed_tmp = self._c2pa_sign_file(ref, src)
        if signed_tmp is not None:
            try:
                return self._save_file_inner(ref, signed_tmp, create=create)
            finally:
                try:
                    os.unlink(signed_tmp)
                except OSError:
                    pass
        return self._save_file_inner(ref, src, create=create)

    def _save_file_inner(self, ref: str, src: str, *, create: bool = False) -> Asset:
        self._require_media_declaration("save_file")
        size = int(os.path.getsize(src))
        _enforce_output_file_size_limit(size)

        local_out = self._resolve_local_output_path(ref)
        if local_out:
            dst = Path(local_out)
            if create and dst.exists():
                raise RuntimeError("output path already exists")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(src, "rb") as fin, open(dst, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=1024 * 1024)
            sha = _sha256_file(str(dst))
            return Asset(
                ref=ref,
                owner=self._owner,
                local_path=str(dst),
                mime_type=None,
                size_bytes=size,
                sha256=sha,
            )
        # Reserve aggregate-bytes budget (issue #269 back-pressure) — held
        # until the upload completes. Reentrant: nested save_file from
        # inside save_checkpoint's non-streaming branch is a no-op for
        # the same thread.
        with self._get_upload_budget_gate().reserve(size):
            stream = self._open_output_stream(ref, create=create, expected_size_bytes=size)
            with open(src, "rb") as fin:
                while True:
                    chunk = fin.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
            out = stream.finalize()
            if isinstance(out, Asset):
                return out
            raise RuntimeError("file save failed (invalid_asset_response)")

    def _open_output_stream(
        self,
        ref: str,
        *,
        create: bool = False,
        expected_size_bytes: Optional[int] = None,
    ) -> _RequestOutputStream:
        """Library-internal: chunk-writable output stream finalizing to an Asset."""
        return _RequestOutputStream(
            ctx=self,
            ref=ref,
            kind="asset",
            create=create,
            expected_size_bytes=expected_size_bytes,
        )

    # Admin-plane visibility toggles deliberately do not live here: visibility
    # flips belong in cozyctl / the tensorhub UI, not on a per-request object.


# ---------------------------------------------------------------------------
# Kind-specific subclasses. RequestContext is the per-inference base;
# conversion, dataset-producing and trainer endpoints get richer subclasses
# sharing `_PublisherMixin` for the producer-contract HTTP helpers (blob fetch
# + materialization by digest). Checkpoint PUBLISHING is not here: producer
# endpoints call gen_worker.convert.publish_flavors (the /commits path).
# ---------------------------------------------------------------------------


class _PublisherMixin:
    """Producer-contract helpers for ``JobContext``: blob fetch by digest and
    ``materialize_blob``.
    Always combined with ``RequestContext`` via multiple inheritance (so
    ``self`` has ``_file_api_base_url`` / ``_owner`` /
    ``_get_worker_capability_token``).

    Producer-only STATE lives here too: the reserved
    ``source``/``destination``/``text_encoder``/``candidate`` payload structs,
    the hf token, and their materialized paths — a plain inference
    ``RequestContext`` never carries them.

    Not a public surface: tenants should never import this directly.
    """

    def __init__(
        self,
        *args: Any,
        source_info: Optional[Dict[str, Any]] = None,
        destination_info: Optional[Dict[str, Any]] = None,
        text_encoder_info: Optional[Dict[str, Any]] = None,
        candidate_info: Optional[Dict[str, Any]] = None,
        resume_from_info: Optional[Dict[str, Any]] = None,
        hf_token: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Reserved-name producer contract attributes, populated by the
        # executor before invoking tenant code when the payload declares the
        # reserved `source`/`destination` struct fields.
        self._source_info = dict(source_info or {})
        self._destination_info = dict(destination_info or {})
        self._source_path: Optional[str] = None
        self._text_encoder_info = dict(text_encoder_info or {})
        self._text_encoder_path: Optional[str] = None
        self._candidate_info = dict(candidate_info or {})
        self._candidate_path: Optional[str] = None
        self._resume_from_info = dict(resume_from_info or {})
        self._resume_from_path: Optional[str] = None
        self._hf_token = (hf_token or "").strip()
    if TYPE_CHECKING:
        # The host contract: everything this mixin borrows from
        # RequestContext, declared so mypy checks the mixin against the
        # composition instead of erroring attr-defined on every use.
        request_id: str
        cancelled: bool
        _file_api_base_url: Optional[str]
        _worker_capability_token: Optional[str]
        _job_id: Optional[str]

        def save_bytes(self, ref: str, data: bytes) -> Asset: ...
        def save_file(
            self, ref: str, local_path: "str | os.PathLike[str]",
            *, create: bool = ...,
        ) -> Asset: ...
        def _open_output_stream(
            self, ref: str, *, create: bool = ...,
            expected_size_bytes: Optional[int] = ...,
        ) -> "_RequestOutputStream": ...
        def _emit_checkpoint_saved(
            self, ref: str, *, step_number: Optional[int] = ...,
            epoch_number: Optional[int] = ..., output_kind: Optional[str] = ...,
            size_bytes: Optional[int] = ...,
        ) -> None: ...
        def _get_upload_budget_gate(self) -> "BudgetGate": ...
        def _get_worker_capability_token(self) -> str: ...
        def _repo_job_upload_scope(self) -> "Optional[tuple[str, str, str]]": ...
        def _repo_job_release(self) -> str: ...
        def _require_repo_job_scope_for_tensors(self, ref: str) -> None: ...
        def _require_publish_declaration(self, surface: str) -> None: ...
        def _should_stream_output_to_file_api(self, ref: str) -> bool: ...

    @property
    def hf_token(self) -> str:
        """HuggingFace API token for gen_worker.convert / conversion helpers.

        Empty string when unconfigured — helpers fall back to
        unauthenticated calls (public repos work)."""
        return self._hf_token

    # Reserved-name conversion/training contract. `source` and `destination`
    # come from the job payload's reserved fields; `source_path` is populated
    # by the library after it materializes the source snapshot locally.
    @property
    def source(self) -> dict[str, Any]:
        return dict(self._source_info)

    @property
    def source_path(self) -> Optional[str]:
        return self._source_path

    @property
    def destination(self) -> dict[str, Any]:
        return dict(self._destination_info)

    def _set_source_path(self, path: str) -> None:
        """Library-internal: called after source materialization."""
        self._source_path = str(path) if path else None

    # Second reserved-name model input: a wholly independent repo from
    # `source`, materialized the same way. Empty/None when the payload
    # declares no `text_encoder`.
    @property
    def text_encoder(self) -> dict[str, Any]:
        return dict(self._text_encoder_info)

    @property
    def text_encoder_path(self) -> Optional[str]:
        return self._text_encoder_path

    def _set_text_encoder_path(self, path: str) -> None:
        """Library-internal: called after text_encoder materialization."""
        self._text_encoder_path = str(path) if path else None

    # Fourth reserved-name model input: the arm a two-ref eval COMPARES
    # against `source`, rather than a component it builds from. Same
    # materialization path; empty/None when the payload declares none.
    @property
    def candidate(self) -> dict[str, Any]:
        return dict(self._candidate_info)

    @property
    def candidate_path(self) -> Optional[str]:
        return self._candidate_path

    def _set_candidate_path(self, path: str) -> None:
        """Library-internal: called after candidate materialization."""
        self._candidate_path = str(path) if path else None

    # pgw#1242/te#185: the fifth reserved model input — a previously PUBLISHED
    # checkpoint to CONTINUE from, which is what lets a training endpoint resume
    # across pod loss instead of restarting a multi-hour run from zero. Absent on
    # every existing payload struct — stays {} and is a no-op.
    @property
    def resume_from(self) -> dict[str, Any]:
        return dict(self._resume_from_info)

    @property
    def resume_from_path(self) -> Optional[str]:
        return self._resume_from_path

    def _set_resume_from_path(self, path: str) -> None:
        """Library-internal: called after resume_from materialization."""
        self._resume_from_path = str(path) if path else None

    def save_checkpoint(
        self,
        ref: str,
        local_path: str | os.PathLike[str],
        format: Optional[str] = None,
        *,
        step_number: Optional[int] = None,
        epoch_number: Optional[int] = None,
        output_kind: Optional[str] = None,
    ) -> Tensors:
        """Save checkpoint/model-weight bytes and return a tensor artifact."""
        src = str(os.fspath(local_path) if local_path else "").strip()
        if not src:
            raise ValueError("local_path is required")
        if not os.path.exists(src):
            raise FileNotFoundError(src)

        def _feed(stream: _RequestOutputStream) -> None:
            with open(src, "rb") as fin:
                while True:
                    chunk = fin.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)

        return self._publish_checkpoint(
            ref,
            size=int(os.path.getsize(src)),
            format=format,
            feed=_feed,
            fallback=lambda r: self.save_file(r, src),
            step_number=step_number,
            epoch_number=epoch_number,
            output_kind=output_kind,
        )

    def _publish_checkpoint(
        self,
        ref: str,
        *,
        size: int,
        format: Optional[str],
        feed: Callable[[_RequestOutputStream], object],
        fallback: Callable[[str], Asset],
        step_number: Optional[int],
        epoch_number: Optional[int],
        output_kind: Optional[str],
    ) -> Tensors:
        """Shared checkpoint-publish core.

        Job-scoped writes publish through the /commits stream so
        the returned Tensors carries a blake3 digest + blob_digest and each
        save materializes one finalized repo revision; everything else falls
        back to the plain asset save the ``fallback`` callable provides.
        """
        ref = _normalize_output_ref(ref)
        self._require_publish_declaration("save_checkpoint")
        self._require_repo_job_scope_for_tensors(ref)
        _enforce_output_file_size_limit(size)
        fmt = str(format or "").strip() or _infer_tensors_format(ref)

        def _emit() -> None:
            self._emit_checkpoint_saved(
                ref, step_number=step_number, epoch_number=epoch_number,
                output_kind=output_kind, size_bytes=size,
            )

        # Reserve aggregate-bytes budget (issue #269 back-pressure). Held
        # across either branch (streaming or asset-save fallthrough); the
        # fallback saves are reentrancy-aware so their inner reserve() is a
        # no-op for the same thread.
        with self._get_upload_budget_gate().reserve(size):
            if self._repo_job_upload_scope() is not None and self._should_stream_output_to_file_api(ref):
                stream = self.open_checkpoint_stream(
                    ref,
                    format=fmt,
                    expected_size_bytes=size,
                    step_number=step_number,
                    epoch_number=epoch_number,
                )
                feed(stream)
                out = stream.finalize()
                if isinstance(out, Tensors):
                    _emit()
                    return out
                raise RuntimeError("file save failed (invalid_tensors_response)")

            asset = fallback(ref)
        _emit()
        return Tensors(
            ref=asset.ref,
            owner=asset.owner,
            local_path=asset.local_path,
            format=fmt,
            size_bytes=asset.size_bytes,
            sha256=asset.sha256,
            download_token=asset.download_token,
        )

    def open_checkpoint_stream(
        self,
        ref: str,
        *,
        format: Optional[str] = None,
        expected_size_bytes: Optional[int] = None,
        step_number: Optional[int] = None,
        epoch_number: Optional[int] = None,
    ) -> _RequestOutputStream:
        """Open a chunk-writable output stream that finalizes to Tensors."""
        ref = _normalize_output_ref(ref)
        self._require_publish_declaration("open_checkpoint_stream")
        self._require_repo_job_scope_for_tensors(ref)

        return _RequestOutputStream(
            ctx=_cast("RequestContext[Any]", self),
            ref=ref,
            kind="checkpoint",
            format=format,
            expected_size_bytes=expected_size_bytes,
            step_number=step_number,
            epoch_number=epoch_number,
        )

    def _download_blob_by_digest(
        self, digest: str, dest: Path, *, origin: str = REF_ORIGIN_PAYLOAD,
    ) -> None:
        """Fetch a blob by ``<algo>:<hex>`` digest to ``dest``.

        Uses the by-digest CAS read endpoint — works for any blob uploaded
        via ``save_checkpoint`` regardless of whether it is a checkpoint file
        or a dataset file. The digest must be ALGORITHM-TAGGED: the hub keys
        two CAS namespaces on different algorithms (repo-CAS is sha256,
        dataset-CAS is blake3), so a bare hex string does not name a blob and
        is refused here for the same reason the hub refuses it.

        ``origin`` is the provenance of the ADDRESS, and it is the only thing
        that decides how a terminal miss classifies. See
        :data:`REF_ORIGIN_PAYLOAD`.

        The cap and the verification are both enforced DURING the stream:

        * the response's declared length is the cap, checked per chunk. A
          source with no ``Content-Length`` is refused — both arms of the hub's
          by-digest route declare one (a 302 to a presigned object, or
          ``DataFromReader`` with the object's size), so its absence means the
          responder is not the one this contract describes. ``identity``
          encoding is demanded so the declared length and the delivered length
          are the same quantity, which also makes a gzip-bomb body impossible
          rather than merely detectable.
        * the digest ADDRESSES the bytes, so hashing them costs one pass and
          refusing a mismatch costs nothing. It lands via a ``.part`` rename,
          so a refused fetch cannot leave a partial file that a resume check
          would read as complete.
        """
        # Lazy: request_context is on the `import gen_worker` path, which the
        # `python -m gen_worker.discover` build step must keep requests-free.
        import requests

        from ..bounded_stream import copy_bounded

        base = (self._file_api_base_url or "").strip().rstrip("/")
        token = self._get_worker_capability_token()
        digest_norm = _parse_cas_digest(digest, origin=origin)
        url = f"{base}/api/v1/blobs/{urllib.parse.quote(digest_norm, safe=':')}/content"
        headers = {"Authorization": f"Bearer {token}", "Accept-Encoding": "identity"}
        caller_supplied = origin == REF_ORIGIN_PAYLOAD
        algo, _, want_hex = digest_norm.partition(":")
        hasher = _cas_hasher(algo)
        tmp = dest.with_name(dest.name + ".part")
        with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
            if resp.status_code in (401, 403):
                if caller_supplied:
                    raise BlobForbiddenError(digest, resp.status_code)
                from ..hub_error import hub_error_of

                raise AuthError(
                    f"blob fetch unauthorized ({resp.status_code}) digest={digest}: "
                    f"{hub_error_of(resp).detail()}".rstrip(": ")
                )
            if resp.status_code == 404:
                if caller_supplied:
                    raise BlobNotFoundError(digest)
                raise RuntimeError(f"blob fetch 404 for digest={digest}")
            if resp.status_code < 200 or resp.status_code >= 300:
                raise RuntimeError(f"blob fetch failed ({resp.status_code}) digest={digest}: {resp.text[:256]}")
            raw_length = str(resp.headers.get("Content-Length") or "").strip()
            if not raw_length.isdigit():
                raise RuntimeError(
                    f"blob fetch refused: digest={digest_norm} came back with no "
                    f"declared length ({raw_length!r}) — nothing bounds the transfer"
                )
            declared = int(raw_length)
            _refuse_without_disk_room(tmp.parent, declared, digest_norm)
            try:
                with open(tmp, "wb") as f:
                    # An empty blob is a legal object with a real digest, and
                    # zero is not a bound `copy_bounded` will accept — write
                    # the file and let the digest check speak.
                    total = 0 if declared == 0 else copy_bounded(
                        resp.iter_content(chunk_size=1024 * 1024),
                        f.write,
                        limit_bytes=declared,
                        what=f"blob {digest_norm}",
                        hasher=hasher,
                    )
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        if total != declared:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"blob fetch truncated: digest={digest_norm} declared {declared} "
                f"bytes, delivered {total}"
            )
        got = hasher.hexdigest()
        if got != want_hex:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"blob digest mismatch: {digest_norm} addresses bytes that hash "
                f"to {algo}:{got[:16]}…"
            )
        tmp.replace(dest)

    def _fetch_platform_blob(self, digest: str, dest: Path) -> None:
        """`_download_blob_by_digest` bound to PLATFORM provenance — for
        addresses the hub itself produced (dataset manifests). A miss there
        is a platform fault, not the caller's."""
        self._download_blob_by_digest(digest, dest, origin=REF_ORIGIN_PLATFORM)

    def materialize_blob(
        self, digest: str, dest: "str | os.PathLike[str]",
        *, origin: str = REF_ORIGIN_PAYLOAD,
    ) -> Path:
        """Fetch a blob by ``<algo>:<hex>`` content-addressed digest.

        Returns the ``Path`` the blob was written to. Public so tenants that
        handle a digest directly (e.g. consuming a snapshot manifest emitted
        by an earlier conversion) can pull the bytes themselves.

        ``origin`` defaults to :data:`REF_ORIGIN_PAYLOAD` because that is the
        untrusted case and the safe default: a bad address fails the REQUEST
        typed rather than indicting the release. Pass
        :data:`REF_ORIGIN_PLATFORM` for a digest the platform produced.

        The bytes are capped at the response's declared length and verified
        against the digest before ``dest`` exists.
        """
        dest_path = Path(os.fspath(dest))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self._download_blob_by_digest(digest, dest_path, origin=origin)
        return dest_path

    def checkpoint_dir(self, *, key: str) -> Path:
        """Return a JOB-SCOPED SCRATCH dir keyed by (job_id, key) — a stable
        working directory for trainer ``output_dir`` use within one job.

        NOT persistent storage: it lives under ``tempfile.gettempdir()`` —
        pod-local ``/tmp``, gone at pod churn/eviction. Do not park resume
        state here; durable resume goes through published checkpoints
        (``save_checkpoint`` / the job's source repo). It IS a deterministic
        path that survives handler retries within the same pod/process, so a
        trainer can wipe-and-recreate it at start without colliding with
        other jobs.
        """
        job_id = self._job_id or self.request_id or "x"
        base = Path(tempfile.gettempdir()) / "txform-persistent" / str(job_id)
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        dir_path = base / safe_key
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    # ----- dataset materialization ------------------------------------

    @property
    def dataset_paths(self) -> Dict[str, str]:
        """Local snapshot roots of resolved datasets, keyed by ref.

        Populated by ``resolve_dataset`` (the executor calls it for every
        ``payload.datasets`` entry before the handler runs).
        """
        d = getattr(self, "_dataset_paths", None)
        if d is None:
            d = {}
            self._dataset_paths = d
        return d

    def resolve_dataset(
        self, ref: str, *, hub_silence_window_s: Optional[float] = None,
    ) -> str:
        """Materialize a dataset by bare dataset-id or ``owner/name`` ref;
        return the local root.

        Production refs are bare dataset UUIDs — the hub rewrites
        ``payload.datasets[].ref`` at submit and mints the ``read_dataset``
        grant by UUID, and a grant-scoped token cannot list — so those hit
        materialize directly. ``owner/name`` refs stay for local/dev via the
        ``?tenant=`` list lookup. Flow:

        1. Slash-less ref → dataset_id verbatim; otherwise
           ``GET /api/v1/datasets?tenant=<owner>`` → the row's ``dataset_id``.
        2. ``GET /api/v1/datasets/:id/materialize?format=files&include_urls=true``
           → a rows.jsonl-style entry index (raw CAS blobs by digest) with
           presigned URLs, sizes and blake3 checksums. A 202 (async snapshot
           build) is polled for as long as the hub keeps answering —
           there is no wall-clock budget; a typed
           ``snapshot_build_failed`` raises ``SnapshotBuildFailedError``, and
           ``hub_silence_window_s`` bounds only how long a hub that answers
           NOTHING is tolerated.
        3. Stream each entry to disk (bounded memory), digest-verified, with
           bounded retries. Entries lacking a presigned URL fall back to the
           repo-CAS by-digest reader.

        The REF is caller-supplied, so a ref that resolves to nothing raises
        ``DatasetNotFoundError`` — a typed request error, never
        release-health evidence. Everything downstream of a resolved ref (an
        empty manifest, a silent hub, an exhausted download) is the
        platform's and still raises ``RuntimeError``.
        """
        # Deferred: _datasets is +129 modules on the `import gen_worker` path.
        from ._datasets import (
            DatasetRefNotFound,
            download_entries,
            fetch_materialize_manifest,
            lookup_dataset_id,
        )

        cached = self.dataset_paths.get(ref)
        if cached:
            return cached
        base = (self._file_api_base_url or "").strip().rstrip("/")
        if not base:
            raise RuntimeError(f"resolve_dataset({ref!r}): no file_api_base_url")
        token = self._get_worker_capability_token()

        fetch_kwargs: Dict[str, Any] = {"cancelled": lambda: self.cancelled}
        if hub_silence_window_s is not None:
            fetch_kwargs["hub_silence_window_s"] = hub_silence_window_s
        # `ref` came from the caller, so THIS is the boundary that knows the
        # provenance — the lookup helpers only see an opaque id. A ref that
        # resolves to nothing is a payload verdict; everything after
        # resolution keeps its platform-fault classification.
        try:
            if "/" in ref:
                owner, name = _parse_owner_repo(ref)
                dataset_id = lookup_dataset_id(base, token, owner, name)
                cache_key = (owner, name)
            else:
                dataset_id = ref.strip()
                if not dataset_id:
                    raise DatasetRefNotFound("empty ref")
                cache_key = ("by-id", dataset_id)
            snapshot_id, entries = fetch_materialize_manifest(
                base, token, dataset_id, **fetch_kwargs,
            )
        except DatasetRefNotFound as exc:
            raise DatasetNotFoundError(ref, str(exc)) from exc

        cache_root = Path(tempfile.gettempdir()) / "gen_worker_datasets"
        target_root = cache_root.joinpath(*cache_key) / (snapshot_id or dataset_id)
        target_root.mkdir(parents=True, exist_ok=True)
        download_entries(
            entries, target_root,
            # These digests come from the HUB's own manifest, not the
            # payload — a miss is a platform fault, not the caller's.
            fetch_blob=self._fetch_platform_blob,
            cancelled=lambda: self.cancelled,
        )
        self.dataset_paths[ref] = str(target_root)
        return str(target_root)


class TrainingMetric(msgspec.Struct, frozen=True, kw_only=True):
    """Typed per-step training metric, payload of a
    ``request.training_metric`` event. tensorhub downsample-persists these
    as ``job.training.metric`` request_events rows."""

    step: int
    total: int
    loss: float
    lr: Optional[float] = None
    it_s: Optional[float] = None
    eta_s: Optional[float] = None
    #: Validation fields: periodic val loss, step of the best val so far, and
    #: a short trainer hint (e.g. "val rising; consider best_step").
    val_loss: Optional[float] = None
    best_step: Optional[int] = None
    advice: Optional[str] = None


class JobContext(_PublisherMixin, RequestContext[GenerationDefaults]):
    """The context a ``@job`` body receives — and a strict SUPERSET of what a
    producer-shaped ``@endpoint`` handler may use, under the same names.

    That superset property is what makes the two decorators portable
    (pgw#1294 / th#2049 charter constraint 1): a job promoted to a
    bounded-recipe serverless endpoint is the SAME body wrapped in
    ``@endpoint`` and priced, with zero body edits. It is enforced by a test
    that registers one body both ways and runs it under both harnesses.

    The ONLY producer context (pgw#1294 / pgw#1306): no kind selects a
    different class, because no kind decides what a body may write — the
    ``@job``/``@endpoint`` declaration
    does (``publishes`` / ``emits_media``), and the hub mints the write grant
    off that declaration. It carries:

    * the publisher surface from ``_PublisherMixin`` — ``save_checkpoint`` /
      ``open_checkpoint_stream`` (and ``gen_worker.convert.publish_flavors``),
      all of which refuse typed unless the function declared ``publishes=True``
    * ``mktemp`` (auto-cleaned scratch) and ``checkpoint_dir`` (deterministic
      job-scoped scratch)
    * ``resolve_dataset`` / ``dataset_paths`` — datasets via tensorhub only
    * ``cancelled`` / ``raise_if_cancelled`` and ``call_endpoint``
    * ``progress`` with a MONOTONIC position (see
      :meth:`RequestContext.progress`) and :meth:`metric`

    Delegated trainers (subprocess DiffSynth and friends) run through
    ``gen_worker.subproc.run_process`` with ``ctx=self`` for cancellation.
    """

    #: Min seconds between emitted metric events; first and last (step>=total)
    #: always emit. Trainers call every step, the throttle keeps the wire sane.
    metric_min_interval_s: float = 5.0

    _last_metric_monotonic: Optional[float] = None
    _last_named_metric_monotonic: Optional[float] = None

    def __init__(self, *args: Any, source: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # ``source`` is the resolved input model handle (a gen_worker.convert
        # ``Source``) for tenants that operate on a checkpoint; None otherwise.
        self._source = source
        self._mktemp_root: Optional[Path] = None

    def mktemp(self) -> Path:
        """Return a job-scoped scratch directory. Contents are NOT persisted.

        Auto-cleaned at job end. Each call returns a fresh subdir so tenants
        can use it as ``out_dir`` for ``model.save_pretrained(ctx.mktemp())``
        without collision.
        """
        if self._mktemp_root is None:
            self._mktemp_root = Path(
                tempfile.mkdtemp(
                    prefix=f"txform-{self.request_id or 'x'}-",
                    dir=tempfile.gettempdir(),
                )
            )
        return Path(tempfile.mkdtemp(dir=str(self._mktemp_root)))

    def metric(
        self,
        values: Optional[Mapping[str, float]] = None,
        *,
        step: Optional[int] = None,
        total: Optional[int] = None,
        phase: Optional[str] = None,
        **scalars: float,
    ) -> None:
        """Emit named scalar measurements (throttled), riding ``job.metric``.

        ``training_metric`` generalized: a trainer's loss curve, a
        quantization's per-rung cosine, a bake's kernel timings — anything a
        UI charts. ``ctx.progress`` answers "is this advancing"; this answers
        "how is it going"::

            ctx.metric({"loss": 0.31, "lr": 1e-4}, step=120, total=2000)
            ctx.metric(cosine=0.9987, phase="rung:w8a8")

        Same throttle as :meth:`training_metric` (``metric_min_interval_s``),
        with the first call and the last (``step >= total``) always emitted so
        a short job is never silent and a finished one always lands its final
        numbers.
        """
        merged: Dict[str, float] = {}
        for key, value in dict(values or {}).items():
            merged[str(key)] = float(value)
        for key, value in scalars.items():
            merged[str(key)] = float(value)
        if not merged:
            raise ValueError(
                "ctx.metric: nothing to report — pass a {name: value} mapping "
                "or keyword scalars"
            )
        now = time.monotonic()
        last = self._last_named_metric_monotonic
        is_last = step is not None and total is not None and total > 0 and step >= total
        if last is not None and not is_last and (now - last) < self.metric_min_interval_s:
            return
        self._last_named_metric_monotonic = now
        # ONE EVENT PER NAMED SCALAR, `{name, value, ...}`. This is the shape
        # th#2050's landed `forkJobProgress` parses — it reads `name` and
        # `value` off the payload and DROPS anything else, so a `{values: {…}}`
        # envelope would have been silently discarded by the hub. They are the
        # producer of the contract; this reconciles to what landed.
        for name, value in sorted(merged.items()):
            payload: Dict[str, Any] = {"name": name, "value": value}
            if step is not None:
                payload["step"] = int(step)
            if total is not None:
                payload["total"] = int(total)
            if phase is not None:
                payload["phase"] = str(phase)
            self._emit_event("request.metric", payload)

    def training_metric(
        self,
        *,
        step: int,
        total: int,
        loss: float,
        lr: Optional[float] = None,
        it_s: Optional[float] = None,
        eta_s: Optional[float] = None,
        val_loss: Optional[float] = None,
        best_step: Optional[int] = None,
        advice: Optional[str] = None,
    ) -> None:
        """Emit a typed ``request.training_metric`` event (throttled).

        Keep ``ctx.progress`` for human-readable stage text; this is the
        machine channel a UI charts (loss curve, it/s, ETA). Events carrying
        ``val_loss`` bypass the throttle like first/last — val points are
        sparse and every one must reach the hub.
        """
        now = time.monotonic()
        last = self._last_metric_monotonic
        is_last = total > 0 and step >= total
        has_val = val_loss is not None
        if (
            last is not None and not is_last and not has_val
            and (now - last) < self.metric_min_interval_s
        ):
            return
        self._last_metric_monotonic = now
        metric = TrainingMetric(
            step=int(step),
            total=int(total),
            loss=float(loss),
            lr=None if lr is None else float(lr),
            it_s=None if it_s is None else float(it_s),
            eta_s=None if eta_s is None else float(eta_s),
            val_loss=None if val_loss is None else float(val_loss),
            best_step=None if best_step is None else int(best_step),
            advice=advice if advice is None else str(advice),
        )
        payload = {k: v for k, v in msgspec.to_builtins(metric).items() if v is not None}
        self._emit_event("request.training_metric", payload)


# pgw#1306: `ConversionContext` / `DatasetContext` / `TrainingContext` are GONE.
# pgw#1294 merged them into JobContext and left the three names as thin aliases
# with a sentence naming th#2052 as executioner; th#2052 is a tensorhub commit
# and cannot delete Python, so this is where the sentence is carried out. The
# names never crossed a wire — `kind` is an author declaration read from local
# source (`@endpoint(kind=...)`, validated against a closed set in
# `discovery/validation.py`), and `execution_hints["kind"]` is outbound-only —
# so there is no retired shape to refuse, only a name to stop exporting.
# `tests/test_producer_context_cut_pgw1306.py` is the text fence that keeps it
# out.
