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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
)

if TYPE_CHECKING:  # heavy deps stay import-time-free; methods import lazily
    import numpy as np
    import torch
    from PIL import Image

    from ._concurrent_upload import BudgetGate
    from ..callout import CalloutClient


class LoraOverlay(TypedDict):
    """One per-request LoRA overlay riding a model slot (gw#393)."""

    ref: str
    weight: float


LogLevel = Literal["debug", "info", "warning", "error"]
"""Severity for :meth:`RequestContext.log` (pgw#508's operator stream)."""

from ..api.errors import AuthError
from ..api.slot import ResolvedSlot
from ..io import DEFAULT_IMAGE_FORMAT, DEFAULT_IMAGE_QUALITY, encode_image
from ..stage_timing import StageTimer
from ..api.types import (
    Asset,
    AudioAsset,
    ImageAsset,
    Tensors,
    VideoAsset,
)


class _SlotTable(Mapping):
    """``ctx.slots`` — a read-only mapping of slot name -> ResolvedSlot.

    Built once at context construction (executor.py::_run_job_pinned / the
    CLI's hub-less dispatch); resolution FAILURES (missing repo metadata +
    no code fallback, no ref for the slot) are stored per-key and raised
    lazily on ``__getitem__`` — "clear error at request time" means when the
    HANDLER actually reads that slot, not a blanket failure for every
    Slot-declared endpoint whose handler never touches an unresolved one."""

    __slots__ = ("_resolved", "_errors")

    def __init__(
        self,
        resolved: Mapping[str, "ResolvedSlot[Any]"],
        errors: Mapping[str, str],
    ) -> None:
        self._resolved = dict(resolved)
        self._errors = dict(errors)

    def __getitem__(self, key: str) -> "ResolvedSlot[Any]":
        if key in self._resolved:
            return self._resolved[key]
        if key in self._errors:
            raise ValueError(self._errors[key])
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter({**self._resolved, **self._errors})

    def __len__(self) -> int:
        return len(set(self._resolved) | set(self._errors))


def _copy_context_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _copy_context_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_context_metadata(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_context_metadata(v) for v in value)
    return value


def _as_asset(asset: Asset, cls: type) -> Any:
    """Re-type a plain Asset as a media Asset subclass (same fields)."""
    kw = {f: getattr(asset, f) for f in asset.__struct_fields__}
    return cls(**kw)


logger = logging.getLogger(__name__)


# Helpers, constants, and JWT/SSRF utilities live in _helpers.py. They are
# re-exported here so existing `from gen_worker.request_context import _foo`
# call sites (worker.py, tests) keep working.
from ._helpers import (
    _MAX_OUTPUT_FILE_BYTES,
    _FILE_API_HTTP_TIMEOUT_S,
    _FILE_API_STREAM_ABORT_TIMEOUT_S,
    _FILE_API_STREAM_CHUNK_TIMEOUT_S,
    _FILE_API_STREAM_FINALIZE_TIMEOUT_S,
    _FILE_API_STREAM_REPLAY_TIMEOUT_S,
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

class RequestContext:
    """Context object passed to request handlers, allowing cancellation."""

    def __init__(
        self,
        request_id: str,
        job_id: Optional[str] = None,
        emitter: Optional[Callable[[Dict[str, Any]], None]] = None,
        owner: Optional[str] = None,
        invoker_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        file_api_base_url: Optional[str] = None,
        worker_capability_token: Optional[str] = None,
        local_output_dir: Optional[str] = None,
        execution_hints: Optional[Dict[str, Any]] = None,
        models: Optional[Dict[str, Any]] = None,
        loras: Optional[Dict[str, Any]] = None,
        resolved_slots: Optional[Mapping[str, "ResolvedSlot[Any]"]] = None,
        slot_errors: Optional[Mapping[str, str]] = None,
        boot_warmup: bool = False,
    ) -> None:
        self._request_id = str(request_id or "").strip()
        self._job_id = str(job_id or "").strip() or None
        self._owner = owner
        self._invoker_id = invoker_id
        self._timeout_ms = timeout_ms
        self._file_api_base_url = (file_api_base_url or "").strip() or None
        self._worker_capability_token = (worker_capability_token or "").strip() or None
        self._local_output_dir = (local_output_dir or "").strip() or None
        self._execution_hints = dict(execution_hints or {})
        self._started_at = time.time()
        self._deadline: Optional[float] = None
        if timeout_ms is not None and timeout_ms > 0:
            self._deadline = self._started_at + (timeout_ms / 1000.0)
        self._canceled = False
        self._boot_warmup = bool(boot_warmup)
        self._lane = ""  # th#1050: executing lane, set by the executor
        self._config: Dict[str, Any] = {}  # th#1087: effective config params
        self._config_snapshot: Optional[bytes] = None
        self._cancel_event = threading.Event()
        self._emitter = emitter
        self._cached_repo_job_scope: Optional[tuple[str, str, str]] = None
        self._models = _copy_context_metadata(models or {})
        self._loras = _copy_context_metadata(loras or {})
        self._slots = _SlotTable(resolved_slots or {}, slot_errors or {})

        # Capability-budget gate (issue #269 back-pressure). Lazy-built from
        # the worker_capability_token's max_total_bytes + max_bytes_per_file
        # claims on first upload. Lives on the base (not the producer mixin)
        # because the base save_file path reserves against it too. The pool's
        # per-file fan-out can over-commit if multiple 30+ GiB shards run in
        # parallel; the gate blocks new reservations until in-flight bytes
        # fit the aggregate budget.
        self._upload_budget_gate = None  # type: Optional["BudgetGate"]
        self._upload_budget_gate_lock = threading.Lock()

        # GPU-slot lease (#382). Set by the executor for GPU jobs; lets
        # blocking uploads release the GPU slot while they wait on the
        # network. None for CPU jobs and local (CLI) runs.
        self._gpu_slot_lease: Optional[Any] = None
        # gw#516: executor callback fired on the TERMINAL slot release at the
        # decode->finalize handoff, so the worker's finalizing-job count (and
        # the hub's StateDelta view of it) tracks the encode/upload tail.
        self._on_finalize_release: Optional[Callable[[], None]] = None

        # th#1111: per-stage timing for this request. Framework hooks
        # (permit wait, input fetch, encode, stamp, upload, denoise steps)
        # record into it unconditionally; endpoints refine with ctx.stage().
        self._stages = StageTimer()

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def boot_warmup(self) -> bool:
        """True when this call is the worker's boot-time synthetic warmup
        (gw#470): the output is discarded, so a handler MAY cheapen the run
        (e.g. ``steps = 1 if ctx.boot_warmup else steps``) — the allocator
        peak is shape-driven, not step-driven."""
        return self._boot_warmup

    @property
    def deadline(self) -> Optional[float]:
        """Absolute unix-time deadline, or None when the request is unbounded."""
        return self._deadline

    @property
    def lane(self) -> str:
        """The EXECUTING precision lane of this call (th#1050), a full
        descriptor id like ``"fp8-w8a8-dynamic+compiled"`` — post-degrade
        truth, the same value JobMetrics.lane reports (th#1043 consistent).
        Read-only; always available. Handlers that declared
        ``handles=[...]`` branch on it; everyone else may ignore it."""
        return self._lane or "bf16-w16a16+eager"

    def _set_lane(self, lane: str) -> None:
        self._lane = str(lane or "").strip()

    @property
    def config(self) -> Dict[str, Any]:
        """Effective values for this endpoint's declared config parameters
        (th#1087, ``@endpoint(config=[ConfigParam(...)])``) at dispatch time:
        declared defaults overlaid with the deployer-set values at the
        worker's observed config generation. Read-only; a returned copy."""
        return dict(self._config)

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
        """Per-request LoRA overlays riding each model slot (gw#393):
        slot name -> tuple of ``{"ref", "weight"}``. Empty for adapter-free
        requests. The worker applies/removes the adapters around the handler
        call; this surface is read-only metadata."""
        return _copy_context_metadata(self._loras)

    @property
    def slots(self) -> Mapping[str, "ResolvedSlot[Any]"]:
        """The pgw#520 resolution chain, one entry per ``Slot``-declared
        model slot: ``ctx.slots["pipeline"].ref`` / ``.defaults`` — repo
        metadata merged over the endpoint's code ``default_config`` preset
        (which LOSES to repo metadata when both are present). A slot with
        no repo metadata and no ``Slot(default_config=...)`` raises on
        access (not at dispatch) — read it only when your handler needs it.
        """
        return self._slots

    def _set_resolved_slots(
        self,
        resolved: Mapping[str, "ResolvedSlot[Any]"],
        errors: Optional[Mapping[str, str]] = None,
    ) -> None:
        """CLI-only mutator (``gen-worker run``/``serve``): the hub-less
        resolve step runs after context construction, unlike the executor
        which has every input up front."""
        self._slots = _SlotTable(resolved, errors or {})

    @property
    def device(self) -> "torch.device":
        """Torch device for this worker runtime (e.g. cuda:0 or cpu)."""
        try:
            import torch
        except Exception:
            raise RuntimeError("torch is not available in this runtime") from None
        if torch.cuda.is_available():
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
                "Worker did not propagate Settings.tensorhub_public_url"
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

    def _media_upload_owner(self) -> str:
        """Owner segment for /api/v1/media/:owner/uploads calls.

        The capability token's `tenant` claim is the canonical org uuid its
        upload_media grant is bound to — the only owner tensorhub authorizes
        media writes for. Falls back to ctx.owner for dev/local paths where
        the token is absent or not a JWT.
        """
        token = self._worker_capability_token or ""
        if token:
            claim = str(_decode_unverified_jwt_claims(token).get("tenant") or "").strip()
            if claim:
                return claim
        return (self._owner or "").strip()

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

        # Scope resolves whenever destination_repo + job_id are present.
        # Previously gated on kind=="training", which broke publish for
        # @inference clone jobs that still emit checkpoints.
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
            return None

        result = (owner, repo, job_id)
        self._cached_repo_job_scope = result
        return result

    def _tensor_upload_execution_kind(self) -> str:
        hints = dict(self._execution_hints or {})
        return str(hints.get("kind", "") or "").strip().lower()

    def _require_repo_job_scope_for_tensors(self, ref: str) -> None:
        """
        For training/conversion checkpoints, remote tensor uploads must be job-scoped
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

    # #321: preferred_batch_size() / prefetch_depth() removed alongside
    # RuntimeBatchingConfigCommand — they only ever read state set by the
    # orchestrator's runtime override, and that producer never landed.

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

    # -- th#826 call-out primitive ------------------------------------------

    def _callout_client(self) -> "CalloutClient":
        from ..callout import CalloutClient

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
        tag: str = "prod",
        wait: bool = True,
        timeout_s: Optional[float] = 3600.0,
        tier: Optional[str] = None,
        poll_interval_s: float = 2.0,
    ) -> Any:
        """Call another endpoint's function as a CHILD request (th#826).

        The function must be declared ``@endpoint(child_calls=True)`` — the
        platform then scopes this invocation's credential for child calls.
        Children bill the parent request's payer, inherit its availability
        tier (``tier=`` may name a CHEAPER class, never escalate), count
        against the tree's depth/budget ceilings, and die with the parent
        when the tree is cancelled.

        ``wait=True`` (default) blocks to a terminal state and returns the
        child's output items (asset refs stay refs — pass them straight into
        the next call's payload). ``wait=False`` returns a
        :class:`~gen_worker.callout.ChildRequest` handle
        (``.status()`` / ``.result()`` / ``.cancel()``).

        Raises ``ChildCallRefusedError`` (typed admission refusals),
        ``ChildRequestFailedError`` / ``ChildRequestCanceledError``,
        ``ChildCallTimeoutError``, and ``CanceledError`` when this invocation
        itself is cancelled mid-wait.
        """
        self.raise_if_cancelled()
        client = self._callout_client()
        request_id = client.submit(endpoint, function, payload, tag=tag, tier=tier)
        from ..callout import ChildRequest

        handle = ChildRequest(client, request_id)
        if not wait:
            return handle
        return handle.result(timeout_s, poll_interval_s=poll_interval_s)

    def workflow_checkpoint(self, key: str, fn: Callable[[], Any]) -> Any:
        """Memoize one workflow step's result under this request (th#826).

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
        tenant code (#382). No-op when there is no lease (CPU jobs, local
        runs) or the slot is already yielded (executor freed it post-handler).

        If the job was cancelled while yielded (deadline / CancelJob), the
        re-acquired slot is released again immediately: the executor's final
        release already saw ``held == False`` and skipped, so the balance
        stays exact and the freed slot isn't captured by a dying job.
        """
        lease = self._gpu_slot_lease
        if lease is None or not lease.yield_slot():
            yield
            return
        try:
            yield
        finally:
            lease.reacquire()
            if self._canceled:
                lease.yield_slot()

    def _release_gpu_slot_for_finalize(self) -> None:
        """Worker-internal: TERMINAL GPU-slot release at the decode->finalize
        handoff (gw#476 / gw#516). The handler is done with GPU compute; the
        encode + upload tail proceeds slotless so the next request's denoise
        starts now instead of idling the GPU (measured up to 179s on a
        CPU-contended host). Unlike :meth:`_gpu_slot_yielded` there is no
        reacquire — a finishing request must never block behind the next
        request's denoise just to return. The executor's post-handler release
        no-ops (lease transitions are once-only), so the semaphore balance
        stays exact. Tenant GPU work after this call runs unscheduled —
        finalize helpers call it only once frames are on the host. No-op
        without a lease (CPU jobs, local runs) or when already yielded."""
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
        (th#1111)::

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
        progress: float,
        stage: Optional[str] = None,
        *,
        step: Optional[int] = None,
        total: Optional[int] = None,
    ) -> None:
        """Report request progress (best-effort, rides ``request.progress``).

        This is the USER-facing stream — the cozy-art job feed renders it
        directly. For platform/operator-only diagnostics use :meth:`log`.

        ``progress`` is a 0..1 fraction; ``step``/``total`` carry the exact
        step counter when known (e.g. denoise step 5 of 20) so UIs can render
        "5 / 20" instead of a bare percentage.
        """
        payload: Dict[str, Any] = {"progress": progress}
        if stage is not None:
            payload["stage"] = stage
        if step is not None:
            payload["step"] = int(step)
        if total is not None:
            payload["total"] = int(total)
        self._emit_event("request.progress", payload)

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

        pgw#508: this is the PLATFORM/OPERATOR debug stream, full stop —
        never user-facing. tensorhub persists it under an operator-only event
        kind and never serves it on a tenant-facing surface (SSE job feed,
        events.bin, poll); it does not reach the cozy-art job card. See
        proto/CONTRACT.md § "The ctx event lane" for the wire-level routing
        contract.

        One-line rule for authors: module-level ``logging.getLogger(__name__)``
        for boot-time/cross-request logging; ``ctx.log`` for anything scoped
        to THIS request you'd want when debugging it (resolved model/
        scheduler choice, retry/degradation detail, malformed-input detail);
        ``ctx.progress`` for what the human watching the job should see.
        There is no user-visible counterpart to ``ctx.log`` — a product
        surface for extra user-facing text would be a deliberate addition,
        not an overload of this method (YAGNI until a real surface asks).

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
        """C2PA-sign media payloads at the finalize seam (th#714).

        Returns ``data`` unchanged when signing is unconfigured or the
        payload is not a signable media format; raises when signing is
        configured but fails (an unlabeled asset must not ship silently).
        """
        from .. import content_credentials

        with self._stages.stage("credential_stamp"):
            return content_credentials.sign_media_bytes(
                data, ref=ref, **self._c2pa_manifest_kwargs())

    def _c2pa_sign_file(self, ref: str, src: str) -> Optional[str]:
        """File variant of :meth:`_c2pa_sign_bytes` — returns a signed temp
        path (caller unlinks) or None when signing doesn't apply."""
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
        if output_format == "inline" and len(data) <= self._SAVE_BYTES_INLINE_THRESHOLD:
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
        format: str = DEFAULT_IMAGE_FORMAT,
        quality: int = DEFAULT_IMAGE_QUALITY,
        lossless: bool = False,
        **encode_kwargs: Any,
    ) -> ImageAsset:
        """Encode + save an image; returns a typed :class:`ImageAsset`.

        ``format`` is ``webp`` (the platform default), ``png``, or ``jpg``.
        ``quality`` applies to webp/jpg; ``lossless`` is webp-only. The
        extension is derived from the format when ``ref`` has no suffix.
        """
        with self._stages.stage("image_encode"):
            payload, ext = encode_image(
                image, format=format, quality=quality, lossless=lossless,
                **encode_kwargs,
            )
        if ref is None or str(ref).strip() == "":
            ref = f"outputs/{self.request_id}/image{ext}"
        else:
            ref = _normalize_output_ref(str(ref))
            if Path(ref).suffix == "":
                ref += ext
        return _as_asset(self.save_bytes(ref, payload), ImageAsset)

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

        # C2PA signing (th#714): media files upload as a signed temp copy;
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

    # Issue #1 (slim-request-context): admin-plane visibility toggles
    # (publish_checkpoint / publish_dataset / publish_endpoint /
    # publish_endpoint_release / publish_media + their unpublish_ counterparts)
    # were deleted as a hard cut. They were not used by any worker-author
    # endpoint; visibility flips belong in cozyctl / the tensorhub UI, not on
    # a per-request object.



# ---------------------------------------------------------------------------
# Issue #1 (slim-request-context): kind-specific subclasses.
#
# RequestContext is the per-inference base. Conversion, dataset-producing,
# and trainer endpoints get richer subclasses that carry the
# producer-contract RPCs (publish_dataset_revision, resolve_dataset,
# materialize_blob).
#
# ConversionContext / DatasetContext / TrainingContext share `_PublisherMixin`
# for the producer-contract HTTP helpers (blob fetch + materialization by
# digest). Checkpoint publishing is NOT here:
# producer endpoints call gen_worker.convert.publish_flavors (the /commits path).
# ---------------------------------------------------------------------------


class _PublisherMixin:
    """Producer-contract helpers shared by ConversionContext, DatasetContext
    and TrainingContext: blob fetch by digest and ``materialize_blob``.
    Always combined with ``RequestContext`` via multiple
    inheritance (so ``self`` has ``_file_api_base_url`` / ``_owner`` /
    ``_get_worker_capability_token``).

    Producer-only STATE lives here too (pgw#526): the reserved
    ``source``/``destination``/``text_encoder`` payload structs, the hf
    token, and their materialized paths initialize in this ``__init__`` — a
    plain inference ``RequestContext`` never carries them.

    Not a public surface: tenants should never import this directly.
    """

    def __init__(
        self,
        *args: Any,
        source_info: Optional[Dict[str, Any]] = None,
        destination_info: Optional[Dict[str, Any]] = None,
        text_encoder_info: Optional[Dict[str, Any]] = None,
        hf_token: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Reserved-name producer contract attributes. Populated by the
        # executor's ctx construction (executor.py::_run_job_pinned) before
        # invoking tenant code when the payload declares the reserved
        # `source`/`destination` struct fields.
        self._source_info = dict(source_info or {})
        self._destination_info = dict(destination_info or {})
        self._source_path: Optional[str] = None
        self._text_encoder_info = dict(text_encoder_info or {})
        self._text_encoder_path: Optional[str] = None
        self._hf_token = (hf_token or "").strip()
    if TYPE_CHECKING:
        # The host contract (gw#497): everything this mixin borrows from
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
        def _require_repo_job_scope_for_tensors(self, ref: str) -> None: ...
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

    # Second reserved-name model input (pgw#594, te#70): a wholly independent
    # repo from `source`, materialized the same way. Absent for every
    # existing producer payload — `text_encoder`/`text_encoder_path` stay
    # empty/None and no behavior changes.
    @property
    def text_encoder(self) -> dict[str, Any]:
        return dict(self._text_encoder_info)

    @property
    def text_encoder_path(self) -> Optional[str]:
        return self._text_encoder_path

    def _set_text_encoder_path(self, path: str) -> None:
        """Library-internal: called after text_encoder materialization."""
        self._text_encoder_path = str(path) if path else None

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

        Job-scoped writes publish through the /commits stream (gw#471) so
        the returned Tensors carries a blake3 digest + blob_digest and each
        save materializes one finalized repo revision; everything else falls
        back to the plain asset save the ``fallback`` callable provides.
        """
        ref = _normalize_output_ref(ref)
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
        self._require_repo_job_scope_for_tensors(ref)
        from typing import cast as _cast

        return _RequestOutputStream(
            ctx=_cast("RequestContext", self),
            ref=ref,
            kind="checkpoint",
            format=format,
            expected_size_bytes=expected_size_bytes,
            step_number=step_number,
            epoch_number=epoch_number,
        )

    def _download_blob_by_digest(self, digest: str, dest: Path) -> None:
        """Fetch a blob by ``<algo>:<hex>`` digest to ``dest``.

        Uses the repo-CAS by-digest read endpoint — works for any blob
        uploaded via ``save_checkpoint`` regardless of whether it's a
        checkpoint file or a dataset file. The server indexes all CAS
        content by blake3 digest; callers that know the digest can fetch
        without needing to know which subsystem the blob belongs to.
        """
        import requests
        base = (self._file_api_base_url or "").strip().rstrip("/")
        token = self._get_worker_capability_token()
        # Normalize digest format for URL.
        digest_norm = digest if ":" in digest else f"blake3:{digest}"
        url = f"{base}/api/v1/blobs/{urllib.parse.quote(digest_norm, safe=':')}/content"
        headers = {"Authorization": f"Bearer {token}"}
        with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
            if resp.status_code in (401, 403):
                raise AuthError(f"blob fetch unauthorized ({resp.status_code}) digest={digest}")
            if resp.status_code == 404:
                raise RuntimeError(f"blob fetch 404 for digest={digest}")
            if resp.status_code < 200 or resp.status_code >= 300:
                raise RuntimeError(f"blob fetch failed ({resp.status_code}) digest={digest}: {resp.text[:256]}")
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)


    def materialize_blob(self, digest: str, dest: "str | os.PathLike[str]") -> Path:
        """Fetch a blob by ``<algo>:<hex>`` content-addressed digest.

        Returns the ``Path`` the blob was written to. Replacement for the
        private ``_download_blob_by_digest`` — exposed publicly so tenants
        that handle a digest directly (e.g. consuming a snapshot manifest
        emitted by an earlier conversion) can pull the bytes themselves.
        """
        dest_path = Path(os.fspath(dest))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self._download_blob_by_digest(digest, dest_path)
        return dest_path

    def checkpoint_dir(self, *, key: str) -> Path:
        """Return a JOB-SCOPED SCRATCH dir keyed by (job_id, key) — a stable
        working directory for trainer ``output_dir`` use within one job.

        NOT persistent storage (pgw#527): it lives under
        ``tempfile.gettempdir()`` — pod-local ``/tmp``, gone at pod
        churn/eviction. Do not park resume state here; durable resume goes
        through published checkpoints (``save_checkpoint`` / the job's
        source repo). What it IS good for: a deterministic path that
        survives handler retries within the same pod/process, so a trainer
        can wipe-and-recreate it at start (the image_lora_finetuner
        pattern) without colliding with other jobs.
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

    def resolve_dataset(self, ref: str, *, budget_s: Optional[float] = None) -> str:
        """Materialize a dataset by bare dataset-id or ``owner/name`` ref;
        return the local root.

        Production refs are bare dataset UUIDs (th#641: the hub rewrites
        ``payload.datasets[].ref`` at submit and mints the ``read_dataset``
        grant by UUID; a grant-scoped token can't list) — those hit
        materialize directly. ``owner/name`` refs stay for local/dev via the
        ``?tenant=`` list lookup. Flow (th#698 blob-manifest wire format):

        1. Slash-less ref → dataset_id verbatim; otherwise
           ``GET /api/v1/datasets?tenant=<owner>`` → the row's ``dataset_id``.
        2. ``GET /api/v1/datasets/:id/materialize?format=files&include_urls=true``
           → a rows.jsonl-style entry index (raw CAS blobs by digest) with
           presigned URLs, sizes and blake3 checksums. A 202 (async snapshot
           build, th#691) is polled until ready within ``budget_s`` (default
           30 min, ≥ the hub's 20-min build budget); a typed
           ``snapshot_build_failed`` raises ``SnapshotBuildFailedError``.
        3. Stream each entry to disk (bounded memory), digest-verified, with
           bounded retries. Entries lacking a presigned URL fall back to the
           repo-CAS by-digest reader.

        Raises ``RuntimeError`` when the dataset isn't found, the manifest is
        empty, the poll budget runs out, or any download exhausts its retries.
        """
        from ._datasets import (
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

        if "/" in ref:
            owner, name = _parse_owner_repo(ref)
            dataset_id = lookup_dataset_id(base, token, owner, name)
            cache_key = (owner, name)
        else:
            dataset_id = ref.strip()
            if not dataset_id:
                raise RuntimeError("resolve_dataset: empty ref")
            cache_key = ("by-id", dataset_id)
        fetch_kwargs: Dict[str, Any] = {"cancelled": lambda: self.cancelled}
        if budget_s is not None:
            fetch_kwargs["budget_s"] = budget_s
        snapshot_id, entries = fetch_materialize_manifest(
            base, token, dataset_id, **fetch_kwargs,
        )

        cache_root = Path(tempfile.gettempdir()) / "gen_worker_datasets"
        target_root = cache_root.joinpath(*cache_key) / (snapshot_id or dataset_id)
        target_root.mkdir(parents=True, exist_ok=True)
        download_entries(
            entries, target_root,
            fetch_blob=self._download_blob_by_digest,
            cancelled=lambda: self.cancelled,
        )
        self.dataset_paths[ref] = str(target_root)
        return str(target_root)


class ConversionContext(_PublisherMixin, RequestContext):
    """RequestContext for ``@conversion(sub_kind="format-conversion")``
    and similar conversion endpoints.

    Carries the producer-contract RPCs needed to publish new repo revisions,
    plus the conversion-helper surface (``mktemp``, ``checkpoint_dir``,
    ``cancelled``). The ETL itself (ingest / cast / quant / clone / writers)
    lives in the ``gen_worker.convert`` module — this class only carries what the
    worker API needs.

    Inference handlers receive ``RequestContext`` instead — they never need
    these methods.
    """

    def __init__(self, *args: Any, source: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # ``source`` is the resolved input model handle (a gen_worker.convert
        # ``Source``) for tenants that operate on a checkpoint; None otherwise.
        self._source = source
        self._mktemp_root: Optional[Path] = None

    # ----- producer-contract RPCs -------------------------------------

    # ----- conversion-helper wrapper API ------------------------------
    #
    # Previously lived in ``gen_worker.conversion.context.ConversionContext``
    # as a wrapper around RequestContext. Subclassed in (since both classes
    # share the same name and the wrapper has no per-method state that
    # RequestContext doesn't already track).

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

class DatasetContext(_PublisherMixin, RequestContext):
    """RequestContext for dataset-producing endpoints (``@dataset``).

    Adds ``publish_dataset_revision``; ``resolve_dataset`` comes from
    ``_PublisherMixin``.
    """

    def publish_dataset_revision(
        self,
        *,
        destination_dataset: str,
        features_json: Dict[str, Any],
        row_artifacts_json: Optional[Dict[str, Any]] = None,
        snapshot_manifest: Optional[List[Dict[str, Any]]] = None,
        visibility: str = "private",
        kind: str = "",
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Publish a dataset revision into ``tensorhub.datasets``.

        Writes to the datasets subsystem instead of ``repo_checkpoints``.
        The individual file bytes are expected to already be in CAS via
        prior ``save_checkpoint`` calls — this method just records the
        dataset-level metadata pointing at those blobs. The server
        cross-references by blob digest at materialize time.

        Args:
            destination_dataset: ``owner/name`` or ``owner/name:tag`` ref.
            features_json: HF-style features schema, e.g.
                ``{"prompt": {"_type": "Value", "dtype": "string"}, ...}``.
            row_artifacts_json: Optional mapping of row IDs → artifact
                refs for datasets that reference external image blobs.
            snapshot_manifest: Optional list of ``{path, digest, size_bytes}``
                entries — the data shards + any sidecar files that
                comprise this dataset revision. Used for provenance /
                content-identity tracking (naming-based versioning means
                the dataset row is mutable, but the manifest captures what
                content was active at publish time).
            visibility: ``"private"`` (default) or ``"public"``.
            kind: Free-form kind string (``"prompt_corpus"`` / ``"eval_set"``).
                Stored in features_json.__cozy_kind__ for now until
                tensorhub grows a dedicated kind column.
            dataset_info: Full ``dataset_info.json`` payload to record
                as tenant metadata.

        Returns:
            ``{ok: True, dataset_id: str, owner: str, name: str, existed: bool}``.

        Raises ``AuthError`` on 401/403 and ``RuntimeError`` on any other
        HTTP failure. The hub-API plumbing lives next to ``HubClient``
        (``gen_worker.convert.hub.publish_dataset_revision``).
        """
        from ..convert.hub import publish_dataset_revision

        return publish_dataset_revision(
            base_url=(self._file_api_base_url or "").strip(),
            token=self._get_worker_capability_token(),
            destination_dataset=destination_dataset,
            features_json=features_json,
            row_artifacts_json=row_artifacts_json,
            snapshot_manifest=snapshot_manifest,
            visibility=visibility,
            kind=kind,
            dataset_info=dataset_info,
        )


class TrainingMetric(msgspec.Struct, frozen=True, kw_only=True):
    """Typed per-step training metric (pgw#450), payload of a
    ``request.training_metric`` event. tensorhub downsample-persists these
    as ``job.training.metric`` request_events rows (th#681)."""

    step: int
    total: int
    loss: float
    lr: Optional[float] = None
    it_s: Optional[float] = None
    eta_s: Optional[float] = None
    #: pgw#459 validation fields: periodic val loss, step of the best val so
    #: far, and a short trainer hint (e.g. "val rising; consider best_step").
    val_loss: Optional[float] = None
    best_step: Optional[int] = None
    advice: Optional[str] = None


class TrainingContext(_PublisherMixin, RequestContext):
    """RequestContext for ``@endpoint(kind="training")`` endpoints.

    From ``_PublisherMixin``: ``save_checkpoint``,
    ``resolve_dataset`` / ``dataset_paths`` (the executor materializes
    ``payload.datasets`` before the handler runs) and ``checkpoint_dir``
    (job-scoped scratch, NOT durable — see its docstring).
    Delegated trainers (subprocess ai-toolkit and friends) run through
    ``gen_worker.subproc.run_process`` with ``ctx=self`` for cancellation.
    """

    #: Min seconds between emitted metric events; first and last (step>=total)
    #: always emit. Trainers call every step, the throttle keeps the wire sane.
    metric_min_interval_s: float = 5.0

    _last_metric_monotonic: Optional[float] = None

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
        ``val_loss`` bypass the throttle like first/last (pgw#459) — val
        points are sparse and every one must reach the hub.
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
