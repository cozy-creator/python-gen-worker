from __future__ import annotations

import asyncio
import contextvars
import functools
import importlib
import itertools
import json
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import msgspec

from . import boot_phases as boot_mod
from . import content_credentials
from . import hostfacts
from . import process_role
from . import receipts
from . import serve_posture
from . import worker_credential
from .api.errors import ValidationError as ApiValidationError
from .capability_renewal import renew_capability_while_running
from .config import Settings
from .failure_traceback import MAX_BYTES as MAX_TRACEBACK_BYTES
from .failure_traceback import traceback_tail
from .input_assets import cleanup_input_assets, manifest_from_run_job
from .stage_timing import stage_ms_for_metrics
from .host_move_guard import install as _install_host_move_guard
from .models.cache_paths import tensorhub_cache_dir, tensorhub_cas_dir
from .models.cozy_snapshot import snapshot_dir_key
from .models.disk_gc import tree_bytes
from .models.projection import SNAPSHOTS_DIR
from .models.refs import WireRef
from .models.store import ModelStore, bind_active_store
from .boot_materialize import (
    REASON_MODEL_UNAVAILABLE,
    CheckpointConfig,
    CheckpointMaterialization,
)
from .discovery.names import slugify_name
from . import postmortem
from .procsplit import is_compute_child
from .pb import worker_scheduler_pb2 as pb
from .serving.context import DeployBinding
from .serving.entrypoints import ENTRYPOINT_ATTR, EntrypointSpec
from .serving.envelope import EnvelopeError
from .serving.host import ServeDispatchError
from .serving.loader import EndpointLoadError, LoadedEndpoint
from .serving.model import ModelDeclarationError, lane_handle, model_lanes, model_type
from .serving.residency import NeverFits, ResidencyError, ResidencyManager
from .serving.serve_loop import ServeLoop
from .transport import (
    DEFAULT_QUEUE_MAXSIZE as _DEFAULT_QUEUE_MAXSIZE,
    FatalTransportError,
    PROTOCOL_VERSION,
    Transport,
)
from .v1_deleted import MIGRATION, refuse_module
from .wire_snapshots import resolved_repos

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_MS = 10_000

_SIGNAL_DRAIN_DEADLINE_MS = 30_000

_HEADROOM_DIVISOR = 4

EVENT_CONTENT_TYPE = "application/x-request-event+json"

_V1_ATTRS: Tuple[str, ...] = (
    "__gen_worker_endpoint__",
    "__gen_worker_job__",
    "__gen_worker_function__",
    "__gen_worker_variant__",
)


class WorkerBootError(RuntimeError):
    """This process cannot state a serve surface."""


class NoEntrypointsDeclared(WorkerBootError):
    """The imported modules declare no ``@entrypoint``."""


class DuplicateEntrypoint(WorkerBootError):
    """Two modules declare the same entrypoint name."""


class CheckpointUnresolved(RuntimeError):
    """A checkpoint pick could not be turned into a local binding."""


class UnexpectedWorkerExit(RuntimeError):
    """The run loop ended without a hub Drain or a shutdown signal."""


def residency_budget(stated: int = 0) -> int:
    """The residency budget, in the order the answers are trustworthy: a stated budget (a CONFIG the caller owns), then the card's own headroom, then — on a host with NO CUDA at all — available host RAM."""
    if stated:
        return int(stated)
    headroom = hostfacts.headroom_bytes()
    if headroom:
        return int(headroom)
    if hostfacts.cuda_ready():
        raise WorkerBootError(
            "this host has CUDA but no VRAM reading "
            "(torch.cuda.mem_get_info gave nothing) and no vram_budget_bytes "
            "was stated: residency admits before it allocates, and it cannot "
            "make that decision against an unknown budget on a card it can see"
        )
    from .models.memory import get_available_ram_gb

    budget = int(get_available_ram_gb() * (1024 ** 3))
    if not budget:
        raise WorkerBootError(
            "no VRAM reading and no readable host RAM: residency admits "
            "before it allocates and has no budget to admit against"
        )
    logger.warning(
        "no CUDA on this host: sizing the residency budget from AVAILABLE "
        "HOST RAM (%.1f GiB). This is a CPU serving process — correct for "
        "cozy-local and CI, and never what a GPU pod should be doing.",
        budget / (1024 ** 3),
    )
    return budget


def harvest_entrypoints(module_names: List[str]) -> LoadedEndpoint:
    """Import the author modules and state their combined serve surface."""
    entrypoints: Dict[str, EntrypointSpec] = {}
    owners: Dict[str, str] = {}
    models: Dict[type, None] = {}
    for name in module_names:
        module = importlib.import_module(name)
        for attr in _V1_ATTRS:
            if any(hasattr(v, attr) for v in vars(module).values()):
                raise refuse_module(module.__name__, attr)
        for value in vars(module).values():
            spec = getattr(value, ENTRYPOINT_ATTR, None)
            if not isinstance(spec, EntrypointSpec):
                continue
            if spec.fn.__module__ != module.__name__:
                continue
            # THE WIRE ROUTE IS THE SLUG, not the python name: the hub dispatches RunJob.function_name as slugify_name(name) — the same normalization discovery.validation checks collisions against — so `steal_credentials` is reached as `steal-credentials`. Keying this table on the raw name makes every multi-word entrypoint unroutable while single-word ones work.
            route = slugify_name(spec.name)
            prior = owners.get(route)
            if prior is not None and prior != module.__name__:
                raise DuplicateEntrypoint(
                    f"entrypoint route {route!r} is declared by both {prior} "
                    f"and {module.__name__}; a dispatch carries only the name, "
                    f"so the two are unroutable"
                )
            entrypoints[route] = spec
            owners[route] = module.__name__
            for cls in spec.model_classes:
                models.setdefault(cls)
    if not entrypoints:
        raise NoEntrypointsDeclared(
            f"modules {list(module_names)!r} declare no @entrypoint — an "
            f"entrypoint is a module-level @entrypoint function "
            f"(ctx: RequestContext, payload: msgspec.Struct) plus zero or more "
            f"slots. {MIGRATION}."
        )
    for cls in models:
        try:
            model_type(cls)
            model_lanes(cls)
        except ModelDeclarationError as exc:
            raise WorkerBootError(str(exc)) from exc
    return LoadedEndpoint(
        module_name=", ".join(module_names),
        entrypoints=entrypoints,
        models=tuple(models),
    )


@dataclass(frozen=True, slots=True)
class _Pick:

    slot: str
    ref: str
    manifest_digest: str
    model: str
    inference_defaults: str


@dataclass(frozen=True, slots=True)
class _DispatchPicks:
    by_ref: Mapping[str, _Pick]
    by_slot: Mapping[str, str]


_EMPTY_PICKS = _DispatchPicks(by_ref={}, by_slot={})

_DISPATCH: contextvars.ContextVar[_DispatchPicks] = contextvars.ContextVar(
    "gen_worker_dispatch_picks", default=_EMPTY_PICKS
)


def _picks_of(run: pb.RunJob) -> _DispatchPicks:
    by_ref: Dict[str, _Pick] = {}
    by_slot: Dict[str, str] = {}
    for binding in run.models:
        digest = str(binding.manifest_digest).strip()
        if not digest:
            snapshot = run.snapshots.get(str(binding.ref))
            if snapshot is not None:
                digest = str(snapshot.digest).strip()
        pick = _Pick(
            slot=str(binding.slot),
            ref=str(binding.ref),
            manifest_digest=digest,
            model=str(binding.model).strip(),
            inference_defaults=str(binding.inference_defaults),
        )
        by_ref[pick.ref] = pick
        by_slot[pick.slot] = pick.ref
    return _DispatchPicks(by_ref=by_ref, by_slot=by_slot)


def _picks_of_bindings(bindings: Any) -> _DispatchPicks:
    by_ref: Dict[str, _Pick] = {}
    by_slot: Dict[str, str] = {}
    for binding in bindings or ():
        pick = _Pick(
            slot=str(binding.slot),
            ref=str(binding.ref),
            manifest_digest=str(binding.manifest_digest).strip(),
            model=str(binding.model).strip(),
            inference_defaults=str(binding.inference_defaults),
        )
        by_ref[pick.ref] = pick
        by_slot[pick.slot] = pick.ref
    return _DispatchPicks(by_ref=by_ref, by_slot=by_slot)


def boot_picks(
    desired: Any, loaded: Any, config: "CheckpointConfig"
) -> Dict[str, _DispatchPicks]:
    functions = sorted(getattr(loaded, "entrypoints", {}) or {})
    picks: Dict[str, _DispatchPicks] = {}
    for instance in getattr(desired, "hot", ()) or ():
        name = str(getattr(instance, "function_name", "")).strip()
        if not name:
            continue
        table = _picks_of_bindings(getattr(instance, "models", ()))
        if table.by_slot:
            picks[name] = table
    refs = [str(ref) for ref in config.refs]
    if len(refs) != 1:
        return picks
    only_ref = refs[0]
    snapshot = config.snapshots.get(WireRef(only_ref))
    digest = str(getattr(snapshot, "digest", "") or "").strip()
    for name in functions:
        if name in picks:
            continue
        spec = loaded.entrypoints[name]
        slots = [slot for slot, _cls in spec.model_params]
        if len(slots) != 1:
            continue
        pick = _Pick(
            slot=slots[0], ref=only_ref, manifest_digest=digest,
            model="", inference_defaults="",
        )
        picks[name] = _DispatchPicks(
            by_ref={only_ref: pick}, by_slot={slots[0]: only_ref}
        )
    return picks


class HubBindingResolver:
    """The hub's half of the deploy state, per dispatch."""

    def __init__(self, snapshots_root: Optional[Path] = None) -> None:
        self.snapshots_root = (
            Path(snapshots_root)
            if snapshots_root is not None
            else tensorhub_cas_dir() / SNAPSHOTS_DIR
        )
        self._store: Optional[Any] = None

    def bind_store(self, store: Any) -> None:
        """Hand the resolver the store that materializes refs."""
        self._store = store

    def _pick(self, model_cls: type, checkpoint_ref: str) -> _Pick:
        picks = _DISPATCH.get()
        pick = picks.by_ref.get(checkpoint_ref)
        if pick is None:
            raise CheckpointUnresolved(
                f"{model_cls.__name__}: this dispatch carries no ModelBinding "
                f"for checkpoint {checkpoint_ref!r} "
                f"(bound refs: {sorted(picks.by_ref) or '[]'})"
            )
        return pick

    def _digest_trees(self, digest: str) -> List[Path]:
        digest = str(digest or "").strip()
        if not digest:
            return []
        bare = digest.split(":", 1)[-1]
        keys = [snapshot_dir_key(bare)]
        if bare != digest:
            keys.append(snapshot_dir_key(digest))
        return [self.snapshots_root / key for key in keys]

    def tree_for(self, model_cls: type, checkpoint_ref: str) -> Path:
        pick = self._pick(model_cls, checkpoint_ref)
        tried: List[Path] = []
        if self._store is not None:
            resident = self._store.disk_local_path(checkpoint_ref)
            if resident is not None and Path(resident).is_dir():
                return Path(resident)
            snapshot = self._store.banked_snapshot(checkpoint_ref)
            if snapshot is not None and snapshot.digest:
                for tree in self._digest_trees(str(snapshot.digest)):
                    if tree.is_dir():
                        return tree
                    tried.append(tree)
        for tree in self._digest_trees(pick.manifest_digest):
            if tree.is_dir():
                return tree
            tried.append(tree)
        materialized = (
            sorted(str(r) for r in self._store.disk_refs())
            if self._store is not None else []
        )
        raise CheckpointUnresolved(
            f"{model_cls.__name__}: checkpoint {checkpoint_ref!r} is not "
            f"materialized on this worker. Boot materialized "
            f"{materialized or '[]'}; the dispatch's manifest_digest is "
            f"{pick.manifest_digest or '(unset — the hub sends none)'}; "
            f"tried {[str(p) for p in tried] or '[]'}"
        )

    def resolve(self, model_cls: type, checkpoint_ref: str) -> DeployBinding:
        pick = self._pick(model_cls, checkpoint_ref)
        defaults: Mapping[str, Any] = {}
        if pick.inference_defaults.strip():
            try:
                parsed = json.loads(pick.inference_defaults)
            except ValueError as exc:
                raise CheckpointUnresolved(
                    f"{model_cls.__name__}: inference_defaults for "
                    f"{checkpoint_ref!r} is not JSON: {exc}"
                ) from None
            if not isinstance(parsed, dict):
                raise CheckpointUnresolved(
                    f"{model_cls.__name__}: inference_defaults for "
                    f"{checkpoint_ref!r} is {type(parsed).__name__}, not an object"
                )
            defaults = parsed
        if defaults and not pick.model:
            raise CheckpointUnresolved(
                f"{model_cls.__name__}: the binding for {checkpoint_ref!r} "
                f"carries an inference_defaults row but no `model` "
                f"classification. The hub's two-column defaults surface stores "
                f"the two together (`model` is NOT NULL beside the JSONB), so "
                f"this is a broken pair on the wire, not an unclassified "
                f"checkpoint — serving type fallbacks here would silently "
                f"discard this checkpoint's own tuned defaults"
            )
        return DeployBinding(
            checkpoint_ref=pick.ref,
            checkpoint_dir=self.tree_for(model_cls, checkpoint_ref),
            model=pick.model or None,
            defaults=defaults,
        )

    def default_pick(self, model_cls: type, slot_name: str) -> str:
        """The dispatch's own per-slot ref."""
        return _DISPATCH.get().by_slot.get(slot_name, "")


class _Admission:
    """The name `tree_for`'s refusals carry when ADMISSION is the caller."""


class SnapshotSizer:
    """Weight bytes from the materialized tree's manifest — the WHOLE tree."""

    def __init__(self, resolver: HubBindingResolver) -> None:
        self._resolver = resolver
        self._cache: Dict[str, int] = {}

    def _bytes(self, checkpoint_ref: str) -> int:
        cached = self._cache.get(checkpoint_ref)
        if cached is not None:
            return cached
        tree = self._resolver.tree_for(_Admission, checkpoint_ref)
        size = int(tree_bytes(tree))
        self._cache[checkpoint_ref] = size
        return size

    def resident_bytes(self, checkpoint_ref: str, lane: str) -> int:
        return self._bytes(checkpoint_ref)

    def activation_headroom_bytes(self, checkpoint_ref: str, lane: str) -> int:
        return max(self._bytes(checkpoint_ref) // _HEADROOM_DIVISOR, 1)


class Worker:
    """Serve-stack + transport for one pod, in one event loop."""

    def __init__(
        self,
        settings: Settings,
        user_module_names: List[str],
        *,
        manifest: Optional[Dict[str, Any]] = None,
        vram_budget_bytes: int = 0,
        lane: str = "",
        output_dir: Optional[Path] = None,
        queue_maxsize: int = _DEFAULT_QUEUE_MAXSIZE,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 30.0,
    ) -> None:
        if not (settings.orchestrator_public_addr or "").strip():
            raise ValueError("Settings.orchestrator_public_addr is required")
        self.settings = settings
        _install_host_move_guard()

        if manifest:
            from .models.download import (
                build_provider_index_from_manifest,
                set_provider_index,
            )

            set_provider_index(build_provider_index_from_manifest(manifest))

        self.loaded = harvest_entrypoints(list(user_module_names))
        self.resolver = HubBindingResolver()
        budget = residency_budget(int(vram_budget_bytes))
        self.residency = ResidencyManager(int(budget), SnapshotSizer(self.resolver))
        hub_base = str(getattr(settings, "tensorhub_url", "") or "").strip()
        if hub_base:
            receipts.configure(hub_base, worker_credential.current)
        else:
            logger.warning(
                "receipts: no TENSORHUB_URL, so the gate is UNARMED for the "
                "boot-adopt window — hub-delivered artifacts will be refused "
                "and self-minted until HelloAck arms it"
            )
        try:
            self.adoption = self._build_adoption()
        except Exception:  # noqa: BLE001 — adoption never costs a boot
            logger.exception("adopt: could not be set up; this pod serves eager")
            self.adoption = None
        try:
            self.serve = ServeLoop(
                self.loaded, residency=self.residency, resolver=self.resolver,
                lane_contract=lane, output_dir=output_dir,
                compile_sink_for=(
                    self.adoption.sink_for if self.adoption is not None else None
                ),
                on_loaded=(
                    self.adoption.loaded if self.adoption is not None else None
                ),
                hf_token=str(getattr(settings, "hf_token", "") or ""),
            )
        except EndpointLoadError as exc:
            raise WorkerBootError(str(exc)) from exc
        self.lanes: frozenset[str] = frozenset(
            lane_handle(lane) for lane in self.serve.lanes.values() if lane is not None
        )

        if is_compute_child():
            from .procsplit.child import ChildTransport

            self.transport: Any = ChildTransport(settings, self)
        else:
            self.transport = Transport(
                settings,
                self,
                queue_maxsize=queue_maxsize,
                backoff_base_s=backoff_base_s,
                backoff_cap_s=backoff_cap_s,
            )

        self.worker_id = (
            settings.worker_id.strip() or f"py-worker-{os.getpid()}"
        )
        self.release_id = settings.worker_release_id.strip()
        self.worker_session_id = uuid.uuid4().hex
        self.file_base_url = ""

        boot_mod.mark_once(
            boot_mod.PHASE_SDK_READY, function=",".join(self.functions)
        )
        self.store = ModelStore(
            self._send,
            cache_dir=self.resolver.snapshots_root.parent,
            vram_budget_bytes=int(budget) or None,
        )
        self.resolver.bind_store(self.store)
        bind_active_store(self.store)
        self.materialization = CheckpointMaterialization(
            self.store,
            announce=self._announce_readiness,
            warm=self._run_boot_warmup,
        )
        self._boot_picks: Dict[str, _DispatchPicks] = {}

        self.draining = False
        self.drained = asyncio.Event()
        self._jobs: Dict[Tuple[str, int], asyncio.Task[None]] = {}
        self._canceled: set[Tuple[str, int]] = set()
        self._dispatch: Optional[Any] = None
        self._drain_task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_requested = False

    def _build_adoption(self) -> Any:
        release_id = (self.settings.worker_release_id or "").strip()
        _, sm = hostfacts.device_identity()
        if not release_id or not sm:
            logger.info(
                "adopt: serving eager (release_id=%r sm=%r) — a pod adopts "
                "for its own release, on the card it can see",
                release_id, sm,
            )
            return None
        from .serving.self_mint import production_mint
        from .serving.serve_adoption import ServeAdoption

        artifacts = tensorhub_cache_dir() / "compiled-graphs"
        self._mint_box: List[Any] = []

        def _arm(adoption: Any) -> Any:
            mint = production_mint(
                store=adoption.store, artifacts_dir=artifacts,
                cas_dir=tensorhub_cas_dir(), sm=sm,
            )
            self._mint_box.append(mint)
            return mint.arm(adoption)

        return ServeAdoption(
            release_id, sm=sm, artifacts_dir=artifacts,
            cas_dir=tensorhub_cas_dir(), on_adopted=_arm,
        )

    def mint_facts(self) -> Dict[str, Any]:
        """The counted observable, for anything that asks this worker what its background mint is doing."""
        adoption = getattr(self, "adoption", None)
        if adoption is None:
            return {"adopting": False, "mint": "not_armed",
                    "refusal": "no release id or no visible GPU"}
        facts: Dict[str, Any] = dict(adoption.facts())
        boxed = getattr(self, "_mint_box", [])
        facts["mint"] = (
            boxed[-1].status().facts() if boxed else "not_armed"
        )
        return facts

    @property
    def functions(self) -> List[str]:
        return sorted(self.loaded.entrypoints)

    @property
    def phase(self) -> "pb.WorkerPhase":
        """This worker's startup phase."""
        return self.materialization.phase()

    def _state_delta(self) -> pb.StateDelta:
        ready = self.materialization.ready
        delta = pb.StateDelta(
            phase=self.phase,
            available_functions=self.functions if ready else [],
            loading_functions=[] if ready else self.functions,
        )
        free = hostfacts.free_vram_bytes()
        if free is not None:
            delta.free_vram_bytes = int(free)
        return delta

    async def _announce_readiness(self) -> None:
        if self.materialization.ready and self.functions:
            boot_mod.mark_once(
                boot_mod.PHASE_FIRST_REQUEST_SERVABLE,
                function=",".join(self.functions),
            )
        try:
            await self._send(pb.WorkerMessage(state_delta=self._state_delta()))
        except Exception:  # noqa: BLE001 — the heartbeat re-states this
            logger.warning("readiness state delta send failed", exc_info=True)
        if not self.materialization.failed:
            return
        for fn in self.functions:
            try:
                await self._send(pb.WorkerMessage(fn_unavailable=pb.FnUnavailable(
                    function_name=fn,
                    reason=REASON_MODEL_UNAVAILABLE,
                    detail=(
                        "boot-time checkpoint materialization failed: "
                        f"{self.materialization.failure}"
                    ),
                )))
            except Exception:  # noqa: BLE001
                logger.warning("fn_unavailable send failed for %s", fn, exc_info=True)

    async def _run_boot_warmup(self) -> None:
        await asyncio.to_thread(self.serve.boot_warmup, prepare=self._bind_boot_picks)

    def _bind_boot_picks(self, function: str) -> str:
        picks = self._boot_picks.get(function)
        if picks is None or not picks.by_slot:
            return (
                "no boot-time checkpoint binding: the HelloAck's "
                "DesiredResidency seeded no per-function ModelBinding for "
                f"{function!r} and its slot/ref shape is not unambiguous "
                "(the worker never guesses which bytes to serve)"
            )
        _DISPATCH.set(picks)
        return ""

    def build_hello(self) -> pb.Hello:
        return pb.Hello(
            protocol_version=PROTOCOL_VERSION,
            worker_id=self.worker_id,
            release_id=self.release_id,
            models=self.store.residency_snapshot(),
            state=self._state_delta(),
            in_flight=[
                pb.InFlightJob(request_id=rid, attempt=att)
                for rid, att in sorted(self._jobs)
            ],
            heartbeat_interval_ms=HEARTBEAT_INTERVAL_MS,
            worker_session_id=self.worker_session_id,
        )

    async def on_hello_ack(self, ack: pb.HelloAck) -> None:
        self.file_base_url = str(ack.file_base_url or "")
        logger.info(
            "hello acked: functions=%s file_base_url=%s",
            self.functions, self.file_base_url or "(none)",
        )
        boot_mod.mark_once(boot_mod.PHASE_HELLO)
        config = CheckpointConfig.from_wire(ack.desired_residency)
        self._boot_picks = boot_picks(ack.desired_residency, self.loaded, config)
        self.materialization.configure(config)
        if self.functions and self.materialization.ready:
            boot_mod.mark_once(
                boot_mod.PHASE_FIRST_REQUEST_SERVABLE,
                function=",".join(self.functions),
            )
        if not self.file_base_url:
            logger.error(
                "hello acked with NO file_base_url: the compiled-graph receipt "
                "gate cannot arm, so every hub-delivered artifact will be "
                "refused and self-minted, and C2PA remote signing is "
                "unavailable"
            )
            return
        receipts.configure(self.file_base_url, worker_credential.current)
        content_credentials.configure_remote_signer(
            self.file_base_url, worker_credential.current
        )

    async def on_disconnect(self) -> None:
        logger.warning("hub stream disconnected; %d job(s) in flight", len(self._jobs))

    async def on_message(self, msg: pb.SchedulerMessage) -> None:
        which = msg.WhichOneof("msg")
        if which == "run_job":
            await self._on_run_job(msg.run_job)
        elif which == "cancel_job":
            self._on_cancel_job(msg.cancel_job)
        elif which == "drain":
            self.start_drain(int(msg.drain.deadline_ms))
        elif which == "serve_posture":
            posture = msg.serve_posture
            serve_posture.apply_command(
                bool(posture.eager_only),
                actor=str(getattr(posture, "actor", "") or ""),
                reason=str(getattr(posture, "reason", "") or ""),
            )
        else:
            logger.warning("no handler for SchedulerMessage.%s; ignored", which)

    async def _send(self, msg: pb.WorkerMessage) -> None:
        await self.transport.send(msg)

    async def _send_result(
        self,
        request_id: str,
        attempt: int,
        status: "pb.JobStatus",
        *,
        inline: Optional[bytes] = None,
        safe_message: str = "",
        metrics: Optional[pb.JobMetrics] = None,
        adjustments: Tuple[Dict[str, str], ...] = (),
        traceback_tail: str = "",
    ) -> None:
        result = pb.JobResult(
            request_id=request_id,
            attempt=attempt,
            status=status,
            safe_message=safe_message[:512],
        )
        if traceback_tail:
            result.traceback = traceback_tail[:MAX_TRACEBACK_BYTES]
        if inline is not None:
            result.inline = inline
        if metrics is not None:
            result.metrics.CopyFrom(metrics)
        for adj in adjustments:
            result.adjustments.add(
                field=str(adj.get("field", "")),
                requested=str(adj.get("requested", "")),
                applied=str(adj.get("applied", "")),
                reason=str(adj.get("reason", "")),
            )
        await self._send(pb.WorkerMessage(job_result=result))

    async def _on_run_job(self, run: pb.RunJob) -> None:
        key = (str(run.request_id), int(run.attempt))
        if key in self._jobs:
            await self._send(
                pb.WorkerMessage(
                    job_accepted=pb.JobAccepted(request_id=key[0], attempt=key[1])
                )
            )
            return
        if self.draining:
            await self._send_result(
                *key, pb.JOB_STATUS_RETRYABLE, safe_message="worker draining"
            )
            return
        await self._send(
            pb.WorkerMessage(
                job_accepted=pb.JobAccepted(request_id=key[0], attempt=key[1])
            )
        )
        task = asyncio.create_task(self._run_one(run, key), name=f"job-{key[0]}")
        self._jobs[key] = task

        def _retire(_done: "asyncio.Task[None]", k: Tuple[str, int] = key) -> None:
            self._jobs.pop(k, None)

        task.add_done_callback(_retire)

    def _on_cancel_job(self, cancel: pb.CancelJob) -> None:
        key = (str(cancel.request_id), int(cancel.attempt))
        self._canceled.add(key)
        task = self._jobs.get(key)
        if task is not None and not task.done():
            task.cancel()
            logger.warning(
                "cancel for in-flight %s attempt=%d: terminal shipped now; the "
                "author's call keeps running on its worker thread to completion",
                *key,
            )

    def _envelope_of(self, run: pb.RunJob) -> Dict[str, Any]:
        payload: Any = {}
        if run.input_payload:
            payload = msgspec.msgpack.decode(run.input_payload)
        spec = self.loaded.entrypoints.get(str(run.function_name))
        envelope: Dict[str, Any] = {"input": payload}
        if spec is None:
            return envelope
        picks = {b.slot: b.ref for b in run.models if b.slot and b.ref}
        slots = [name for name, _ in spec.model_params]
        if len(slots) == 1:
            if picks.get(slots[0]):
                envelope["model"] = picks[slots[0]]
        elif slots:
            named = {s: picks[s] for s in slots if picks.get(s)}
            if named:
                envelope["models"] = named
        return envelope

    def _request_context_facts(self, run: pb.RunJob) -> Dict[str, Any]:
        hints: Dict[str, Any] = {}
        if run.media_bytes == pb.MEDIA_BYTES_INLINE:
            hints["output_format"] = "inline"
        emitter, chunk_sink = self._progress_channel(run)
        return {
            "owner": str(run.org or "") or None,
            "invoker_id": str(run.invoker_id or "") or None,
            "file_api_base_url": self.file_base_url or None,
            "worker_capability_token": str(run.capability_token or "") or None,
            "execution_hints": hints or None,
            "emitter": emitter,
            "chunk_sink": chunk_sink,
        }

    def _progress_channel(
        self, run: pb.RunJob
    ) -> Tuple[
        Callable[[Dict[str, Any]], None], Callable[[bytes, str], None]
    ]:
        loop = asyncio.get_running_loop()
        seq = itertools.count(1)
        request_id, attempt = str(run.request_id), int(run.attempt)

        async def _send_progress(data: bytes, content_type: str) -> None:
            await self._send(
                pb.WorkerMessage(
                    job_progress=pb.JobProgress(
                        request_id=request_id,
                        attempt=attempt,
                        seq=next(seq),
                        data=data,
                        content_type=content_type,
                    )
                )
            )

        def _put(data: bytes, content_type: str) -> None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    _send_progress(data, content_type), loop
                )
            except RuntimeError:
                return
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                return
            future.result()

        def _emit_event(event: Dict[str, Any]) -> None:
            try:
                data = msgspec.json.encode(event)
            except Exception:
                logger.debug(
                    "unencodable ctx event dropped for %s", request_id
                )
                return
            _put(data, EVENT_CONTENT_TYPE)

        return _emit_event, _put

    def _check_lane(self, run: pb.RunJob) -> None:
        lane = str(run.lane or "").strip()
        if lane and lane not in self.lanes:
            raise ServeDispatchError(
                f"dispatch names lane {lane!r}; this release serves "
                f"{sorted(self.lanes) or '[]'} — never a silent fallback"
            )

    def _dispatch_counter(self) -> Any:
        counter = getattr(self, "_dispatch", None)
        if counter is None:
            from .serving.dispatch_counter import DispatchCounter

            counter = DispatchCounter()
            self._dispatch = counter
        counter.install(self)
        return counter

    def _served_lane(self, ctx_box: List[Any], *, solo: bool) -> str:
        ctx = ctx_box[0] if ctx_box else None
        resolved = getattr(ctx, "_resolved_lane", None) if ctx is not None else None
        body = str(getattr(resolved, "body", "") or "")
        if not body:
            return ""
        if not solo or len(self._jobs) > 1:
            return body
        counts = self._dispatch_counter().take()
        if counts.armed_modules == 0:
            return f"{body}+eager"
        return (
            f"{body}+compiled" if counts.compiled_graph_calls > 0
            else f"{body}+eager"
        )

    async def _run_one(self, run: pb.RunJob, key: Tuple[str, int]) -> None:
        accepted_at = time.monotonic()
        if key in self._canceled:
            self._canceled.discard(key)
            await self._send_result(
                *key, pb.JOB_STATUS_CANCELED, safe_message="canceled before start"
            )
            return
        status = pb.JOB_STATUS_FATAL
        inline: Optional[bytes] = None
        message = ""
        tb = ""
        adjustments: Tuple[Dict[str, str], ...] = ()
        stages: Optional[Any] = None
        started = accepted_at
        ctx_box: List[Any] = []
        solo = len(self._jobs) <= 1
        self._dispatch_counter().reset()
        try:
            self._check_lane(run)
            envelope = self._envelope_of(run)
            _DISPATCH.set(_picks_of(run))
            started = time.monotonic()
            inflight = postmortem.note_inflight(
                "request", str(run.function_name), request_id=str(run.request_id)
            )
            renew: Optional[asyncio.Task[None]] = None

            def _bind_context(ctx: Any, box: List[Any] = ctx_box) -> None:
                box.append(ctx)

            if run.capability_token and self.file_base_url:
                renew = asyncio.create_task(
                    renew_capability_while_running(
                        file_base_url=self.file_base_url,
                        request_id=str(run.request_id),
                        attempt=int(run.attempt),
                        get_worker_jwt=worker_credential.current,
                        get_token=lambda: (
                            ctx_box[0]._worker_capability_token or ""
                            if ctx_box else str(run.capability_token or "")
                        ),
                        set_token=lambda t: (
                            setattr(ctx_box[0], "_worker_capability_token", t)
                            if ctx_box else None
                        ),
                    ),
                    name=f"cap-renew-{run.request_id}",
                )
            try:
                outcome = await asyncio.to_thread(
                    functools.partial(
                        self.serve.invoke,
                        str(run.function_name),
                        envelope,
                        request_id=str(run.request_id),
                        attempt=int(run.attempt),
                        input_assets=manifest_from_run_job(run.input_assets),
                        snapshots=resolved_repos(run.snapshots, run.models),
                        context=self._request_context_facts(run),
                        on_context=_bind_context,
                    )
                )
            finally:
                postmortem.clear_inflight(inflight)
                if renew is not None:
                    renew.cancel()
                cleanup_input_assets(str(run.request_id), int(run.attempt))
            inline = msgspec.msgpack.encode(outcome.result)
            adjustments = outcome.adjustments
            stages = outcome.stages
            status = pb.JOB_STATUS_OK
        except asyncio.CancelledError:
            await self._send_result(*key, pb.JOB_STATUS_CANCELED, safe_message="canceled")
            raise
        except (
            EnvelopeError,
            ServeDispatchError,
            msgspec.ValidationError,
            ApiValidationError,
        ) as exc:
            status, message = pb.JOB_STATUS_INVALID, f"{type(exc).__name__}: {exc}"
            tb = traceback_tail(exc)
            logger.warning("job %s attempt=%d rejected: %s", *key, exc)
        except NeverFits as exc:
            status, message = pb.JOB_STATUS_FATAL, f"{type(exc).__name__}: {exc}"
            tb = traceback_tail(exc)
            logger.error("job %s attempt=%d cannot ever fit: %s", *key, exc)
        except (CheckpointUnresolved, ResidencyError) as exc:
            status, message = pb.JOB_STATUS_RETRYABLE, f"{type(exc).__name__}: {exc}"
            tb = traceback_tail(exc)
            logger.error("job %s attempt=%d unplaceable: %s", *key, exc)
        except Exception as exc:  # noqa: BLE001 — the terminal must still ship
            status, message = pb.JOB_STATUS_FATAL, f"{type(exc).__name__}: {exc}"
            tb = traceback_tail(exc)
            logger.exception("job %s attempt=%d failed", *key)
        finally:
            self._canceled.discard(key)
        now = time.monotonic()
        runtime_ms = int((now - started) * 1000)
        metrics = pb.JobMetrics(
            runtime_ms=runtime_ms,
            queue_ms=int((started - accepted_at) * 1000),
        )
        metrics.lane = self._served_lane(ctx_box, solo=solo)
        metrics.stage_ms.update(stage_ms_for_metrics(stages, runtime_ms))
        await self._send_result(
            *key,
            status,
            inline=inline,
            safe_message=message,
            metrics=metrics,
            adjustments=adjustments,
            traceback_tail=tb,
        )

    def start_drain(self, deadline_ms: int = 0) -> None:
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(
                self._drain(int(deadline_ms)), name="drain"
            )

    async def _drain(self, deadline_ms: int) -> None:
        self.draining = True
        logger.info(
            "draining: %d job(s) in flight, deadline_ms=%d", len(self._jobs), deadline_ms
        )
        budget = (deadline_ms / 1000.0) if deadline_ms > 0 else None
        if self._jobs:
            await asyncio.wait(set(self._jobs.values()), timeout=budget)
        await self.transport.close_after_flush(budget)
        self.drained.set()

    def stop(self) -> None:
        """Thread-safe stop (tests / embedding); production exits via Drain."""
        self._stop_requested = True
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self.transport.stop)

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_MS / 1000.0)
            logger.debug("mint: %s", self.mint_facts())
            try:
                await self._send(pb.WorkerMessage(state_delta=self._state_delta()))
            except Exception:
                logger.warning("heartbeat send failed", exc_info=True)

    async def arun(self) -> int:
        loop = self._loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig, self.start_drain, _SIGNAL_DRAIN_DEADLINE_MS
                )
            except (NotImplementedError, RuntimeError):
                pass
        from . import activity as activity_mod
        from . import boot_phases as boot_mod

        activity_mod.bind_sink(self._send, loop)
        boot_mod.bind_sink(self._send, loop)
        boot_mod.bind_servable_probe(lambda: self.materialization.ready)

        self.store.bind_loop()
        try:
            self.store.rescan_disk()
        except Exception as exc:  # noqa: BLE001 — a cold CAS is not a boot failure
            logger.warning("disk rescan at boot failed: %s: %s", type(exc).__name__, exc)

        process_role.declare(process_role.ROLE_SERVING)
        process_role.emit_boot_role()

        heartbeat = asyncio.create_task(self._heartbeat(), name="heartbeat")
        transport_task = asyncio.create_task(self.transport.run(), name="transport")
        try:
            await transport_task
        except FatalTransportError as exc:
            logger.error("worker exiting: %s", exc)
            raise
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        if self.drained.is_set():
            logger.info("worker drained; exiting 0")
            return 0
        if self._stop_requested:
            return 0
        raise UnexpectedWorkerExit(
            "transport loop ended without a Drain command or shutdown signal "
            f"(connected={self.transport.connected} draining={self.draining})"
        )

    def run(self) -> int:
        """Always returns an exit code."""
        try:
            return asyncio.run(self.arun())
        except (FatalTransportError, UnexpectedWorkerExit) as exc:
            from .worker_fatal import report_worker_fatal

            logger.error("worker exiting on a fatal: %s", exc, exc_info=True)
            report_worker_fatal(self.settings, "run_loop", exc, exit_code=1)
            return 1


__all__ = [
    "CheckpointUnresolved",
    "DuplicateEntrypoint",
    "HEARTBEAT_INTERVAL_MS",
    "HubBindingResolver",
    "NoEntrypointsDeclared",
    "SnapshotSizer",
    "UnexpectedWorkerExit",
    "Worker",
    "WorkerBootError",
    "harvest_entrypoints",
]
