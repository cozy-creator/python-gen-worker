"""v2 worker wiring (pgw#1373): ``@entrypoint`` modules -> ServeLoop -> hub stream.

The connector the hardcut left missing. One asyncio loop owns three things:

* the **surface** — the author modules are imported and their ``@entrypoint``
  declarations harvested exactly the way :mod:`gen_worker.serving.loader`
  reads one endpoint directory, only driven by MODULE NAMES;
* the **serve stack** — one :class:`~gen_worker.serving.serve_loop.ServeLoop`
  over a :class:`~gen_worker.serving.residency.ResidencyManager`, with a
  :class:`BindingResolver` that materializes the hub's own per-dispatch
  ``ModelBinding`` rows against this pod's local snapshot store;
* the **wire** — :class:`gen_worker.transport.Transport`, unchanged, driven
  by the handler contract in its docstring.

Nothing here guesses. A pick with no binding, no manifest digest or no local
tree is a typed refusal naming the pick; modules with no ``@entrypoint`` (or
with a v1 declaration still stamped on them) refuse at boot naming the
migration.
"""

from __future__ import annotations

import asyncio
import contextvars
import importlib
import json
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import msgspec

from . import hostfacts
from .config import Settings
from .host_move_guard import install as _install_host_move_guard
from .models.cache_paths import tensorhub_cache_dir, tensorhub_cas_dir
from .models.cozy_snapshot import snapshot_dir_key
from .models.disk_gc import tree_bytes
from .models.projection import SNAPSHOTS_DIR
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

logger = logging.getLogger(__name__)

#: Promised in Hello; the hub reaps after 6 consecutive misses (~60s). The
#: cadence the deleted lifecycle promised — unchanged, because the hub-side
#: reaper was tuned against it.
HEARTBEAT_INTERVAL_MS = 10_000

#: Bounds only the SHUTDOWN half of a signal drain, never tenant work.
_SIGNAL_DRAIN_DEADLINE_MS = 30_000

#: Activation reservation alongside the weights, as a divisor of weight bytes.
#: The same estimate ``python -m gen_worker.serving`` reserves locally; the
#: exact per-(model type, resolution class) number rides the pgw#1380 sizer.
_HEADROOM_DIVISOR = 4

#: v1 decorator stamps. A module still carrying one is a build failure with a
#: name attached, never an empty manifest.
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


# --------------------------------------------------------------------------
# surface harvesting
# --------------------------------------------------------------------------



def residency_budget(stated: int = 0) -> int:
    """The residency budget, in the order the answers are trustworthy:
    a stated budget (a CONFIG the caller owns), then the card's own headroom,
    then — on a host with NO CUDA at all — available host RAM.

    The CPU arm is not a fallback papering over a missing reading; it is the
    correct budget for a machine that serves on the CPU, which cozy-local, CI
    and every fake-weights drive are. Refusing there made the worker unbootable
    on all three while `python -m gen_worker.serving` served happily beside it,
    and the two paths must not disagree about whether a host can serve. What
    stays a refusal is the case that is genuinely UNKNOWN: a host that HAS a
    card whose memory cannot be read.

    A MODULE-LEVEL FUNCTION, not a branch inside `__init__`, and that is the
    point (pgw#1411). Which arm runs depends on the box: this workspace's dev
    machine reports `cuda_ready()` true with real headroom, CI reports it
    false, and the `cuda_ready and not headroom` refusal therefore executed on
    NEITHER — it shipped untested because no environment could reach it. A
    decision buried in a constructor can only be tested by the machine that
    runs it; this one can be tested by naming the two readings.
    """
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
    """Import the author modules and state their combined serve surface.

    Mirrors ``serving.loader._surface_of`` per module: read
    :data:`ENTRYPOINT_ATTR` off every module-level object, keep only the
    declarations this module owns, and collect the ``Model`` classes the
    specs reference.
    """
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
                continue  # re-exported from elsewhere: not this module's surface
            # THE WIRE ROUTE IS THE SLUG, not the python name. The hub
            # dispatches `RunJob.function_name` as `slugify_name(name)` — the
            # same normalization `discovery.validation` checks for collisions
            # against — so `steal_credentials` is reached as
            # `steal-credentials`. Keying this table on the raw name made every
            # multi-word entrypoint unroutable while single-word ones worked,
            # which is the shape of bug that ships.
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


# --------------------------------------------------------------------------
# per-dispatch binding resolution
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Pick:
    """One hub-resolved ``ModelBinding`` row, as the resolver needs it."""

    slot: str
    ref: str
    manifest_digest: str
    #: The hub's two-column defaults pair, RAW and TOGETHER (th#2140): the
    #: recognized model name and its JSONB row. They travel as a pair because
    #: the name is what chooses the schema the row decodes against — split
    #: them and the row is undecodable, which is pgw#1415's whole defect.
    model: str
    inference_defaults: str


@dataclass(frozen=True, slots=True)
class _DispatchPicks:
    by_ref: Mapping[str, _Pick]
    by_slot: Mapping[str, str]


_EMPTY_PICKS = _DispatchPicks(by_ref={}, by_slot={})

#: The dispatch a resolution belongs to. Set per job Task; ``asyncio.to_thread``
#: copies the context, so the serve thread reads the same dispatch.
_DISPATCH: contextvars.ContextVar[_DispatchPicks] = contextvars.ContextVar(
    "gen_worker_dispatch_picks", default=_EMPTY_PICKS
)


def _picks_of(run: pb.RunJob) -> _DispatchPicks:
    by_ref: Dict[str, _Pick] = {}
    by_slot: Dict[str, str] = {}
    for binding in run.models:
        pick = _Pick(
            slot=str(binding.slot),
            ref=str(binding.ref),
            manifest_digest=str(binding.manifest_digest),
            model=str(binding.model).strip(),
            inference_defaults=str(binding.inference_defaults),
        )
        by_ref[pick.ref] = pick
        by_slot[pick.slot] = pick.ref
    return _DispatchPicks(by_ref=by_ref, by_slot=by_slot)


class HubBindingResolver:
    """The hub's half of the deploy state, per dispatch.

    ``resolve`` materializes a pick the hub already validated: the dispatch's
    own ``ModelBinding`` row names the composed manifest digest, and this
    worker's snapshot store either holds that tree or it does not. There is no
    fallback — a miss names the pick and the path it looked for.
    """

    def __init__(self, snapshots_root: Optional[Path] = None) -> None:
        self.snapshots_root = (
            Path(snapshots_root)
            if snapshots_root is not None
            else tensorhub_cas_dir() / SNAPSHOTS_DIR
        )

    def _pick(self, model_cls: type, checkpoint_ref: str) -> _Pick:
        picks = _DISPATCH.get()
        pick = picks.by_ref.get(checkpoint_ref)
        if pick is None:
            raise CheckpointUnresolved(
                f"{model_cls.__name__}: this dispatch carries no ModelBinding "
                f"for checkpoint {checkpoint_ref!r} "
                f"(bound refs: {sorted(picks.by_ref) or '[]'})"
            )
        if not pick.manifest_digest:
            raise CheckpointUnresolved(
                f"{model_cls.__name__}: binding for {checkpoint_ref!r} carries "
                f"no manifest_digest; the worker has no other fetch pointer"
            )
        return pick

    def tree_for(self, model_cls: type, checkpoint_ref: str) -> Path:
        pick = self._pick(model_cls, checkpoint_ref)
        digest = pick.manifest_digest.split(":", 1)[-1]
        tree = self.snapshots_root / snapshot_dir_key(digest)
        if not tree.is_dir():
            raise CheckpointUnresolved(
                f"{model_cls.__name__}: checkpoint {checkpoint_ref!r} "
                f"(manifest {pick.manifest_digest}) is not materialized on this "
                f"worker — no tree at {tree}"
            )
        return tree

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
        # pgw#1415's FENCE. The hub's `model` column is NOT NULL beside the
        # defaults JSONB (th#2140 migration 0104), so a row WITHOUT a name
        # cannot be a checkpoint the hub left unclassified — it is a pair that
        # broke in transit, which is exactly how this defect served 28-step
        # CFG-on for a 4-step guidance-free Turbo checkpoint for a week. The
        # unclassified arm (no name, NO row) keeps warn-and-serve, pgw#1377's
        # read-side matrix, untouched; THIS arm refuses, because serving
        # platform fallbacks while holding the checkpoint's own tuned row is
        # the one outcome nobody can want.
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
        """The dispatch's own per-slot ref. '' = this dispatch bound nothing
        to the slot, which decode refuses naming the slot."""
        return _DISPATCH.get().by_slot.get(slot_name, "")


class SnapshotSizer:
    """Weight bytes from the materialized tree's tensorfs manifest.

    ``tree_bytes`` sizes a PROJECTED tree from its manifest rather than
    walking stubs, so this is the exact resident cost, known before any byte
    moves — which is what admission requires.
    """

    def __init__(self, resolver: HubBindingResolver) -> None:
        self._resolver = resolver
        self._cache: Dict[str, int] = {}

    def _bytes(self, checkpoint_ref: str) -> int:
        cached = self._cache.get(checkpoint_ref)
        if cached is not None:
            return cached
        picks = _DISPATCH.get()
        pick = picks.by_ref.get(checkpoint_ref)
        if pick is None:
            raise CheckpointUnresolved(
                f"admission asked for the size of {checkpoint_ref!r}, which "
                f"this dispatch does not bind "
                f"(bound refs: {sorted(picks.by_ref) or '[]'})"
            )
        digest = pick.manifest_digest.split(":", 1)[-1]
        tree = self._resolver.snapshots_root / snapshot_dir_key(digest)
        if not tree.is_dir():
            raise CheckpointUnresolved(
                f"admission needs the size of {checkpoint_ref!r}; no "
                f"materialized tree at {tree}"
            )
        size = int(tree_bytes(tree))
        self._cache[checkpoint_ref] = size
        return size

    def resident_bytes(self, checkpoint_ref: str, lane: str) -> int:
        return self._bytes(checkpoint_ref)

    def activation_headroom_bytes(self, checkpoint_ref: str, lane: str) -> int:
        return max(self._bytes(checkpoint_ref) // _HEADROOM_DIVISOR, 1)


# --------------------------------------------------------------------------
# the worker
# --------------------------------------------------------------------------


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
        # An author-side oversized `.to("cpu")` becomes a typed job error
        # instead of a cgroup OOM SIGKILL of the whole worker.
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
        # pgw#1371/pgw#1372: ADOPT-FIRST BOOT, then fill this pod's own holes.
        # Both halves existed and neither was constructed here, so every
        # `ctx.compile` on every real serving pod was a pass-through and the
        # mint had no work-list to read. `None` back is the eager bridge, with
        # a stated reason — never a boot failure.
        self.adoption = self._build_adoption()
        try:
            # The deploy's active lane. A single-lane model needs none; a
            # multi-lane one has no boot-time wire field yet (RunJob.lane is
            # per-dispatch), so it refuses below naming the lanes it declares.
            self.serve = ServeLoop(
                self.loaded, residency=self.residency, resolver=self.resolver,
                lane_contract=lane, output_dir=output_dir,
                compile_sink_for=(
                    self.adoption.sink_for if self.adoption is not None else None
                ),
                # The mint's trigger: fired when the author's load(ctx) has
                # RETURNED, which is the first instant the hole list is whole.
                on_loaded=(
                    self.adoption.loaded if self.adoption is not None else None
                ),
            )
        except EndpointLoadError as exc:
            # A multi-lane model needs the deploy's lane pick, which no wire
            # field delivers to boot yet — refuse naming it.
            raise WorkerBootError(str(exc)) from exc
        #: Declared lane handles, so a RunJob naming a lane this release does
        #: not serve refuses instead of silently serving another one.
        self.lanes: frozenset[str] = frozenset(
            lane_handle(lane) for lane in self.serve.lanes.values() if lane is not None
        )

        # pgw#763: IN THE SPLIT'S COMPUTE CHILD THE PARENT OWNS THE gRPC
        # STREAM. This process speaks frames to it over the child socket and
        # never dials the orchestrator — a child that constructed the real
        # Transport would dial the placeholder address the parent passes it
        # (`127.0.0.1:1`) and hang there forever, never reaching hello, which
        # is exactly what it did. Same handler contract either way, so the
        # only thing that changes is which object carries the frames.
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

        # pgw#763 delta 1: this process is the COMPUTE CHILD and holds no
        # credential, so reading a bootstrap JWT for identity refuses on every
        # real serving pod. The parent RELAYS the two claims as WORKER_ID /
        # WORKER_RELEASE_ID; those are the identity, and their absence is a
        # named pid fallback rather than a silent empty string.
        self.worker_id = (
            settings.worker_id.strip() or f"py-worker-{os.getpid()}"
        )
        self.release_id = settings.worker_release_id.strip()
        self.worker_session_id = uuid.uuid4().hex
        self.file_base_url = ""

        self.phase = pb.WORKER_PHASE_READY
        self.draining = False
        self.drained = asyncio.Event()
        self._jobs: Dict[Tuple[str, int], asyncio.Task[None]] = {}
        self._canceled: set[Tuple[str, int]] = set()
        self._drain_task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_requested = False

    # ---- adopt-first boot + the background mint ---------------------------

    def _build_adoption(self) -> Any:
        """This pod's :class:`ServeAdoption`, or ``None`` — the eager bridge.

        The two facts an adopt needs are the pod's OWN release id (the hub
        refuses a credential adopting for a sibling release) and this card's
        sm — artifacts are per-sm and there is no such thing as adopting for
        a GPU you cannot see. Missing either is a stated eager pod, not a
        refusal: a CPU-only worker and an unstamped release are both ordinary.

        The mint is constructed WITH the adoption and armed BY it — the hook
        fires the instant the session registers its holes, which is the first
        model load. It is never armed from here, because at this point in boot
        no model has loaded and the work-list does not exist yet.
        """
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
        #: Filled by the hook below, at the first model load. A list rather
        #: than an attribute assignment so `mint_facts()` reads either a whole
        #: mint or none, never a half-built one.
        self._mint_box: List[Any] = []

        def _arm(adoption: Any) -> None:
            # The trigger (pgw#1371): holes are registered, so the mint has a
            # work-list. It runs on its own daemon thread, CPU-reserved and
            # niced against this process — the serving loop is never involved.
            mint = production_mint(
                store=adoption.store, artifacts_dir=artifacts,
                cas_dir=tensorhub_cas_dir(), sm=sm,
            )
            self._mint_box.append(mint)
            mint.arm(adoption)

        return ServeAdoption(
            release_id, sm=sm, artifacts_dir=artifacts,
            cas_dir=tensorhub_cas_dir(), on_adopted=_arm,
        )

    def mint_facts(self) -> Dict[str, Any]:
        """The counted observable, for anything that asks this worker what
        its background mint is doing.

        Three answers, all distinct and none of them a bare zero: no adoption
        was attempted at all, an adoption was attempted and refused (with the
        reason), or a mint exists and has a named state plus its counts.
        """
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

    # ---- state snapshots --------------------------------------------------

    @property
    def functions(self) -> List[str]:
        return sorted(self.loaded.entrypoints)

    def _state_delta(self) -> pb.StateDelta:
        delta = pb.StateDelta(
            phase=self.phase,
            available_functions=self.functions,
        )
        # 0 is the wire's "unmeasured"; only a real reading is reported.
        free = hostfacts.free_vram_bytes()
        if free is not None:
            delta.free_vram_bytes = int(free)
        return delta

    def build_hello(self) -> pb.Hello:
        # NO `resources`: the split parent measures the silicon in a process
        # that imported no tenant code and stamps every relayed Hello, so
        # anything measured here would be replaced before the hub saw it.
        # NO `models`/`lifecycle_snapshot`: this build carries no residency
        # ledger and no intent registry, and an invented one is worse than an
        # absent one.
        return pb.Hello(
            protocol_version=PROTOCOL_VERSION,
            worker_id=self.worker_id,
            release_id=self.release_id,
            state=self._state_delta(),
            in_flight=[
                pb.InFlightJob(request_id=rid, attempt=att)
                for rid, att in sorted(self._jobs)
            ],
            heartbeat_interval_ms=HEARTBEAT_INTERVAL_MS,
            worker_session_id=self.worker_session_id,
        )

    # ---- transport handlers ------------------------------------------------

    async def on_hello_ack(self, ack: pb.HelloAck) -> None:
        self.file_base_url = str(ack.file_base_url or "")
        logger.info(
            "hello acked: functions=%s file_base_url=%s",
            self.functions, self.file_base_url or "(none)",
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
        else:
            # Named, not swallowed: this build implements no residency
            # reconciliation (model_op) and no posture command.
            logger.warning("no handler for SchedulerMessage.%s; ignored", which)

    # ---- dispatch ----------------------------------------------------------

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
    ) -> None:
        result = pb.JobResult(
            request_id=request_id,
            attempt=attempt,
            status=status,
            safe_message=safe_message[:512],
        )
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
            # Retransmit of an accepted attempt: re-ack, never re-run.
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
            # CANCEL THE AWAITING TASK, which is what makes the CANCELED result
            # prompt. It does NOT interrupt the author's code: the serve loop
            # runs it on a worker thread and no v2 surface can stop a thread
            # mid-call — exactly as the v1 executor could not stop a CPU-bound
            # sync handler either. What the hub gets back is the terminal it
            # asked for, on time, instead of nothing until the handler happens
            # to finish. Marking without cancelling produced NO job_result at
            # all, which reads to the hub as a hung pod.
            task.cancel()
            logger.warning(
                "cancel for in-flight %s attempt=%d: terminal shipped now; the "
                "author's call keeps running on its worker thread to completion",
                *key,
            )

    def _envelope_of(self, run: pb.RunJob) -> Dict[str, Any]:
        """The signature-derived envelope for this dispatch.

        ``input_payload`` is the caller's input; the checkpoint picks come off
        the hub's own ``ModelBinding`` rows, which is where deploy state lives.
        """
        payload: Any = {}
        if run.input_payload:
            payload = msgspec.msgpack.decode(run.input_payload)
        spec = self.loaded.entrypoints.get(str(run.function_name))
        envelope: Dict[str, Any] = {"input": payload}
        if spec is None:
            return envelope  # invoke() refuses naming the function
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

    def _check_lane(self, run: pb.RunJob) -> None:
        lane = str(run.lane or "").strip()
        if lane and lane not in self.lanes:
            raise ServeDispatchError(
                f"dispatch names lane {lane!r}; this release serves "
                f"{sorted(self.lanes) or '[]'} — never a silent fallback"
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
        adjustments: Tuple[Dict[str, str], ...] = ()
        started = accepted_at
        try:
            self._check_lane(run)
            envelope = self._envelope_of(run)
            _DISPATCH.set(_picks_of(run))
            started = time.monotonic()
            # pgw#676: STAMP THE IN-FLIGHT MARKER around tenant execution. A
            # SIGKILL mid-handler leaves this marker behind, and it is the only
            # thing that lets the supervisor's post-mortem attribute the death
            # to a FUNCTION and build the native-crash streak that eventually
            # refuses it. Without it a signal death is charged to nothing, the
            # streak never forms, and a reliably-crashing handler is served
            # forever.
            inflight = postmortem.note_inflight(
                "request", str(run.function_name), request_id=str(run.request_id)
            )
            try:
                outcome = await asyncio.to_thread(
                    self.serve.invoke,
                    str(run.function_name),
                    envelope,
                    request_id=str(run.request_id),
                )
            finally:
                postmortem.clear_inflight(inflight)
            inline = msgspec.msgpack.encode(outcome.result)
            adjustments = outcome.adjustments
            status = pb.JOB_STATUS_OK
        except asyncio.CancelledError:
            await self._send_result(*key, pb.JOB_STATUS_CANCELED, safe_message="canceled")
            raise
        except (EnvelopeError, ServeDispatchError, msgspec.ValidationError) as exc:
            status, message = pb.JOB_STATUS_INVALID, f"{type(exc).__name__}: {exc}"
            logger.warning("job %s attempt=%d rejected: %s", *key, exc)
        except NeverFits as exc:
            status, message = pb.JOB_STATUS_FATAL, f"{type(exc).__name__}: {exc}"
            logger.error("job %s attempt=%d cannot ever fit: %s", *key, exc)
        except (CheckpointUnresolved, ResidencyError) as exc:
            status, message = pb.JOB_STATUS_RETRYABLE, f"{type(exc).__name__}: {exc}"
            logger.error("job %s attempt=%d unplaceable: %s", *key, exc)
        except Exception as exc:  # noqa: BLE001 — the terminal must still ship
            status, message = pb.JOB_STATUS_FATAL, f"{type(exc).__name__}: {exc}"
            logger.exception("job %s attempt=%d failed", *key)
        finally:
            self._canceled.discard(key)
        now = time.monotonic()
        await self._send_result(
            *key,
            status,
            inline=inline,
            safe_message=message,
            metrics=pb.JobMetrics(
                runtime_ms=int((now - started) * 1000),
                queue_ms=int((started - accepted_at) * 1000),
            ),
            adjustments=adjustments,
        )

    # ---- drain / shutdown --------------------------------------------------

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
        # The hub's deadline is a COMMAND budget, not a stall detector: 0 waits
        # for the work itself.
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

    # ---- run loop ----------------------------------------------------------

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_MS / 1000.0)
            # The mint's own counted observable, once per beat. Its counter
            # already rides the hub's activity stream from the mint's own
            # thread; this is the LOCAL reading, and it exists so that
            # "adopting=False" and "mint=not_armed" are legible on a pod
            # somebody is looking at rather than only inferable from an
            # absence of events.
            logger.debug("mint: %s", self.mint_facts())
            try:
                await self._send(pb.WorkerMessage(state_delta=self._state_delta()))
            except Exception:  # a missed beat must not kill the loop
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
        # ROUTE ACTIVITY AND BOOT-PHASE REPORTS ONTO THE STREAM. This is not
        # telemetry decoration: the control parent's hang watchdog HOLDS its
        # verdict while an activity is open, so a child that publishes none is
        # a child whose long-but-healthy work reads as a hang. pgw#771's
        # `compute_hang_verdict_held` never fired without it — the parent had
        # nothing to hold on.
        from . import activity as activity_mod
        from . import boot_phases as boot_mod

        activity_mod.bind_sink(self._send, loop)
        boot_mod.bind_sink(self._send, loop)

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
        # Falling out of the reconnect loop any other way ended the process
        # clean with NOTHING on the wire; the hub saw only a young-worker
        # death. An unexplained exit is a fatal.
        raise UnexpectedWorkerExit(
            "transport loop ended without a Drain command or shutdown signal "
            f"(connected={self.transport.connected} draining={self.draining})"
        )

    def run(self) -> int:
        """Always returns an exit code. A fatal end to the run loop is reported
        to the HUB here — pod stdout is unreadable on RunPod, so this is the
        only channel that survives the process."""
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
