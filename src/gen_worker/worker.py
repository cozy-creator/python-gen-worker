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

#: pgw#1576: the content type that marks a ``JobProgress`` chunk as a typed
#: ctx-event envelope rather than streamed output. Verbatim from the hub's own
#: ``runtimestore.RequestEventContentType``; a chunk carrying anything else is
#: fanned out to SSE subscribers as ``output.delta``.
EVENT_CONTENT_TYPE = "application/x-request-event+json"

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
        # pgw#1490: THE DISPATCH ALREADY CARRIES ITS OWN FETCH POINTER, and it
        # was being ignored. `ModelBinding.manifest_digest` has never had a
        # sender, but `RunJob.snapshots` ships on every dispatch and the hub
        # keys an uncomposed artifact in it BY REF (`attachJobSourceSnapshots`;
        # pgw#1475 already relies on exactly that for reserved repos). So when
        # the wire's digest field is empty — which is always, today — the ref's
        # own snapshot supplies it. Nothing is fetched here: this is the
        # identity of a tree, not a request for one.
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
    """``_picks_of``'s table over a bare ``ModelBinding`` list.

    ``DesiredInstance.models`` and ``RunJob.models`` are the SAME message, so
    the boot warm pass reads its picks through the same builder the dispatch
    does. The difference is only where the digest fallback comes from: a
    dispatch has ``RunJob.snapshots`` beside the bindings, boot has the
    config's, so the caller supplies it below.
    """
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
    """Per-function checkpoint picks for the BOOT WARM PASS (pgw#1584).

    Boot has no dispatch, and ``default_pick`` reads one — so without this the
    warm pass would decode every envelope into *"model slot has no envelope
    pick and no deployment default"*. The picks exist on the wire the pod
    already receives: ``DesiredResidency.Hot`` is ``repeated DesiredInstance
    {function_name, models}``, and ``models`` is the very ``ModelBinding`` the
    dispatch carries — slot, ref, the recognized ``model`` name and its
    ``inference_defaults`` row. Nothing is invented; the hub's own boot seed is
    read.

    **The ONE inference, and its fence.** The hub seeds ``Hot`` for dynamic
    slot defaults and compile-cache prewarm; a release whose bindings are all
    static may arrive with ``Hot`` empty. For that case, and ONLY when the
    entrypoint declares exactly one model slot and the config materialized
    exactly one ref, the two are bound: with one slot and one ref there is
    exactly one possible answer, so this is arithmetic rather than a guess.
    Every other shape yields NO entry, the warm pass skips that function with
    a reason, and the worker keeps `decode_envelope`'s rule intact — the worker
    never guesses which bytes to serve.
    """
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
            # UNKNOWN, and left so. `ctx.defaults()`'s unclassified arm (no
            # name, no row) is pgw#1377's warn-and-serve platform fallback; a
            # fabricated classification would warm under a recipe the hub never
            # resolved, and `resolve()`'s pgw#1415 fence only fires on the
            # broken pair (a row with no name), which this is not.
            model="", inference_defaults="",
        )
        picks[name] = _DispatchPicks(
            by_ref={only_ref: pick}, by_slot={slots[0]: only_ref}
        )
    return picks


class HubBindingResolver:
    """The hub's half of the deploy state, per dispatch.

    ``resolve`` materializes a pick the hub already validated. ``tree_for`` is
    a PURE LOOKUP and stays one: pgw#1490 makes boot materialize every ref the
    runtime config names before this worker advertises a function, so by the
    time a dispatch arrives the tree is there. Dispatch-time fetching is not
    missing — it is deliberately absent, because the local chain (``download``,
    ``up``, then ``run``) has the same shape and for the same reason.

    Which tree, though, is answered by THE STORE THAT PUT IT THERE, not by a
    wire field. Two things made the digest route unusable on v2:

    * ``ModelBinding.manifest_digest`` has NEVER had a sender (its own proto
      comment says so — th#1941's hub leg is a parked draft), so every v2
      dispatch reached a refusal that named a fetch pointer nobody sends; and
    * the two producers of a tree name disagreed about the algorithm prefix.
      ``Snapshot.digest`` is algorithm-tagged (``sha256:<hex>``) and the proto
      calls it *"the worker's snapshot directory name — one key, one meaning"*;
      ``cozy_snapshot`` writes exactly that, and this class used to STRIP the
      prefix before looking, so the lookup could not succeed even for bytes
      the pod held.

    So the ref is resolved through the ``ModelStore``'s own residency first —
    the same object, the same map, the same tree the boot pull produced — and
    the digest path survives only as a fallback for a dispatch that does carry
    one. A miss names the ref, what boot materialized, and every path tried.
    """

    def __init__(self, snapshots_root: Optional[Path] = None) -> None:
        self.snapshots_root = (
            Path(snapshots_root)
            if snapshots_root is not None
            else tensorhub_cas_dir() / SNAPSHOTS_DIR
        )
        #: Set by `Worker` once the store exists. `None` in the bare-resolver
        #: unit paths, where the digest fallback below is the only route.
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
        """Both spellings of one digest's tree, authoritative one first."""
        digest = str(digest or "").strip()
        if not digest:
            return []
        bare = digest.split(":", 1)[-1]
        # BARE FIRST, because that is what production writes: all 13 trees in
        # this box's CAS are bare hex, and the hub's volume manifest is 38/38
        # bare. The tagged spelling is tried second because the hub also stores
        # 1999/1999 tagged digests in its artifact metadata, so a route that
        # answers from there would hand the worker the other spelling.
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


class _Admission:
    """The name `tree_for`'s refusals carry when ADMISSION is the caller.

    `tree_for` takes the asking class so a refusal says who could not be
    served; admission is not a model class, and saying so beats borrowing an
    unrelated one.
    """


class SnapshotSizer:
    """Weight bytes from the materialized tree's manifest — the WHOLE tree.

    ``tree_bytes`` sizes a PROJECTED tree from its manifest rather than
    walking stubs, so this is known before any byte moves, which is what
    admission requires.

    IT IS AN UPPER BOUND, NOT THE RESIDENT COST, and that is deliberate as of
    pgw#1599. Three narrower answers were tried and MEASURED against the
    vendored contract library, and all three can UNDER-count — the one
    direction that OOMs a rented card rather than refusing:

    * **the lane's declared VRAM floor** (pgw#1590) — deleted with every other
      floor string (Paul: *"there is no required VRAM"*).
    * **summing the tensors the lane's CONTRACT claims** — a contract is a
      layout TEMPLATE describing a matching SET, not an inventory. h3's bf16
      contract declares 10 patterns; anything in the DiT they do not name goes
      uncounted.
    * **charging only the FILES the contract claims a tensor in** — measured
      across four shipped contracts and the coverage is not consistent enough
      to decide residency from: ``sdxl.diffusers-bf16`` covers unet + vae +
      text encoders (the whole pipeline), ``sd15.diffusers-bf16`` covers the
      UNET ONLY, and ``minimax.h3-dit-diffusers`` the DiT only. Narrowing sd15
      to its contract would drop the VAE and both text encoders that its model
      class actually holds resident.

    So the tree stands until the number has a producer that cannot be wrong in
    the OOM direction. What that producer is, is now clear and is not an
    admission change at all: a lane whose contract states the precision the
    weights ACTUALLY land at. h3 charges 133 GB of bf16 here and holds ~66.5
    because it `quantize_()`s inside `setup()` — a runtime quantization that
    no manifest, header or contract can see, and that the two-arena design
    (pgw#1605 exclusion 1) bans outright. Its close is h3 serving a real
    ``minimax.h3-dit-fp8-rowwise@1`` lane, not a cleverer sizer.
    """

    def __init__(self, resolver: HubBindingResolver) -> None:
        self._resolver = resolver
        self._cache: Dict[str, int] = {}

    def _bytes(self, checkpoint_ref: str) -> int:
        cached = self._cache.get(checkpoint_ref)
        if cached is not None:
            return cached
        # ONE resolution, not a parallel one. Sizing a tree by re-deriving its
        # path was the second place the prefix asymmetry lived, and a sizer
        # that disagrees with the loader about which tree it is admits a model
        # that is not the one that gets loaded.
        tree = self._resolver.tree_for(_Admission, checkpoint_ref)
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
        # a stated reason — never a boot failure. The try is the invariant, not
        # defensiveness: a pod that cannot adopt can still SERVE, and a worker
        # that refuses to boot over its compiled-graph story is strictly worse
        # than one that serves eager and says why.
        # pgw#1425: ARM THE RECEIPT GATE BEFORE THE ADOPT, not only at HelloAck.
        # Boot-adopt runs before the stream is up, so a gate armed only at
        # HelloAck leaves the pod's FIRST artifacts — the ones it pulls by key
        # from the hub store — outside it. v1 had the same window and it was
        # silently OPEN; with the gate failing closed it would instead refuse
        # every boot-adopt and self-mint the lot. Neither is right, and the
        # fix is neither: `HelloAck.file_base_url` IS the tensorhub base URL
        # (`worker_scheduler.proto`: "tensorhub base URL for capability-token
        # HTTP calls"), which this pod already knows as TENSORHUB_URL. So arm
        # from settings now and re-arm from the ack later — `configure` is
        # idempotent and drops the JWKS cache, so the authoritative value
        # simply replaces this one.
        hub_base = str(getattr(settings, "tensorhub_url", "") or "").strip()
        if hub_base:
            receipts.configure(hub_base, worker_credential.current)
        else:
            # Not a refusal to boot: the pod serves, and every hub-delivered
            # artifact refuses until the ack arms the gate. Said out loud
            # because "nothing adopted" would otherwise read as "nothing to
            # adopt".
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
                # pgw#1475: the pod's HF credential — `ctx.hf_token` for the
                # producer contract, and what an upstream-mirror reserved repo
                # downloads with.
                hf_token=str(getattr(settings, "hf_token", "") or ""),
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

        # pgw#1425: THE SDK IS UP. Author modules are imported, every
        # `@entrypoint` is harvested and the serve stack is built — the phase
        # the whole boot table is anchored on, and the one the v2 rewrite never
        # marked, which is why every pod's phase series was empty.
        boot_mod.mark_once(
            boot_mod.PHASE_SDK_READY, function=",".join(self.functions)
        )
        # pgw#1483 / th#2204: THE MATERIALIZATION SEAM. `ModelStore` was never
        # constructed anywhere in `src/` — and `WorkerMessage.model_event` is
        # built INSIDE it, so this pod had no live object capable of fetching
        # a weight or stating that it held one. `_send` is the store's emit:
        # the same wire every other WorkerMessage rides. `rescan_disk()` is
        # boot-time truth — on a warm pod with the endpoint volume attached,
        # this is what turns already-staged bytes into a residency the pod can
        # answer with instead of re-downloading.
        self.store = ModelStore(
            self._send,
            cache_dir=self.resolver.snapshots_root.parent,
            vram_budget_bytes=int(budget) or None,
        )
        # ONE resolution of "where is this ref's tree". The store materializes
        # it; the resolver hands it to `ctx.load`. Two answers to that question
        # is how a pod ends up holding a checkpoint it cannot find.
        self.resolver.bind_store(self.store)
        # pgw#1543: and the SERVING path too, which holds only a binding.
        bind_active_store(self.store)
        # NOTE: the disk rescan is NOT here. `Worker` is constructed with no
        # running event loop (`entrypoint.py`), and `ModelStore`'s event path
        # closes its coroutine when it can find neither a running loop nor a
        # bound one — so a rescan at construction populates the disk tier
        # perfectly and delivers ZERO ModelEvents. It runs in `arun`, inside
        # the loop, where the events it produces can actually reach the hub.
        # pgw#1490: BOOT IS `up`. The refs this release serves are materialized
        # before this worker advertises a single function, exactly as `run`
        # requires `up` locally. Until then the worker is connected and NOT
        # routable — which is what makes "never fetch inside a user request"
        # true by construction, with no hub-side parking to enforce it.
        self.materialization = CheckpointMaterialization(
            self.store,
            announce=self._announce_readiness,
            # pgw#1584: one synthetic forward per entrypoint, run between the
            # last weight landing and this worker calling itself ready.
            warm=self._run_boot_warmup,
        )
        #: pgw#1584: per-function checkpoint picks for the warm pass, read off
        #: the HelloAck's own `DesiredResidency` (see `boot_picks`). Empty
        #: until the ack arrives, which is also before the warm pass can run.
        self._boot_picks: Dict[str, _DispatchPicks] = {}

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

        def _arm(adoption: Any) -> Any:
            # The trigger (pgw#1371): holes are registered, so the mint has a
            # work-list. It runs on its own daemon thread, CPU-reserved and
            # niced against this process — the serving loop is never involved.
            mint = production_mint(
                store=adoption.store, artifacts_dir=artifacts,
                cas_dir=tensorhub_cas_dir(), sm=sm,
            )
            self._mint_box.append(mint)
            # pgw#1480: the STATUS is returned, not dropped. `arm` answers
            # `unavailable` when this pod has no compiler — a mint that never
            # starts leaves the pod eager FOREVER, which is the terminal state
            # `boot_ended_uncompiled` exists to name. Returning nothing made
            # that indistinguishable from a mint that is running.
            return mint.arm(adoption)

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

    @property
    def phase(self) -> "pb.WorkerPhase":
        """This worker's startup phase. DERIVED, never assigned.

        It used to be the constant `WORKER_PHASE_READY`, set in `__init__`
        before a single weight existed on the pod — which is the worker half
        of th#2204: the hub was told READY by a process that could not serve
        anything, so the only thing left that could gate a dispatch was the
        hub's own residency bookkeeping.
        """
        return self.materialization.phase()

    def _state_delta(self) -> pb.StateDelta:
        # pgw#1490: READINESS IS THE ROUTING SIGNAL. A worker whose configured
        # checkpoints are not on disk advertises them as LOADING, not
        # AVAILABLE — the hub's `applyStateDeltaLocked` already renders that
        # as `availability=starting` and routes elsewhere, like any web
        # service in front of a pool. No residency accounting, no parking, no
        # transfer-owner election: the pod says when it can serve.
        ready = self.materialization.ready
        delta = pb.StateDelta(
            phase=self.phase,
            available_functions=self.functions if ready else [],
            loading_functions=[] if ready else self.functions,
        )
        # 0 is the wire's "unmeasured"; only a real reading is reported.
        free = hostfacts.free_vram_bytes()
        if free is not None:
            delta.free_vram_bytes = int(free)
        return delta

    async def _announce_readiness(self) -> None:
        """Put this worker's readiness on the wire the instant it changes.

        Called by `CheckpointMaterialization` on every transition, so the hub
        learns "routable" at materialization time rather than up to one
        heartbeat later — and learns "failed" as a TYPED per-function fact
        rather than as an absence that looks identical to a slow boot.
        """
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
        # LOUD AND TYPED. `model_unavailable` is already in FnUnavailable's
        # closed vocabulary and the hub's th#1100 policy re-probes it with
        # backoff; the alternative — a pod that keeps retrying a hopeless pull
        # forever — is th#2204's livelock wearing a different hat.
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

    # ---- the boot warm pass (pgw#1584) -------------------------------------

    async def _run_boot_warmup(self) -> None:
        """Drive :meth:`ServeLoop.boot_warmup` off the event loop.

        `invoke` is synchronous and does real work — an admission, a weight
        load, a forward — so it goes to a thread for the same reason every
        dispatch does. `asyncio.to_thread` COPIES the context, which is what
        lets `_bind_boot_picks` set `_DISPATCH` per function from in here.

        Never raises: `boot_warmup` confesses per entrypoint with a
        `serve_degrade` event and `CheckpointMaterialization._warm` is the
        second belt. A failed OPTIMIZATION must never cost this pod its boot.
        """
        await asyncio.to_thread(self.serve.boot_warmup, prepare=self._bind_boot_picks)

    def _bind_boot_picks(self, function: str) -> str:
        """Bind the deploy's picks for one warm invocation; '' = proceed.

        The warm pass runs with NO dispatch, and `HubBindingResolver` reads its
        bindings off `_DISPATCH`. This sets that contextvar to the function's
        boot picks so `default_pick`/`resolve`/`tree_for` answer exactly as
        they would for a real request against the same checkpoint.

        A function the ack seeded no bindings for returns a SKIP REASON rather
        than an empty table: an empty table decodes into "model slot has no
        envelope pick and no deployment default", which is a correct refusal
        wearing the costume of a warm-pass defect.
        """
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
        # NO `resources`: the split parent measures the silicon in a process
        # that imported no tenant code and stamps every relayed Hello, so
        # anything measured here would be replaced before the hub saw it.
        # NO `lifecycle_snapshot`: this build carries no intent registry, and
        # an invented one is worse than an absent one.
        # `models` IS carried now (pgw#1483): the store's residency snapshot is
        # this pod's BOOT BASELINE, replayed hub-side through
        # ApplyModelResidency. On a warm pod with the endpoint volume attached
        # this is the cheapest possible answer to th#2204 — the hub learns the
        # 134 GB is already here before it ever declares a goal, so the park
        # never happens rather than being unparked later.
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

    # ---- transport handlers ------------------------------------------------

    async def on_hello_ack(self, ack: pb.HelloAck) -> None:
        self.file_base_url = str(ack.file_base_url or "")
        logger.info(
            "hello acked: functions=%s file_base_url=%s",
            self.functions, self.file_base_url or "(none)",
        )
        # pgw#1425: THE HELLOACK WIRING MOMENT. Three modules take their hub
        # half here and nowhere else, and the v2 rewrite called none of them.
        # `mark_once`, not `mark`: this coroutine runs again on every
        # RECONNECT, and "process start -> hello" measured on the third
        # reconnect of a six-hour-old worker is not a boot number.
        boot_mod.mark_once(boot_mod.PHASE_HELLO)
        # pgw#1490 / th#2204: TAKE THE CHECKPOINT CONFIG AND PULL. This must
        # sit ABOVE the `file_base_url` early return below — a session with no
        # file API is a hub-side fact that has nothing to do with weights, and
        # putting the materialization under that return would strand exactly
        # the pods least able to explain themselves. Non-blocking: the pull
        # runs on its own task while the transport keeps reading, and the
        # worker stays connected-and-unroutable until it finishes.
        config = CheckpointConfig.from_wire(ack.desired_residency)
        # pgw#1584: BEFORE `configure`, because `configure` is what starts the
        # materialization task that ends in the warm pass. The picks come off
        # the same ack — `DesiredResidency.Hot` carries the hub's own
        # per-function `ModelBinding` rows.
        self._boot_picks = boot_picks(ack.desired_residency, self.loaded, config)
        self.materialization.configure(config)
        if self.functions and self.materialization.ready:
            # THE cold-boot number, and it now means what it says: the pod
            # holds its weights AND the hub has a Hello advertising these
            # functions, so from this instant it may dispatch here. When the
            # config named refs this pod had to fetch, the mark is stamped by
            # the materialization's own readiness transition instead.
            boot_mod.mark_once(
                boot_mod.PHASE_FIRST_REQUEST_SERVABLE,
                function=",".join(self.functions),
            )
        if not self.file_base_url:
            # The gate stays UNSET, which now refuses (pgw#1425). Say so: a
            # session with no file API is a hub-side fact this pod cannot fix,
            # and the operator must be able to see why nothing arms.
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
            # pgw#1425: the operator's eager-only order (§4.32 item 4).
            # `apply_command` owns idempotence — the hub REPLAYS the order to a
            # reconnecting worker — so this handler must not try to dedupe.
            posture = msg.serve_posture
            serve_posture.apply_command(
                bool(posture.eager_only),
                actor=str(getattr(posture, "actor", "") or ""),
                reason=str(getattr(posture, "reason", "") or ""),
            )
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
        traceback_tail: str = "",
    ) -> None:
        result = pb.JobResult(
            request_id=request_id,
            attempt=attempt,
            status=status,
            safe_message=safe_message[:512],
        )
        # pgw#1474: already bounded by `failure_traceback.traceback_tail`; the
        # slice is this layer declining to put an unbounded string on a wire it
        # owns, not a second policy. `safe_message` above is the same posture.
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

    def _request_context_facts(self, run: pb.RunJob) -> Dict[str, Any]:
        """The per-request half of the handler's ``RequestContext``.

        pgw#1438: the v2 rewrite carried the PAYLOAD across and nothing else,
        so `RunJob.capability_token` reached no context and every file
        operation raised ``worker_capability_token is required for file
        operations`` — that is `ctx.save_image`, `save_bytes`, `save_file`,
        `save_audio`, `save_video`, i.e. the OUTPUT of every media endpoint.
        The token is minted per request and expires, so it can never ride
        `ServeLoop`'s construction-time `context_kwargs`.

        `file_base_url` comes off the HelloAck rather than the RunJob: the hub
        names one file API per session, and a per-dispatch override would let
        one job's blobs land somewhere the session never agreed to.
        """
        hints: Dict[str, Any] = {}
        if run.media_bytes == pb.MEDIA_BYTES_INLINE:
            # The client asked for bytes back in the result rather than a link.
            hints["output_format"] = "inline"
        emitter, chunk_sink = self._progress_channel(run)
        return {
            "owner": str(run.org or "") or None,
            "invoker_id": str(run.invoker_id or "") or None,
            "file_api_base_url": self.file_base_url or None,
            "worker_capability_token": str(run.capability_token or "") or None,
            "execution_hints": hints or None,
            # pgw#1576: THIS REQUEST'S JobProgress LANE, both halves. Per
            # request because `seq` is per (request_id, attempt) — it can never
            # ride `ServeLoop`'s construction-time context_kwargs, which is
            # exactly the reason the v2 rewrite ended up wiring neither.
            "emitter": emitter,
            "chunk_sink": chunk_sink,
        }

    def _progress_channel(
        self, run: pb.RunJob
    ) -> Tuple[
        Callable[[Dict[str, Any]], None], Callable[[bytes, str], None]
    ]:
        """This request's ``JobProgress`` lane: ``(ctx event emitter, chunk sink)``.

        ONE seam and ONE ``seq`` counter for both, because they are one wire
        message. ``JobProgress.seq`` is "strictly increasing per (request_id,
        attempt)" and it is stamped ON THE LOOP, so send order and seq order
        cannot disagree no matter which thread produced the frame.

        * the CTX EVENT lane — ``ctx.progress``/``log``/``warning``/
          ``checkpoint`` as a JSON envelope under
          ``application/x-request-event+json``, which the hub parses into
          request-progress positions and `request_events` rows.
          **pgw#1576: the v2 rewrite wired NO emitter at all**, so
          ``_emit_event`` hit its `no emitter configured` branch on every pod
          and the hub's liveness sweep read positions nobody was sending.
        * the OUTPUT DELTA lane — ``ctx.emit(chunk)`` frames, content-typed by
          the chunk itself, fanned out live to SSE subscribers.

        Both are best-effort by contract: the send queue sheds progress under
        pressure (and confesses a ``serve_degrade`` row where it does), and a
        producer never fails a request over a chunk that did not fit.
        """
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
                return  # loop closed: the worker is shutting down
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                # Called ON the loop (a sync entrypoint invoked in-process, a
                # test). Waiting here would deadlock the very loop that has to
                # do the send, so the frame is scheduled and not awaited.
                return
            # WAIT, deliberately: it orders this thread's frames and applies
            # whatever backpressure the send path has. It cannot hang on queue
            # capacity — the send queue's policy for a full queue is to DROP
            # progress, never to block its producer (transport.py).
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
        # Empty on the success path and on every terminal that carries no
        # exception; the hub reads an empty traceback as `no_traceback_reported`
        # rather than as an absent field.
        tb = ""
        adjustments: Tuple[Dict[str, str], ...] = ()
        stages: Optional[Any] = None
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
            # pgw#1425: the capability token this job writes with expires
            # MID-FLIGHT on any long request, and nothing else refreshes it —
            # the token carries no refresh and the hub does not push. The
            # renewal loop lives on the event loop while the author's body runs
            # on a worker thread; `on_context` is how it reaches the context
            # object that holds the token.
            renew: Optional[asyncio.Task[None]] = None
            ctx_box: List[Any] = []

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
                        # pgw#1418: the ordered, credential-free input manifest.
                        # Without it the serve path materialized nothing and
                        # every Image/Video/AudioAsset reached the author with
                        # `local_path` unset.
                        input_assets=manifest_from_run_job(run.input_assets),
                        # pgw#1475: the dispatch's ref-keyed snapshot map, the
                        # pin every RESERVED repo field materializes against.
                        # The hub ships them unconditionally
                        # (`ExtractReservedRepoBindingsFromPayload` ->
                        # `attachJobSourceSnapshots`) and keys an
                        # uncomposed artifact — which a payload `source` is —
                        # by ref. Without this the map is empty and every
                        # tensorhub source refuses `missing_snapshot`.
                        snapshots=resolved_repos(run.snapshots, run.models),
                        context=self._request_context_facts(run),
                        on_context=_bind_context,
                    )
                )
            finally:
                postmortem.clear_inflight(inflight)
                if renew is not None:
                    renew.cancel()
                # The attempt's input directory is worker-owned scratch and
                # nothing downstream reads it once the body has returned.
                cleanup_input_assets(str(run.request_id), int(run.attempt))
            inline = msgspec.msgpack.encode(outcome.result)
            adjustments = outcome.adjustments
            stages = outcome.stages
            status = pb.JOB_STATUS_OK
        except asyncio.CancelledError:
            await self._send_result(*key, pb.JOB_STATUS_CANCELED, safe_message="canceled")
            raise
        # pgw#1474 / th#2201: EVERY arm below ships the traceback tail beside
        # the repr. `f"{type(exc).__name__}: {exc}"` was the whole diagnostic
        # surface of a body failure, and it cost a $0.50, 455-GPU-second
        # flagship run to learn nothing but the five characters `'keys'` — the
        # exception object was in hand at this exact site the entire time.
        except (
            EnvelopeError,
            ServeDispatchError,
            msgspec.ValidationError,
            # pgw#1475: a reserved repo field naming an unparseable or empty
            # ref is BAD INPUT, and retrying it costs a pod-minute to reach
            # the same refusal. `api.errors.ValidationError` says "do not
            # retry" in its own docstring; the wire status has to agree.
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
        # pgw#1425: the per-stage breakdown, closed against `runtime_ms` so
        # every emitted stage plus `resid.unattributed` sums to it. Before this
        # the `JobResult.metrics` of every v2 request carried two numbers and
        # the whole stage series was empty.
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
        # pgw#1555: the boot recorder asks THIS object whether the worker can
        # serve, because this is the object the hub's routing already reads
        # (`_state_delta`). Without it `in_boot()` latched shut ~1 ms after
        # `hello` on every pod whose first ack named no refs, and the
        # `weights_fetch` span — the one phase the boot table exists for — was
        # never opened on any of them.
        boot_mod.bind_servable_probe(lambda: self.materialization.ready)

        # BOOT-TIME TRUTH, and it must run HERE. On a warm pod with the
        # endpoint volume attached this is what turns already-staged bytes into
        # a residency this pod can answer with instead of re-downloading — and
        # every ON_DISK it produces is a ModelEvent, which only reaches the hub
        # from inside the loop (see the construction-site note above).
        self.store.bind_loop()
        try:
            self.store.rescan_disk()
        except Exception as exc:  # noqa: BLE001 — a cold CAS is not a boot failure
            logger.warning("disk rescan at boot failed: %s: %s", type(exc).__name__, exc)

        # pgw#1425: THIS is the serving process, and it says so where the wire
        # that carries the claim is bound — so the fact and its transport
        # arrive together. Compile children's pid rows are read beside this
        # session; without the declaration they are attributed to nothing.
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
