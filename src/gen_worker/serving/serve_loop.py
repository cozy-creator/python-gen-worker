"""The entrypoint dispatch loop (pgw#1372): envelope in, leases around, result out.

The wire-facing serving path over the pgw#1382 primitives:

1. Route by function name (the route IS the entrypoint name).
2. Decode the SIGNATURE-DERIVED envelope (:mod:`.envelope`): per-slot
   checkpoint picks, hub-resolved adapter rows into typed slot values,
   ``input`` against the entrypoint's own schema — every refusal typed,
   before any author code runs.
3. For every model slot in STABLE SLOT-NAME ORDER (the multi-model deadlock
   rule), take a :class:`~gen_worker.serving.residency.ResidencyManager`
   lease on ``(model class x checkpoint x lane)``: admission before
   allocation, LRU two-tier eviction between requests, loads serialized,
   single-flight per instance — Paul's residency ruling, wired around every
   invocation.
4. Invoke ctx-first: ``spec.fn(ctx, payload, *slot values in signature
   order)``; release the leases; return the result with the request's
   accumulated ``ctx.warn`` rows.

:meth:`ServeLoop.boot_warmup` (pgw#1584) runs that same walk once per
entrypoint AT BOOT, on a payload synthesized from the entrypoint's own schema
at its neutral defaults and a context carrying ``boot_warmup=True`` — so the
first-call tax is paid by the pod's boot rather than by a paying request. It
is not a second serving path; it is this one, called earlier.

Instances are keyed by (model class, checkpoint ref, lane): one resident
author object per key, built inside the lease's admission window via
``Model()`` + ``model.load(LoadContext)`` and dropped via the author's
best-effort ``unload``. The deploy state comes through a
:class:`BindingResolver` — the hub wiring implements it; tests and
cozy-local implement it over local trees.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import tempfile
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import msgspec

from .. import activity, boot_phases
from ..warm_payload import neutral_payload
from ..input_assets import (
    InputManifestEntry,
    cleanup_input_assets,
    materialize_input_assets,
)
from ..stage_timing import StageTimer
from .context import DeployBinding, LoadContext, LoaderEngine, RequestContext
from .envelope import DecodedRequest, decode_envelope
from .host import ServeDispatchError
from .loader import EndpointLoadError, LoadedEndpoint
from .model import Model, lane_handle, model_type
from .placement import warn_if_degraded
from .reserved_repos import (
    materialize_reserved_inputs,
    reserved_context_kwargs,
)
from .residency import InstanceSizer, ResidencyError, ResidencyManager
from .worker_context import worker_load_context

logger = logging.getLogger(__name__)


class BindingResolver(Protocol):
    """Deploy state per (model class, checkpoint pick) — the hub's half.

    ``resolve`` returns the worker-resolved binding for a pick the hub
    already validated against the slot's ``allowed_checkpoints`` (the
    worker re-refuses nothing; it materializes). ``default_pick`` is the
    deployment's per-slot default ref ('' = none bound).
    """

    def resolve(self, model_cls: type, checkpoint_ref: str) -> DeployBinding: ...

    def default_pick(self, model_cls: type, slot_name: str) -> str: ...


@dataclass(frozen=True, slots=True)
class InvokeOutcome:
    """One served request: the entrypoint's result + the request facts the
    envelope reply carries (``ctx.warn`` rows, adjustments, stage timings)."""

    result: Any
    warnings: Tuple[str, ...]
    adjustments: Tuple[Dict[str, str], ...]
    #: pgw#1425: this request's :class:`~gen_worker.stage_timing.StageTimer`.
    #: ``RequestContext`` has always FILLED one; until this field existed
    #: nothing read it out, so every served request reported a bare
    #: ``runtime_ms`` and the stage breakdown died with the context object.
    #: Handed out RAW rather than rendered because
    #: :func:`~gen_worker.stage_timing.stage_ms_for_metrics` closes the
    #: breakdown against ``runtime_ms``, and the dispatch — not the loop — is
    #: what measures that.
    stages: Optional[StageTimer] = None


@dataclass(frozen=True, slots=True)
class WarmPass:
    """What the boot warm pass did about ONE entrypoint.

    Three outcomes and they are deliberately distinguishable: ``warmed`` (a
    real forward ran and its cost is off the first paying request),
    ``skipped`` (nothing could honestly be warmed — the reason says what) and
    ``failed`` (the pass ran and raised; the pod SERVES and the first real
    request pays the cold cost). "Nobody warmed" and "warming was free" are
    different answers, so a skip never renders as a success.
    """

    function: str
    outcome: str
    reason: str = ""
    duration_ms: int = 0

    @property
    def warmed(self) -> bool:
        return self.outcome == WARM_OK


#: A real synthetic forward ran under ``boot_warmup=True``.
WARM_OK = "warmed"
#: Nothing was warmed and nothing went wrong — the reason names which.
WARM_SKIPPED = "skipped"
#: The warm pass raised. LOUD (``serve_degrade``), never fatal.
WARM_FAILED = "failed"


class _InstanceBackend:
    """One resident (model class x checkpoint x lane) as residency sees it.

    ``load`` runs the author's ``Model()`` + ``load(ctx)`` (inside the
    manager's serialized load gate); ``drop`` runs the author's ``unload``
    — best-effort tidiness, never correctness.

    The host tiers ride pgw#1497's
    :class:`~gen_worker.models.stream_residency.StreamedResidency`: a demote
    is a re-plan at a budget of zero (every leaf to pinned host RAM), a
    promote a re-plan at the instance's full weight size. The SAME object and
    the SAME arithmetic serve the partial case, so the warm tier is not a
    second implementation of the offload rung — it is that rung at its two
    end stops."""

    def __init__(
        self,
        model_cls: type,
        load_context: LoadContext[Any],
        lane: Any = None,
        on_loaded: Optional[Callable[[type, Any], None]] = None,
    ) -> None:
        self.model_cls = model_cls
        self.load_context = load_context
        self.model: Optional[Model[Any]] = None
        self.lane = lane
        #: Called once the author's ``load(ctx)`` has RETURNED — the first
        #: instant an adopt session's hole list is complete, because holes are
        #: registered by the author's own ``ctx.compile`` calls inside that
        #: load. pgw#1371's background mint triggers here; anything earlier
        #: reads an empty work-list and mints nothing.
        self._on_loaded = on_loaded
        #: pgw#1497's partial-residency handle for this instance, built on the
        #: first tier move (or handed over by a rung engaged during load).
        self._stream_residency: Optional[Any] = None
        #: Unmet machine floors, measured once at residency-admit. Non-empty
        #: means this instance is serving DEGRADED, and every request it
        #: serves says so (`invoke` warns the caller with these).
        self.degraded: Tuple[str, ...] = ()

    def load(self) -> None:
        # Residency-admit is where the lane is picked and the load begins, so
        # it is where the machine is compared against the lane's declared
        # floors — BEFORE any weight moves, so the warning precedes the
        # slowness it predicts rather than explaining it afterwards. It never
        # refuses: any model, any machine (Paul, 2026-08-18).
        self.degraded = warn_if_degraded(self.model_cls, self.lane)
        model: Model[Any] = self.model_cls()  # cheap __init__ — no GPU, by contract
        model.load(self.load_context)
        self.model = model
        # pgw#1425: the FIRST user-visible "could have answered a request".
        # `mark_once` because this runs on every residency admission and only
        # the first one is a boot number.
        boot_phases.mark_once(
            boot_phases.PHASE_EAGER_READY, function=self.model_cls.__name__
        )
        if self._on_loaded is not None:
            try:
                self._on_loaded(self.model_cls, self.lane)
            except Exception:  # noqa: BLE001 — a hook never fails a load
                logger.exception(
                    "post-load hook raised for %s; the instance is loaded and "
                    "serves", self.model_cls.__name__)

    def drop(self) -> None:
        residency, self._stream_residency = self._stream_residency, None
        if residency is not None:
            try:
                # Un-hook before the author's unload: a live forward hook
                # holding a cast-buffer view outlives the tree it was
                # installed on and keeps the pinned host copies alive.
                residency.release()
            except Exception:
                logger.exception(
                    "releasing the streamed residency of %s raised; the drop "
                    "proceeds", self.model_cls.__name__,
                )
        model, self.model = self.model, None
        if model is None:
            return
        try:
            model.unload(self.load_context)
        except Exception:
            logger.exception(
                "unload(%s) raised; eviction proceeds (best-effort, never "
                "correctness)", self.model_cls.__name__,
            )

    def _residency(self) -> Any:
        """This instance's :class:`StreamedResidency`, built on first use.

        Built lazily and over the LIVE object, because the tree only exists
        once the author's ``load`` has run, and a rung engaged during that
        load may already own it.
        """
        if self._stream_residency is not None:
            return self._stream_residency
        if self.model is None:
            raise ResidencyError(
                f"{self.model_cls.__name__}: a tier move was ordered on an "
                f"instance that holds no model object"
            )
        # A rung armed during the author's own load already owns this tree.
        # Building a second handle over the same modules would give two
        # planners two disagreeing views of one resident set.
        from ..models.memory import stream_residency_of
        from ..models.stream_residency import StreamedResidency

        for component in (getattr(self.model, "pipe", None), self.model):
            armed = stream_residency_of(component) if component is not None else None
            if armed is not None:
                self._stream_residency = armed
                return armed

        # No `device=`: the execution device is DERIVED from where this
        # instance's weights already are. Probing the machine instead would
        # answer "cuda" for a pipeline the CPU rung deliberately put on the
        # host, and the first promote would move it somewhere nobody asked.
        residency = StreamedResidency.over(self.model, budget_bytes=0)
        if not residency.costs:
            raise ResidencyError(
                f"{self.model_cls.__name__}: no nn.Module tree found under the "
                f"author's model object, so its weights cannot be tiered — "
                f"run this instance with host_budget_bytes=0 so eviction drops "
                f"it instead"
            )
        self._stream_residency = residency
        return residency

    def demote_to_host(self) -> None:
        residency = self._residency()
        moved = (
            residency.demote_to_host()
            if residency.plan is not None
            else self._engage_at(residency, 0)
        )
        logger.info(
            "residency: %s demoted to host (%d bytes to pinned host RAM)",
            self.model_cls.__name__, moved,
        )

    def promote_to_device(self) -> None:
        residency = self._residency()
        if residency.plan is None:
            self._engage_at(residency, residency.total_bytes)
            return
        residency.promote_to_device()

    @staticmethod
    def _engage_at(residency: Any, budget_bytes: int) -> int:
        """First tier move on an un-engaged instance: engage at ``budget``."""
        from ..models.stream_residency import MemoryBudget

        residency.budget = MemoryBudget.of(int(budget_bytes))
        plan = residency.engage()
        return int(plan.streamed_bytes)


class ServeLoop:
    """The worker's wire-facing dispatcher for one loaded endpoint."""

    def __init__(
        self,
        loaded: LoadedEndpoint,
        *,
        residency: ResidencyManager,
        resolver: BindingResolver,
        engine: Optional[LoaderEngine] = None,
        lane_contract: str = "",
        compile_sink_for: Optional[Callable[[type, Any], Any]] = None,
        on_loaded: Optional[Callable[[type, Any], None]] = None,
        output_dir: Optional[Path] = None,
        context_kwargs: Optional[Mapping[str, Any]] = None,
        hf_token: str = "",
    ) -> None:
        self.loaded = loaded
        self.residency = residency
        self._resolver = resolver
        self._engine = engine
        #: pgw#1606: an OPERATOR PIN, not the pick. Empty on every pod — only
        #: the local CLI and the daemon ever write it — and when it is empty a
        #: multi-lane model is resolved by the ladder at load, not refused.
        self._lane_contract = str(lane_contract or "")
        #: The deploy's per-class lane pick (contract handle; '' = the single
        #: declared lane / eager-permanent). ``None`` for a multi-lane model,
        #: whose pick is not knowable until the binding names what is staged —
        #: `_resolve_for` below answers it then.
        self.lanes: Dict[type, Any] = {
            cls: self._single_lane(cls) for cls in loaded.models
        }
        #: Every lane each class DECLARED — the candidate set the ladder ranks,
        #: and the set `worker.py` validates a dispatch's `lane=` against.
        self.declared_lanes: Dict[type, Tuple[Any, ...]] = {
            cls: tuple(loaded.lanes_of(cls)) for cls in loaded.models
        }
        #: Ladder answers, cached per (class, checkpoint). The kernel gates
        #: behind a resolution are `lru_cache(1)`'d micro-benchmarks, so this
        #: cache is about not re-deciding, not about not re-measuring.
        self._resolved: Dict[Tuple[type, str], Any] = {}
        self._compile_sink_for = compile_sink_for
        self._on_loaded = on_loaded
        self._output_dir = output_dir
        self._context_kwargs = dict(context_kwargs or {})
        #: The pod's HF credential — a producer-contract fact (`ctx.hf_token`)
        #: AND what an upstream-mirror reserved repo downloads with. Pod-wide,
        #: never per-request: it is the machine's credential, not the caller's.
        self._hf_token = str(hf_token or "")
        #: Live backends by residency key, so a lease hit reuses the author
        #: object instead of rebuilding it. Guarded: leases serialize per
        #: key, but two DIFFERENT keys mutate this table concurrently.
        self._backends: Dict[Tuple[type, str, str], _InstanceBackend] = {}
        self._backends_lock = threading.Lock()

    # -- placement ----------------------------------------------------------

    def _single_lane(self, model_cls: type) -> Any:
        """The lane, when there is exactly one (or the operator pinned one).

        pgw#1606: a multi-lane model answers ``None`` here rather than raising,
        because "which of these" is a question the LADDER answers once the
        binding says what is staged — and until this issue there was no ladder,
        which is why the raise made multi-lane endpoints unbootable.
        """
        try:
            return self.loaded.lane(model_cls, self._lane_contract)
        except EndpointLoadError:
            return None

    def _resolve_for(self, model_cls: type, binding: DeployBinding) -> Any:
        """The boot ladder's answer for this (class, binding), cached."""
        key = (model_cls, str(binding.checkpoint_ref))
        hit = self._resolved.get(key)
        if hit is not None:
            return hit
        from .lane_host import BindingVerdicts, HostKernelGates, host_card_facts

        verdicts = BindingVerdicts.of(binding)
        single = self.lanes.get(model_cls)
        if single is not None:
            # The single-lane deployment carries no lane map; its one staged
            # tree IS the answer for its one declared contract. Stated
            # explicitly rather than inferred, so a MULTI-lane binding that
            # shipped without its map cannot inherit "everything is staged".
            verdicts = verdicts.for_single_lane(
                lane_handle(single), binding.checkpoint_dir)
        resolved = self.loaded.resolve(
            model_cls, card=host_card_facts(), verdicts=verdicts,
            gates=HostKernelGates(), contract=self._lane_contract,
        )
        if resolved is not None:
            logger.info("lane ladder: %s", resolved.confession())
            self._resolved[key] = resolved
        return resolved

    def _lane_of(self, model_cls: type) -> Tuple[Any, str]:
        lane = self.lanes[model_cls]
        return lane, (lane_handle(lane) if lane is not None else "eager")

    def _backend_factory(
        self, model_cls: type, binding: DeployBinding, key: Tuple[type, str, str]
    ) -> Callable[[], _InstanceBackend]:
        def make() -> _InstanceBackend:
            lane, _ = self._lane_of(model_cls)
            # pgw#1606: the ladder runs HERE, at the one moment both facts are
            # in hand — the card (process-wide) and what the deploy staged
            # (per binding). A multi-lane model has no `lane` until now.
            resolved = self._resolve_for(model_cls, binding)
            if lane is None and resolved is not None:
                lane = getattr(resolved.declared, "contract", None)
            sink = (
                self._compile_sink_for(model_cls, lane)
                if self._compile_sink_for is not None
                else None
            )
            backend = _InstanceBackend(
                model_cls,
                # pgw#1549: THE ONE PRODUCTION LOAD CONTEXT. This call used to
                # be a hand-assembled `LoadContext(...)` that named no
                # `device=` and no `io=`, so pgw#1452's placement decision —
                # landed on `EndpointHost` and asserted there — never reached
                # a pod at all: `_placed` saw `_device == ""` and returned
                # every eagerly-bridged pipeline on the CPU. Same shape as
                # pgw#1544's missing engine ask, on the same caller, silent
                # instead of loud.
                worker_load_context(
                    binding=binding,
                    model_type=model_type(model_cls),
                    lane=lane,
                    resolved=resolved,
                    engine=self._engine,
                    compile_sink=sink,
                    # pgw#1497: ADMISSION-FIRST. The `partial_stream` rung
                    # sizes its resident set from the bytes residency admitted
                    # this instance for, not from an activation estimate, so
                    # that number travels with the load moment.
                    weight_budget_bytes=self.residency.weight_budget_bytes(
                        binding.checkpoint_ref, key[2]
                    ),
                ),
                lane,
                self._on_loaded,
            )
            with self._backends_lock:
                self._backends[key] = backend
            return backend

        return make

    # -- the one public operation -------------------------------------------

    def invoke(
        self,
        function: str,
        envelope: Any,
        *,
        request_id: str,
        attempt: int = 0,
        input_assets: Sequence[InputManifestEntry] = (),
        snapshots: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
        on_context: Optional[Callable[[RequestContext[Any]], None]] = None,
    ) -> InvokeOutcome:
        """Serve one request end-to-end. See the module docstring for the
        walk; every step before ``spec.fn`` refuses typed.

        ``context`` carries the PER-REQUEST context facts the dispatch owns
        rather than the loop — the caller's org, the capability token that
        authorizes this request's writes, the file API base the HelloAck named,
        and the inline-output preference. They cannot ride ``context_kwargs``,
        which is constructed once per ServeLoop: a capability token is minted
        per request and expires.

        ``input_assets`` is the dispatch's ordered, credential-free input
        manifest (``RunJob.input_assets``) and ``attempt`` its attempt number;
        together with the token and file API base out of ``context`` they are
        everything :func:`~gen_worker.input_assets.materialize_input_assets`
        needs. pgw#1418: the v1 executor called it and the v2 rewrite did not,
        so every typed media input reached the author with ``local_path``
        unset and every asset-taking endpoint failed the request.

        ``snapshots`` is the dispatch's ref-keyed resolved-repo map
        (``wire_snapshots.resolved_repos(run.snapshots, run.models)``), which
        the RESERVED REPO fields materialize against. pgw#1475: the same
        hardcut deleted that step too, so ``ctx.source_path`` had a reader in
        every conversion producer and no writer anywhere in ``src/``, and 25 of
        27 died on their own first line at 0 GPU-seconds.

        ``on_context`` is handed this request's :class:`RequestContext` the
        instant it exists — the seam a caller needs to renew the capability
        token mid-flight, since the token lives on the context and the context
        is built in here.
        """
        spec = self.loaded.entrypoints.get(function)
        if spec is None:
            raise ServeDispatchError(
                f"{self.loaded.module_name} serves no function {function!r} "
                f"(functions: {sorted(self.loaded.entrypoints)})"
            )
        model_slots = dict(spec.model_params)
        # Adapter overlays decode against the PRIMARY model type's adapter
        # schema (pgw#1377 adapter-of-base scoping: SDXL.Lora.Defaults).
        # pgw#1392: a WEIGHTLESS entrypoint has no primary model and so no
        # adapter vocabulary to overlay against — absent, never invented.
        lora_scope = (
            getattr(model_type(spec.model_params[0][1]), "Lora", None)
            if spec.model_params
            else None
        )
        decoded: DecodedRequest = decode_envelope(
            spec,
            envelope,
            default_picks={
                name: self._resolver.default_pick(cls, name)
                for name, cls in model_slots.items()
            },
            adapter_defaults_schema=getattr(lora_scope, "Defaults", None),
        )
        picks = dict(decoded.model_picks)
        adapters = dict(decoded.adapter_values)

        # THE PAYLOAD SEAM (pgw#1418). Everything the author's body sees is
        # decoded now, so this is the one instant where a typed media input
        # can be turned into bytes on disk — and it happens BEFORE any weight
        # moves, exactly where the v1 executor put it: a request whose input
        # cannot be fetched must not first pay for a model load.
        #
        # The context is built here rather than after the leases for the same
        # reason it was: `materialize_input_assets` needs the request's
        # capability token, `on_context` needs to hand it to the renewal loop
        # before the first long wait, and the primary binding is a pure
        # resolver lookup that never needed a lease to compute.
        primary_binding: Optional[DeployBinding] = None
        if spec.model_params:
            first_slot = spec.model_params[0][0]
            primary_binding = self._resolver.resolve(
                model_slots[first_slot], picks[first_slot]
            )
        ctx = self._make_context(
            request_id, primary_binding, spec, context, payload=decoded.payload
        )
        if on_context is not None:
            on_context(ctx)

        per_request = dict(context or {})
        input_fetch_t0 = time.monotonic()
        # pgw#1584: THE BOOT WARM PASS CARRIES NO FETCHABLE INPUT. Its payload
        # is the platform's own synthesis (`warm_payload`) and any asset in it
        # is a file this process just wrote, referenced by `local_path` — there
        # is no dispatch, no manifest, no capability token and nothing hub-side
        # to resolve a ref against. `materialize_input_assets` NULLS every
        # `local_path` first, on the correct reasoning that a path in RunJob
        # input is caller-controlled wire data; a boot warm payload is the one
        # input on this path that is not caller-controlled, and `boot_warmup`
        # is the same authority `output_integrity.judged` reads to decide the
        # blank-render exemption. Skipped, never faked with a stub manifest.
        if not ctx.boot_warmup:
            try:
                materialize_input_assets(
                    decoded.payload,
                    request_id,
                    attempt=int(attempt),
                    manifest=tuple(input_assets),
                    file_base_url=str(per_request.get("file_api_base_url") or ""),
                    capability_token=str(
                        per_request.get("worker_capability_token") or ""
                    ),
                )
            except BaseException:
                # `materialize_input_assets` already cleared its own attempt
                # directory; this only guarantees it for anything that raised
                # around it. The refusal itself is typed and propagates.
                cleanup_input_assets(request_id, int(attempt))
                raise
        # A PRE-handler stage: input fetch is not the author's runtime, and
        # folding it in makes every asset-taking endpoint look slow.
        ctx._stages.record_pre("input_fetch", time.monotonic() - input_fetch_t0)

        # THE SAME SEAM, ONE FIELD OVER (pgw#1475). A reserved `source` /
        # `text_encoder` / `candidate` / `resume_from` names a REPO the
        # platform materializes before the body runs, and the body reads only
        # `ctx.source_path`. Here, and not after the leases, for the v1 reason
        # and one more: a weightless producer takes no lease at all, so
        # anything placed inside the ExitStack below would never run for the
        # 27 conversion producers this exists for.
        source_fetch_t0 = time.monotonic()
        # pgw#1524: no HF token travels here any more — every reserved repo
        # comes out of the platform CAS, never off an upstream registry.
        materialize_reserved_inputs(ctx, decoded.payload, snapshots or {})
        ctx._stages.record_pre(
            "source_fetch", time.monotonic() - source_fetch_t0
        )

        with ExitStack() as leases:
            models: Dict[str, Model[Any]] = {}
            #: Unmet machine floors of the instances this request actually
            #: uses — a load-time fact, so it stains EVERY request the
            #: degraded instance serves, not just the one that loaded it.
            degraded: list[str] = []
            # STABLE SLOT-NAME ORDER — the multi-model deadlock rule: every
            # request over the same slot set acquires in one global order.
            # pgw#1392: a WEIGHTLESS entrypoint's slot set is EMPTY, so this
            # loop does not run — no lease, no admission, no load, and no
            # compile subject. That is the whole of its serve path, and it
            # is deliberate: there are no weights to make resident.
            for slot_name in sorted(model_slots):
                model_cls = model_slots[slot_name]
                binding = self._resolver.resolve(model_cls, picks[slot_name])
                lane_object, lane_key = self._lane_of(model_cls)
                key = (model_cls, binding.checkpoint_ref, lane_key)
                with self._backends_lock:
                    known = self._backends.get(key)
                factory: Callable[[], Any]
                if known is not None and known.model is not None:
                    def factory(live: _InstanceBackend = known) -> _InstanceBackend:
                        return live
                else:
                    # First residency, or dropped by eviction: fresh build.
                    factory = self._backend_factory(model_cls, binding, key)
                lease = self.residency.lease(
                    binding.checkpoint_ref,
                    f"{model_cls.__name__}/{lane_key}",
                    factory,
                )
                leases.enter_context(lease)
                backend = lease.backend
                if backend.model is None:  # pragma: no cover — defensive
                    raise ServeDispatchError(
                        f"model slot {slot_name!r} has no live instance after "
                        f"admission; the residency ledger is inconsistent"
                    )
                models[slot_name] = backend.model
                degraded.extend(backend.degraded)

            for warning in degraded:
                ctx.warn(warning)
            arguments = [
                models[slot.name] if slot.kind == "model" else adapters[slot.name]
                for slot in spec.slots
            ]
            # pgw#1475, SAME FAMILY, found by the writer-less-setter fence this
            # issue added rather than by a rented pod: `_set_execution_lane`
            # also lost its only caller with `executor.py`, so every v2 request
            # read the property's "bf16-w16a16+eager" DEFAULT no matter what it
            # ran — and a body that declared `handles=[...]` branched on it.
            # The executing lane is the PRIMARY slot's; a weightless entrypoint
            # has none and keeps the default.
            if spec.model_params:
                primary_lane, primary_handle = self._lane_of(spec.model_params[0][1])
                if primary_lane is not None:
                    ctx._set_execution_lane(primary_handle)

            # The author's own span. Everything outside it — envelope decode,
            # input fetch, admission, weight load — is platform time, and
            # `handler_open`/`handler_close` are what let a slow request be
            # attributed to the right side of that line.
            ctx._stages.handler_open()
            try:
                result = spec.fn(ctx, decoded.payload, *arguments)
                if inspect.iscoroutine(result):
                    # AN `async def` ENTRYPOINT IS DRIVEN HERE, INSIDE ITS
                    # LEASE. `@entrypoint` accepts one (an async function IS a
                    # function), and returning the coroutine unawaited made the
                    # result a coroutine OBJECT — msgpack then failed the job
                    # with "Encoding objects of type coroutine is unsupported",
                    # after the leases had already been released. Awaiting it
                    # anywhere above this line would run author code outside
                    # its residency lease, which is the one thing the lease
                    # exists to prevent.
                    #
                    # `asyncio.run` is correct rather than lucky: `invoke` is
                    # called from `asyncio.to_thread`, so this thread has no
                    # running loop of its own to conflict with.
                    result = asyncio.run(result)
            finally:
                # CLOSE THE SPAN ON THE FAILING PATH TOO: a handler that raised
                # after 90 s is the sample most worth having.
                ctx._stages.handler_close()
        return InvokeOutcome(
            result=result,
            warnings=ctx.warnings,
            adjustments=ctx.adjustments,
            stages=ctx._stages,
        )

    # -- the boot warm pass (pgw#1584) --------------------------------------

    def boot_warmup(
        self, *, prepare: Optional[Callable[[str], str]] = None
    ) -> Tuple[WarmPass, ...]:
        """One synthetic invocation per entrypoint, at boot, before servable.

        **What it is.** The worker's first-call tax is EAGER cost — allocator
        pool growth to the activation peak plus cuBLAS/cuDNN heuristic
        selection — and today a PAYING request pays it. This runs that forward
        once at boot instead, through :meth:`invoke`: the real envelope decode,
        the real residency lease, the real author body. Not a second serving
        path; the same one, called earlier.

        **The context carries ``boot_warmup=True``, and that is load-bearing
        in two directions.** An endpoint cheapens the run off it (musicgen: one
        second of tokens; the qwens: one token) — the surface
        ``RequestContext.boot_warmup``'s docstring has advertised since v1 and
        which had no writer at all between the v2 hardcut and pgw#1584. And
        :func:`gen_worker.output_integrity.judged` reads the SAME object to
        exempt this pass from the blank-render floor: a degenerate output from
        a degenerate input is the expected result here, and without the flag
        reaching the reader the warm pass would fail on its own discarded
        output.

        **The payload is the schema's neutral defaults** (:mod:`.warm_payload`)
        — v1's warm plan shape, *"a single run at the schema's neutral
        defaults, at BOOT, before any request exists"*. Not the largest preset,
        not an author declaration (``NoWarmup`` is tombstoned): one run at the
        point every request is measured against.

        **FAILURE DEGRADES, IT DOES NOT BRICK.** A warm pass that raises emits
        a ``serve_degrade`` event naming the function and the exception, and
        the boot CONTINUES — the pod serves, and the first real request simply
        takes the cold cost it would have taken anyway. The alternative, v1's
        posture of failing the load, turns a warm-pass defect into an
        unservable pod, which is strictly worse than the tax this exists to
        remove.

        ``prepare(function)`` is the caller's per-entrypoint hook, called just
        before each warm invocation: it returns ``""`` to proceed, or a SKIP
        REASON to decline this one. It is where the worker binds the deploy's
        checkpoint picks for the function — boot has no dispatch to read them
        off, and a release the hub seeded no per-function bindings for is a
        skip with a reason, never a guess about which bytes to serve.

        Returns one :class:`WarmPass` per entrypoint, in route-name order.

        Never raises (except a `KeyboardInterrupt`/`SystemExit` tearing the
        process down, which is not a warm-pass failure). The caller places it
        after weights materialize and
        before ``first_request_servable`` is stamped, which is what makes it a
        REAL readiness probe rather than a stamp asserting the process is up
        (th#2233).
        """
        results: list[WarmPass] = []
        with tempfile.TemporaryDirectory(prefix="gw-boot-warmup-") as scratch:
            for name in sorted(self.loaded.entrypoints):
                results.append(self._warm_one(name, scratch, prepare))
        warmed = [row for row in results if row.warmed]
        logger.info(
            "boot warmup: %d/%d entrypoint(s) warmed (%s)",
            len(warmed), len(results),
            ", ".join(f"{row.function}={row.outcome}" for row in results) or "none",
        )
        return tuple(results)

    def _warm_one(
        self,
        function: str,
        scratch: str,
        prepare: Optional[Callable[[str], str]] = None,
    ) -> WarmPass:
        spec = self.loaded.entrypoints[function]
        # A WEIGHTLESS entrypoint takes no lease, loads nothing and allocates
        # no activation peak (pgw#1392) — there is no cold cost to move, so
        # warming it would spend a boot second to save nothing.
        if not spec.model_params:
            return WarmPass(function, WARM_SKIPPED,
                            "weightless: nothing is resident to warm")
        kind = str(getattr(spec, "kind", "") or "inference")
        if kind != "inference":
            # A producer's cost is its own job's, and its payload names
            # reserved repos the platform materializes per REQUEST — there is
            # no boot-time answer for what `ctx.source_path` would point at.
            return WarmPass(function, WARM_SKIPPED,
                            f"kind={kind!r}: only inference pays a first-call tax")
        payload, reason = neutral_payload(spec.payload_type, scratch)
        if payload is None:
            # Stated rather than faked. A required VideoAsset has no honest
            # 2 KB stand-in, and inventing one warms a path with bytes no
            # request will ever carry.
            return WarmPass(function, WARM_SKIPPED, reason)
        if prepare is not None:
            declined = str(prepare(function) or "")
            if declined:
                return WarmPass(function, WARM_SKIPPED, declined)
        envelope = {"input": msgspec.to_builtins(payload)}
        started = time.monotonic()
        try:
            # SPANNED, and this is the phase's first producer: `PHASE_WARMUP`
            # has been in `boot_phases`' vocabulary — with the module's own
            # rule that "a declared phase with no producer is a default read as
            # a fact" printed above it — and nothing in `src/` has emitted one
            # since the hardcut.
            with boot_phases.span(boot_phases.PHASE_WARMUP, function=function):
                self.invoke(
                    function,
                    envelope,
                    request_id=f"boot-warmup-{function}",
                    context={
                        "boot_warmup": True,
                        # Discarded by construction: whatever the body saves
                        # lands in the scratch directory this pass owns and
                        # dies with it. Nothing is uploaded and nothing banks.
                        "local_output_dir": scratch,
                    },
                )
        except (KeyboardInterrupt, SystemExit):
            # NOT a warm-pass failure — the process is being torn down, and
            # swallowing that would make a boot un-interruptible.
            raise
        except BaseException as exc:  # noqa: BLE001 — a warm pass never bricks
            duration_ms = int((time.monotonic() - started) * 1000)
            detail = (
                f"function={function} payload=schema-defaults "
                f"{type(exc).__name__}: {exc}"
            )
            logger.warning(
                "boot warmup FAILED for %s (%s: %s); this pod SERVES and the "
                "first real request pays the cold cost",
                function, type(exc).__name__, exc, exc_info=True,
            )
            # pgw#760: a fail-soft outcome that changes what this worker
            # serves rides a TYPED event, never only a log line — a hub-spawned
            # worker exposes no stdout, so a warning here is invisible.
            try:
                activity.emit_event(
                    activity.KIND_SERVE_DEGRADE, detail,
                    phase="boot_warmup_failed", duration_ms=duration_ms,
                )
            except Exception:  # noqa: BLE001 — the confession never fails boot
                logger.debug("serve_degrade emit failed", exc_info=True)
            return WarmPass(function, WARM_FAILED,
                            f"{type(exc).__name__}: {exc}", duration_ms)
        return WarmPass(
            function, WARM_OK, "", int((time.monotonic() - started) * 1000)
        )

    def _make_context(
        self,
        request_id: str,
        binding: Optional[DeployBinding],
        spec: Optional[Any] = None,
        per_request: Optional[Mapping[str, Any]] = None,
        payload: Any = None,
    ) -> RequestContext[Any]:
        kwargs: Dict[str, Any] = dict(self._context_kwargs)
        # Per-request facts WIN over the loop's construction-time defaults:
        # the token, the owner and the file API base belong to this dispatch.
        kwargs.update({k: v for k, v in (per_request or {}).items() if v is not None})
        if self._output_dir is not None:
            kwargs.setdefault("local_output_dir", str(self._output_dir))
        if self._hf_token:
            kwargs.setdefault("hf_token", self._hf_token)
        if payload is not None:
            # pgw#1475: the reserved STRUCTS, stamped at construction because
            # the mixin that holds them takes them as constructor arguments.
            # `ctx.source` is read beside `ctx.source_path` — `source_from_ctx`
            # builds its `Source` from `info["ref"]` and `info["attributes"]`
            # — so a path with no info is the same defect one field over.
            kwargs.update(reserved_context_kwargs(payload))
        if spec is not None:
            # pgw#1406: the AUTHORITY declarations, stamped from the spec onto
            # the context the body receives. This is the SDK half of "one
            # fact, two enforcers": the hub mints the repo-write grant off the
            # same `publishes` it read on the manifest row, and the publisher
            # surface refuses undeclared code here, before a byte moves.
            #
            # `emits_media` is stamped only when the author DECLARED it. Left
            # absent it stays `None` — the base reads that as "not job-scoped",
            # media is the product of a request, and behaviour is unchanged for
            # every endpoint that never says the word.
            kwargs["publishes"] = bool(getattr(spec, "publishes", False))
            # pgw#1576: the SAME stamp for the streaming declaration. `emit`
            # refuses without it, so an undeclared function cannot stream past
            # a manifest that says `incremental_output: false`.
            kwargs["streams"] = getattr(spec, "delta_arms", ()) or None
            declared_media = getattr(spec, "emits_media", None)
            if declared_media is not None:
                kwargs["emits_media"] = bool(declared_media)
            # pgw#1579/pgw#1580, the same wire for the two declarations the
            # hardcut dropped. Unstamped they read as undeclared, which is the
            # fail-closed answer the hub's own grant minter uses.
            kwargs["child_calls"] = bool(getattr(spec, "child_calls", False))
            kwargs["handles"] = tuple(getattr(spec, "handles", ()) or ())
        return RequestContext(
            request_id,
            binding=binding
            or DeployBinding(checkpoint_ref="", checkpoint_dir=Path(".")),
            **kwargs,
        )


def manifest_sizer(
    weights: Mapping[str, int], *, headroom_bytes: int
) -> InstanceSizer:
    """An :class:`InstanceSizer` over a static ref->bytes table.

    The production sizer reads the tensorfs manifest per (checkpoint, lane)
    — exact bytes, known ahead of load; this helper is the local/cozy-local
    shape (and the test double) until that wiring lands with the pgw#1380
    loader engine.
    """

    class _Sizer:
        def resident_bytes(self, checkpoint_ref: str, lane: str) -> int:
            if checkpoint_ref not in weights:
                raise KeyError(
                    f"no manifest bytes for {checkpoint_ref!r}; admission "
                    f"needs the real size"
                )
            return weights[checkpoint_ref]

        def activation_headroom_bytes(self, checkpoint_ref: str, lane: str) -> int:
            return headroom_bytes

    return _Sizer()


__all__ = [
    "BindingResolver",
    "InvokeOutcome",
    "ServeLoop",
    "WARM_FAILED",
    "WARM_OK",
    "WARM_SKIPPED",
    "WarmPass",
    "manifest_sizer",
]
