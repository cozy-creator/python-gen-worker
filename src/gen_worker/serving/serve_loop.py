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
import threading
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

from .context import DeployBinding, LoadContext, LoaderEngine, RequestContext
from .envelope import DecodedRequest, decode_envelope
from .host import ServeDispatchError
from .loader import LoadedEndpoint
from .model import Model, lane_handle, model_type
from .placement import warn_if_degraded
from .residency import InstanceSizer, ResidencyManager

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
    envelope reply carries (``ctx.warn`` rows, adjustments)."""

    result: Any
    warnings: Tuple[str, ...]
    adjustments: Tuple[Dict[str, str], ...]


class _InstanceBackend:
    """One resident (model class x checkpoint x lane) as residency sees it.

    ``load`` runs the author's ``Model()`` + ``load(ctx)`` (inside the
    manager's serialized load gate); ``drop`` runs the author's ``unload``
    — best-effort tidiness, never correctness. The host tiers
    (``demote_to_host``/``promote_to_device``) belong to the pgw#1380
    native-loader wave: until it lands the loop runs the manager with a
    zero host budget, so eviction is always a drop, and these arms refuse
    loudly rather than pretend."""

    def __init__(
        self,
        model_cls: type,
        load_context: LoadContext[Any],
        lane: Any = None,
    ) -> None:
        self.model_cls = model_cls
        self.load_context = load_context
        self.model: Optional[Model[Any]] = None
        self.lane = lane
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

    def drop(self) -> None:
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

    def demote_to_host(self) -> None:
        raise NotImplementedError(
            "host-tier staging rides the pgw#1380 native loader; run the "
            "ResidencyManager with host_budget_bytes=0 until it lands"
        )

    def promote_to_device(self) -> None:
        raise NotImplementedError(
            "host-tier staging rides the pgw#1380 native loader; run the "
            "ResidencyManager with host_budget_bytes=0 until it lands"
        )


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
        output_dir: Optional[Path] = None,
        context_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.loaded = loaded
        self.residency = residency
        self._resolver = resolver
        self._engine = engine
        #: The deploy's per-class lane pick (contract handle; '' = the single
        #: declared lane / eager-permanent).
        self.lanes: Dict[type, Any] = {
            cls: loaded.lane(cls, lane_contract) for cls in loaded.models
        }
        self._compile_sink_for = compile_sink_for
        self._output_dir = output_dir
        self._context_kwargs = dict(context_kwargs or {})
        #: Live backends by residency key, so a lease hit reuses the author
        #: object instead of rebuilding it. Guarded: leases serialize per
        #: key, but two DIFFERENT keys mutate this table concurrently.
        self._backends: Dict[Tuple[type, str, str], _InstanceBackend] = {}
        self._backends_lock = threading.Lock()

    # -- placement ----------------------------------------------------------

    def _lane_of(self, model_cls: type) -> Tuple[Any, str]:
        lane = self.lanes[model_cls]
        return lane, (lane_handle(lane) if lane is not None else "eager")

    def _backend_factory(
        self, model_cls: type, binding: DeployBinding, key: Tuple[type, str, str]
    ) -> Callable[[], _InstanceBackend]:
        def make() -> _InstanceBackend:
            lane, _ = self._lane_of(model_cls)
            sink = (
                self._compile_sink_for(model_cls, lane)
                if self._compile_sink_for is not None
                else None
            )
            backend = _InstanceBackend(
                model_cls,
                LoadContext(
                    binding=binding,
                    model_type=model_type(model_cls),
                    lane=lane,
                    engine=self._engine,
                    compile_sink=sink,
                ),
                lane,
            )
            with self._backends_lock:
                self._backends[key] = backend
            return backend

        return make

    # -- the one public operation -------------------------------------------

    def invoke(
        self, function: str, envelope: Any, *, request_id: str
    ) -> InvokeOutcome:
        """Serve one request end-to-end. See the module docstring for the
        walk; every step before ``spec.fn`` refuses typed."""
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

        with ExitStack() as leases:
            models: Dict[str, Model[Any]] = {}
            primary_binding: Optional[DeployBinding] = None
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
                _, lane_key = self._lane_of(model_cls)
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
                if primary_binding is None:
                    # The FIRST slot in signature order is the request's
                    # primary routing fact (ctx.checkpoint_ref).
                    first_slot = spec.model_params[0][0]
                    primary_binding = self._resolver.resolve(
                        model_slots[first_slot], picks[first_slot]
                    )

            ctx = self._make_context(request_id, primary_binding)
            for warning in degraded:
                ctx.warn(warning)
            arguments = [
                models[slot.name] if slot.kind == "model" else adapters[slot.name]
                for slot in spec.slots
            ]
            result = spec.fn(ctx, decoded.payload, *arguments)
            if inspect.iscoroutine(result):
                # AN `async def` ENTRYPOINT IS DRIVEN HERE, INSIDE ITS LEASE.
                # `@entrypoint` accepts one (an async function IS a function),
                # and returning the coroutine unawaited made the result a
                # coroutine OBJECT — msgpack then failed the job with
                # "Encoding objects of type coroutine is unsupported", after
                # the leases had already been released. Awaiting it anywhere
                # above this line would run author code outside its residency
                # lease, which is the one thing the lease exists to prevent.
                #
                # `asyncio.run` is correct rather than lucky: `invoke` is
                # called from `asyncio.to_thread`, so this thread has no
                # running loop of its own to conflict with.
                result = asyncio.run(result)
        return InvokeOutcome(
            result=result,
            warnings=ctx.warnings,
            adjustments=ctx.adjustments,
        )

    def _make_context(
        self, request_id: str, binding: Optional[DeployBinding]
    ) -> RequestContext[Any]:
        kwargs: Dict[str, Any] = dict(self._context_kwargs)
        if self._output_dir is not None:
            kwargs.setdefault("local_output_dir", str(self._output_dir))
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
    "manifest_sizer",
]
