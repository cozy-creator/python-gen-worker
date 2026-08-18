"""The endpoint serve host (pgw#1372/pgw#1382): eager-first boot, adopt via
``ctx.compile``, single-flight per model instance.

Boot, in order (Paul's flow, pgw#1367 + the imperative-marking ruling): when
release metadata is at hand, build the torchcg ``AdoptSession`` FIRST — its
constructor runs the exact-env audit, so a mismatched pod refuses loudly
before any author code runs — then instantiate each referenced Model class
(cheap ``__init__``) and call ``model.load(LoadContext)``. The author's own
``ctx.compile(module)`` calls are where compiled graphs swap in: hit ->
armed, miss -> eager + an ordered :class:`Hole` for the background mint
(pgw#1371). No trace, no derivation, no compile happens here, ever.

Concurrency: SINGLE-FLIGHT PER MODEL INSTANCE — the admission lock lives
with the instance (the object that HAS the state); a multi-model entrypoint
acquires all its slots' admissions in deterministic slot-name order.
Entrypoints, being stateless, have no concurrency property at all.

Eviction (``evict``/``teardown``): drain (acquire the admission), call the
author's ``unload(ctx)`` — BEST-EFFORT, NEVER CORRECTNESS: an exception is
logged and eviction proceeds; a failing unload cannot pin VRAM.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import msgspec

from .. import boot_stages
from .context import DeployBinding, LoadContext, LoaderEngine, RequestContext
from .entrypoints import EntrypointSpec
from .loader import LoadedEndpoint
from .model import Model, model_type

logger = logging.getLogger(__name__)


class ServeDispatchError(RuntimeError):
    """A request names a function this endpoint does not serve."""


class ModelInstance:
    """One resident model: the author's object + its admission + load ctx."""

    def __init__(self, model: Model[Any], load_context: LoadContext[Any]) -> None:
        self.model = model
        self.load_context = load_context
        #: SINGLE-FLIGHT: the worker acquires this before invoking an
        #: entrypoint against the instance; drained = acquirable.
        self.admission = threading.Lock()


class EndpointHost:
    """One loaded endpoint, booted and routable.

    The host owns worker-side state only: the deploy binding (mutable hub
    state — ``rebind`` swaps it), each model class's active lane (the
    deploy's pick), the live model instances with their admissions, and the
    adoption session. Author state lives on the author's model instances,
    untouched.
    """

    def __init__(
        self,
        loaded: LoadedEndpoint,
        binding: DeployBinding,
        *,
        lane_contract: str = "",
        engine: Optional[LoaderEngine] = None,
        device: str = "cuda",
        io: str = "buffered",
        output_dir: Optional[Path] = None,
        context_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.loaded = loaded
        self.binding = binding
        self.lanes: Dict[type, Any] = {
            cls: loaded.lane(cls, lane_contract) for cls in loaded.models
        }
        self.instances: Dict[type, ModelInstance] = {}
        self.adoption: Any = None
        #: pgw#1392: "has boot run" is the real question. `instances` used to
        #: stand in for it, which a WEIGHTLESS endpoint falsifies forever —
        #: it references no Model class, so it is never resident and has
        #: nothing to make resident.
        self._booted = False
        if engine is None:
            # pgw#1380: the native store->VRAM engine, bound off the
            # checkpoint tree the worker already resolved — a projected
            # snapshot carries its own chunk store, so nothing extra is
            # plumbed here. `None` back means this tree has no store behind
            # it (a bare download, a fixture) and `ctx.load` keeps the eager
            # bridge; PLACEMENT is decided here, by the worker, never by
            # author code.
            from .streaming import engine_for

            engine = engine_for(binding.checkpoint_dir, device=device, io=io)
        self._engine = engine
        self._output_dir = output_dir
        self._context_kwargs = dict(context_kwargs or {})

    def _stream_attributes(self) -> Dict[str, Any]:
        """The streamed load's own numbers, or nothing when no engine ran."""
        report = getattr(self._engine, "last_report", None)
        if report is None:
            return {}
        attributes: Dict[str, Any] = report.attributes()
        return attributes

    # -- contexts -----------------------------------------------------------

    def make_context(self, request_id: str, **overrides: Any) -> RequestContext[Any]:
        kwargs: Dict[str, Any] = dict(self._context_kwargs)
        if self._output_dir is not None:
            kwargs.setdefault("local_output_dir", str(self._output_dir))
        kwargs.update(overrides)
        return RequestContext(request_id, binding=self.binding, **kwargs)

    def _load_context(
        self, model_cls: type, *, compile_sink: Any = None
    ) -> LoadContext[Any]:
        return LoadContext(
            binding=self.binding,
            model_type=model_type(model_cls),
            lane=self.lanes[model_cls],
            engine=self._engine,
            compile_sink=compile_sink,
        )

    # -- boot ---------------------------------------------------------------

    def setup(
        self,
        *,
        store: Any = None,
        document: Any = None,
        sm: str = "",
        loader: Any = None,
        artifacts_dir: Optional[Path] = None,
        installed: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Instantiate each referenced Model class and run its ``load(ctx)``.

        With no ``document`` this is the EAGER bridge: ``ctx.compile`` is a
        transparent pass-through and the endpoint serves after load,
        unconditionally. With release metadata (``document`` + ``sm``), the
        adopt session forms FIRST — the exact-env audit refuses loudly
        before any author code runs — and the author's own ``ctx.compile``
        calls arm what the store holds; ``self.holes`` afterwards is the
        ordered mint work-list. ``store`` may be ``None`` with a document:
        every claimed graph is then a hole (mint everything).
        """
        from .._vendor.torchcg import EnvironmentMismatch
        from .._vendor.torchcg.adopt import AdoptSession
        from .._vendor.torchcg.graph_identity import installed_closure

        started = time.monotonic()

        def span(**attrs: object) -> None:
            boot_stages.record_ending_now(
                boot_stages.Stage.ADOPT_PULL,
                duration_ms=int((time.monotonic() - started) * 1000),
                label=self.loaded.module_name,
                sm=sm,
                **attrs,
            )

        session = None
        installed_map: Optional[Mapping[str, str]] = None
        if document is not None and getattr(document, "eager_permanent", False):
            span(graphs_from="eager_permanent")
            document = None
        if document is not None:
            lane_bearing = [cls for cls, lane in self.lanes.items() if lane is not None]
            if not lane_bearing:
                raise RuntimeError(
                    "setup(): release metadata offered but no model class "
                    "declared lanes; an eager-permanent endpoint has nothing "
                    "to adopt"
                )
            if len(lane_bearing) > 1:
                raise RuntimeError(
                    "setup(): release metadata offered for a multi-model "
                    "endpoint; per-slot adoption sessions are not designed "
                    "yet (one lane-bearing model class per endpoint for now)"
                )
            from .model import lane_handle

            lane_contract = lane_handle(self.lanes[lane_bearing[0]])
            installed_map = dict(installed) if installed is not None else installed_closure()
            try:
                session = AdoptSession(
                    store,
                    document,
                    lane_contract,
                    sm,
                    loader=loader,
                    artifacts_dir=artifacts_dir
                    or Path(".compiled-graphs"),
                    installed=installed_map,
                )
            except EnvironmentMismatch as exc:
                # The audit fired BEFORE any author code ran — a
                # build-system bug surfacing, recorded then re-raised loudly.
                span(graphs_from="release", lane=lane_contract,
                     refusal="environment_mismatch")
                logger.error("adopt refused: %s", exc)
                raise

        for model_cls in self.loaded.models:
            load_started = time.monotonic()
            model: Model[Any] = model_cls()  # cheap __init__ — no GPU, by contract
            load_context = self._load_context(
                model_cls,
                compile_sink=session.adopt if session is not None else None,
            )
            model.load(load_context)
            self.instances[model_cls] = ModelInstance(model, load_context)
            # pgw#1380: `model_load` IS the streaming span now. The fill's
            # write-then-reread time did not move to a new stage — it left
            # the boot, and what remains here is store->VRAM. The attributes
            # say how fast and through what, so a regression is legible
            # without a second measurement.
            boot_stages.record_ending_now(
                boot_stages.Stage.MODEL_LOAD,
                duration_ms=int((time.monotonic() - load_started) * 1000),
                label=f"{self.loaded.module_name}:{model_cls.__name__}",
                checkpoint=self.binding.checkpoint_ref,
                **self._stream_attributes(),
            )
        if session is not None:
            # The mint-written manifest is the artifact's COMPATIBILITY SET
            # (Paul, 2026-08-18): same-major ``>=`` floors for the libraries
            # the artifact actually LINKS, read off its own ELF at mint. This
            # host adopts when it SATISFIES them — an unrelated package bump
            # can no longer reject bytes that still load, and a runtime older
            # than the one that built them still cannot adopt.
            if store is not None:
                from .mint import assert_satisfied

                for record in session.adopted:
                    manifest = store.get_manifest(record.graph, session.env)
                    if manifest is not None:
                        assert_satisfied(manifest, sm=sm)
            span(
                graphs_from="release",
                lane=lane_contract,
                artifact_from_store=len(session.adopted),
                artifact_from_eager=len(session.holes),
                ambiguous=len(session.ambiguous),
                unclaimed=len(session.unclaimed),
            )
        self.adoption = session
        self._booted = True

    # -- eviction -----------------------------------------------------------

    def evict(self, model_cls: type) -> None:
        """Evict one instance: DRAIN (acquire the admission — single-flight
        defines drained), call the author's ``unload(ctx)``, drop the
        reference. The unload is best-effort tidiness, never correctness:
        exceptions are logged and eviction proceeds — a failing or slow
        unload cannot pin the instance resident."""
        instance = self.instances.get(model_cls)
        if instance is None:
            return
        with instance.admission:  # drain: in-flight requests finish first
            try:
                instance.model.unload(instance.load_context)
            except Exception:
                logger.exception(
                    "unload(%s) raised; eviction proceeds (best-effort, "
                    "never correctness)", model_cls.__name__,
                )
            del self.instances[model_cls]

    def teardown(self) -> None:
        """Evict every resident instance (reverse residency order)."""
        for model_cls in list(reversed(list(self.instances))):
            self.evict(model_cls)
        self._booted = False

    def rebind(self, binding: DeployBinding) -> None:
        """Swap the deploy binding (hub deploy state changed). Contexts made
        after this read the new binding; release identity is untouched —
        graphs a rebind introduces are holes the mint fills (partial-hit)."""
        self.binding = binding

    @property
    def holes(self) -> tuple[Any, ...]:
        """The ordered mint work-list (pgw#1371's input): torchcg ``Hole``
        rows in canonical document order, each carrying its full
        ``GraphRecord`` (graph hash + ingress) and reason. The mint arms
        each landed artifact via ``self.adoption.arm(record, path)``."""
        return tuple(self.adoption.holes) if self.adoption is not None else ()

    # -- serving ------------------------------------------------------------

    def dispatch(
        self,
        function: str,
        payload: Any,
        *,
        request_id: str,
        ctx: Optional[RequestContext[Any]] = None,
        loras: Sequence[Any] = (),
    ) -> Any:
        """Route one request to the named entrypoint — function name IS the
        route. ``payload`` is the wire mapping (or an already-typed struct);
        it decodes against the entrypoint's own msgspec schema, so a payload
        that does not fit is a typed ``msgspec.ValidationError`` naming the
        field, before author code runs. The call runs under every slot
        model's admission, acquired in slot-name order — a WEIGHTLESS
        entrypoint (pgw#1392) declares no slot, so it acquires nothing and
        is called directly; there is no residency to admit it to."""
        spec = self.loaded.entrypoints.get(function)
        if spec is None:
            raise ServeDispatchError(
                f"{self.loaded.module_name} serves no function {function!r} "
                f"(functions: {sorted(self.loaded.entrypoints)})"
            )
        if not self._booted:
            raise RuntimeError("dispatch(): boot the endpoint first (setup())")
        missing = [
            cls.__name__ for cls in spec.model_classes if cls not in self.instances
        ]
        if missing:
            raise RuntimeError(
                f"dispatch({function!r}): model instance(s) not resident: "
                f"{missing} (evicted or never loaded)"
            )
        decoded = self._decode(spec, payload)
        if ctx is None:
            ctx = self.make_context(request_id)
        arguments = [
            self.instances[slot.annotation].model if slot.kind == "model"
            else list(loras) if slot.kind == "adapters"
            else self._resolve_adapter(function, slot)
            for slot in spec.slots
        ]
        with ExitStack() as admissions:
            # Deterministic slot-name order — the multi-model deadlock rule.
            for slot_name, model_cls in sorted(spec.model_params):
                admissions.enter_context(self.instances[model_cls].admission)
            return spec.fn(ctx, decoded, *arguments)

    def _resolve_adapter(self, function: str, slot: Any) -> Any:
        """Fill one declared adapter slot from deploy state (hub-resolved).
        A required slot with nothing bound is a typed refusal BEFORE author
        code runs; an optional (`... | None`) slot passes None — the author
        owns that branch. The slot's ANNOTATION is its KIND (Paul's
        structural guard): a `DistillationAdapter` slot takes only a
        distillation-marked adapter — a style LoRA cannot seize the serving
        config even on a misconfigured deployment."""
        bound = self.binding.adapter
        if bound is None:
            if slot.required:
                raise ServeDispatchError(
                    f"{function!r} requires adapter slot {slot.name!r} and "
                    "this deployment binds no adapter"
                )
            return None
        if not isinstance(bound, slot.annotation):
            raise ServeDispatchError(
                f"{function!r} slot {slot.name!r} takes "
                f"{slot.annotation.__name__}; the bound adapter "
                f"{bound.ref or bound.name!r} is not distillation-marked "
                "(a misdeploy the hub should have refused)"
            )
        return bound

    def _decode(self, spec: EntrypointSpec, payload: Any) -> Any:
        if isinstance(payload, spec.payload_type):
            return payload
        if isinstance(payload, (bytes, bytearray)):
            return msgspec.json.decode(payload, type=spec.payload_type)
        return msgspec.convert(payload, type=spec.payload_type)


__all__ = ["EndpointHost", "ModelInstance", "ServeDispatchError"]
