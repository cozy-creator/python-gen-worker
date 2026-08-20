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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

#: A compile stack, in EITHER of its two canonical shapes. torchcg's own
#: `require_stack` has always accepted both — `document.stack` and
#: `compile_stack()` return the ORDERED tuple-of-pairs, and a caller that
#: happens to hold a dict should not have to convert. Annotating only the
#: Mapping half made every caller passing a real stack a type error while
#: working perfectly at runtime (both call sites do `dict(stack)`).
CompileStack = Union[Mapping[str, str], Sequence[Tuple[str, str]]]

import msgspec

from .. import boot_stages
from .context import DeployBinding, LoadContext, LoaderEngine, RequestContext
from .entrypoints import EntrypointSpec
from .loader import LoadedEndpoint
from .model import Model, model_type
from .placement import serving_device, warn_if_degraded
from .worker_context import worker_load_context

logger = logging.getLogger(__name__)


def _artifacts_root() -> Path:
    """The box artifacts cache — pgw#1526's ONE address.

    Imported inside the function, not at module scope: this module is inside
    the adopt-only serve closure (pgw#1328) and must not acquire the CLI
    package merely to learn a path. The answer still comes from
    `cli/workspace.py` and is not restated here — a second spelling of a store
    address is how a build and a lookup end up at different directories.
    """
    from ..cli.workspace import artifacts_root

    return artifacts_root()



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
        device: str = "",
        io: str = "buffered",
        output_dir: Optional[Path] = None,
        context_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # pgw#1452: the default was the literal `"cuda"` — an enumeration of
        # what a pod usually is, not a measurement of what THIS machine is. It
        # feeds both arms (`engine_for(device=)` and the eager bridge's
        # `_placed`), so on a CPU-only host it turned a boot into a bare torch
        # device error instead of a named fallback. An explicit `device=` from
        # a caller that has already decided still wins; only the DEFAULT is
        # probed.
        # `device or ""` and not `str(device)`: a caller handing `None` means
        # "you decide", and `str(None)` is the string "None" — a device name
        # torch would reject three frames later with nothing pointing here.
        device = str(device or "") or serving_device()
        self.loaded = loaded
        self.binding = binding
        self.lanes: Dict[type, Any] = {
            cls: loaded.lane(cls, lane_contract) for cls in loaded.models
        }
        self.instances: Dict[type, ModelInstance] = {}
        #: Per class, the unmet machine floors measured at residency-admit.
        #: Non-empty = this endpoint serves DEGRADED, and says so on every
        #: request rather than only in the boot log.
        self.degraded: Dict[type, Tuple[str, ...]] = {}
        self.adoption: Any = None
        #: pgw#1392: "has boot run" is the real question. `instances` used to
        #: stand in for it, which a WEIGHTLESS endpoint falsifies forever —
        #: it references no Model class, so it is never resident and has
        #: nothing to make resident.
        self._booted = False
        # pgw#1549: THIS HOST NO LONGER BUILDS AN ENGINE, and that deletion is
        # the fix rather than a tidy-up.
        #
        # `engine_for` had two production call sites — here, and (since
        # pgw#1544) `ctx.load`. Two constructions of one capability is exactly
        # the state that kept the fleet down: the pod went through `ServeLoop`,
        # which had NEITHER, so every local red/green passed against a caller
        # the production path is not. Deleting this one leaves `ctx.load` — the
        # one place that always has the tree, and the one place BOTH hosts
        # reach — as the sole binder, so the local CLI and the pod now execute
        # the same code and a local verdict finally means something about a pod.
        #
        # `engine=` survives as an EXPLICIT override (endpoint suites inject a
        # fake). Production passes nothing.
        self._engine = engine
        self._io = str(io or "buffered")
        #: pgw#1452: the SAME decision reaches the eager bridge. Handing it
        #: only to `engine_for` meant a tree with no chunk store behind it
        #: was built by `from_pretrained` and left wherever that put it —
        #: the CPU — which is the substrate cozy-local, fixtures and every
        #: local run use.
        self._device = str(device or "")
        self._output_dir = output_dir
        self._context_kwargs = dict(context_kwargs or {})

    def _stream_attributes(self, ctx: LoadContext[Any]) -> Dict[str, Any]:
        """The streamed load's own numbers, or nothing when no engine ran.

        pgw#1549: read off the CONTEXT that actually loaded, not off a
        host-held engine. The host no longer builds one — `ctx.load` binds it —
        so a host-held handle would report the numbers of a load that did not
        happen, which is worse than reporting none.
        """
        report = getattr(ctx.loader_engine, "last_report", None)
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
        ctx: RequestContext[Any] = RequestContext(
            request_id, binding=self.binding, **kwargs
        )
        # Degradation is a LOAD-time fact, so it rides every request the
        # degraded instance serves — a caller must not have to read the pod's
        # boot log to learn why its request took ten minutes.
        for warnings in self.degraded.values():
            for warning in warnings:
                ctx.warn(warning)
        return ctx

    def _load_context(
        self, model_cls: type, *, compile_sink: Any = None
    ) -> LoadContext[Any]:
        # pgw#1549: THE ONE PRODUCTION LOAD CONTEXT, shared with `ServeLoop`.
        return worker_load_context(
            binding=self.binding,
            model_type=model_type(model_cls),
            lane=self.lanes[model_cls],
            engine=self._engine,
            compile_sink=compile_sink,
            device=self._device,
            io=self._io,
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
        stack: Optional[CompileStack] = None,
    ) -> None:
        """Instantiate each referenced Model class and run its ``load(ctx)``.

        ``stack`` is this env's compile stack (the torch/triton/nvidia rows of
        the endpoint's ``uv.lock``); absent, the document's own stamp is used.

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
        from . import adapter_guard
        from ..env_identity import installed_stack_drift

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
        stack_rows: Optional[Mapping[str, str]] = None
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
            # The compile stack this boot claims (pgw#1489): the caller's,
            # read from the endpoint's own uv.lock, or the document's own
            # stamp when nobody stated one. The venv is compared against it
            # DIAGNOSTICALLY — a torch that does not match what the artifacts
            # were compiled against costs a failed load, not a wrong answer,
            # so it warns by name and never gates.
            stack_rows = dict(stack) if stack is not None else dict(document.stack)
            for row in installed_stack_drift(dict(document.stack)):
                logger.warning("adopt: compile-stack drift vs this venv: %s", row)
            try:
                session = AdoptSession(
                    store,
                    document,
                    lane_contract,
                    sm,
                    loader=loader,
                    # pgw#1526: the box cache, never cwd. `compile` writes
                    # there and this is the reader that arms what it wrote.
                    artifacts_dir=artifacts_dir or _artifacts_root(),
                    stack=stack_rows,
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
            # Residency-admit: the machine meets the lane's declared floors,
            # or it is told loudly that it does not and loads anyway (Paul,
            # 2026-08-18 — any model on any machine). Measured before the
            # first weight moves; the warning rides every request this
            # instance serves via `make_context`.
            self.degraded[model_cls] = warn_if_degraded(
                model_cls, self.lanes[model_cls]
            )
            model: Model[Any] = model_cls()  # cheap __init__ — no GPU, by contract
            # GUARDED (pgw#1573/pgw#1571): a peft adapter attached after
            # arming never executes inside a compiled graph, so an unguarded
            # sink serves the base model bit-identically and says nothing.
            load_context = self._load_context(
                model_cls,
                compile_sink=(
                    adapter_guard.sink(session.adopt)
                    if session is not None else None
                ),
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
                **self._stream_attributes(load_context),
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
            marks = tuple(session.unclaimed_marks)
            span(
                graphs_from="release",
                lane=lane_contract,
                artifact_from_store=len(session.adopted),
                artifact_from_eager=len(session.holes),
                ambiguous=len(session.ambiguous),
                unclaimed=len(session.unclaimed),
                unmatched_marks=len(marks),
            )
            # THE VERDICT REACHES A TERMINAL, not only a span (pgw#1534).
            #
            # This is the path `up` and every in-process host take;
            # `ServeAdoption` — which does emit a boot_adopt_summary — is the
            # pod-side adopt-only object and does not run here. So on this
            # path every one of these counts was going into a span no operator
            # reads, and the boot printed nothing at all. Measured: 14 graphs
            # in the document, 14 artifacts present in the store,
            # `adopted: 0, holes: 0, armed: 0`, and not one line saying why.
            #
            # THREE DIFFERENT FACTS COLLAPSE ONTO THOSE TWO ZEROS, and each is
            # loud below, because "which one" is the entire diagnosis:
            #   * nothing was marked          -> an eager endpoint, correct
            #   * marks matched no record     -> the compiled path is off and
            #                                    no mint can turn it on
            #   * every record was AMBIGUOUS  -> the artifacts are present and
            #                                    fine, and the dispatcher
            #                                    refuses to guess between
            #                                    literal-twins, so it disarms
            #                                    BOTH — adopted drops back to
            #                                    zero and holes never rises
            if session.ambiguous:
                logger.warning(
                    "adopt: %d graph(s) in lane %s share a tensor structure "
                    "with another graph and were ALL disarmed — the dispatcher "
                    "cannot tell literal-twins apart at call time and refuses "
                    "to guess. The artifacts are present and valid; they are "
                    "unusable as specialized. Serving EAGER for them.",
                    len(session.ambiguous), lane_contract,
                )
                for record in session.ambiguous[:8]:
                    logger.warning(
                        "adopt:   ambiguous %s (target %s)",
                        record.graph[-16:], record.target,
                    )
            if marks:
                logger.warning(
                    "adopt: %d module(s) marked with ctx.compile matched NO "
                    "graph in lane %s. They serve EAGER, permanently — and no "
                    "mint can change that, because the graphs are not missing, "
                    "the MATCH is.", len(marks), lane_contract,
                )
                for mark in marks:
                    logger.warning("adopt:   %s", mark.describe())
            # THE ZERO MUST NAME ITSELF (pgw#1564). This summary sat at INFO,
            # and `gen-worker up` surfaces WARNING+ only (`logging.lastResort`)
            # — so on 2026-08-20 09:41 a boot claimed 14 graphs, holed all 14
            # with `cannot decompress` sitting RIGHT THERE on each Hole, and
            # the resident log carried not one adoption line. The failure was
            # then investigated as a NEW silent-adoption defect for hours; it
            # was pgw#1561's unloadable blobs wearing this path's silence. A
            # declared-compiled boot that armed NOTHING is a WARNING with its
            # hole reasons attached; a boot that armed is one INFO line.
            if session.adopted:
                logger.info(
                    "adopt: lane=%s sm=%s — %d armed, %d hole(s) to mint, "
                    "%d ambiguous, %d record(s) unclaimed, %d marked module(s) "
                    "matched nothing",
                    lane_contract, sm, len(session.adopted), len(session.holes),
                    len(session.ambiguous), len(session.unclaimed), len(marks),
                )
            else:
                logger.warning(
                    "adopt: ZERO of %d claimed graph(s) armed for lane=%s "
                    "sm=%s (%d hole(s), %d ambiguous, %d unclaimed, %d "
                    "unmatched mark(s)) — this boot serves EAGER for a lane "
                    "that declared compiled serving",
                    len(session.adopted) + len(session.holes), lane_contract,
                    sm, len(session.holes), len(session.ambiguous),
                    len(session.unclaimed), len(marks),
                )
                seen: set[str] = set()
                reasons: list[str] = []
                for hole in session.holes:
                    # One line per distinct reason, not per graph: fourteen
                    # copies of one sentence is how a reason gets skimmed past.
                    reason_class = str(hole.reason).split(" ", 3)[0]
                    if reason_class in seen:
                        continue
                    seen.add(reason_class)
                    reasons.append(
                        f"{hole.record.graph[-16:]}: {str(hole.reason)[:300]}"
                    )
                    logger.warning("adopt:   hole %s", reasons[-1])
                    if len(seen) >= 6:
                        break
                # DURABLE, not log-only (pgw#1564, second lesson): the 09:41
                # zero was diagnosable from this exact list, and the list
                # lived in a resident log; the NEXT instance is on a pod
                # whose SSH is dead two rentals running. The row is what a
                # $0.07 rental reads from the hub instead of buying blind.
                try:
                    from .. import activity as activity_mod
                    from .self_mint import KIND_SKIPPED

                    activity_mod.emit_event(
                        KIND_SKIPPED,
                        f"lane={lane_contract} sm={sm}: boot armed ZERO of "
                        f"{len(session.adopted) + len(session.holes)} claimed "
                        f"graph(s) ({len(session.holes)} hole(s), "
                        f"{len(session.ambiguous)} ambiguous, "
                        f"{len(marks)} unmatched mark(s)); hole reasons: "
                        + ("; ".join(reasons) or "none recorded"),
                        phase="armed_zero",
                        step=0,
                        total_steps=len(session.adopted) + len(session.holes),
                    )
                except Exception:  # noqa: BLE001 — the row never costs the boot
                    logger.debug("adopt: armed-zero row failed to emit",
                                 exc_info=True)
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
            # pgw#1421: engine-hosted teardown is STRUCTURAL, not remembered.
            # An external engine subprocess (`llama-server`, `vllm serve`) is
            # invisible to torch's allocator, so an unload that forgot to stop
            # it — or raised before reaching the stop — leaves VRAM the next
            # admit can neither see nor reclaim. This runs whatever `unload`
            # did, and `EngineHandle.stop` is idempotent so an author who did
            # stop it pays nothing.
            instance.load_context.stop_engines()
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
        # pgw#1579/pgw#1580: the local host builds a context BEFORE it knows
        # the function, so the spec's declarations are stamped here instead of
        # at construction. Without this the CLI and the daemon refuse
        # `ctx.execution_lane` and `ctx.call_endpoint` on an endpoint that
        # declared both — a declaration only the serverless dispatcher honours
        # is one an author cannot test.
        ctx._declare_from_spec(spec)
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
