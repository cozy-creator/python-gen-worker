from __future__ import annotations

import logging
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

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
    from ..cli.workspace import artifacts_root

    return artifacts_root()



class ServeDispatchError(RuntimeError):
    """A request names a function this endpoint does not serve."""


class ModelInstance:
    """One resident model: the author's object + its admission + load ctx."""

    def __init__(self, model: Model[Any], load_context: LoadContext[Any]) -> None:
        self.model = model
        self.load_context = load_context
        self.admission = threading.Lock()


class EndpointHost:
    """One loaded endpoint, booted and routable."""

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
        device = str(device or "") or serving_device()
        self.loaded = loaded
        self.binding = binding
        self.lanes: Dict[type, Any] = {
            cls: loaded.lane(cls, lane_contract) for cls in loaded.models
        }
        self.instances: Dict[type, ModelInstance] = {}
        self.degraded: Dict[type, Tuple[str, ...]] = {}
        self.adoption: Any = None
        self._booted = False
        self._engine = engine
        self._io = str(io or "buffered")
        self._device = str(device or "")
        self._output_dir = output_dir
        self._context_kwargs = dict(context_kwargs or {})

    def _stream_attributes(self, ctx: LoadContext[Any]) -> Dict[str, Any]:
        report = getattr(ctx.loader_engine, "last_report", None)
        if report is None:
            return {}
        attributes: Dict[str, Any] = report.attributes()
        return attributes

    def make_context(self, request_id: str, **overrides: Any) -> RequestContext[Any]:
        kwargs: Dict[str, Any] = dict(self._context_kwargs)
        if self._output_dir is not None:
            kwargs.setdefault("local_output_dir", str(self._output_dir))
        kwargs.update(overrides)
        ctx: RequestContext[Any] = RequestContext(
            request_id, binding=self.binding, **kwargs
        )
        for warnings in self.degraded.values():
            for warning in warnings:
                ctx.warn(warning)
        return ctx

    def _load_context(
        self, model_cls: type, *, compile_sink: Any = None
    ) -> LoadContext[Any]:
        return worker_load_context(
            binding=self.binding,
            model_type=model_type(model_cls),
            lane=self.lanes[model_cls],
            engine=self._engine,
            compile_sink=compile_sink,
            device=self._device,
            io=self._io,
        )

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
        """Instantiate each referenced Model class and run its ``load(ctx)``."""
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
                    artifacts_dir=artifacts_dir or _artifacts_root(),
                    stack=stack_rows,
                )
            except EnvironmentMismatch as exc:
                span(graphs_from="release", lane=lane_contract,
                     refusal="environment_mismatch")
                logger.error("adopt refused: %s", exc)
                raise

        for model_cls in self.loaded.models:
            load_started = time.monotonic()
            self.degraded[model_cls] = warn_if_degraded(
                model_cls, self.lanes[model_cls]
            )
            model: Model[Any] = model_cls()
            load_context = self._load_context(
                model_cls,
                compile_sink=(
                    adapter_guard.sink(session.adopt)
                    if session is not None else None
                ),
            )
            model.load(load_context)
            self.instances[model_cls] = ModelInstance(model, load_context)
            boot_stages.record_ending_now(
                boot_stages.Stage.MODEL_LOAD,
                duration_ms=int((time.monotonic() - load_started) * 1000),
                label=f"{self.loaded.module_name}:{model_cls.__name__}",
                checkpoint=self.binding.checkpoint_ref,
                **self._stream_attributes(load_context),
            )
        if session is not None:
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

    def evict(self, model_cls: type) -> None:
        """Evict one instance: DRAIN (acquire the admission — single-flight defines drained), call the author's ``unload(ctx)``, drop the reference."""
        instance = self.instances.get(model_cls)
        if instance is None:
            return
        with instance.admission:
            try:
                instance.model.unload(instance.load_context)
            except Exception:
                logger.exception(
                    "unload(%s) raised; eviction proceeds (best-effort, "
                    "never correctness)", model_cls.__name__,
                )
            instance.load_context.stop_engines()
            del self.instances[model_cls]

    def teardown(self) -> None:
        """Evict every resident instance (reverse residency order)."""
        for model_cls in list(reversed(list(self.instances))):
            self.evict(model_cls)
        self._booted = False

    def rebind(self, binding: DeployBinding) -> None:
        """Swap the deploy binding (hub deploy state changed)."""
        self.binding = binding

    @property
    def holes(self) -> tuple[Any, ...]:
        return tuple(self.adoption.holes) if self.adoption is not None else ()

    def dispatch(
        self,
        function: str,
        payload: Any,
        *,
        request_id: str,
        ctx: Optional[RequestContext[Any]] = None,
        loras: Sequence[Any] = (),
    ) -> Any:
        """Route one request to the named entrypoint — function name IS the route."""
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
        ctx._declare_from_spec(spec)
        arguments = [
            self.instances[slot.annotation].model if slot.kind == "model"
            else list(loras) if slot.kind == "adapters"
            else self._resolve_adapter(function, slot)
            for slot in spec.slots
        ]
        with ExitStack() as admissions:
            for slot_name, model_cls in sorted(spec.model_params):
                admissions.enter_context(self.instances[model_cls].admission)
            return spec.fn(ctx, decoded, *arguments)

    def _resolve_adapter(self, function: str, slot: Any) -> Any:
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
