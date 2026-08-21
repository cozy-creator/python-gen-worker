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
    """Deploy state per (model class, checkpoint pick) — the hub's half."""

    def resolve(self, model_cls: type, checkpoint_ref: str) -> DeployBinding: ...

    def default_pick(self, model_cls: type, slot_name: str) -> str: ...


@dataclass(frozen=True, slots=True)
class InvokeOutcome:
    """One served request: the entrypoint's result + the request facts the envelope reply carries (``ctx.warn`` rows, adjustments, stage timings)."""

    result: Any
    warnings: Tuple[str, ...]
    adjustments: Tuple[Dict[str, str], ...]
    stages: Optional[StageTimer] = None


@dataclass(frozen=True, slots=True)
class WarmPass:
    """What the boot warm pass did about ONE entrypoint."""

    function: str
    outcome: str
    reason: str = ""
    duration_ms: int = 0

    @property
    def warmed(self) -> bool:
        return self.outcome == WARM_OK


WARM_OK = "warmed"
WARM_SKIPPED = "skipped"
WARM_FAILED = "failed"


class _InstanceBackend:

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
        self._on_loaded = on_loaded
        self._stream_residency: Optional[Any] = None
        self.degraded: Tuple[str, ...] = ()

    def load(self) -> None:
        self.degraded = warn_if_degraded(self.model_cls, self.lane)
        model: Model[Any] = self.model_cls()
        model.load(self.load_context)
        self.model = model
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
        if self._stream_residency is not None:
            return self._stream_residency
        if self.model is None:
            raise ResidencyError(
                f"{self.model_cls.__name__}: a tier move was ordered on an "
                f"instance that holds no model object"
            )
        from ..models.memory import stream_residency_of
        from ..models.stream_residency import StreamedResidency

        for component in (getattr(self.model, "pipe", None), self.model):
            armed = stream_residency_of(component) if component is not None else None
            if armed is not None:
                self._stream_residency = armed
                return armed

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
        self._lane_contract = str(lane_contract or "")
        self.lanes: Dict[type, Any] = {
            cls: self._single_lane(cls) for cls in loaded.models
        }
        self.declared_lanes: Dict[type, Tuple[Any, ...]] = {
            cls: tuple(loaded.lanes_of(cls)) for cls in loaded.models
        }
        self._resolved: Dict[Tuple[type, str], Any] = {}
        self._compile_sink_for = compile_sink_for
        self._on_loaded = on_loaded
        self._output_dir = output_dir
        self._context_kwargs = dict(context_kwargs or {})
        self._hf_token = str(hf_token or "")
        self._backends: Dict[Tuple[type, str, str], _InstanceBackend] = {}
        self._backends_lock = threading.Lock()

    def _single_lane(self, model_cls: type) -> Any:
        try:
            return self.loaded.lane(model_cls, self._lane_contract)
        except EndpointLoadError:
            return None

    def _resolve_for(self, model_cls: type, binding: DeployBinding) -> Any:
        key = (model_cls, str(binding.checkpoint_ref))
        hit = self._resolved.get(key)
        if hit is not None:
            return hit
        from .lane_host import BindingVerdicts, HostKernelGates, host_card_facts

        verdicts = BindingVerdicts.of(binding)
        single = self.lanes.get(model_cls)
        if single is not None:
            verdicts = verdicts.for_single_lane(
                lane_handle(single), binding.checkpoint_dir)
        resolved = self.loaded.resolve(
            model_cls, card=host_card_facts(), verdicts=verdicts,
            gates=HostKernelGates(), contract=self._lane_contract,
        )
        if resolved is not None:
            logger.info("lane ladder: %s", resolved.confession())
            try:
                from .. import activity

                activity.emit_event(
                    activity.KIND_APPLIED_LANE,
                    f"{model_cls.__name__} <- {binding.checkpoint_ref}: "
                    f"{resolved.confession()}",
                    phase=str(resolved.body or "?"),
                    family=str(resolved.contract_id or ""),
                )
            except Exception:  # noqa: BLE001 — a confession never fails a boot
                logger.debug("lane ladder: could not emit applied_lane",
                             exc_info=True)
            self._resolved[key] = resolved
        return resolved

    def resolved_lane_for(self, model_cls: type, binding: DeployBinding) -> Any:
        """The ladder's pick for this (class, binding), if it has run."""
        return self._resolved.get((model_cls, str(binding.checkpoint_ref)))

    def _lane_of(self, model_cls: type) -> Tuple[Any, str]:
        lane = self.lanes[model_cls]
        return lane, (lane_handle(lane) if lane is not None else "eager")

    def _backend_factory(
        self, model_cls: type, binding: DeployBinding, key: Tuple[type, str, str]
    ) -> Callable[[], _InstanceBackend]:
        def make() -> _InstanceBackend:
            lane, _ = self._lane_of(model_cls)
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
                worker_load_context(
                    binding=binding,
                    model_type=model_type(model_cls),
                    lane=lane,
                    resolved=resolved,
                    engine=self._engine,
                    compile_sink=sink,
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
        """Serve one request end-to-end."""
        spec = self.loaded.entrypoints.get(function)
        if spec is None:
            raise ServeDispatchError(
                f"{self.loaded.module_name} serves no function {function!r} "
                f"(functions: {sorted(self.loaded.entrypoints)})"
            )
        model_slots = dict(spec.model_params)
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
                cleanup_input_assets(request_id, int(attempt))
                raise
        ctx._stages.record_pre("input_fetch", time.monotonic() - input_fetch_t0)

        source_fetch_t0 = time.monotonic()
        materialize_reserved_inputs(ctx, decoded.payload, snapshots or {})
        ctx._stages.record_pre(
            "source_fetch", time.monotonic() - source_fetch_t0
        )

        with ExitStack() as leases:
            models: Dict[str, Model[Any]] = {}
            degraded: list[str] = []
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
            if spec.model_params:
                primary_cls = spec.model_params[0][1]
                primary_lane, primary_handle = self._lane_of(primary_cls)
                if primary_lane is not None:
                    ctx._set_execution_lane(primary_handle)
                if primary_binding is not None:
                    ctx._set_resolved_lane(
                        self.resolved_lane_for(primary_cls, primary_binding))

            ctx._stages.handler_open()
            try:
                result = spec.fn(ctx, decoded.payload, *arguments)
                if inspect.iscoroutine(result):
                    result = asyncio.run(result)
            finally:
                ctx._stages.handler_close()
        return InvokeOutcome(
            result=result,
            warnings=ctx.warnings,
            adjustments=ctx.adjustments,
            stages=ctx._stages,
        )

    def boot_warmup(
        self, *, prepare: Optional[Callable[[str], str]] = None
    ) -> Tuple[WarmPass, ...]:
        """One synthetic invocation per entrypoint, at boot, before servable."""
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
        if not spec.model_params:
            return WarmPass(function, WARM_SKIPPED,
                            "weightless: nothing is resident to warm")
        kind = str(getattr(spec, "kind", "") or "inference")
        if kind != "inference":
            return WarmPass(function, WARM_SKIPPED,
                            f"kind={kind!r}: only inference pays a first-call tax")
        payload, reason = neutral_payload(spec.payload_type, scratch)
        if payload is None:
            return WarmPass(function, WARM_SKIPPED, reason)
        if prepare is not None:
            declined = str(prepare(function) or "")
            if declined:
                return WarmPass(function, WARM_SKIPPED, declined)
        envelope = {"input": msgspec.to_builtins(payload)}
        started = time.monotonic()
        try:
            with boot_phases.span(boot_phases.PHASE_WARMUP, function=function):
                self.invoke(
                    function,
                    envelope,
                    request_id=f"boot-warmup-{function}",
                    context={
                        "boot_warmup": True,
                        "local_output_dir": scratch,
                    },
                )
        except (KeyboardInterrupt, SystemExit):
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
        kwargs.update({k: v for k, v in (per_request or {}).items() if v is not None})
        if self._output_dir is not None:
            kwargs.setdefault("local_output_dir", str(self._output_dir))
        if self._hf_token:
            kwargs.setdefault("hf_token", self._hf_token)
        if payload is not None:
            kwargs.update(reserved_context_kwargs(payload))
        if spec is not None:
            kwargs["publishes"] = bool(getattr(spec, "publishes", False))
            kwargs["streams"] = getattr(spec, "delta_arms", ()) or None
            declared_media = getattr(spec, "emits_media", None)
            if declared_media is not None:
                kwargs["emits_media"] = bool(declared_media)
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
    """An :class:`InstanceSizer` over a static ref->bytes table."""

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
