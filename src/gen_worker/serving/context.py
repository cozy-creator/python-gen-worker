from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Callable, Generic, List, Mapping, Optional, Protocol,
    Sequence, Tuple, Type, TypeVar, cast,
)

import msgspec

from ..families.base import GenerationDefaults
from ..request_context import RequestContext as _BaseRequestContext
from ..request_context import _PublisherMixin
from .deltas import frame_of

if TYPE_CHECKING:
    from ..models.defaults_decode import CarriesDefaults
    from ..models.model_types import ModelType
    from ..models.refs import WireRef

P = TypeVar("P")
D = TypeVar("D", bound=GenerationDefaults)
DT = TypeVar("DT", bound=msgspec.Struct)
MT_co = TypeVar("MT_co", covariant=True)

logger = logging.getLogger(__name__)

_DTYPE_REJECTIONS = ("torch_dtype", "'dtype'", '"dtype"')


def _torch_dtype_from_name(name: str) -> Any:
    try:
        import torch
    except Exception:  # pragma: no cover - torch-free serve role
        return None
    candidate = getattr(torch, name.replace("torch.", "", 1), None)
    return candidate if isinstance(candidate, torch.dtype) else None


def _lane_torch_dtype(lane: Any, *, checkpoint_dir: Any = None) -> Any:
    """The lane's load dtype as a REAL ``torch.dtype``, or None.

    Shared by both eager bridges (`serving/context.py` and
    `release/trace_context.py`) so the two cannot drift — pgw#1447 already
    proved they drift when they are two copies.

    ``checkpoint_dir`` (pgw#1488) is the fallback source for a lane that
    declares no dtype — a DERIVED lane, which by construction has no contract
    to read one from. Precision is graph identity, so "no answer" is not an
    outcome the trace can take: the checkpoint's own dtype is read instead
    (``serving.checkpoint_dtype``). Both bridges pass their tree, so the trace
    and the serve resolve the SAME precision for the same lane.

    Prefers a ``torch_dtype`` attribute (a real object) over ``dtype`` (the
    SPELLING), and resolves a spelling through ``_torch_dtype_from_name``.
    pgw#1448 is why the order matters: diffusers REFUSES a string
    ``torch_dtype`` with a scroll-past warning and falls back to fp32, so a
    spelling handed straight to a kwargs-accepting pipeline loaded every model
    at the wrong precision, silently.

    pgw#1621: a DECLARED lane now carries the quant RULE's ``declared_dtype``
    (``"bfloat16"``, ``"float8_e4m3fn"``) and cannot be dtype-less — the rule
    schema requires the field and there are eight rules for the fleet. The
    protective arms below are kept anyway, because the OTHER caller is a
    DERIVED lane, which by construction has no rule to read a dtype from and
    falls through to the checkpoint's own precision.
    """
    from .checkpoint_dtype import checkpoint_dtype

    if lane is None:
        return checkpoint_dtype(checkpoint_dir)
    for attr in ("torch_dtype", "dtype"):
        try:
            value = getattr(lane, attr)
        except AttributeError:
            continue
        except Exception:
            logger.info(
                "ctx.load: lane %r declares no load dtype; the eager bridge "
                "takes the checkpoint's own", lane,
            )
            return checkpoint_dtype(checkpoint_dir)
        if value is None:
            continue
        if isinstance(value, str):
            resolved = _torch_dtype_from_name(value)
            if resolved is not None:
                return resolved
            logger.warning(
                "ctx.load: lane %r names dtype %r, which does not resolve to a "
                "torch dtype; loading in the checkpoint's own precision "
                "(pgw#1448)", lane, value,
            )
            return checkpoint_dtype(checkpoint_dir)
        return value
    return checkpoint_dtype(checkpoint_dir)


def _rejected_torch_dtype(exc: TypeError) -> bool:
    text = str(exc)
    return any(token in text for token in _DTYPE_REJECTIONS)


@dataclass(frozen=True, slots=True)
class Adapter:
    """A hub-resolved adapter (LoRA)."""

    name: str
    path: Path
    defaults: Any = None
    scale: float = 1.0
    ref: str = ""


@dataclass(frozen=True, slots=True)
class DistillationAdapter(Adapter):
    """A distillation adapter — runtime-identical to :class:`Adapter`; its MEANING is the SLOT KIND (Paul's structural guard)."""


@dataclass(slots=True)
class DeployBinding:
    """One deployment's checkpoint binding — hub deploy state, rebindable."""

    checkpoint_ref: str
    checkpoint_dir: Path
    model: Optional[str] = None
    defaults: Mapping[str, Any] = field(default_factory=dict)
    adapter: Optional[Adapter] = None
    lane_trees: Mapping[str, Path] = field(default_factory=dict)
    lane_verdicts: Mapping[str, str] = field(default_factory=dict)
    lane_bytes: Mapping[str, int] = field(default_factory=dict)


class LoaderEngine(Protocol):

    def build(
        self, pipeline_cls: type, *, checkpoint_dir: Path, lane: Any
    ) -> Any: ...


class DefaultsError(msgspec.ValidationError):
    """A hub defaults row does not fit the model type's schema."""


class ProjectedTreeNotStreamable(RuntimeError):
    """The eager bridge was handed a tree whose weights are POINTERS."""

    def __init__(
        self,
        tree: Path,
        stubs: Sequence[Tuple[str, int, int]],
        declined: str = "",
    ) -> None:
        self.tree = Path(tree)
        self.stubs = list(stubs)
        self.declined = declined
        shown = ", ".join(
            f"{path} ({on_disk} B on disk, names {named:,} B)"
            for path, on_disk, named in self.stubs[:3]
        )
        more = "" if len(self.stubs) <= 3 else f" (+{len(self.stubs) - 3} more)"
        repair = _projection_pin_outcome(self.tree)
        super().__init__(
            f"ENGINE DECLINED: {declined or 'unknown'} "
            f"| repair attempted: {repair} "
            f"| PROJECTED TREE, {len(self.stubs)} pointer stub(s), NOT weights: "
            f"{shown}{more} "
            f"| tree={self.tree} "
            f"| The eager `from_pretrained` bridge reads with the stock "
            f"safetensors reader and would report `header too large`, which "
            f"describes a corrupt checkpoint — and this checkpoint is NOT "
            f"corrupt: its bytes are in the CAS. The streaming engine "
            f"(pgw#1380) is the reader for this tree and declined to bind. "
            f"Refusing rather than serving a lie about the weights."
        )


def _projection_pin_outcome(tree: Path) -> str:
    try:
        from ..models.projection import pin_outcome

        return pin_outcome(Path(tree).name) or "not attempted"
    except Exception:  # noqa: BLE001
        return "unknown"


def _serving_device() -> str:
    try:
        from .placement import serving_device

        return serving_device()
    except Exception:  # noqa: BLE001 — an unprobeable host is not a refusal
        return "cuda"


def _projection_pinned(tree: Path) -> bool:
    from ..models import projection

    try:
        return projection.resolve_projection(tree) is not None
    except Exception:  # noqa: BLE001 — a probe never fails a load
        return False


def _projection_declined_because(
    tree: Path, *, pinned: Optional[bool] = None
) -> str:
    from ..models.projection import SNAPSHOTS_DIR

    root = Path(tree)
    if root.parent.name != SNAPSHOTS_DIR:
        return (
            f"the tree is at {root}, whose parent directory is "
            f"{root.parent.name!r} and not {SNAPSHOTS_DIR!r} — "
            f"`resolve_projection` locates the store by walking UP from the "
            f"tree, so a correctly-built tree in the wrong place is invisible "
            f"to it"
        )
    base = root.parent.parent
    missing = [d for d in ("refs", "objects") if not (base / d).is_dir()]
    if missing:
        return (
            f"the store root {base} has no {'/'.join(missing)} directory, so "
            f"there is no CAS behind this tree"
        )
    if pinned is None:
        pinned = _projection_pinned(root)
    if pinned:
        return (
            f"the manifest pin `snapshot:{root.name}` RESOLVES at {base} and "
            f"this tree is STREAMABLE — the store is not the problem. No "
            f"streaming engine was bound for this load, so the eager bridge "
            f"was handed a projected tree. That is a WIRING defect on the path "
            f"that built this load context, not a missing pin (pgw#1544)"
        )
    return (
        f"the manifest pin `snapshot:{root.name}` is MISSING from the store at "
        f"{base}. The tree and its bytes are fine; what is absent is the ref "
        f"that lets a reader recover the manifest, and it is keyed on the "
        f"tree's own DIRECTORY NAME. Re-materializing this ref re-pins it "
        f"without moving any bytes"
    )


def _projection_artifacts(tree: Path) -> List[Tuple[str, int, int]]:
    from ..models import projection

    found: List[Tuple[str, int, int]] = []
    root = Path(tree)
    if not root.is_dir():
        return found
    for path in sorted(root.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        stub = projection.stub_at(path)
        if stub is None:
            continue
        try:
            on_disk = path.lstat().st_size
        except OSError:
            on_disk = 0
        found.append((path.relative_to(root).as_posix(), on_disk, int(stub.size)))
    return found


class LoadContext(Generic[MT_co]):
    """What ``Model.load``/``Model.unload`` receive — the load moment."""

    def __init__(
        self,
        *,
        binding: DeployBinding,
        model_type: Optional[type] = None,
        lane: Any = None,
        resolved: Any = None,
        engine: Optional[LoaderEngine] = None,
        compile_sink: Optional[Callable[[Any], Any]] = None,
        device: str = "",
        io: str = "buffered",
        weight_budget_bytes: int = 0,
    ) -> None:
        self._binding = binding
        self._model_type = model_type
        self._lane = lane
        self._resolved = resolved
        self._engine = engine
        self._compile_sink = compile_sink
        self._engines: list[Any] = []
        self._device = str(device or "")
        self._io = str(io or "buffered")
        self._engaged_rung = ""
        self._weight_budget_bytes = max(0, int(weight_budget_bytes))

    @property
    def loader_engine(self) -> Optional[LoaderEngine]:
        """The streaming engine THIS load actually bound, or ``None``."""
        return self._engine

    @property
    def checkpoint_dir(self) -> Path:
        """The worker-resolved checkpoint tree for the RESOLVED lane."""
        tree = self._lane_tree()
        return Path(tree if tree is not None else self._binding.checkpoint_dir)

    def _lane_tree(self) -> Optional[Path]:
        resolved = self._resolved
        if resolved is None:
            return None
        trees = getattr(self._binding, "lane_trees", None) or {}
        if not trees:
            return None
        for key in (getattr(resolved, "fetch_contract", ""),
                    getattr(resolved, "contract_id", "")):
            if key and key in trees:
                return Path(trees[key])
        return None

    @property
    def resolved_lane(self) -> Any:
        """The boot ladder's answer, or ``None`` if the ladder never ran."""
        return self._resolved

    @property
    def checkpoint_ref(self) -> str:
        """The hub ref of the bound checkpoint."""
        return self._binding.checkpoint_ref

    @property
    def lane(self) -> Any:
        """The active lane (the deploy's pick) — a tensorfs layout contract; ``ctx.lane.dtype`` is the load dtype."""
        if self._lane is None:
            raise RuntimeError(
                "ctx.lane: no active lane was bound for this load. Every "
                "model class declares real lanes (pgw#1599), so this is a "
                "platform wiring defect, not an authored state — the caller "
                "that built this LoadContext passed no `lane=`."
            )
        return self._lane

    def _bind_streaming_engine(self, *, pinned: bool) -> Tuple[bool, bool]:
        try:
            from ..models.store import active_store
            from . import streaming

            if not pinned:
                store = active_store()
                if store is None:
                    return False, pinned
                ref = cast("WireRef", self.checkpoint_ref)
                snapshot = store.banked_snapshot_for_tree(
                    self.checkpoint_dir.name)
                if not store.ensure_pinned(ref, self.checkpoint_dir, snapshot):
                    return False, pinned
                pinned = True
            engine = streaming.engine_for(
                self.checkpoint_dir,
                device=self._device or _serving_device(),
                io=self._io or "buffered",
            )
            if engine is None:
                return False, pinned
            self._engine = engine
            return True, pinned
        except Exception:  # noqa: BLE001 — this must never mask the refusal
            logger.warning(
                "pgw#1544: binding the streaming engine raised for %s; falling "
                "through to the refusal", self.checkpoint_dir, exc_info=True,
            )
            return False, pinned

    def load_pipeline(self, pipeline_cls: Type[P]) -> P:
        built = self.load(pipeline_cls)
        resolved = self._resolved
        if resolved is None:
            logger.info(
                "ctx.load_pipeline: no lane was resolved for this context "
                "(fixture/derive/local run) — built %s with no lane "
                "materialization (pgw#1606)", pipeline_cls.__name__,
            )
            return built
        from . import lane_materialize

        lane_materialize.materialize(
            built, resolved, tree=self.checkpoint_dir,
            compute_dtype=self._lane_dtype(),
        )
        logger.info("ctx.load_pipeline: %s", resolved.confession())
        return built

    def load(self, pipeline_cls: Type[P]) -> P:
        """Build ``pipeline_cls`` with this checkpoint's weights resident."""
        if self._engine is not None:
            built: P = self._engine.build(
                pipeline_cls,
                checkpoint_dir=self.checkpoint_dir,
                lane=self._lane,
            )
            return built
        from_pretrained = getattr(pipeline_cls, "from_pretrained", None)
        if from_pretrained is None:
            raise RuntimeError(
                f"ctx.load({pipeline_cls.__name__}): no loader engine is "
                "bound (pgw#1380 lands the native store->VRAM engine here) "
                "and the class has no from_pretrained for the eager bridge"
            )
        logger.info(
            "ctx.load: eager from_pretrained bridge for %s (pgw#1380's "
            "native loader engine is not bound)", pipeline_cls.__name__,
        )
        from ..models import projection

        collected = projection.collected_entries(self.checkpoint_dir)
        if collected:
            raise ProjectedTreeNotStreamable(
                self.checkpoint_dir,
                _projection_artifacts(self.checkpoint_dir),
                projection.collected_refusal(self.checkpoint_dir, collected),
            )
        stubbed = _projection_artifacts(self.checkpoint_dir)
        if stubbed:
            pinned = _projection_pinned(self.checkpoint_dir)
            bound, pinned = self._bind_streaming_engine(pinned=pinned)
            if bound:
                engine = self._engine
                assert engine is not None
                rebound: P = engine.build(
                    pipeline_cls,
                    checkpoint_dir=self.checkpoint_dir,
                    lane=self._lane,
                )
                return rebound
            raise ProjectedTreeNotStreamable(
                self.checkpoint_dir,
                stubbed,
                _projection_declined_because(self.checkpoint_dir, pinned=pinned),
            )
        from .variants import detect_variant

        extra: dict[str, Any] = {}
        variant = detect_variant(self.checkpoint_dir)
        if variant is not None:
            extra["variant"] = variant
        dtype = self._lane_dtype()
        if dtype is None:
            no_lane: P = from_pretrained(self.checkpoint_dir, **extra)
            return self._placed(no_lane)
        try:
            bridged: P = from_pretrained(
                self.checkpoint_dir, torch_dtype=dtype, **extra)
            return self._placed(bridged)
        except TypeError as exc:
            if not _rejected_torch_dtype(exc):
                raise
        logger.info(
            "ctx.load: %s.from_pretrained does not accept torch_dtype; "
            "loading without it and applying the lane dtype post-load "
            "(pgw#1447)", pipeline_cls.__name__,
        )
        loaded: P = from_pretrained(self.checkpoint_dir, **extra)
        to = getattr(loaded, "to", None)
        if callable(to):
            to(dtype)
        return self._placed(loaded)

    def _placed(self, pipeline: P) -> P:
        if not self._device:
            return pipeline
        to = getattr(pipeline, "to", None)
        if not callable(to):
            return pipeline
        from ..models.rung import touches_host_ram

        mode = self._fit_rung(pipeline)
        if mode and touches_host_ram(mode):
            logger.info(
                "ctx.load: %r engaged pre-placement; the rung places its own "
                "components (pgw#1486)", mode,
            )
            return pipeline
        logger.info(
            "ctx.load: placing the bridged pipeline on %s — the worker's "
            "decision, inherited (pgw#1452); fit rung %r",
            self._device, mode or "not consulted",
        )
        moved = to(self._device)
        return pipeline if moved is None else moved

    def _fit_rung(self, pipeline: P) -> str:
        try:
            import torch

            if torch.device(self._device).type != "cuda":
                return ""
        except Exception:
            return ""
        from ..api.errors import HostRamMoveRefusedError

        try:
            from ..models.memory import apply_low_vram_config

            applied = apply_low_vram_config(
                pipeline,
                mode="auto",
                stream_budget_bytes=self._weight_budget_bytes,
                # pgw#1627: COMPILE INTENT, for the regime-split headroom
                # probe. A bound compile sink is what makes `ctx.compile`
                # consult the graph store at all, so it IS "this load may
                # serve compiled" — and it is the only signal available at
                # admission: adopt arms dispatchers AFTER load, so there is
                # nothing armed to interrogate yet (the forensics ordering
                # proof). The compiled demand stamp (pgw#1601) is not wired
                # yet; until it is, the compiled refusal is inert and the
                # basis confession says which arithmetic ran.
                regime=(
                    "compiled" if self._compile_sink is not None else "eager"
                ),
            )
        except HostRamMoveRefusedError:
            raise
        except Exception as exc:
            logger.warning(
                "ctx.load: the offload ladder could not size this pipeline "
                "(%s: %s); placing it whole, as before (pgw#1486)",
                type(exc).__name__, exc,
            )
            return ""
        mode = str(applied.get("mode") or "")
        self._engaged_rung = mode
        return mode

    def _lane_dtype(self) -> Any:
        """The lane's load dtype as a real ``torch.dtype``, or None.

        pgw#1448 — IT MUST RETURN A ``torch.dtype``, NOT A STRING. A lane's
        dtype is a SPELLING (``'bfloat16'``); diffusers REFUSES a string
        ``torch_dtype`` with a warning and falls back to **fp32**, so every
        kwargs-accepting pipeline loaded through the eager bridge was loading
        at the wrong precision — silently. Measured by the local lane on a
        4070: sd1.5 at **13 s/it**, ~20-40x off, because fp32 doubles the
        weights on a 7.63 GiB card. **The precision bug IS the performance
        bug**, and it is invisible: a scroll-past warning, then a model that
        works and is slow.

        FLEET CONSEQUENCE: any timing ever taken through this bridge was fp32
        timing and must be re-baselined, not compared.

        pgw#1386's no-dtype arm survives in ``_lane_torch_dtype`` for the
        DERIVED-lane caller. It no longer has a declared-lane case to protect:
        under tensor-layout v2 a lane's dtype is the quant rule's
        ``declared_dtype``, which the rule schema requires.
        """
        return _lane_torch_dtype(self._lane, checkpoint_dir=self.checkpoint_dir)

    def engine(self, spec: Any) -> Any:
        from .engine_runtime import EngineSpec, boot_engine

        if not isinstance(spec, EngineSpec):
            raise TypeError(
                f"ctx.engine({spec!r}): expected an EngineSpec declaration "
                "(gen_worker.LlamaServer / gen_worker.VllmServer), not a "
                "command or a base URL. The spec states WHAT engine and WHICH "
                "flags; the checkpoint tree, the port and the supervision are "
                "the platform's half."
            )
        handle = boot_engine(spec, self.checkpoint_dir)
        self._engines.append(handle)
        return handle

    @property
    def engines(self) -> tuple[Any, ...]:
        """Every engine this context started, in boot order."""
        return tuple(self._engines)

    def stop_engines(self) -> None:
        """Stop every engine this context started, newest first."""
        while self._engines:
            handle = self._engines.pop()
            try:
                handle.stop()
            except Exception:
                logger.exception(
                    "stopping engine %r raised; continuing with the rest",
                    getattr(handle, "runtime", handle),
                )

    def compile(self, target: Any) -> Any:
        """Imperative module marking, the torch.compile idiom (Paul ruling):: self.pipe.unet = ctx.compile(self.pipe.unet) Everything behind the call is torchcg (tcg#42): at publish the hollow session records..."""
        if self._compile_sink is None:
            return target
        from ..models.partial_resident import parks_module
        from ..models.rung import moves_every_component, touches_host_ram

        rung = self._engaged_rung
        if moves_every_component(rung):
            logger.warning(
                "ctx.compile: serving EAGER for %s — the %r offload rung is "
                "engaged, and a compiled graph cannot bind constants to "
                "weights accelerate moves between host and device per forward "
                "(pgw#1486). Fit the model on the card to get compiled speed.",
                type(target).__name__, rung,
            )
            return target
        if touches_host_ram(rung):
            if parks_module(target):
                logger.warning(
                    "ctx.compile: serving EAGER for %s — the %r rung parks "
                    "this component, so its weights move between pinned host "
                    "RAM and the card per forward and a compiled graph cannot "
                    "bind constants to them (pgw#1587). The components the "
                    "plan left resident still compile.",
                    type(target).__name__, rung,
                )
                return target
            logger.info(
                "ctx.compile: compiling %s UNDER the %r rung — this target is "
                "in the plan's resident set, so its device pointers are stable "
                "for the life of the load; the components the rung parked are "
                "what made room for it (pgw#1587).",
                type(target).__name__, rung,
            )
        return self._compile_sink(target)

    def defaults(self: "LoadContext[ModelType[DT]]") -> DT:
        """THIS checkpoint's defaults, typed via the model class's generic: the hub's per-checkpoint row decoded as a field-level overlay on the model type's ``Defaults`` struct (zero-arg = platform fallbacks..."""
        from ..models.defaults_decode import (
            DefaultsDecodeError,
            ModelTypeMismatch,
            decode_model_defaults,
        )

        schema = getattr(self._model_type, "Defaults", None)
        if self._model_type is None or schema is None:
            raise RuntimeError(
                "ctx.defaults(): no model type is bound to this LoadContext "
                "(the model class header's generic parameter carries it)"
            )
        carrier = cast("CarriesDefaults[DT]", self._model_type)
        try:
            decoded: DT = decode_model_defaults(
                carrier,
                model=self._binding.model,
                defaults=self._binding.defaults,
            )
        except (DefaultsDecodeError, ModelTypeMismatch) as exc:
            raise DefaultsError(
                f"per-checkpoint defaults for "
                f"{self._binding.checkpoint_ref!r}: {exc}"
            ) from None
        return decoded


class RequestContext(_PublisherMixin, _BaseRequestContext[D]):
    """What entrypoints receive — request facts + the salvaged base surface."""

    def __init__(
        self,
        request_id: str,
        *,
        binding: Optional[DeployBinding] = None,
        streams: Optional[Tuple[type, ...] | type] = None,
        chunk_sink: Optional[Callable[[bytes, str], None]] = None,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(request_id, **base_kwargs)
        self._serve_binding = binding
        self._mktemp_root: Optional[Path] = None
        self._delta_types: Tuple[type, ...] = (
            () if streams is None
            else (tuple(streams) if isinstance(streams, tuple) else (streams,))
        )
        self._chunk_sink = chunk_sink

    def emit(self, chunk: Any) -> None:
        declared = self._delta_types
        if not declared:
            raise RuntimeError(
                "ctx.emit: this entrypoint declared no chunk type, so the hub "
                "publishes incremental_output=false for it and no client will "
                "subscribe. Declare @entrypoint(streams=<msgspec.Struct>) "
                "(pgw#1576)"
            )
        if not isinstance(chunk, declared):
            names = " | ".join(arm.__name__ for arm in declared)
            raise TypeError(
                f"ctx.emit: this entrypoint declared streams={names} and "
                f"delta_output_schema was published from it; got "
                f"{type(chunk).__name__}"
            )
        sink = self._chunk_sink
        if sink is None:
            logger.debug(
                "ctx.emit dropped for %s: no chunk sink (local run)",
                self.request_id,
            )
            return
        data, content_type = frame_of(chunk)
        sink(data, content_type)

    def mktemp(self) -> Path:
        """A request-scoped scratch directory."""
        if self._mktemp_root is None:
            self._mktemp_root = Path(
                tempfile.mkdtemp(
                    prefix=f"txform-{self.request_id or 'x'}-",
                    dir=tempfile.gettempdir(),
                )
            )
        return Path(tempfile.mkdtemp(dir=str(self._mktemp_root)))

    @property
    def checkpoint_ref(self) -> str:
        """The RESOLVED, fully-pinned hub ref of the checkpoint that actually served this request — honest structured output evidence (the ``model`` field of an output struct)."""
        if self._serve_binding is None:
            raise RuntimeError(
                "ctx.checkpoint_ref: no deploy binding rides this context "
                "(local construction without binding=)"
            )
        return self._serve_binding.checkpoint_ref

    def step_callback(self, num_inference_steps: int, **kwargs: Any) -> Any:
        """A diffusers ``callback_on_step_end`` wired to progress + cancel — the context spelling of :func:`gen_worker.diffusers_step_callback`."""
        from ..api.progress import diffusers_step_callback

        return diffusers_step_callback(self, num_inference_steps, **kwargs)

    def warn(self, message: str) -> None:
        """A caller-visible WARNING into the response envelope — the generalization of clamp visibility, same delivery path (the adjustment ledger rides ``JobResult.adjustments`` + the hub's request record/ev..."""
        self.adjusted("", None, None, str(message))

    @property
    def warnings(self) -> tuple[str, ...]:
        """The accumulated :meth:`warn` messages, in emission order (the field-less rows of the adjustment ledger)."""
        return tuple(
            row["reason"] for row in self.adjustments if not row["field"]
        )


__all__ = [
    "Adapter",
    "DefaultsError",
    "DistillationAdapter",
    "DeployBinding",
    "LoadContext",
    "LoaderEngine",
    "RequestContext",
]
