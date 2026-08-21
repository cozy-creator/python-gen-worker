"""The ctx split (pgw#1382): ``LoadContext[MT]`` + ``RequestContext``.

Each half carries only what its moment can answer for:

* :class:`LoadContext` -> ``Model.load``/``Model.unload``: the checkpoint
  tree, the lane (a tensorfs layout contract), and the three seams —
  ``load()`` (pgw#1380's native store->VRAM engine lands here; until then a
  ``from_pretrained`` eager bridge), ``compile()`` (tcg#42's AdoptSession
  delegation), ``defaults()`` (pgw#1377's typed decode).
* :class:`RequestContext` -> entrypoints: the ``checkpoint_ref`` fact
  (RESOLVED pinned ref of what actually serves) + the salvaged base surface
  (``raise_if_cancelled``, ``save_image``, ``clamp``, ``generator``,
  ``progress``, ``log`` …) + ``step_callback`` + ``warn``.

Adapters are EXPLICIT ENTRYPOINT PARAMETERS (Paul ruling), not ctx facts:
``def generate(ctx, payload, model: SdxlModel, turbo: Adapter | None,
loras: list[Adapter])`` — the hub resolves what rides per deployment/request
into the declared slots; APPLYING them stays a model mutation through the
model-owned scope (``with model.adapters(riding): ...``).

There is deliberately NO trace flag on any ctx (Paul ruling): author code is
trace-oblivious by construction — the publish-time derive varies BINDINGS
(input enumeration: synthesized adapters, cfg-false defaults) and its
harness-private context no-ops the output surfaces; it never asks author
code to cooperate.

Per-checkpoint serving values are MUTABLE PLATFORM DEPLOY STATE (hub DB),
not release metadata: the host holds one :class:`DeployBinding` and stamps
it on each context, so a rebind swaps the binding without touching release
identity — new graphs a rebind introduces are simply holes (partial-hit).
"""

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

#: Substrings that identify a ``TypeError`` raised because the loader (or the
#: class it constructs) would not take ``torch_dtype``. Narrow ON PURPOSE: a
#: TypeError from INSIDE a model's own __init__ must keep propagating, so the
#: retry below fires only when the rejected keyword is named.
_DTYPE_REJECTIONS = ("torch_dtype", "'dtype'", '"dtype"')


def _torch_dtype_from_name(name: str) -> Any:
    """``'bfloat16'`` -> ``torch.bfloat16``, or None if it is not one.

    Guarded with ``isinstance`` because ``getattr(torch, 'load')`` is a
    perfectly good attribute that is not a dtype, and handing a FUNCTION to
    ``torch_dtype=`` would be a worse failure than handing it a string.
    """
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

    Prefers ``Contract.torch_dtype`` (the object) over ``Contract.dtype`` (the
    spelling). Every non-AttributeError read is treated as "this contract
    declares no top-level dtype", which is a LEGAL state tensorfs signals by
    RAISING (``MissingDtype``) rather than answering None — the pgw#1386
    protective arm, kept identical and applied to BOTH spellings, since a
    dtype-less document raises the same way whichever attribute is asked for.
    """
    from .checkpoint_dtype import checkpoint_dtype

    if lane is None:
        return checkpoint_dtype(checkpoint_dir)
    for attr in ("torch_dtype", "dtype"):
        try:
            value = getattr(lane, attr)
        except AttributeError:
            continue  # this contract object does not carry that spelling
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
    """Did this ``TypeError`` come from passing ``torch_dtype``?

    WHY THIS IS A TRY/EXCEPT AND NOT A SIGNATURE CHECK — measured, because the
    obvious fix does not work. Both loader families take ``**kwargs`` and
    NEITHER names ``torch_dtype`` in its signature::

        ModularPipeline.from_pretrained   torch_dtype named: False  **kwargs: True
        StableDiffusionXLPipeline...      torch_dtype named: False  **kwargs: True

    so `"torch_dtype" in signature.parameters` is False for both (SDXL would
    lose its dtype) and `has **kwargs` is True for both (H3 would still break).
    The real difference is what the implementation DOES with the keyword:
    ``DiffusionPipeline.from_pretrained`` reads ``torch_dtype`` (3 mentions in
    its source) and consumes it; ``ModularPipeline.from_pretrained`` never
    mentions it (0), so ``**kwargs`` funnels it through ``load_config`` into
    the constructor — where a strict pipeline correctly refuses an argument
    that is neither a pipeline argument nor a component (se#754/se#766's
    ``MiniMaxH3StreamingPipeline``, which is where this was found).

    Behaviour, not shape, is the discriminator — and asking is the only way to
    read behaviour. Every loader that works today still takes the first branch
    and is untouched; only the family that REFUSES gets the second.
    """
    text = str(exc)
    return any(token in text for token in _DTYPE_REJECTIONS)


@dataclass(frozen=True, slots=True)
class Adapter:
    """A hub-resolved adapter (LoRA). ``ref`` is its FULLY-PINNED hub
    identity (``org/repo@release`` — what evidence/output reports; hub
    resolution fills it pinned even when the pick was floating); ``name`` is
    the diffusers adapter-registry label (``load_lora_weights``'s
    ``adapter_name``); ``defaults`` is the decoded overlay (pgw#1377's
    ``adapter.defaults``, e.g. ``SDXL.Lora.Defaults``); ``scale`` is
    envelope-resolved, platform-clamped through the LoRA's strength Knob
    before the worker sees it. Entrypoints declare adapter SLOTS as explicit
    parameters — ``turbo: Adapter | None`` (single, optional) or
    ``loras: list[Adapter]`` (the request's picks; empty when none ride) —
    param name = slot name; the worker fills them from deploy/request state.
    Applying adapters is a MODEL mutation and goes through a model-owned
    scope (``model.adapters(riding)``)."""

    name: str
    path: Path
    defaults: Any = None
    scale: float = 1.0
    ref: str = ""


@dataclass(frozen=True, slots=True)
class DistillationAdapter(Adapter):
    """A distillation adapter — runtime-identical to :class:`Adapter`; its
    MEANING is the SLOT KIND (Paul's structural guard). A param annotated
    ``DistillationAdapter | None`` is a distillation-adapter slot: release
    metadata records the kind, the hub refuses envelope picks whose adapter
    row lacks the distillation marker (typed 400 naming the adapter), and
    resolution constructs THIS class for marked rows. Takeover power over
    the serving config used to be positional; now it is typed — a style
    LoRA cannot seize the config even on a misconfigured deployment."""


@dataclass(slots=True)
class DeployBinding:
    """One deployment's checkpoint binding — hub deploy state, rebindable.

    ``checkpoint_dir`` is the worker-resolved checkpoint tree, converted
    through the active lane's tensor-layout contract by tensorfs before any
    model sees it. ``defaults`` is the hub's per-checkpoint JSONB row (raw
    mapping) — typed by ``LoadContext.defaults()`` against the model type's
    ``Defaults`` schema."""

    checkpoint_ref: str
    checkpoint_dir: Path
    #: The hub row's ``model`` classification column. ``None``/empty = the
    #: checkpoint is unclassified: it serves on platform fallbacks with the
    #: named visible warning (pgw#1377's read-side matrix), never a guess.
    model: Optional[str] = None
    defaults: Mapping[str, Any] = field(default_factory=dict)
    adapter: Optional[Adapter] = None
    #: pgw#1606 — THE MULTI-LANE HALF. A binding used to carry exactly one
    #: tree, which is why a model declaring two lanes could not boot at all
    #: (``loader.lane`` raised: "the active lane must be named by contract").
    #: These three say what the deploy staged PER LANE CONTRACT, so the
    #: boot-time ladder has a candidate set to rank instead of an ambiguity to
    #: refuse. All three empty is the single-lane deployment — every one in the
    #: fleet today — and ``checkpoint_dir`` remains its answer.
    #:
    #: ``lane_trees``   contract handle -> the staged tree for that lane
    #: ``lane_verdicts`` contract handle -> the hub bind gate's own tri-state
    #:                  (``satisfies``/``derivable``/``incompatible``), which
    #:                  it computed at BIND time with the real Go
    #:                  ``Contract.Verdict``; the worker consumes it and never
    #:                  recomputes a rival answer.
    #: ``lane_bytes``   contract handle -> tree size on the wire, which is what
    #:                  makes the upcast rung's saving a measurement.
    lane_trees: Mapping[str, Path] = field(default_factory=dict)
    lane_verdicts: Mapping[str, str] = field(default_factory=dict)
    lane_bytes: Mapping[str, int] = field(default_factory=dict)


class LoaderEngine(Protocol):
    """The pgw#1380 seam: build a pipeline with weights streamed chunk-store
    -> pinned -> CUDA (serve) or on meta (derive). gen-worker's native
    loader lands behind this; tests inject theirs."""

    def build(
        self, pipeline_cls: type, *, checkpoint_dir: Path, lane: Any
    ) -> Any: ...


class DefaultsError(msgspec.ValidationError):
    """A hub defaults row does not fit the model type's schema."""


class ProjectedTreeNotStreamable(RuntimeError):
    """The eager bridge was handed a tree whose weights are POINTERS.

    pgw#1513. Raised instead of letting ``from_pretrained`` open a ~128 B
    TFSSTUB1 stub with the stock safetensors reader, which reports
    ``SafetensorError: header too large`` — a message about a corrupt
    checkpoint, for a checkpoint that is perfectly intact and whose bytes are
    sitting in the CAS. Two days of this incident were spent looking at
    volumes and download paths because of that sentence.

    The numbers in the message are what identify a stub on sight: a stub's
    size is fixed (~128 B) no matter how large the model it names, which is
    why a 3.4 GB and a 68 GB checkpoint failed identically.
    """

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
        # pgw#1513 follow-up: THE DECLINE REASON GOES FIRST, and this ordering
        # is load-bearing rather than editorial.
        #
        # The first field run of this refusal proved the fix works and then
        # LOST the one clause it exists to deliver: `JobResult.safe_message` is
        # bounded at 512 chars on the wire (`worker.py::_send_result`, a
        # deliberate slice — this layer declining to put an unbounded string on
        # a wire it owns), and the decline reason sat at the END of a longer
        # message. It was truncated away, so the pod said WHAT was wrong and
        # not WHY, which is the half nobody can reconstruct afterwards.
        #
        # Leading with it is structural: it survives ANY cap, at any layer,
        # including caps nobody has told us about yet. The explanation that
        # follows is the part a reader can afford to lose, because it is the
        # same every time — the reason is not.
        # pgw#1542: THE REPAIR OUTCOME RIDES IN POSITION TWO, NOT AT THE END.
        #
        # The ask was to append it. Appending would have reproduced the exact
        # bug the comment below describes: at 512 chars the tail is sliced off,
        # and a long `tree=` plus three stub paths already spends most of the
        # budget before the boilerplate starts. The repair outcome is
        # per-incident and unreconstructable — the same property that earned
        # the decline reason its front position — so it goes directly behind
        # it, ahead of everything a reader can afford to lose.
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
    """WHICH named exit `ModelStore.ensure_pinned` took for this tree.

    pgw#1542. Never raises and never returns empty: a blank here would read as
    "no repair happened", and telling a post-mortem reader that the repair did
    not run when in fact it ran and failed is worse than the silence it
    replaces.
    """
    try:
        from ..models.projection import pin_outcome

        return pin_outcome(Path(tree).name) or "not attempted"
    except Exception:  # noqa: BLE001
        return "unknown"


def _serving_device() -> str:
    """The device to stream onto when nobody handed one down (pgw#1544).

    ``ServeLoop`` builds its load contexts with no ``device=``, and the literal
    ``"cuda"`` is an enumeration of what a pod usually is rather than a
    measurement of this machine — pgw#1452's whole point. So the same probe the
    host uses answers here.
    """
    try:
        from .placement import serving_device

        return serving_device()
    except Exception:  # noqa: BLE001 — an unprobeable host is not a refusal
        return "cuda"


def _projection_pinned(tree: Path) -> bool:
    """Does this tree's manifest pin RESOLVE. The one measurement (pgw#1544).

    Both the repair's precondition and the refusal's decline reason are derived
    from this single call, passed down. Two independent lookups of one fact is
    how a refusal came to contradict itself in the same sentence.
    """
    from ..models import projection

    try:
        return projection.resolve_projection(tree) is not None
    except Exception:  # noqa: BLE001 — a probe never fails a load
        return False


def _projection_declined_because(
    tree: Path, *, pinned: Optional[bool] = None
) -> str:
    """WHICH of `resolve_projection`'s three silent Nones fired, in words.

    pgw#1513. Without this the reader gets a beautiful message about stubs and
    still does not know why the streaming engine passed on a tree it should
    have taken — the same two-day hunt, one layer up.

    **pgw#1544: the last branch used to ASSERT the pin was missing without ever
    looking.** It was the else of two structural checks, so any tree in the
    right place with a CAS behind it got the sentence "the manifest pin ... is
    MISSING" whether or not it was. On the fleet's pods the pin was PRESENT, so
    the refusal read `pin ... is MISSING | repair attempted: not needed:
    already pinned` — one string contradicting itself, on every request, for
    21 hours, sending three lanes to look at the store while the defect was in
    the wiring. A refusal that names an unmeasured cause is worse than one that
    names none.

    ``pinned`` is the caller's own measurement, handed in so that the sentence
    and the decision it explains can never key on two different lookups.
    """
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
    """Every tensor container in ``tree`` that is a pointer, not weights.

    Returns ``(relative path, bytes on disk, bytes it names)`` so the refusal
    can show the fixed-size-stub signature rather than assert it.
    """
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
        #: pgw#1606: the boot ladder's answer — the chosen lane, the reason,
        #: and the rejected rungs in order. ``None`` for a context built
        #: before the ladder ran (a fixture, a derive), which is why every
        #: read of it degrades rather than raising.
        self._resolved = resolved
        self._engine = engine
        self._compile_sink = compile_sink
        self._engines: list[Any] = []
        #: The WORKER's placement decision, handed down (pgw#1452). Empty
        #: means none was handed down — a bare `LoadContext(...)` places
        #: nothing, exactly as before.
        self._device = str(device or "")
        #: pgw#1543: the host's IO mode, kept so a repair-at-decline re-asks
        #: `engine_for` with the SAME arguments the host used. Retrying with a
        #: guessed mode would make the retry a different question.
        self._io = str(io or "buffered")
        #: The offload rung `_placed` engaged for this load, or "" for "the
        #: ladder was never consulted" (pgw#1486). Read by `compile()`, which
        #: may not arm a compiled graph over hook-managed weights.
        self._engaged_rung = ""
        #: pgw#1497: the DEVICE bytes this instance's weights were ADMITTED
        #: for — the residency lease's own number, handed down so the
        #: `partial_stream` rung can size its resident set from it. 0 means no
        #: lease was in scope (a local run, a fixture), and that rung then
        #: refuses rather than invent a budget.
        self._weight_budget_bytes = max(0, int(weight_budget_bytes))

    @property
    def loader_engine(self) -> Optional[LoaderEngine]:
        """The streaming engine THIS load actually bound, or ``None``.

        pgw#1549. `EndpointHost` used to keep its own handle and report its
        numbers; since `ctx.load` is the one binder, the only truthful source
        for "did an engine run, and what did it do" is the context that ran.
        """
        return self._engine

    @property
    def checkpoint_dir(self) -> Path:
        """The worker-resolved checkpoint tree for the RESOLVED lane.

        pgw#1606: a binding used to carry exactly one tree. It now carries one
        per staged lane contract, and this answers the one the boot ladder
        picked — including the upcast rung's case, where the bytes fetched are
        a quantized lane's and the modules served are baseline. A single-lane
        deployment has no map and gets ``binding.checkpoint_dir``, unchanged.
        """
        tree = self._lane_tree()
        return Path(tree if tree is not None else self._binding.checkpoint_dir)

    def _lane_tree(self) -> Optional[Path]:
        """The staged tree for the resolved lane, or ``None`` for "the one"."""
        resolved = self._resolved
        if resolved is None:
            return None
        trees = getattr(self._binding, "lane_trees", None) or {}
        if not trees:
            return None
        # The upcast rung fetches the QUANTIZED tree and serves baseline
        # modules out of it, so the bytes to open are `fetch_contract`'s.
        for key in (getattr(resolved, "fetch_contract", ""),
                    getattr(resolved, "contract_id", "")):
            if key and key in trees:
                return Path(trees[key])
        return None

    @property
    def resolved_lane(self) -> Any:
        """The boot ladder's answer, or ``None`` if the ladder never ran.

        Read by the platform, not by endpoints — an endpoint that branches on
        this is exactly the code pgw#1606 exists to delete.
        """
        return self._resolved

    @property
    def checkpoint_ref(self) -> str:
        """The hub ref of the bound checkpoint."""
        return self._binding.checkpoint_ref

    @property
    def lane(self) -> Any:
        """The active lane (the deploy's pick) — a tensorfs layout contract;
        ``ctx.lane.dtype`` is the load dtype.

        pgw#1599: every model class declares at least one REAL lane, so a
        missing active lane is now a WIRING defect (nothing passed one in),
        never an authored state. Resolution among several declared lanes is
        platform machinery (pgw#1606)."""
        if self._lane is None:
            raise RuntimeError(
                "ctx.lane: no active lane was bound for this load. Every "
                "model class declares real lanes (pgw#1599), so this is a "
                "platform wiring defect, not an authored state — the caller "
                "that built this LoadContext passed no `lane=`."
            )
        return self._lane

    def _bind_streaming_engine(self, *, pinned: bool) -> Tuple[bool, bool]:
        """ASK for this projected tree's engine, repairing the pin if it is
        gone. Returns ``(bound, pinned_after)``.

        pgw#1543 put the REPAIR here. pgw#1544 puts the ASK here, which is the
        half that was missing and the reason the fleet stayed down.

        **A projected tree reaching this method means no engine was bound for
        this load, and until now that was read as "the engine declined".** It
        is not. ``engine_for`` has two production call sites: ``EndpointHost``
        (the local CLI and the daemon) and this one. The serverless worker
        builds its ``LoadContext`` in ``ServeLoop._backend_factory`` with
        ``engine=self._engine``, and ``worker.py`` constructs ``ServeLoop``
        with no ``engine=`` at all — so on a POD nothing ever asked. Every
        projected tree fell to the eager bridge and refused, on every request,
        with a message blaming the store.

        So the engine is asked HERE, at the one place that always has the tree,
        rather than at each of the N paths that build a context — the same
        defect class pgw#1543 fixed one layer up, and the reason this fix does
        not simply add an ``engine=`` to ``ServeLoop``: the next path to build
        a context would omit it again.

        ``pinned`` is the caller's measurement, not a second lookup. When it is
        True the pin is already good and the repair is not merely unnecessary,
        it is **not a precondition** — pgw#1543 returned False here because
        `ensure_pinned` answers "did I repair", and "already pinned" is False.
        A correctly-pinned tree therefore never re-asked for an engine and
        refused anyway. That single conflation is the outage's second half.

        Never raises: a failure must leave the ORIGINAL refusal intact rather
        than replace a precise diagnosis with a repair traceback. The outcome
        is recorded either way (pgw#1542) and rides the refusal.
        """
        try:
            from ..models.store import active_store
            from . import streaming

            if not pinned:
                store = active_store()
                if store is None:
                    return False, pinned
                ref = cast("WireRef", self.checkpoint_ref)
                # The snapshot is looked up BY TREE first, not by ref. The
                # serving path holds the resolver's `pick.ref`, which need not
                # be the exact string the store banked under, and a ref-only
                # lookup would make this repair a SILENT NO-OP on any spelling
                # mismatch — refusing exactly as before while every stubbed
                # test passes. The tree's directory name is its digest, which
                # is a fact about disk.
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
                # Pinned and STILL no engine: a different fault, and the
                # refusal below is the honest answer. Not logging this as a
                # success is the whole point of asking rather than assuming.
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
        """Build ``pipeline_cls`` on the RESOLVED lane — the ONE spelling
        (pgw#1606), and the only one an endpoint should write::

            self.pipe = ctx.load_pipeline(StableDiffusionXLPipeline)

        There is no dtype argument, no quantization if-tree and no hardware
        read at the author's site. The platform resolved a lane at boot out of
        the lanes this Model declared (`serving.lane_ladder`), and this call
        materializes the pipeline per that lane's contract document: baseline
        is a plain build; fp8 swaps the denoiser's linears to the scaled-mm
        modules; nvfp4 swaps them to the block-scaled fp4 modules.

        **It then proves the modules are the lane.** `lane_materialize` runs a
        marker census and refuses a quantized lane that converted nothing, a
        half-converted one, and a baseline lane carrying stray quantized
        leaves. A loader must not be the only witness of what it did — the
        audit behind this issue found four separate places where it was, and
        each of them could serve the wrong numerics in silence.

        A context whose ladder never ran (a fixture, a derive, a local CLI
        run) has no resolved lane, and this degrades to :meth:`load` rather
        than inventing one. That is the same discipline `ctx.lane` already
        applies: no silent default lane, ever.
        """
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
        """Build ``pipeline_cls`` with this checkpoint's weights resident.

        The materialization half of :meth:`load_pipeline`, and the older
        spelling. Prefer ``ctx.load_pipeline`` in endpoint code: this one puts
        weights in memory, that one puts them in the right MODULES and proves
        it. Kept because the derive, the local CLI and every fixture build a
        context with no resolved lane, and for them the two are the same call.

        The original contract (pgw#1380)::

            self.pipe = ctx.load(StableDiffusionXLPipeline)

        No ``torch_dtype=`` (the lane contract IS the dtype), no
        ``.to("cuda")`` (weights land on device). Behind the call is
        pgw#1380's ``gen_worker.serving.streaming`` engine: the pipeline
        built from configs on ``meta``, then its weights walked out of the
        chunk store in FILE order through pinned staging onto the device,
        with no tensor file written or read anywhere. The host binds it off
        the projected checkpoint tree (which carries its own store), so
        PLACEMENT stays a worker decision and this code names no device.

        The eager ``from_pretrained`` bridge survives for a tree with no
        chunk store behind it — a bare download, a local run, a fixture —
        where there is nothing to stream from."""
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
        # pgw#1513: THE EAGER BRIDGE MUST NOT BE HANDED A PROJECTED TREE.
        #
        # Its own contract, three lines up, is "a tree with no chunk store
        # behind it — a bare download, a local run, a fixture". A PROJECTED
        # tree is the opposite: every tensor container in it is a ~128 B
        # TFSSTUB1 pointer stub whose bytes live in the CAS, and
        # `from_pretrained` reads with the stock safetensors reader, which
        # knows nothing about stubs. It reads the stub's first 8 bytes as a
        # header length and raises `SafetensorError: header too large`.
        #
        # That error is a LIE ABOUT THE CHECKPOINT, and it cost two days
        # pointed at poisoned volumes and truncated downloads. It is identical
        # for a 3.4 GB model and a 68 GB one — the stub is a fixed size — which
        # is precisely the signature that ruled out every short-write theory
        # and should have named this on day one. So the bridge refuses, by
        # name, with the numbers that identify a stub on sight.
        #
        # Reaching here means the streaming engine declined this tree
        # (`engine_for` -> `store_for` -> `resolve_projection` returned None),
        # which for a tree that IS projected means its manifest pin could not
        # be resolved — `resolve_projection` keys the pin on the tree's own
        # DIRECTORY NAME, so a tree selected under a different spelling than
        # the one it was pinned under resolves to nothing. This refusal turns
        # that into one named failure instead of an opaque loader crash.
        # ONE source of truth for this wording, shared with `skeleton.build`
        # (pgw#1514) so the two refusals cannot drift into teaching different
        # mental models. Checked BEFORE the stub condition: "the objects were
        # collected" is the real cause and must win over "this is a stub".
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
            # pgw#1543: REPAIR AT THE DECLINE, because this is the only place
            # the failure actually happens. pgw#1544: and ASK FOR THE ENGINE
            # here too, because on a pod nobody else ever does.
            #
            # `ensure_pinned` existed since pgw#1526 and had two callers —
            # `announce_resident` (boot) and `_materialize_local`
            # (materialization). NEITHER is on this path. A pod whose tree was
            # already materialized, or whose announce ran before the repair
            # existed, reaches here on EVERY request and refuses, forever,
            # while a working repair sits two modules away. That is a ~19 h
            # fleet-wide serve outage explained without a single log line: the
            # fix was never on the failing path.
            #
            # Ordering is load-bearing: `collected` is checked ABOVE and never
            # reaches here, because a tree whose objects were GC'd has nothing
            # to re-pin (se#790 measured 5.6 GB / 134 GB gone) and writing a
            # pin over absent bytes would convert an honest refusal into a
            # corrupt serve. Only the stubbed-but-INTACT case is repairable —
            # exactly "the objects are present and only the pin is gone".
            #
            # pgw#1544: ONE LOOKUP OF "IS THIS TREE PINNED", derived from here
            # and handed to both the decision and the sentence that explains
            # it. The pod's refusal read `pin ... is MISSING | repair
            # attempted: not needed: already pinned` because the repair MEASURED
            # the pin and the message ASSUMED it — and the message was wrong.
            pinned = _projection_pinned(self.checkpoint_dir)
            bound, pinned = self._bind_streaming_engine(pinned=pinned)
            if bound:
                engine = self._engine
                assert engine is not None  # set by the bind above
                # Returned RAW, exactly as the primary streaming arm above
                # returns it. `_placed` is the EAGER bridge's placement — the
                # streaming engine already streamed these weights onto the
                # device it was handed, and running the fit ladder over a
                # pipeline that is already resident is how pgw#1486's OOM
                # happens. One placement story for one engine.
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
        # pgw#1473: a VARIANT-ONLY tree (`*.fp16.safetensors`, which is what
        # every fp16 mirror ships) is invisible to `from_pretrained` unless it
        # is told. Detected off the tree the worker already resolved, never
        # configured — an author stating a fact about bytes they did not
        # publish is the second source of truth that drifts. `None` for every
        # published/converted checkpoint, which is why it runs unconditionally.
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
        # This loader does not SPEAK `torch_dtype`. Load without it and apply
        # the lane's dtype afterwards, so the lane is honoured rather than
        # silently dropped — `.to(dtype)` moves floating parameters and leaves
        # ints/bools alone, which is what `from_pretrained(torch_dtype=)` does.
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
        """Apply the WORKER's placement decision to a bridged pipeline.

        pgw#1452. The streaming arm streams weights ONTO the device
        (`engine_for(..., device=device)`), and `host.py` says why in the same
        breath: "PLACEMENT is decided here, by the worker, never by author
        code". The eager arm built a pipeline and placed nothing — so every
        tree with no chunk store behind it (a bare download, a local run, a
        fixture: the entire cozy-local substrate) ran on the CPU, while
        `ctx.load`'s own docstring four lines above promised "weights land on
        device". Nothing failed. It was simply the wrong processor, which is
        why every timing taken through this bridge was measuring a CPU.

        This names NO device, which is the whole point — it applies the one it
        was handed. An empty `_device` places nothing, so a bare
        `LoadContext(...)` and any caller that never had a placement decision
        behave exactly as before.

        pgw#1486: the placement decision is now a FIT decision, and the order
        is the whole point. `models.memory.select_auto_mode` nets the
        requirement against `estimate_cuda_resident_gb(pipeline)` (pgw#1025,
        correct: a pipeline must not be charged twice for bytes already on the
        card) — so asking it AFTER a full `.to(device)` makes the requirement
        net to zero, "it fits" trivially true, and the ladder answers its most
        memory-hungry rung for a pipeline that is about to OOM. Measured on one
        SDXL pipeline object at 1024^2 on a 7.62 GiB card: `model_offload`
        asked before placement, `vae_only` asked after. The second OOMs.
        """
        if not self._device:
            return pipeline
        to = getattr(pipeline, "to", None)
        if not callable(to):
            return pipeline
        from ..models.rung import touches_host_ram

        mode = self._fit_rung(pipeline)
        if mode and touches_host_ram(mode):
            # The rung armed accelerate's hooks and OWNS placement from here:
            # each component is onloaded for its own forward and evicted after.
            # A `.to(device)` now would either undo the hooks or re-land the
            # very bytes the rung just moved off — which is the OOM.
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
        # diffusers returns self; some component-wise `.to` return None.
        return pipeline if moved is None else moved

    def _fit_rung(self, pipeline: P) -> str:
        """Ask the shipped offload ladder which rung this pipeline needs,
        while it is still on the host, and engage it.

        Returns the engaged mode, or `""` when the ladder was not consulted at
        all (no CUDA target, or the ladder could not read the pipeline). `""`
        is deliberately distinct from `"off"`: "nobody asked" and "asked, and
        the answer was full residency" are different facts, and only the
        second licenses the unconditional placement below.

        EAGER-BRIDGE SCOPE, stated here rather than inferred. The streaming
        engine returns before `_placed` is ever reached and streams weights
        onto the device itself; giving it a fit rung is a separate mechanism
        (it would have to choose per-component destinations mid-stream, not
        re-place a built pipeline). And this is an ADMISSION check, never a
        catch-and-retry: an OOM inside a compiled graph is not catchable — it
        is process death (pgw#1255 leg 2) — so the only honest place to decide
        is before the weights land.
        """
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
            )
        except HostRamMoveRefusedError:
            # The guard did its job: this host cannot hold what the rung wanted
            # to move. Re-raising is right — the alternative is placing the
            # whole pipeline on a card that already refused it.
            raise
        except Exception as exc:
            # A ladder that cannot read this pipeline must not stop it loading.
            # Placement then proceeds exactly as it did before pgw#1486, which
            # is the behaviour every non-diffusers `ctx.load` caller has today.
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
        """The lane's load dtype, or None when the contract declares none.

        pgw#1386, measured on se#754's minimax-h3: tensorfs' ``Contract.dtype``
        RAISES (``MissingDtype``) for a document with no top-level ``dtype``
        rather than answering None — a deliberate "never guess" refusal on
        tensorfs' side. That is a LEGAL state for a lane
        (``minimax.h3-dit-diffusers@1`` declares per-tensor dtypes only, while
        ``sdxl.diffusers-bf16@1`` declares ``"dtype": "bfloat16"``), and this
        bridge already HAS a no-dtype arm — so the read must not be able to
        kill the load before reaching it. ``getattr(lane, "dtype", None)`` does
        NOT swallow a non-AttributeError, which is why the plain read did.
        Any other error still propagates.

        pgw#1448 — AND IT MUST RETURN A ``torch.dtype``, NOT A STRING. This
        read used to answer ``Contract.dtype``, which is the contract's
        SPELLING (``'bfloat16'``), while the sibling attribute
        ``Contract.torch_dtype`` is the real object (``torch.bfloat16``)::

            contracts.MINIMAX_H3_DIT_DIFFUSERS.dtype        -> 'bfloat16'  (str)
            contracts.MINIMAX_H3_DIT_DIFFUSERS.torch_dtype  -> torch.bfloat16

        diffusers REFUSES a string ``torch_dtype`` with a warning and falls
        back to **fp32**, so every kwargs-accepting pipeline loaded through the
        eager bridge was loading at the wrong precision — silently. Measured by
        the local lane on a 4070: sd1.5 at **13 s/it**, ~20-40x off, because
        fp32 doubles the weights on a 7.63 GiB card. **The precision bug IS the
        performance bug**, and it is invisible: a scroll-past warning, then a
        model that works and is slow.

        FLEET CONSEQUENCE: any timing ever taken through this bridge was fp32
        timing and must be re-baselined, not compared.
        """
        return _lane_torch_dtype(self._lane, checkpoint_dir=self.checkpoint_dir)

    def engine(self, spec: Any) -> Any:
        """Boot the EXTERNAL engine that serves this checkpoint — the one
        spelling for the engine-hosted tier (pgw#1421)::

            self.engine = ctx.engine(LlamaServer(extra_args=["-ngl", "99"]))

        The sibling of :meth:`load`, and the same division of labour: the
        author declares (which engine, which flags), the platform supplies
        (which checkpoint tree, which port, which environment) and supervises
        (boot ladder, real-liveness health wait, typed events, teardown).

        Returns an ``EngineHandle`` whose ``base_url`` an entrypoint POSTs to.

        THE HANDLE IS REGISTERED HERE, which is what makes teardown
        structural: ``EndpointHost.evict`` stops every engine this context
        started AFTER the author's ``unload``, whether or not ``unload`` was
        written. An engine subprocess is invisible to torch's allocator, so a
        forgotten stop is not a tidy-up debt — it is VRAM the next residency
        admit cannot see and cannot reclaim. ``EngineHandle.stop`` is
        idempotent, so an author who stops it in ``unload`` anyway is right
        and costs nothing.

        This is the ONLY tier that gets a real file out of the store: Paul's
        2026-08-19 ruling narrowed the #1303 ladder's tier 3 to external
        binaries and AOT ``.so`` delivery, and made it permanent there. A
        serving pytorch endpoint that reaches a materialized view is a defect
        signal; ``llama-server -m`` reaching one is the design.
        """
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
        """Stop every engine this context started, newest first.

        BEST-EFFORT AND UNCONDITIONAL: called by the host after the author's
        ``unload``, so an author who never wrote one still gets the process
        reaped. Each stop is idempotent and a raising stop cannot prevent the
        next one — the whole point is that no single failure strands a
        VRAM-holding subprocess.
        """
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
        """Imperative module marking, the torch.compile idiom (Paul ruling)::

            self.pipe.unet = ctx.compile(self.pipe.unet)

        Everything behind the call is torchcg (tcg#42): at publish the
        hollow session records + hooks; at serve the AdoptSession arms what
        the store holds for (lane, sm) and returns the module UNCHANGED
        (eager) on a miss while the graph joins the ordered hole list the
        background mint consumes (pgw#1371). ``ctx.compile(pipe)`` walks a
        pipeline's nn.Module components. With no adoption source bound
        (local runs, the eager bridge) it returns ``target`` untouched —
        marking is always safe."""
        if self._compile_sink is None:
            return target
        from ..models.partial_resident import parks_module
        from ..models.rung import moves_every_component, touches_host_ram

        rung = self._engaged_rung
        if moves_every_component(rung):
            # pgw#1486, the ADMISSION half. An offload rung hands each
            # component's weights to accelerate, which onloads them for a
            # forward and frees the device copy after — so the device pointers
            # a compiled graph binds its constants to are dangling by the
            # second call. That is not an OOM to catch; it is a use-after-free,
            # and on the compiled path it is the SIGSEGV that takes the whole
            # worker down (pgw#1255 leg 2). Serving eager here is a real
            # degradation and is therefore said out loud, not logged at debug.
            logger.warning(
                "ctx.compile: serving EAGER for %s — the %r offload rung is "
                "engaged, and a compiled graph cannot bind constants to "
                "weights accelerate moves between host and device per forward "
                "(pgw#1486). Fit the model on the card to get compiled speed.",
                type(target).__name__, rung,
            )
            return target
        if touches_host_ram(rung):
            # pgw#1587. A rung that parks only the components it NAMED is not
            # the case above, and treating it as one is what made a small card
            # mean "no compiled graph, ever". Paul, 2026-08-20: *"For SDXL in
            # particular, we need to offload the text encoders to free up room
            # for the Unet, and then it works, during inference. This doesn't
            # conflict with compilation however because [we] are only running
            # the compiled UNet."*
            #
            # So the question is per-TARGET and the armed plan answers it: a
            # parked target serves eager (its pointers really do move), and
            # everything the plan left resident compiles exactly as it would
            # with no rung engaged. The rung's whole reason for parking those
            # encoders was to make room for THIS.
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
        """THIS checkpoint's defaults, typed via the model class's generic:
        the hub's per-checkpoint row decoded as a field-level overlay on the
        model type's ``Defaults`` struct (zero-arg = platform fallbacks —
        pgw#1377's decode matrix; ill-typed values are a typed refusal
        naming the checkpoint, never a silent coercion).

        The decode is pgw#1377's ``defaults_decode`` AUTHORITY, never a second
        msgspec pass: only that path narrows a Knob's [lo, hi] across the
        platform and checkpoint layers and re-stamps each Knob with its field
        name, which is what keeps two clamps distinguishable in the
        caller-visible adjustment ledger."""
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
        # `_model_type` is erased to `type` on the instance; the self-type on
        # this method is what carries `DT` for the caller.
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
    """What entrypoints receive — request facts + the salvaged base surface.

    The base class (`gen_worker.request_context`) contributes
    ``raise_if_cancelled``, ``save_image``, ``clamp``, ``generator``,
    ``progress``, ``log`` and the rest; this subclass adds the pgw#1382
    request facts. Constructible bare (``RequestContext("req-1")``) for
    local/unit use — deployment facts then read as absent.

    **IT IS ALSO THE PRODUCER CONTEXT (pgw#1406).** ``_PublisherMixin``
    contributes ``save_checkpoint`` / ``open_checkpoint_stream``, the
    reserved ``source``/``destination``/``text_encoder``/``candidate`` payload
    contract, and ``hf_token``; :meth:`mktemp` completes it. There is
    deliberately no second class for producers, because pgw#1294/pgw#1306
    already ruled the shape: *"no kind selects a different class, because no
    kind decides what a body may write — the declaration does"*. Every one of
    those surfaces refuses typed unless the ``@entrypoint`` declared
    ``publishes=True`` (:meth:`_require_publish_declaration`), so an inference
    entrypoint gains reach it cannot use and a ported ``@job`` producer gains
    nothing it did not already have. That is what makes pgw#983's deletion of
    ``@job`` recoverable by re-decorating rather than by rewriting 27
    producers (th#2173)."""

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
        #: pgw#1576: the ``@entrypoint(streams=…)`` chunk type(s), stamped from
        #: the spec exactly like ``publishes`` — the SDK half of the declaration
        #: the manifest reports as ``incremental_output``. A tuple because one
        #: handler may declare several shapes.
        self._delta_types: Tuple[type, ...] = (
            () if streams is None
            else (tuple(streams) if isinstance(streams, tuple) else (streams,))
        )
        #: Where a chunk goes: ``(data, content_type) -> None``, the dispatch's
        #: ``JobProgress`` emitter. Absent on a local run, where a chunk is
        #: logged and dropped rather than refused — the wire says deltas are
        #: droppable, and `gen-worker run` has no wire at all.
        self._chunk_sink = chunk_sink

    def emit(self, chunk: Any) -> None:
        """Put one incremental-output chunk on the wire (pgw#1576).

        Ordered per request, live, never persisted, and DROPPABLE by contract —
        the worker's send queue sheds progress under pressure and says so with
        a ``serve_degrade`` row. What a caller is owed is the struct this
        entrypoint RETURNS; a delta is how they see it arriving.

        Refuses unless the function declared ``@entrypoint(streams=<type>)``,
        and refuses a chunk that is not that type: an undeclared emitter would
        publish ``incremental_output: false`` and stream anyway, which is a
        manifest that lies to every client that reads it.

        Sync and thread-safe on purpose — the body runs on a worker thread
        (``asyncio.to_thread``) and a plain ``def`` entrypoint streams with the
        identical call.
        """
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
        """A request-scoped scratch directory. Contents are NOT persisted.

        Each call returns a fresh subdir, so a producer can use it as
        ``out_dir`` for successive conversions without collision."""
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
        """The RESOLVED, fully-pinned hub ref of the checkpoint that
        actually served this request — honest structured output evidence
        (the ``model`` field of an output struct)."""
        if self._serve_binding is None:
            raise RuntimeError(
                "ctx.checkpoint_ref: no deploy binding rides this context "
                "(local construction without binding=)"
            )
        return self._serve_binding.checkpoint_ref

    def step_callback(self, num_inference_steps: int, **kwargs: Any) -> Any:
        """A diffusers ``callback_on_step_end`` wired to progress + cancel —
        the context spelling of :func:`gen_worker.diffusers_step_callback`."""
        from ..api.progress import diffusers_step_callback

        return diffusers_step_callback(self, num_inference_steps, **kwargs)

    def warn(self, message: str) -> None:
        """A caller-visible WARNING into the response envelope — the
        generalization of clamp visibility, same delivery path (the
        adjustment ledger rides ``JobResult.adjustments`` + the hub's
        request record/events). Warn-and-serve: for a request aspect that
        cannot apply on this serving (e.g. guidance on a cfg-free recipe),
        never a silent drop and never an aborted request."""
        self.adjusted("", None, None, str(message))

    @property
    def warnings(self) -> tuple[str, ...]:
        """The accumulated :meth:`warn` messages, in emission order (the
        field-less rows of the adjustment ledger)."""
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
