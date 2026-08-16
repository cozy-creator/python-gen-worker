"""``@endpoint`` — the one decorator.

* Plain function -> stateless endpoint::

      @endpoint
      def hello(ctx, payload: In) -> Out: ...

* Class (+ optional ``setup()``) -> stateful endpoint. Every public method
  (not ``setup``/``warmup``/``shutdown``, not ``_``-prefixed) is a routable
  function; helpers must be underscore-prefixed::

      @endpoint(model=HF("org/repo", dtype="bf16"), resources=Resources(gpu=True))
      class Generate:
          def setup(self, model: FluxPipeline): self.pipeline = model
          def generate(self, ctx, p: Input) -> Output: ...

* ``kind="conversion" | "training" | "dataset"`` selects the context subclass.
  Producer kinds publish explicitly — write files locally, call
  ``gen_worker.convert.publish_flavors(ctx, flavors)`` (one Tensorhub commit per
  ``ProducedFlavor``), and return a result struct. Generator handlers are
  rejected for producer kinds: nothing is ever published by yielding.
* ``publishes=True`` declares that this function MAY write tensors/repos to
  the hub. ONE declaration surface, shared with ``@job``: the hub mints the
  worker-capability write grant off the declaration, never off the kind, and
  the SDK refuses the publisher surface without it. It says MAY write; the
  request still says WHERE.
* ``kind="eval"`` MEASURES a candidate against a reference. It reads
  reserved refs like every non-inference kind and writes request output assets,
  but publishes no catalog repo — the hub refuses repo writes for eval tokens
  and never asks an eval where its output repo goes. Its input struct declares
  no destination, and that is the point.
* An async-generator handler streams (inference only); there is no separate
  streaming decorator.
* ``runtime="vllm"`` boots an engine-hosting server subprocess before setup.

Checkpoint SELECTION is a runtime payload argument, not a build-time fan-out:
a ``Slot(selected_by=..., default_checkpoint=...)`` is resolved by the hub per
request, so one handler serves every curated checkpoint that shares the
resident base. Divergent WIRE contracts are separate methods; only
weight-sharing forces one class.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, TypeVar, Union, get_args, overload

import msgspec

from .binding import BINDING_TYPES, Binding
from .export_contract import (
    Arg, Dim, Fork, GraphClass, Input, MintBlocker, open_blockers,
    validate_contract, validate_speed_bar,
)
from .formula import RuntimeFormula
from .slot import OBJECTIVES, TASKS, D, Slot
from ..models import execution_lanes as lanespec
from ..runtimes.server import ServerHandle
from .tree import is_introspectable

T = TypeVar("T")
SlotLike = Union[Binding, "Slot[D]"]

KINDS = ("inference", "training", "dataset", "conversion", "eval")

RESERVED_METHODS = frozenset({"setup", "warmup", "shutdown"})


# The dotted capability (8.9) is the author-facing unit, the way NVIDIA writes
# it. `sm_89` is accepted because that is how kernels and error messages spell
# it; the bare SM code is not — 89 and 8.9 are a silent factor-of-ten apart.
_MAX_COMPUTE_CAPABILITY = 20.0


def _normalize_compute_capability(raw: float | str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"compute_capability must be numeric, got {raw!r}")
    if isinstance(raw, str):
        s = raw.strip().lower()
        if not s:
            raise ValueError("compute_capability is empty")
        if s.startswith("sm"):
            code = s[2:].lstrip("_").strip()
            if not code.isdigit():
                raise ValueError(
                    f"compute_capability {raw!r} is not an SM code (e.g. 'sm_89')")
            value = int(code) / 10.0
        else:
            try:
                value = float(s)
            except ValueError:
                raise ValueError(
                    f"compute_capability must be numeric, got {raw!r}") from None
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"compute_capability must be numeric, got {raw!r}") from None
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"compute_capability must be finite, got {raw!r}")
    if value <= 0:
        raise ValueError(f"compute_capability must be positive, got {raw!r}")
    if value > _MAX_COMPUTE_CAPABILITY:
        raise ValueError(
            f"compute_capability {raw!r} looks like a bare SM code — declare the "
            "dotted capability (8.9, 12.0) or the prefixed code ('sm_89')")
    return round(value, 1)


# The parallel mechanisms the PLATFORM implements, kept in lockstep with the
# hub's builder ingest (internal/builder/staffing_envelope.go
# parallelMechanisms) and orchestrator/topology's Parallel* constants — an
# unknown token refuses here, at declaration, before a build is spent.
_PARALLEL_MECHANISMS = frozenset({"sequence", "cfg"})


class Resources(msgspec.Struct, frozen=True, omit_defaults=True):
    """Hardware envelope for one function: ONLY what the endpoint
    CANNOT RUN WITHOUT — ``Resources(gpu, gpu_count, libraries, vcpus)``.

    "What it needs to run WELL" belongs to the fit ladder / residency
    planner / economics gate — all of which have measurements the endpoint
    author does not.

    th#1867 (DESIGN-RULINGS §1.35) deleted the LAST THREE VRAM markers this
    struct carried — ``vram_gb_hint``, ``min_vram_gb`` and ``strict_vram`` —
    and they went as ONE change because §2.4 ruling 4 measured what a partial
    cut does: the hub builder folds an absent ``min_vram_gb`` from ``vram_gb``,
    so deleting one of them silently drops a floor. Paul's ruling in one line:
    *"We should be able to run any model on any GPU. The challenge is not IF it
    can run — it's: is it an EFFICIENT choice?"* A card that looks too small is
    a card whose best (GPU, lane) pair sits further down the operator's ladder,
    and the RUNTIME picks that rung from what it MEASURES (``models/memory.py``
    ``select_auto_mode``), never from what the author guessed. §1.2 measured the
    guess against live serve profiles and it was wrong in BOTH directions —
    anima declared 8 GB against a 10.6 GiB peak, sdxl declared 20 against a
    proven 9.3 GiB run on a 16 GB A4000.

    ``compute_capability`` is NOT one of them and survives untouched: an sm
    floor is a statement about which KERNELS exist in our build, not about how
    big a card is (§2.4 ruling 1).

    ``ram_gb_hint`` (pgw#670) is its HOST-side twin, restored after the v2
    cut deleted ``ram_gb`` on the reasoning that host RAM is an
    opportunistic latency tier. For ltx-video-2.3 it was neither
    opportunistic nor a guess: ie#484 measured 179-301 s mp4-encode and
    147 s VAE-decode tails on host-starved allocations at IDENTICAL GPU
    step-ms, ie#492 sized the floor at 64 GB from that failure, and the hub
    consumed it (pod-create minimum + th#740 read-back-and-reject). Without
    it a starved allocation DEGRADES SILENTLY — a slow request, not a
    refused pod — which is exactly what the declaration existed to prevent.
    It is an ALLOCATION-time ask (the hub's ``min_ram_gb``), never a runtime
    gate: nothing refuses a request because host RAM is low. That distinction
    is why it survives th#1867 while every VRAM marker went — the deleted
    fields were sizing a CARD, which §1.35 rules is never the question. It does not imply ``gpu=True`` — the components are
    pinned host staging, model-load staging, frame/encoder buffers and
    page-cache headroom, and a CPU-only encode lane needs them too.
    Asymmetry note: ``vcpus`` survived the v2 cut and covers the CPU side of
    the very same encode tail, which is what made the RAM deletion read as
    accidental rather than principled.

    ``compute_capability`` (pgw#660) is the HARD GPU-architecture floor,
    restored after the v2 cut deleted it. The cut's reasoning — "precision
    per card is the fit ladder's call, never a placement gate" — is right
    about precision SELECTION and wrong about INCAPABILITY: a producer whose
    kernel is ``torch._scaled_mm`` cannot run below sm_89 at any precision,
    on any rung, ever. With no carrier the hub emitted no
    ``compute_capability`` in ``requirement_payload_json`` and the scheduler
    placed the fp8 producer on sm_80 A100s (th#1155 six times; te#125
    again). This is NOT a hint and has no ``_hint`` suffix: the scheduler
    filters offers on it and refuses to rent below it. th#1867 deleted the
    VRAM markers and deliberately LEFT this one (§2.4 ruling 1): an sm floor
    says which kernels exist in our build, which is a statement about our
    code — a card size is a statement about the card.
    Declare the DOTTED capability the way NVIDIA writes it — ``8.9``,
    ``"8.9"``, or ``"sm_89"`` — never the bare SM code. Declaring it implies
    ``gpu=True``. Declare it ONLY for a genuine incapability; a function that
    merely runs BETTER on newer silicon declares nothing and lets the ladder
    choose.

    There is no disk axis. ``min_disk_gb`` (pgw#732) existed for two releases
    and no endpoint in any repo ever declared it — th#1233 sizes a pod's
    container disk from the bytes the job will materialize, which covers every
    live case — so HARDCUT C3 deleted the emitter rather than keep an unused
    author-facing knob.

    ``vcpus`` declares the host-side vCPU ask (CPU-heavy encode);
    it does not imply ``gpu=True``. ``gpu_count`` declares how many devices
    one instance needs; endpoint code NEVER picks devices.

    ``max_gpu_count`` + ``max_gpus_per_execution_group`` + ``parallel``
    are the author ENVELOPE — the SDK half of the
    hub builder's ``extractStaffingEnvelope`` contract. ``gpu_count`` is the
    floor (devices ONE materialization requires); ``max_gpu_count`` is how
    many GPUs the POD may hold; ``max_gpus_per_execution_group`` is how many
    of them ONE REQUEST may be spread across; ``parallel`` names the platform
    mechanisms the function survives at degree > 1 (``"sequence"`` is the
    only one with a worker runtime). The author never writes a degree, a
    tier, a packing or a device id — those are the hub's decisions.

    The last two are INDEPENDENT axes::

        max_gpu_count=4                                  -> 1x4: one request across 4 cards
        max_gpu_count=4, max_gpus_per_execution_group=2  -> 2x2: two slots, each request across 2

    Omitting ``max_gpus_per_execution_group`` means "no opinion on that axis"
    and lets the hub use the whole ceiling as one group. It is NOT defaulted
    to a legal value: the declared domain is 2 or more, and 0/1 are refused
    here and at ingest, so "declared nothing" can never be confused with a
    declaration. Both are elided from the manifest when they carry nothing
    (``omit_defaults``). Validation mirrors the builder's ingest refusals so a
    contradiction costs a declaration-time ValueError instead of a build.
    """

    gpu: bool = False
    gpu_count: int = 1
    libraries: tuple[str, ...] = ()
    vcpus: int | None = None
    ram_gb_hint: float | None = None
    compute_capability: float | str | None = None
    max_gpu_count: int | None = None
    max_gpus_per_execution_group: int | None = None
    parallel: tuple[str, ...] = ()

    def manifest_dict(self) -> Dict[str, Any]:
        """The manifest ``resources{}`` projection.

        The declaration and the wire name differ for exactly one field: the
        builder's host-floor key is ``ram_gb`` (which it maps to the
        scheduler's ``min_ram_gb``), so ``ram_gb_hint`` is projected under
        that name. One mapping, in one place, rather than a second spelling
        for endpoint authors to get wrong.

        th#1867: nothing VRAM-shaped is projected any more. The hub still has
        a ``min_vram_gb`` fold in ``function_requirements.go`` and a buy-side
        candidate filter behind it; both are th#1867 S4's to delete, and after
        this change no manifest reaches them. That absence is NAMED here rather
        than left to be discovered, because a hub gate reading a key nobody
        emits is exactly how pgw#660's gate vanished silently the first time.

        ``compute_capability`` needs no remap: it is already the key
        ``internal/builder/function_requirements.go`` parses (scalar or
        ``{"min": ...}``) into ``FunctionRequirements.ComputeCapabilityMin``.
        Note ``min_compute_capability`` — v1's author-facing spelling — is
        typed-REJECTED by the builder (th#1015 ``ErrMinComputeCapabilityRemoved``),
        so it must never appear on the wire.
        """
        raw = msgspec.to_builtins(self)
        out: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
        ram = out.pop("ram_gb_hint", None)
        if ram is not None:
            out["ram_gb"] = ram
        return out

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        if self.libraries:
            force(self, "libraries", tuple(
                str(x).strip() for x in self.libraries if str(x).strip()
            ))
        n_gpu = int(self.gpu_count)
        if n_gpu <= 0:
            raise ValueError(f"gpu_count must be positive, got {self.gpu_count}")
        force(self, "gpu_count", n_gpu)
        if self.compute_capability is not None:
            force(self, "compute_capability",
                  _normalize_compute_capability(self.compute_capability))
        if n_gpu > 1 or self.compute_capability is not None:
            force(self, "gpu", True)
        if self.ram_gb_hint is not None:
            r = float(self.ram_gb_hint)
            if r <= 0:
                raise ValueError(f"ram_gb_hint must be positive, got {r}")
            force(self, "ram_gb_hint", r)
        if self.vcpus is not None:
            c = int(self.vcpus)
            if c <= 0:
                raise ValueError(f"vcpus must be positive, got {c}")
            force(self, "vcpus", c)
        # Mirror the builder's extractStaffingEnvelope refusals at declaration
        # time.
        if self.max_gpu_count is not None:
            m = int(self.max_gpu_count)
            if m != self.max_gpu_count or m <= 0:
                raise ValueError(
                    f"max_gpu_count must be a positive whole number of "
                    f"devices, got {self.max_gpu_count}")
            if m < n_gpu:
                raise ValueError(
                    f"max_gpu_count {m} is below gpu_count {n_gpu}")
            force(self, "max_gpu_count", m)
            force(self, "gpu", True)
        # The degree axis: legal domain is [2, max_gpu_count]; absence means
        # "no opinion", never a defaulted legal value.
        if self.max_gpus_per_execution_group is not None:
            d = int(self.max_gpus_per_execution_group)
            if d != self.max_gpus_per_execution_group:
                raise ValueError(
                    f"max_gpus_per_execution_group must be a whole number of "
                    f"devices, got {self.max_gpus_per_execution_group}")
            if d < 2:
                raise ValueError(
                    f"max_gpus_per_execution_group {d} declares a sharding "
                    f"width that does not shard; omit the field to let the hub "
                    f"use the whole max_gpu_count as one group")
            if self.max_gpu_count is None or d > self.max_gpu_count:
                raise ValueError(
                    f"max_gpus_per_execution_group {d} exceeds max_gpu_count "
                    f"{self.max_gpu_count} — one request cannot span more GPUs "
                    f"than the pod may hold")
            force(self, "max_gpus_per_execution_group", d)
            force(self, "gpu", True)
        if self.parallel:
            mechanisms = tuple(
                str(x).strip().lower() for x in self.parallel if str(x).strip()
            )
            unknown = [m for m in mechanisms if m not in _PARALLEL_MECHANISMS]
            if unknown:
                raise ValueError(
                    f"parallel mechanism(s) {unknown} not implemented by the "
                    f"platform; known: {sorted(_PARALLEL_MECHANISMS)}")
            if len(set(mechanisms)) != len(mechanisms):
                raise ValueError(f"parallel repeats a mechanism: {mechanisms}")
            if self.max_gpu_count is None or self.max_gpu_count <= n_gpu:
                raise ValueError(
                    f"parallel {list(mechanisms)} declared without a "
                    f"max_gpu_count above gpu_count {n_gpu} — a mechanism "
                    f"is only reachable with device headroom")
            force(self, "parallel", mechanisms)
            force(self, "gpu", True)
        elif self.max_gpus_per_execution_group is not None:
            # A group width only bounds a MECHANISM; with none declared the
            # degree is always 1 and the field is inert. Mirrors the hub's own
            # ingest refusal.
            raise ValueError(
                f"max_gpus_per_execution_group "
                f"{self.max_gpus_per_execution_group} declared without a "
                f"parallel mechanism — nothing shards a group, so the "
                f"declaration is inert")


class NoWarmup(msgspec.Struct, frozen=True):
    """Opt a class out of the default boot warmup with a recorded
    reason: ``@endpoint(warmup=NoWarmup("engine captures CUDA graphs at
    boot"))``. Warmup is behavior — the opt-out lives in code, never in an
    env knob."""

    reason: str

    def __post_init__(self) -> None:
        reason = str(self.reason or "").strip()
        if not reason:
            raise ValueError("NoWarmup requires a non-empty reason")
        msgspec.structs.force_setattr(self, "reason", reason)


def _validate_warmup_decl(owner: str, warmup: Optional[NoWarmup]) -> Optional[NoWarmup]:
    if warmup is None or isinstance(warmup, NoWarmup):
        return warmup
    if isinstance(warmup, Mapping):
        raise TypeError(
            f"@endpoint {owner}: warmup= payload dicts are DELETED (pgw#654) "
            "— the warm plan is DERIVED: defaulted fields keep their "
            "defaults, CompileAxis fields cross-product their classes' "
            "warm= representatives, required fields synthesize neutral "
            "values. A per-function override for a non-axis field that "
            "genuinely changes tracing rides "
            "@worker_function(warm={...}, warm_reason=...). The only class "
            "form left is warmup=NoWarmup(reason)."
        )
    raise TypeError(
        f"@endpoint {owner}: warmup= must be NoWarmup(reason) or None "
        f"(derived warm plan, pgw#654), got {type(warmup).__name__}"
    )


# Per-function facts live ON the function, never in class-level name-keyed
# dicts; ``@worker_function`` is the one per-handler declaration surface.
WF_ATTR = "__gen_worker_function__"


REFERENCE_MODALITIES = ("image", "video", "audio")


class AcceptsReferences(msgspec.Struct, frozen=True, kw_only=True):
    """This handler OPTS IN to the platform reference layer.

    A reference is a tenant-defined character/object/place/style that a user
    writes as ``@handle`` in a prompt. The concept is PLATFORM-side: the hub
    resolves the handle, rewrites the prompt into plain English and delivers
    the reference's media onto an input this handler ALREADY declares. Nothing
    here reaches the handler at runtime — a reference-bearing request arrives
    as ordinary assets on ``delivery`` and a prompt with no ``@`` in it.

    ``max`` — how many distinct references one request may name.
    ``modalities`` — what this handler can absorb (image/video/audio).
    ``per_ref_images`` — how many media items ONE reference contributes, as an
    int (exactly n) or a ``(min, max)`` pair.
    ``delivery`` — the declared media input the media lands on, in the
    discovery contract's dotted path syntax (``references[].image``). The HUB
    validates at publish that this names a real media field of this function;
    declaring a path that does not exist fails the release, not a request.
    """

    max: int
    modalities: Tuple[str, ...]
    delivery: str
    per_ref_images: Any = 1

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        if not isinstance(self.max, int) or self.max < 1:
            raise ValueError(f"accepts_references: max must be a positive int, got {self.max!r}")
        mods = tuple(str(m).strip().lower() for m in self.modalities)
        if not mods:
            raise ValueError("accepts_references: modalities is required")
        for m in mods:
            if m not in REFERENCE_MODALITIES:
                raise ValueError(
                    f"accepts_references: modality {m!r} is not one of "
                    f"{'|'.join(REFERENCE_MODALITIES)}"
                )
        force(self, "modalities", mods)
        delivery = str(self.delivery or "").strip()
        if not delivery:
            raise ValueError("accepts_references: delivery is required")
        force(self, "delivery", delivery)
        lo, hi = _per_ref_bounds(self.per_ref_images)
        force(self, "per_ref_images", (lo, hi))

    def to_manifest(self) -> Dict[str, Any]:
        lo, hi = self.per_ref_images
        return {
            "max": int(self.max),
            "modalities": list(self.modalities),
            "per_ref_images": {"min": lo, "max": hi},
            "delivery": self.delivery,
        }


def _per_ref_bounds(raw: Any) -> Tuple[int, int]:
    if isinstance(raw, int) and not isinstance(raw, bool):
        lo = hi = int(raw)
    elif isinstance(raw, (tuple, list)) and len(raw) == 2:
        lo, hi = int(raw[0]), int(raw[1])
    elif isinstance(raw, Mapping):
        lo = int(raw.get("min", 1))
        hi = int(raw.get("max", lo))
    else:
        raise ValueError(
            "accepts_references: per_ref_images must be an int, a (min, max) pair "
            f"or a mapping, got {raw!r}"
        )
    if lo < 1 or hi < lo:
        raise ValueError(
            f"accepts_references: per_ref_images must satisfy 1 <= min <= max, got {lo}..{hi}"
        )
    return lo, hi


class WorkerFunctionDecl(msgspec.Struct, frozen=True, kw_only=True):
    """``@worker_function`` marker: this handler's own contract facts.

    ``objectives`` — which checkpoint training objectives the handler's
    scheduling/CFG code path serves (subset of
    :data:`~gen_worker.api.slot.OBJECTIVES`). ``None`` = unrestricted (the
    handler does not branch on the objective; non-diffusion endpoints
    simply never declare it).

    ``tasks`` — the SERVING TASKS this handler performs
    (subset of :data:`~gen_worker.api.slot.TASKS`). Several means one
    handler input-routes them (a merged ``generate()`` that does
    text-to-image when given no image and image-edit when given one).
    The hub scopes its human quality sign-off per task: an fp8 approval
    earned on text-to-image says nothing about image-edit on the same
    weights, so a handler that declares NOTHING can resolve no quant
    approval at all and serves at base precision.

    ``distilled`` — tri-state: ``True`` = serves only distilled
    checkpoints, ``False`` = only non-distilled, ``None`` = either (e.g. a
    turbo lane overlaying a distillation LoRA on standard weights AND
    passing through already-distilled ones).

    ``text_len`` — this handler's pinned text-sequence length,
    overriding the class ``Compile.text_len`` for its lane. The class cell
    contract digests the UNION of all lanes' pins, so a dual-pin class
    (qwen: 512 t2i / 1024 edit) describes both.

    ``warm``/``warm_reason`` — per-function warm-plan override for a
    NON-AXIS payload field that genuinely changes tracing. Needing it
    usually signals the field should be a CompileAxis; the mandatory
    ``warm_reason`` records why it is not.
    """

    objectives: Optional[Tuple[str, ...]] = None
    tasks: Optional[Tuple[str, ...]] = None
    distilled: Optional[bool] = None
    text_len: Optional[int] = None
    warm: Optional[Dict[str, Any]] = None
    warm_reason: str = ""
    # The opt-in reference contract. None = this handler never sees the
    # concept and a ref_text sent to it is refused hub-side.
    accepts_references: Optional[AcceptsReferences] = None

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        if self.objectives is not None:
            objs = tuple(str(v).strip() for v in self.objectives)
            if not objs:
                raise ValueError(
                    "@worker_function objectives= must not be empty (omit it "
                    "for an unrestricted handler)"
                )
            bad = [o for o in objs if o not in OBJECTIVES]
            if bad:
                raise ValueError(
                    f"@worker_function: unknown objective(s) {bad!r} "
                    f"(valid: {OBJECTIVES})"
                )
            if len(set(objs)) != len(objs):
                raise ValueError(f"@worker_function objectives= repeats a value: {objs!r}")
            force(self, "objectives", objs)
        if self.tasks is not None:
            tasks = tuple(str(v).strip().lower().replace("-", "_") for v in self.tasks)
            if not tasks:
                raise ValueError(
                    "@worker_function tasks= must not be empty (omit it for a "
                    "handler with no serving task, e.g. a utility function)"
                )
            bad = [t for t in tasks if t not in TASKS]
            if bad:
                raise ValueError(
                    f"@worker_function: unknown task(s) {bad!r} (valid: {TASKS})"
                )
            if len(set(tasks)) != len(tasks):
                raise ValueError(f"@worker_function tasks= repeats a value: {tasks!r}")
            force(self, "tasks", tasks)
        if self.text_len is not None:
            tl = int(self.text_len)
            if tl < 0:
                raise ValueError(
                    f"@worker_function text_len must be >= 0, got {self.text_len!r}"
                )
            force(self, "text_len", tl)
        if self.warm is not None:
            if not isinstance(self.warm, Mapping) or not self.warm:
                raise TypeError(
                    "@worker_function warm= must be a non-empty "
                    "{field: value} mapping"
                )
            if not str(self.warm_reason or "").strip():
                raise ValueError(
                    "@worker_function warm= requires warm_reason= — a "
                    "one-line justification for overriding the derived warm "
                    "plan on a non-axis field (needing it usually means the "
                    "field should be a CompileAxis)"
                )
            force(self, "warm", dict(self.warm))
        force(self, "warm_reason", str(self.warm_reason or "").strip())


def worker_function(
    *,
    objectives: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    distilled: Optional[bool] = None,
    text_len: Optional[int] = None,
    warm: Optional[Mapping[str, Any]] = None,
    warm_reason: str = "",
    accepts_references: Optional[AcceptsReferences] = None,
) -> Callable[[T], T]:
    """Per-handler contract facts, declared AT the definition
    site — on class handler methods and on bare ``@endpoint`` functions::

        @worker_function(objectives=("epsilon",), tasks=("text_to_image",))
        def generate(self, ctx, payload: In) -> Out: ...

    See :class:`WorkerFunctionDecl` for field semantics. The class-level
    ``regimes={...}`` / ``warmup={...}`` name-keyed dicts are DELETED —
    passing them to ``@endpoint`` is a decoration-time error.
    """
    decl = WorkerFunctionDecl(
        objectives=tuple(objectives) if objectives is not None else None,
        tasks=tuple(tasks) if tasks is not None else None,
        distilled=distilled,
        text_len=text_len,
        warm=dict(warm) if warm is not None else None,
        warm_reason=warm_reason,
        accepts_references=accepts_references,
    )

    def apply(fn: T) -> T:
        if not inspect.isfunction(fn):
            raise TypeError(
                f"@worker_function decorates handler functions/methods, got "
                f"{type(fn).__name__}"
            )
        if getattr(fn, WF_ATTR, None) is not None:
            raise ValueError(
                f"@worker_function: {fn.__name__!r} already carries a "
                "worker_function declaration"
            )
        setattr(fn, WF_ATTR, decl)
        return fn

    return apply


def _validate_handles(owner: str, handles: Any) -> Tuple[str, ...]:
    """Opt-in lane declaration: `handles=["fp8-w8a8-dynamic"]` marks that this
    endpoint's code BRANCHES on the executing lane (ctx.lane) — behavioral
    divergence only, never inventory. Tokens are concrete lane BODIES (no
    `+eager|+compiled`: execution is platform-managed). Nothing declared =
    fully platform-managed lanes."""

    if handles is None:
        return ()
    if isinstance(handles, str) or not isinstance(handles, (list, tuple)):
        raise TypeError(
            f"@endpoint {owner}: handles= must be a list/tuple of lane body "
            f"strings, got {type(handles).__name__}"
        )
    known = lanespec.known_execution_lane_bodies()
    out: list[str] = []
    for token in handles:
        t = str(token or "").strip().lower()
        if "+" in t:
            raise ValueError(
                f"@endpoint {owner}: handles= token {token!r} carries an "
                "execution axis; declare the lane body (execution is "
                "platform-managed)"
            )
        if t in lanespec.FAMILIES:
            raise ValueError(
                f"@endpoint {owner}: handles= token {token!r} is a coarse "
                f"family; declare a concrete lane body (one of {known})"
            )
        if t not in known:
            raise ValueError(
                f"@endpoint {owner}: handles= token {token!r} is not a known "
                f"lane body (known: {known})"
            )
        if t in out:
            raise ValueError(f"@endpoint {owner}: handles= repeats {token!r}")
        out.append(t)
    return tuple(out)


_CONFIG_TYPES: Dict[type, str] = {str: "string", int: "int", float: "float", bool: "bool"}


class ConfigParam(msgspec.Struct, frozen=True):
    """Declared config parameter: code declares the knob (name, type,
    default, constraints); the deployer sets values through the hub config
    API (write-time 422 on unknown/invalid); handlers read the effective
    value via ``ctx.config[name]``::

        @endpoint(config=[
            ConfigParam("scheduler", str, default="ddim", choices=["ddim", "euler_a"]),
            ConfigParam("default_steps", int, default=30, ge=1, le=150),
        ])
    """

    name: str
    type: type
    default: Any
    choices: Tuple[Any, ...] = ()
    ge: Optional[float] = None
    le: Optional[float] = None
    regex: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        name = str(self.name or "").strip()
        if not name or not name.isidentifier():
            raise ValueError(f"ConfigParam name {self.name!r} must be an identifier")
        force(self, "name", name)
        if self.type not in _CONFIG_TYPES:
            raise TypeError(
                f"ConfigParam {name!r}: type must be one of "
                f"{sorted(t.__name__ for t in _CONFIG_TYPES)}, got {self.type!r}"
            )
        self._check_value("default", self.default)
        if self.type is float and isinstance(self.default, int) and not isinstance(self.default, bool):
            force(self, "default", float(self.default))
        choices = tuple(self.choices)
        for c in choices:
            self._check_value("choices entry", c)
        if choices:
            if len(set(choices)) != len(choices):
                raise ValueError(f"ConfigParam {name!r}: choices must be unique")
            if self.default not in choices:
                raise ValueError(
                    f"ConfigParam {name!r}: default {self.default!r} not in "
                    f"choices {list(choices)}"
                )
        force(self, "choices", choices)
        if (self.ge is not None or self.le is not None) and self.type not in (int, float):
            raise ValueError(f"ConfigParam {name!r}: ge/le apply to int/float only")
        if self.ge is not None and self.le is not None and self.ge > self.le:
            raise ValueError(f"ConfigParam {name!r}: ge {self.ge} > le {self.le}")
        regex = str(self.regex or "")
        if regex and self.type is not str:
            raise ValueError(f"ConfigParam {name!r}: regex applies to str only")
        if regex:
            try:
                re.compile(regex)
            except re.error as exc:
                raise ValueError(
                    f"ConfigParam {name!r}: invalid regex {regex!r}: {exc}"
                ) from exc
            if re.search(regex, self.default) is None:
                raise ValueError(
                    f"ConfigParam {name!r}: default {self.default!r} does not "
                    f"match regex {regex!r}"
                )
        force(self, "regex", regex)
        force(self, "description", str(self.description or "").strip())

    def _check_value(self, label: str, value: Any) -> None:
        ok = isinstance(value, self.type) and not (
            self.type is not bool and isinstance(value, bool)
        )
        if self.type is float and isinstance(value, int) and not isinstance(value, bool):
            ok = True
        if not ok:
            raise TypeError(
                f"ConfigParam {self.name!r}: {label} {value!r} is not "
                f"{self.type.__name__}"
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.ge is not None and value < self.ge:
                raise ValueError(
                    f"ConfigParam {self.name!r}: {label} {value!r} < ge {self.ge}"
                )
            if self.le is not None and value > self.le:
                raise ValueError(
                    f"ConfigParam {self.name!r}: {label} {value!r} > le {self.le}"
                )

    def to_manifest(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "type": _CONFIG_TYPES[self.type],
            "default": self.default,
        }
        if self.choices:
            out["choices"] = list(self.choices)
        if self.ge is not None:
            out["ge"] = self.ge
        if self.le is not None:
            out["le"] = self.le
        if self.regex:
            out["regex"] = self.regex
        if self.description:
            out["description"] = self.description
        return out


def _validate_config_decl(owner: str, config: Any) -> Tuple[ConfigParam, ...]:
    if config is None:
        return ()
    if isinstance(config, ConfigParam) or not isinstance(config, (list, tuple)):
        raise TypeError(
            f"@endpoint {owner}: config= must be a list/tuple of ConfigParam, "
            f"got {type(config).__name__}"
        )
    out: list[ConfigParam] = []
    seen: set[str] = set()
    for p in config:
        if not isinstance(p, ConfigParam):
            raise TypeError(
                f"@endpoint {owner}: config= entries must be ConfigParam, "
                f"got {type(p).__name__}"
            )
        if p.name in seen:
            raise ValueError(f"@endpoint {owner}: config= repeats {p.name!r}")
        seen.add(p.name)
        out.append(p)
    return tuple(out)


def _validate_env_decl(owner: str, env: Any) -> Tuple[str, ...]:
    """Code declares the env names it reads; the hub validates config-layer
    env writes against this declaration (undeclared-write 422)."""
    if env is None:
        return ()
    if isinstance(env, str) or not isinstance(env, (list, tuple)):
        raise TypeError(
            f"@endpoint {owner}: env= must be a list/tuple of env-var name "
            f"strings, got {type(env).__name__}"
        )
    out: list[str] = []
    for name in env:
        n = str(name or "").strip()
        if (
            not n
            or len(n) > 64
            or not ("A" <= n[0] <= "Z")
            or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in n)
        ):
            raise ValueError(
                f"@endpoint {owner}: env= name {name!r} is not a valid "
                "environment variable name"
            )
        if n in out:
            raise ValueError(f"@endpoint {owner}: env= repeats {name!r}")
        out.append(n)
    return tuple(out)


class DynamicDim(msgspec.Struct, frozen=True):
    """One DECLARED dynamic dim of the compile shape contract.

    ``dim`` names the logical axis (``"batch"`` | ``"sequence"``); ``min``/
    ``max`` bound it. The SDK translates the declaration into explicit
    ``mark_dynamic(t, dim, min=, max=)`` marks BEFORE the first compile —
    dynamism only where declared, never discovered.

    ``min`` must be >= 2: torch's 0/1 specialization is not overridable
    (marking a size-1 dim dynamic is refused), so size 1 always gets its own
    free specialized graph and a declared range starts at 2. Measured cost of
    a declared-dynamic dim: +0.6% to +5.5%.
    """

    dim: str
    min: int
    max: int

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        d = str(self.dim or "").strip()
        # Any declared Dim name is admissible; Compile.__post_init__
        # cross-validates it against Compile.dims, so an unknown axis is still
        # refused — by cross-reference, not by a two-literal list. The two
        # logical dynamo axes stay case-normalized; declared Dim names match
        # exactly.
        if d.lower() in ("batch", "sequence"):
            d = d.lower()
        if not d.isidentifier():
            raise ValueError(
                f"DynamicDim.dim must name a logical axis (\"batch\", "
                f"\"sequence\", or a declared Compile.dims name), got {self.dim!r}"
            )
        force(self, "dim", d)
        lo, hi = int(self.min), int(self.max)
        if lo < 2:
            raise ValueError(
                f"DynamicDim({d!r}).min must be >= 2 (torch 0/1 "
                f"specialization is not overridable; size 1 gets its own "
                f"free specialized graph), got {lo}"
            )
        if hi <= lo:
            raise ValueError(f"DynamicDim({d!r}).max must exceed min, got {lo}..{hi}")
        force(self, "min", lo)
        force(self, "max", hi)


class Compile(msgspec.Struct, frozen=True):
    """Opt-in torch.compile over pre-built per-SKU cache artifacts, carrying
    the DECLARED shape contract: the compiler is TOLD every static bucket and
    every dynamic range up front; it never discovers a shape from traffic.

    ``Compile(family="flux2-klein-4b", shapes=((768, 768), (1024, 1024)),
    text_len=512)`` names the model FAMILY (caches key on the traced graph,
    so every fine-tune of a family shares one artifact), the spatial shape
    buckets the compile job warms, and the PINNED text-sequence length. A
    shape row is ``(width, height)`` for image models or ``(width, height,
    frames)`` for video models. Two-stage presets contribute BOTH
    their base and refined resolutions as rows.

    SHAPES DERIVE FROM THE PAYLOAD PRESET ENUM (fleet policy): when the
    endpoint's payload uses size buckets, ``shapes`` must be exactly that
    bucket table — one source of truth, 100% cache coverage of legal requests.

    ``text_len`` is the text-sequence axis: a prompt-length-dependent sequence
    dim reaching a statically-compiled denoiser mints a new graph per distinct
    prompt length — unbounded and un-warmable. Every ``compile=`` endpoint
    MUST state it (a walk-time lint enforces this): a positive value pins the
    token length (pad embeddings with :func:`gen_worker.pad_text_sequence` — a
    pin that fixes only the SIZE is not a pin, dynamo guards on STRIDES);
    ``0`` declares "this endpoint has no text conditioning" explicitly;
    alternatively declare the axis dynamic via ``dynamic=(DynamicDim(
    "sequence", min=..., max=...),)``.

    CFG graph classes are graph shapes too, but they are a PAYLOAD-FIELD fact:
    annotate the guidance field with :class:`~gen_worker.CompileAxis`
    equivalence classes. The warm plan derives from the cross-product of axis
    classes x shape buckets.

    Declaring this does NOT force compilation: the worker arms
    torch.compile only when a verified cache artifact for (family, SKU,
    torch, triton) is seeded — otherwise it stays eager. See
    ``gen_worker.compile_cache``.
    """

    shapes: tuple[tuple[int, ...], ...] = ()
    targets: tuple[str, ...] = ("transformer", "vae.decode")
    family: str = ""
    # Text-sequence axis: None = undeclared (walk-time lint failure for
    # compile= endpoints); 0 = explicitly unconditioned; >0 = pinned token
    # length.
    text_len: Optional[int] = None
    # Declared-dynamic dims (min/max ranges -> explicit marks).
    dynamic: tuple[DynamicDim, ...] = ()
    # Regional compilation (diffusers compile_repeated_blocks): compile the
    # target's repeated transformer blocks instead of the whole forward.
    # REQUIRED for big fp8 layerwise-cast models (measured on LTX 22B/H100):
    # whole-graph inductor planning co-materializes per-layer bf16 upcast
    # buffers and OOMs at the largest shapes; per-block graphs bound that to
    # one block. Also much faster cold compile (one block graph per shape,
    # reused across blocks). Cells record the mode — a mode drift consumer
    # stays eager (cache would miss anyway).
    regional: bool = False
    # ------------------------------------------------------------------
    # Export-declaration vocabulary. Per-family content is a DECLARATION
    # here, never worker code: named dims with (input, axis) bindings,
    # graph-class forks, and RESOLVED coordinate rows generated by the
    # endpoint's own legality oracle. See api/export_contract.py.
    # ------------------------------------------------------------------
    dims: tuple[Dim, ...] = ()
    forks: tuple[Fork, ...] = ()
    classes: tuple[GraphClass, ...] = ()
    inputs: tuple[Input, ...] = ()
    args: tuple[Arg, ...] = ()
    # Shape strategy is a per-family DECLARED choice — conv-bearing families
    # "static-rows" (symbolic latent H/W turns off inductor's channels-last
    # layout opt, +7.2% measured on sdxl), DiTs "dynamic-collapse". "" =
    # undeclared; refused at mint-plan time, not defaulted.
    shape_strategy: str = ""
    # Whether pre-warming the module changes the exported graph. A declared
    # per-family FACT, not a ritual — sdxl measured False, z-image measured
    # True (rope tables: 4327 cold vs 4285 warmed). None = unmeasured; refused
    # at mint time for declared families.
    warm_changes_key: Optional[bool] = None
    # Declared-eager lanes: recorded decision, not an omission.
    eager: tuple[str, ...] = ()
    # This family's ADOPTION numerics tolerance. A cell about to arm is run
    # against the eager forward it replaces and judged on the shared verdict
    # ladder; below `numerics_floor` it REFUSES to arm. None = the SDK default
    # (0.98 / 0.999) — see `numerics_ladder.NUMERICS_FLOOR` for the derivation.
    # Declared per family because bf16 attention reassociation makes exact
    # equality the wrong bar, and the right bar is not the same for a 25-block
    # fp8 DiT and a conv-bearing UNet.
    numerics_floor: Optional[float] = None
    numerics_warn: Optional[float] = None
    # This family's own compile-vs-eager SPEED bar — the stage it is read off
    # and the minimum speedup the compiled arm must clear. Declared by the
    # AUTHOR because the hub's publish-time validation session judges a release
    # against the author's bar and holds none of its own. Undeclared (the
    # default) resolves `bar_undeclared` at publish — a named refusal, never a
    # default number. A PAIR: see `validate_speed_bar`. Deliberately NOT a
    # contract axis — declaring or raising a bar must not re-key a cell — but
    # it IS a `derive.OVERRIDE_FACTS` entry, so a migration cannot drop it
    # silently.
    speed_metric: str = ""
    min_speedup: Optional[float] = None
    # DECLARED mint blockers. A family that may not mint yet says so as DATA
    # here, never as a thunk that raises (`compile=` takes a Compile, never a
    # callable, so a folded refusal would simply vanish). While any blocker is
    # unresolved the mint fails CLOSED and names the ids; serving is untouched.
    # Deliberately NOT a contract axis (`contract_axes()`): resolving a blocker
    # must not re-key a cell.
    blockers: tuple[MintBlocker, ...] = ()
    #: The latent divisor the CLASS ROWS were derived at, carried so the mint
    #: can reconcile it against the pipeline's real `vae_scale_factor` before
    #: it spends an export. CELL-LEVEL: one divisor per declaration, not a
    #: fact about any single row.
    #:
    #: NEVER AUTHORED. It is transferred off `derive.cfg_image_classes`'s
    #: return value in `__post_init__`; an author who writes it by hand is
    #: declaring the same number twice. `None` means the class rows did not
    #: come from a deriver that knows the divisor, and the mint reports
    #: UNRECONCILED — never a pass.
    #:
    #: DELIBERATELY ABSENT FROM :meth:`contract_axes`: it is a provenance fact
    #: about how the rows were COMPUTED, not a shape-contract axis (the latent
    #: extents it produced are already digested, via `classes`). Adding it
    #: there would re-key every cell in the fleet.
    latent_basis: Optional[int] = None

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        # BEFORE the row coercion below, which rebuilds `classes` as a PLAIN
        # tuple and drops any subclass attribute with it. The deriver's return
        # value is the transport; this field is the home.
        carried = getattr(self.classes, "latent_basis", None)
        if carried is not None and self.latent_basis is None:
            force(self, "latent_basis", int(carried))
        shapes = tuple(tuple(int(v) for v in s) for s in self.shapes)
        if (
            (not shapes and not self.classes)
            or any(len(s) not in (2, 3) for s in shapes)
            or any(v <= 0 for s in shapes for v in s)
        ):
            raise ValueError(
                "Compile.shapes must be positive (w, h) or (w, h, frames) "
                f"rows (omittable only when classes= carries the coordinate "
                f"rows), got {self.shapes!r}"
            )
        force(self, "shapes", shapes)
        targets = tuple(str(t).strip() for t in self.targets if str(t).strip())
        if not targets:
            raise ValueError("Compile.targets must not be empty")
        force(self, "targets", targets)
        force(self, "family", str(self.family or "").strip())
        if self.text_len is not None:
            tl = int(self.text_len)
            if tl < 0:
                raise ValueError(
                    f"Compile.text_len must be >= 0 (0 = explicitly "
                    f"unconditioned), got {self.text_len!r}"
                )
            force(self, "text_len", tl)
        dyn = tuple(self.dynamic)
        if any(not isinstance(d, DynamicDim) for d in dyn):
            raise TypeError("Compile.dynamic must be a tuple of DynamicDim rows")
        if len({d.dim for d in dyn}) != len(dyn):
            raise ValueError("Compile.dynamic repeats a dim")
        force(self, "dynamic", dyn)
        # `regional=True` + `dynamic=(...)` is ADMITTED and served by both
        # lanes. `regional` is the dynamo/JIT per-block knob — the AOT export
        # lane ignores it entirely; the dynamo regional branch applies the
        # declared marks at the compiled-block ingress
        # (`compile_cache._mark_regional_blocks`).
        for name, typ in (("dims", Dim), ("forks", Fork),
                          ("classes", GraphClass), ("inputs", Input),
                          ("args", Arg), ("blockers", MintBlocker)):
            rows = tuple(getattr(self, name))
            if any(not isinstance(r, typ) for r in rows):
                raise TypeError(
                    f"Compile.{name} must be a tuple of {typ.__name__} rows")
            force(self, name, rows)
        force(self, "shape_strategy", str(self.shape_strategy or "").strip())
        if self.warm_changes_key is not None:
            force(self, "warm_changes_key", bool(self.warm_changes_key))
        force(self, "eager", tuple(str(e).strip() for e in self.eager))
        for name in ("numerics_floor", "numerics_warn"):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Compile.{name} is a COSINE bound and must be in "
                    f"[0, 1], got {value!r}")
            force(self, name, value)
        if (self.numerics_floor is not None and self.numerics_warn is not None
                and self.numerics_floor > self.numerics_warn):
            raise ValueError(
                f"Compile.numerics_floor ({self.numerics_floor}) must not "
                f"exceed numerics_warn ({self.numerics_warn}): the floor "
                f"refuses, the warn only confesses")
        metric, bar = validate_speed_bar(self.speed_metric, self.min_speedup)
        force(self, "speed_metric", metric)
        force(self, "min_speedup", bar)
        validate_contract(self)

    @property
    def open_blockers(self) -> tuple[MintBlocker, ...]:
        """The UNRESOLVED blockers. Non-empty ⟹ this family refuses
        to mint, by declaration, and every mint site fails closed on it."""
        return open_blockers(self)

    @property
    def sequence_dynamic(self) -> Optional[DynamicDim]:
        for d in self.dynamic:
            if d.dim == "sequence":
                return d
        return None

    @property
    def batch_dynamic(self) -> Optional[DynamicDim]:
        for d in self.dynamic:
            if d.dim == "batch":
                return d
        return None

    def contract_axes(self) -> Dict[str, Any]:
        """The canonical shape-contract facts (feeds the cell key's contract
        digest: a worker on a newer contract must never consume an older
        cell)."""
        axes: Dict[str, Any] = {
            "shapes": [list(s) for s in self.shapes],
            "targets": list(self.targets),
            "text_len": self.text_len,
            "dynamic": [
                {"dim": d.dim, "min": d.min, "max": d.max} for d in self.dynamic
            ],
            "regional": bool(self.regional),
        }
        # Declaration facts, present only when declared so a family that has
        # not adopted the vocabulary keeps its existing digest.
        if self.dims:
            axes["dims"] = [d.as_row() for d in self.dims]
        if self.forks:
            axes["forks"] = [f.as_row() for f in self.forks]
        if self.classes:
            axes["classes"] = [c.as_row() for c in self.classes]
        if self.inputs:
            axes["inputs"] = [i.as_row() for i in self.inputs]
        if self.args:
            axes["args"] = [a.as_row() for a in self.args]
        if self.shape_strategy:
            axes["shape_strategy"] = self.shape_strategy
        if self.warm_changes_key is not None:
            axes["warm_changes_key"] = self.warm_changes_key
        if self.eager:
            axes["eager"] = list(self.eager)
        return axes


class EndpointDecl(msgspec.Struct, frozen=True, kw_only=True):
    """Metadata attached by ``@endpoint``; read by the registry walker."""

    kind: str = "inference"
    resources: Resources = msgspec.field(default_factory=Resources)
    models: Mapping[str, Binding] = msgspec.field(default_factory=dict)
    # Slot-declared entries in `models=`/`model=`: a subset of
    # `models`' keys, carrying the Slot's selected_by/default_checkpoint
    # metadata that `models` (a plain Binding map every model-injection
    # call site understands) can't hold.
    slots: Mapping[str, Slot[Any]] = msgspec.field(default_factory=dict)
    runtime: Optional[str] = None
    name: Optional[str] = None  # function-shaped endpoints only
    is_function: bool = False
    compile: Optional[Compile] = None
    # The function makes endpoint-to-endpoint child calls (ctx.call_endpoint /
    # ctx.workflow_checkpoint). The hub mints the invoke_child credential ONLY
    # for declaring functions.
    child_calls: bool = False
    # Handlers on ONE live instance run SINGLE-FLIGHT by default (one binding
    # set = one materialized graph with mutable buffers — e.g. a resident LoRA
    # branch; two concurrent requests would corrupt each other).
    # ``reentrant=True`` is the explicit opt-in for classes whose handlers
    # genuinely mutate no instance state.
    reentrant: bool = False
    # Dynamic-LoRA endpoints declare the resident traced rank bucket at the
    # DECORATOR (it shapes the graph family and the resident branch buffers
    # whether or not compile= is present). The worker serves the branch-bearing
    # graph family: canonical zeroed rank-<bucket> branches enabled at load,
    # only `<lane>-lora<bucket>` cells adopt, and adapter swaps stay buffer
    # copies — never a recompile. 0 = branchless.
    lora_bucket: int = 0
    # None = the DERIVED warm plan (defaults + axis cross-product +
    # synthesized required fields, union-deduped per graph class);
    # NoWarmup(reason) = class-level opt-out.
    warmup: Optional[NoWarmup] = None
    # Opt-in declared lane bodies this endpoint's code branches on (ctx.lane).
    # Empty = platform-managed behavior only.
    handles: Tuple[str, ...] = ()
    # Declared compute-time formulas, attr_name -> RuntimeFormula
    # (function-shaped endpoints key their single handler under "").
    # Declared via the runtime= kwarg overload; payload-field validation
    # happens at registry walk time when payload types are resolved.
    runtime_formula: Mapping[str, RuntimeFormula] = msgspec.field(default_factory=dict)
    # Declared config parameters (ctx.config) + declared env names — the
    # mutable-config surface the hub validates writes against.
    config: Tuple[ConfigParam, ...] = ()
    env: Tuple[str, ...] = ()
    # th#2049/pgw#1294: this function MAY write tensors/repos to the hub. ONE
    # declaration surface for @endpoint and @job alike — Paul's tensor-emission
    # ruling, which is what makes a serverless-LoRA trainer possible. It says
    # MAY write; the request still says WHERE. The hub mints the
    # worker-capability write grant off this declaration, never off the kind.
    publishes: bool = False


ATTR = "__gen_worker_endpoint__"
VARIANT_ATTR = "__gen_worker_variant__"


class VariantDecl(msgspec.Struct, frozen=True):
    """``@variant_of`` marker: this handler is the ``kind`` variant
    of sibling function ``of`` on the same endpoint."""

    of: str
    kind: str

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        of = str(self.of or "").strip()
        kind = str(self.kind or "").strip().lower()
        if not of:
            raise ValueError("@variant_of requires a target function name")
        if not kind:
            raise ValueError("@variant_of requires a non-empty kind")
        force(self, "of", of)
        force(self, "kind", kind)


def variant_of(of: str, *, kind: str = "turbo") -> Callable[[T], T]:
    """Declare a handler as a variant of a sibling function.

    The pairing is emitted into the discovery manifest (``variant_of`` /
    ``variant``) and surfaced on tensorhub's public endpoint info, so a
    platform can render e.g. a regular/turbo toggle instead of modeling the
    variant as a separate product::

        @variant_of("generate")  # kind="turbo"
        def generate_turbo(self, ctx, payload: TurboInput) -> ImageOutput: ...

    The target must be another function on the same endpoint and must not
    itself be a variant (validated at discovery time).
    """
    decl = VariantDecl(of=of, kind=kind)

    def apply(fn: T) -> T:
        if not inspect.isfunction(fn):
            raise TypeError(
                f"@variant_of decorates handler functions/methods, got "
                f"{type(fn).__name__}"
            )
        setattr(fn, VARIANT_ATTR, decl)
        return fn

    return apply


def _handler_params(fn: Callable[..., Any], *, is_method: bool) -> list[inspect.Parameter]:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if is_method:
        params = [p for p in params if p.name != "self"]
    return params


def _validate_handler_shape(owner: str, fn: Callable[..., Any], *, is_method: bool) -> None:
    params = _handler_params(fn, is_method=is_method)
    if len(params) < 2:
        raise TypeError(
            f"@endpoint: {owner} must accept (ctx, payload) "
            f"(got params {[p.name for p in params]}). Handlers take the "
            "request context first and a msgspec.Struct payload second; "
            "prefix non-handler methods with an underscore."
        )
    if is_method and len(params) > 2:
        raise TypeError(
            f"@endpoint: {owner} must accept exactly (self, ctx, payload) — "
            f"got extra params {[p.name for p in params[2:]]}. SDK v2 "
            "(pgw#647): per-handler model args are rejected; setup(self, "
            "pipeline, ...) runs once per instance and stores "
            "self.pipeline — one live instance == one binding set, so the "
            "runtime routed this request here BECAUSE the bindings match."
        )


def _reject_producer_generator(owner: str, fn: Callable[..., Any], kind: str) -> None:
    """Producer kinds (conversion/training/dataset) publish explicitly and
    return a result; a generator handler would stream chunks and publish
    NOTHING. Fail at decoration time instead of silently at runtime."""
    if kind == "inference":
        return
    if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
        raise TypeError(
            f"@endpoint(kind={kind!r}): {owner} must not be a generator. "
            "Producer endpoints write files locally, publish explicitly "
            "(gen_worker.convert.publish_flavors(ctx, flavors)), and return a "
            "result struct; streaming generators are inference-only."
        )


def _split_slot_like(
    key: str, v: "SlotLike[D]",
) -> Tuple[Optional[Binding], Optional[Slot[D]]]:
    """One ``models={}``/``model=`` value -> ``(binding_or_None, slot_or_None)``.

    A ``Slot`` contributes its ``default_checkpoint`` (if any) to the plain
    binding map every existing model-injection call site (executor, CLI,
    prefetch) already understands, PLUS itself to the slots map discovery
    emission and the resolution chain read. A bare binding contributes only to
    the binding map — it carries no Slot metadata."""
    if isinstance(v, Slot):
        return v.default_checkpoint, v
    if isinstance(v, BINDING_TYPES):
        return v, None
    raise TypeError(
        f"@endpoint models[{key!r}] must be a HF/Hub/Civitai/ModelScope "
        f"binding or Slot(...), got {type(v).__name__}"
    )


def _normalize_models(
    model: Optional["SlotLike[D]"], models: Optional[Mapping[str, "SlotLike[D]"]]
) -> Tuple[Dict[str, Binding], Dict[str, Slot[D]]]:
    if model is not None and models is not None:
        raise ValueError("@endpoint: pass model= OR models=, not both")
    if model is not None:
        # Single-binding shorthand: the slot name is resolved from the
        # setup()/handler parameter name at class validation time.
        binding, slot = _split_slot_like("", model)
        out_models: Dict[str, Binding] = {"": binding} if binding is not None else {}
        out_slots: Dict[str, Slot[D]] = {"": slot} if slot is not None else {}
        return out_models, out_slots
    out_models = {}
    out_slots = {}
    for key, v in (models or {}).items():
        k = str(key or "").strip()
        if not k or not k.isidentifier():
            raise ValueError(f"@endpoint models key {key!r} must be an identifier")
        binding, slot = _split_slot_like(k, v)
        if binding is not None:
            out_models[k] = binding
        if slot is not None:
            out_slots[k] = slot
    return out_models, out_slots


def _find_handler_methods(cls: type) -> list[tuple[str, Callable[..., Any]]]:
    out: list[tuple[str, Callable[..., Any]]] = []
    for attr, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        if attr.startswith("_") or attr in RESERVED_METHODS:
            continue
        out.append((attr, member))
    if not out:
        raise TypeError(
            f"@endpoint class {cls.__name__!r}: no handler methods found. "
            "Define at least one public method taking (self, ctx, payload)."
        )
    for handler_name, handler_fn in out:
        _validate_handler_shape(f"{cls.__name__}.{handler_name}", handler_fn, is_method=True)
    return out


def _setup_params(cls: type) -> Optional[dict[str, inspect.Parameter]]:
    """Named params of ``setup`` (excluding self), or None when absent."""
    setup = inspect.getattr_static(cls, "setup", None)
    if setup is None:
        return None
    fn = setup.__func__ if isinstance(setup, (classmethod, staticmethod)) else setup
    sig = inspect.signature(fn)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    return {p.name: p for p in params}


def _is_server_handle_param(p: inspect.Parameter) -> bool:
    """True when the param is annotated ``ServerHandle`` (a runtime= injection
    target, not a model slot)."""
    ann = p.annotation
    if ann is inspect.Parameter.empty:
        return False

    if ann is ServerHandle:
        return True
    return isinstance(ann, str) and ann.split(".")[-1] == "ServerHandle"


def _resolve_single_slot(
    cls: type,
    models: Dict[str, Binding],
    slots: Dict[str, Slot[D]],
    handlers: list[tuple[str, Callable[..., Any]]],
) -> Tuple[Dict[str, Binding], Dict[str, Slot[D]]]:
    """Resolve the ``model=`` shorthand's slot name from setup()/handler
    params. Exactly one of ``models``/``slots`` holds the ``""`` key (a bare
    binding or a Slot, never both — ``_normalize_models`` is the only
    producer)."""
    if "" not in models and "" not in slots:
        return models, slots

    def _name_it() -> str:
        setup_kwargs = _setup_params(cls)
        if setup_kwargs:
            named = [
                n for n, p in setup_kwargs.items()
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                and not _is_server_handle_param(p)
            ]
            if len(named) == 1:
                return named[0]
            raise ValueError(
                f"@endpoint class {cls.__name__!r}: model= needs exactly one "
                f"setup() parameter to name the slot (setup declares {named})."
            )
        raise ValueError(
            f"@endpoint class {cls.__name__!r}: model= requires a setup() "
            "method (SDK v2: instances own their models — setup(self, "
            "pipeline) stores self.pipeline; handlers are (self, ctx, "
            "payload) with no per-handler model args)."
        )

    name = _name_it()
    if "" in models:
        models[name] = models.pop("")
    if "" in slots:
        slots[name] = slots.pop("")
    return models, slots


def _validate_class_models(
    cls: type, models: Dict[str, Binding], slots: Dict[str, Slot[Any]]
) -> None:
    """Model slots must be consumable: by setup() params (stateful) or by
    handler params (per-call injection when there is no setup)."""
    all_keys = set(models) | set(slots)
    if not all_keys:
        return
    setup_kwargs = _setup_params(cls)
    if setup_kwargs is None:
        raise ValueError(
            f"@endpoint class {cls.__name__!r}: models slot(s) "
            f"{sorted(all_keys)} require a setup() method (SDK v2: "
            "instances own their models; per-handler injection is deleted)."
        )
    if setup_kwargs is not None:
        has_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in setup_kwargs.values()
        )
        missing = [k for k in all_keys if k not in setup_kwargs and not has_var_kw]
        if missing:
            raise ValueError(
                f"@endpoint class {cls.__name__!r}: models slot(s) {missing} "
                f"match no setup() parameter (setup declares "
                f"{sorted(setup_kwargs)})."
            )
        _derive_optional_slots(cls, slots, setup_kwargs)


def _derive_optional_slots(
    cls: type, slots: Dict[str, Slot[Any]], setup_kwargs: dict[str, inspect.Parameter]
) -> None:
    """A slot is OPTIONAL exactly when its ``setup()`` parameter has a
    default — the signature is the single declaration, so code and manifest
    can never disagree. Optional slots may be left unbound at deploy time
    (which lanes a release serves is deploy config); setup then runs with the
    parameter's own default.

    A defaulted parameter whose annotation cannot hold that default is a
    decoration-time error: injection is typed off the annotation, and an
    endpoint that writes ``edit: Pipe = None`` would type-lie to every
    reader and to ``typing.get_type_hints``."""
    for name, slot in slots.items():
        param = setup_kwargs.get(name)
        if param is None or param.default is inspect.Parameter.empty:
            slot.optional = False
            continue
        slot.optional = True
        if param.default is None and not _annotation_admits_none(param.annotation):
            raise ValueError(
                f"@endpoint class {cls.__name__!r}: setup parameter {name!r} "
                f"defaults to None but its annotation "
                f"({param.annotation!r}) does not admit None — an optional "
                f"slot must be annotated e.g. "
                f"{name}: {getattr(slot.pipeline_cls, '__name__', 'X')} | None = None"
            )


def _annotation_admits_none(ann: Any) -> bool:
    """True for ``X | None`` / ``Optional[X]`` / bare ``None``, including the
    string form a ``from __future__ import annotations`` module produces."""
    if ann is inspect.Parameter.empty or ann is None or ann is type(None):
        return True
    if isinstance(ann, str):
        return "None" in ann or ann.startswith("Optional[")
    return type(None) in get_args(ann)


def _validate_root_slot(owner: str, slots: Dict[str, Slot[Any]]) -> None:
    """A multi-slot class must designate ONE root — ambiguity is a
    decoration-time error, never a silent runtime fallback (which would drop
    e.g. a v-pred checkpoint's prediction_type)."""
    roots = [n for n, s in slots.items() if s.root]
    if len(roots) > 1:
        raise ValueError(
            f"@endpoint {owner}: more than one Slot(root=True) ({sorted(roots)}) "
            "— exactly one slot may be the root"
        )
    if len(slots) > 1 and not roots and "pipeline" not in slots:
        raise ValueError(
            f"@endpoint {owner}: {len(slots)} model slots ({sorted(slots)}) "
            "and no unambiguous root — mark exactly one with "
            "Slot(root=True) (it backs ctx.defaults / ctx.for_request when "
            "no slot= is passed; handlers on other lanes pass "
            "slot=<name> explicitly)."
        )


def _validate_lora_state_dict(owner: str, slots: Dict[str, Slot[Any]], lora_bucket: int) -> None:
    """A ``lora_bucket=`` endpoint whose root pipeline class
    is NON-introspectable (non-diffusers) must declare
    ``lora_state_dict`` — the adapter normalization path routes through
    ``type(pipe).lora_state_dict``, and without a converter every adapter
    attach fails closed at FIRST LIVE ATTACH (comfy-grammar keys fall to
    the peft path, which raises on pipelines lacking ``load_lora_weights``).
    ``str``/``Path`` self-loading slots are unverifiable here (the real
    pipeline class is endpoint-internal) and are skipped."""
    if not lora_bucket or not slots:
        return

    roots = [s for s in slots.values() if s.root]
    root = roots[0] if roots else slots.get("pipeline") or (
        next(iter(slots.values())) if len(slots) == 1 else None
    )
    if root is None:
        return
    cls = root.pipeline_cls
    if not isinstance(cls, type) or cls in (str, bytes) or issubclass(cls, (str, bytes)):
        return
    try:
        from pathlib import PurePath

        if issubclass(cls, PurePath):
            return
    except TypeError:
        pass
    if is_introspectable(cls):
        return
    if not callable(getattr(cls, "lora_state_dict", None)):
        raise ValueError(
            f"@endpoint {owner}: lora_bucket={lora_bucket} with a "
            f"non-introspectable root pipeline class {cls.__name__!r} that "
            "does not define lora_state_dict. Non-diffusers pipelines MUST "
            "declare a lora_state_dict classmethod converting adapter key "
            "grammars to the canonical 'transformer.'/'unet.' form, or every "
            "adapter (curated and BYO) fails closed at first attach."
        )


def _split_runtime_kwarg(
    owner: str, runtime: Any,
) -> Tuple[Optional[str], Dict[str, RuntimeFormula]]:
    """runtime= overload: a str selects an engine runtime (vllm /
    llama-server, classes only); a RuntimeFormula declares the compute-time
    formula for every handler ("*"); a mapping declares per-handler formulas
    (classes only). Returns (engine_runtime, {attr_or_"*": formula})."""
    if runtime is None:
        return None, {}
    if isinstance(runtime, str):
        return runtime, {}
    if isinstance(runtime, RuntimeFormula):
        return None, {"*": runtime}
    if isinstance(runtime, Mapping):
        out: Dict[str, RuntimeFormula] = {}
        for attr, rf in runtime.items():
            if not isinstance(attr, str) or not isinstance(rf, RuntimeFormula):
                raise TypeError(
                    f"@endpoint {owner}: runtime= mapping must be "
                    f"{{handler_name: RuntimeFormula}}"
                )
            out[attr] = rf
        return None, out
    raise TypeError(
        f"@endpoint {owner}: runtime= must be an engine runtime string, a "
        f"RuntimeFormula, or a mapping of handler name -> RuntimeFormula, "
        f"got {type(runtime).__name__}"
    )


def _expand_formula_map(
    owner: str, formulas: Dict[str, RuntimeFormula], handler_attrs: "list[str]",
) -> Dict[str, RuntimeFormula]:
    if not formulas:
        return {}
    if "*" in formulas:
        if len(formulas) > 1:
            raise ValueError(
                f"@endpoint {owner}: runtime= cannot mix a bare RuntimeFormula "
                f"with per-handler entries"
            )
        return {attr: formulas["*"] for attr in handler_attrs}
    unknown = [a for a in formulas if a not in handler_attrs]
    if unknown:
        raise ValueError(
            f"@endpoint {owner}: runtime= names unknown handler(s) {unknown} "
            f"(handlers: {sorted(handler_attrs)})"
        )
    return dict(formulas)


def _decorate_class(
    cls: type,
    *,
    kind: str,
    resources: Resources,
    models: Dict[str, Binding],
    slots: Dict[str, Slot[Any]],
    runtime: Optional[str],
    runtime_formula: Optional[Dict[str, RuntimeFormula]] = None,
    compile: Optional[Compile] = None,
    child_calls: bool = False,
    reentrant: bool = False,
    lora_bucket: int = 0,
    warmup: Optional[NoWarmup] = None,
    handles: Optional[Any] = None,
    config: Optional[Any] = None,
    env: Optional[Any] = None,
    publishes: bool = False,
) -> type:
    handlers = _find_handler_methods(cls)
    for attr, member in handlers:
        _reject_producer_generator(f"{cls.__name__}.{attr}", member, kind)
    models, slots = _resolve_single_slot(cls, models, slots, handlers)
    _validate_class_models(cls, models, slots)
    _validate_root_slot(cls.__name__, slots)
    _validate_lora_state_dict(cls.__name__, slots, lora_bucket)

    if runtime is not None and runtime not in ("vllm", "llama-server"):
        raise ValueError(
            f"@endpoint class {cls.__name__!r}: runtime must be 'vllm' or "
            f"'llama-server', got {runtime!r}"
        )

    decl = EndpointDecl(
        kind=kind, resources=resources, models=models, slots=slots,
        runtime=runtime, compile=compile, child_calls=child_calls,
        reentrant=reentrant, lora_bucket=lora_bucket,
        warmup=_validate_warmup_decl(cls.__name__, warmup),
        handles=_validate_handles(cls.__name__, handles),
        runtime_formula=_expand_formula_map(
            cls.__name__, runtime_formula or {}, [attr for attr, _ in handlers]
        ),
        config=_validate_config_decl(cls.__name__, config),
        env=_validate_env_decl(cls.__name__, env),
        publishes=bool(publishes),
    )
    setattr(cls, ATTR, decl)
    setattr(cls, "__gen_worker_handlers__", handlers)
    # Default-on boot warmup — fail unwarmable GPU inference classes at import
    # when type hints resolve here (walk time re-checks). Lazy import:
    # warmup.py imports this module.
    from ..warmup import validate_at_decoration

    validate_at_decoration(cls, decl)
    return cls


def _decorate_function(
    fn: Callable[..., Any],
    *,
    kind: str,
    resources: Resources,
    models: Dict[str, Binding],
    slots: Dict[str, Slot[Any]],
    runtime: Optional[str],
    runtime_formula: Optional[Dict[str, RuntimeFormula]] = None,
    name: Optional[str],
    compile: Optional[Compile] = None,
    child_calls: bool = False,
    reentrant: bool = False,
    lora_bucket: int = 0,
    warmup: Optional[NoWarmup] = None,
    handles: Optional[Any] = None,
    config: Optional[Any] = None,
    env: Optional[Any] = None,
    publishes: bool = False,
) -> Callable[..., Any]:
    if reentrant:
        raise ValueError(
            f"@endpoint function {fn.__name__!r}: reentrant= applies to "
            "class endpoints only (stateless functions hold no instance "
            "state to single-flight)."
        )
    _validate_handler_shape(fn.__name__, fn, is_method=False)
    _reject_producer_generator(fn.__name__, fn, kind)
    if "" in models or "" in slots:
        injected = [p.name for p in _handler_params(fn, is_method=False)[2:]]
        if len(injected) != 1:
            raise ValueError(
                f"@endpoint function {fn.__name__!r}: model= needs exactly one "
                f"injected parameter after (ctx, payload) to name the slot "
                f"(found {injected or 'none'})."
            )
        slot_name = injected[0]
        if "" in models:
            models[slot_name] = models.pop("")
        if "" in slots:
            slots[slot_name] = slots.pop("")
    if runtime is not None:
        raise ValueError(
            f"@endpoint function {fn.__name__!r}: an engine runtime= requires "
            "a class with setup() (the engine server outlives single calls)."
        )
    if warmup is not None:
        raise ValueError(
            f"@endpoint function {fn.__name__!r}: warmup= requires a class "
            "with setup() (stateless functions hold nothing to warm)."
        )
    formulas = runtime_formula or {}
    if set(formulas) - {"*"}:
        raise ValueError(
            f"@endpoint function {fn.__name__!r}: runtime= takes a bare "
            "RuntimeFormula (per-handler mappings are for classes)"
        )
    decl = EndpointDecl(
        kind=kind, resources=resources, models=models, slots=slots,
        runtime=None, name=(name or fn.__name__), is_function=True,
        compile=compile, child_calls=child_calls, lora_bucket=lora_bucket,
        handles=_validate_handles(fn.__name__, handles),
        runtime_formula={"": formulas["*"]} if "*" in formulas else {},
        config=_validate_config_decl(fn.__name__, config),
        env=_validate_env_decl(fn.__name__, env),
        publishes=bool(publishes),
    )
    setattr(fn, ATTR, decl)
    return fn


@overload
def endpoint(target: T) -> T: ...  # bare @endpoint on a class/function


@overload
def endpoint(
    *,
    kind: str = ...,
    model: Optional["SlotLike[Any]"] = ...,
    models: Optional[Mapping[str, "SlotLike[Any]"]] = ...,
    resources: Optional[Resources] = ...,
    runtime: Union[str, RuntimeFormula, Mapping[str, RuntimeFormula], None] = ...,
    name: Optional[str] = ...,
    compile: Optional[Compile] = ...,
    child_calls: bool = ...,
    reentrant: bool = ...,
    lora_bucket: int = ...,
    warmup: Optional[NoWarmup] = ...,
    handles: Optional[Any] = ...,
    config: Optional[Sequence[ConfigParam]] = ...,
    env: Optional[Sequence[str]] = ...,
    publishes: bool = ...,
) -> Callable[[T], T]: ...  # configured @endpoint(...) form


def endpoint(
    target: Optional[T] = None,
    *,
    kind: str = "inference",
    model: Optional["SlotLike[Any]"] = None,
    models: Optional[Mapping[str, "SlotLike[Any]"]] = None,
    resources: Optional[Resources] = None,
    runtime: Union[str, RuntimeFormula, Mapping[str, RuntimeFormula], None] = None,
    name: Optional[str] = None,
    compile: Optional[Compile] = None,
    child_calls: bool = False,
    reentrant: bool = False,
    lora_bucket: int = 0,
    warmup: Optional[NoWarmup] = None,
    handles: Optional[Any] = None,
    config: Optional[Sequence[ConfigParam]] = None,
    env: Optional[Sequence[str]] = None,
    publishes: bool = False,
) -> Any:
    """The one endpoint decorator. See the module docstring for shapes.

    ``model=``/``models=`` values are a ``Binding`` (a fixed pick — HF/Hub/
    Civitai/ModelScope) or a :class:`~gen_worker.api.slot.Slot` (a
    hub-resolved slot: ``selected_by=``, ``default_checkpoint=``; SDK v2).
    A bare binding is sugar for
    ``Slot(<inferred pipeline class>, default_checkpoint=ref)`` with no hub
    involvement — both forms can mix in one ``models={}`` dict.
    """
    if kind not in KINDS:
        raise ValueError(f"@endpoint kind must be one of {KINDS}, got {kind!r}")
    resources_value = resources if resources is not None else Resources()
    if resources is not None and not isinstance(resources, Resources):
        raise TypeError(
            f"@endpoint resources= must be a Resources, got {type(resources).__name__}"
        )
    if compile is not None and not isinstance(compile, Compile):
        raise TypeError(
            f"@endpoint compile= must be a Compile, got {type(compile).__name__}"
        )
    if compile is not None and kind == "inference" \
            and compile.text_len is None and compile.sequence_dynamic is None:
        raise ValueError(
            "@endpoint compile=: no text-sequence axis declared (SDK v2 "
            "lint, ie#544). A variable text dim reaching a statically "
            "compiled denoiser mints a new graph per prompt length — "
            "unbounded and un-warmable. Declare Compile(text_len=<pinned "
            "token length>) (pad embeddings with gen_worker."
            "pad_text_sequence), Compile(text_len=0) for an explicitly "
            "unconditioned model, or a dynamic range via "
            "Compile(dynamic=(DynamicDim('sequence', min=.., max=..),))."
        )
    if compile is not None:
        # An export declaration is the family's export contract; registering at
        # decoration is what lets the mint derive export inputs with zero
        # per-family SDK code.
        #
        # The gate asks about INTENT, not about the payload: whether the author
        # reached for the export vocabulary at all. Requiring `classes` of every
        # `compile=` would be wrong — endpoints ship a thin class-less
        # `compile=` for the dynamo lane only and declare no export contract.
        # If the author did reach for it, `register_export_declaration` holds
        # them to it.
        from .export_contract import (
            declares_export_contract, register_export_declaration,
        )

        if declares_export_contract(compile):
            register_export_declaration(compile)
    bucket = int(lora_bucket or 0)
    if bucket:
        from ..models.w8a8_lora import RANK_BUCKETS

        if bucket not in RANK_BUCKETS:
            raise ValueError(
                f"@endpoint lora_bucket must be 0 or one of {RANK_BUCKETS}, "
                f"got {lora_bucket!r}"
            )
    model_map, slot_map = _normalize_models(model, models)
    owner = getattr(target, "__name__", "<endpoint>") if target is not None else "<endpoint>"
    engine_runtime, runtime_formulas = _split_runtime_kwarg(owner, runtime)

    def apply(obj: Any) -> Any:
        if inspect.isclass(obj):
            if name is not None:
                raise ValueError("@endpoint name= applies to functions only; "
                                 "class handlers route by method name")
            return _decorate_class(
                obj, kind=kind, resources=resources_value, models=dict(model_map),
                slots=dict(slot_map), runtime=engine_runtime,
                runtime_formula=runtime_formulas, compile=compile,
                child_calls=child_calls, reentrant=reentrant,
                lora_bucket=bucket, warmup=warmup,
                handles=handles, config=config, env=env,
                publishes=publishes,
            )
        if inspect.isfunction(obj):
            return _decorate_function(
                obj, kind=kind, resources=resources_value, models=dict(model_map),
                slots=dict(slot_map), runtime=engine_runtime,
                runtime_formula=runtime_formulas, name=name, compile=compile,
                child_calls=child_calls, reentrant=reentrant,
                lora_bucket=bucket, warmup=warmup,
                handles=handles, config=config, env=env,
                publishes=publishes,
            )
        raise TypeError(
            f"@endpoint requires a function or class, got {type(obj).__name__}"
        )

    if target is not None:
        return apply(target)
    return apply


__all__ = [
    "Compile", "ConfigParam", "DynamicDim", "EndpointDecl", "NoWarmup",
    "Resources", "SlotLike", "VariantDecl", "WorkerFunctionDecl",
    "endpoint", "variant_of", "worker_function",
]
