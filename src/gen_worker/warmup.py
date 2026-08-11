"""Boot-time warm-plan DERIVATION (gw#470 -> pgw#654).

First-call tax on a fresh worker is EAGER cost — allocator-pool growth to
the activation peak plus cuBLAS/cuDNN heuristic selection (measured 216s vs
63s warm on LTX/H100 pre-expandable-segments; 0-152s host lottery on B200).
The worker runs synthetic requests per GPU inference function after
``setup()``, BEFORE the function reports READY. Output is discarded (never
billing/outputs/CAS); a failure is a load failure (loud).

pgw#654: the warm plan is DERIVED, never developer-written — hand-written
warmup payloads were a coverage CLAIM that drifts (add a guidance class,
forget warmup, ship an untraced graph that compiles at request time).
Derivation per handler:

1. Defaulted payload fields keep their defaults (post-th#1116 those are
   neutral schema values — deterministic).
2. ``CompileAxis`` fields take each class's ``warm=`` representative,
   CROSS-PRODUCTED — that product IS the graph set, so "fully warmed?" is
   computable. Axes partition fields INDEPENDENTLY, so for an endpoint with
   real INTER-FIELD constraints most of that product is not a servable
   request (pgw#669: ltx-video-2.3 has 29 legal combinations of 120). Such an
   endpoint declares the sparsity by raising
   :class:`~gen_worker.IllegalCombination` from its payload
   ``__post_init__``; the plan drops exactly those rows and records the
   coverage it achieved. Every OTHER exception during payload construction is
   still a load failure — that is the whole point of the typed error: a
   sparse legal set and a synthesis bug are no longer the same event.
3. Required no-default fields synthesize neutral values by type: ``str``
   fills ``"warmup"`` (content never affects the graph; ``text_len`` pins
   the traced shape), ``ImageAsset``/``AudioAsset`` get a tiny generated
   PNG/WAV, nested structs/lists recurse. A handler whose schema cannot
   synthesize (e.g. required video input) is skipped with a logged reason.
4. ``@worker_function(warm={...}, warm_reason=...)`` overrides a NON-AXIS
   field that genuinely changes tracing (validated at walk time; needing
   it usually means the field should be an axis).

The graph SET unifies per class via the class-level axis union (pgw#647
gap #1's fix): the cell CONTRACT digests the union, and the forge traces
each graph once. Warm RUNS at boot deliberately stay PER FUNCTION — every
alias proves itself causally through its own handler (a sibling's run
must never certify an alias whose code path was not exercised; pgw#637's
whole blindness class). A deduped-away graph re-executes as a cheap cache
hit, never a second trace.

Handlers SHOULD cheapen non-graph work on ``ctx.boot_warmup`` (e.g.
``steps = 1 if ctx.boot_warmup else steps``): the allocator peak is
shape-driven and the traced graph is step-count-independent. The SDK no
longer trusts that alone: synthesized payloads CLAMP step-count fields
(:data:`_STEP_FIELDS`) to their declared floor, so a warm run never pays a
full recipe's steps even on an endpoint that forgot the clamp.

Warm RUNS are further bounded by :func:`select_runs` (the pgw#654 warm-tax
fix): eager lanes run one shape representative per guidance class instead
of the full cross-product, and a checkpoint INSTANCE whose warm contract
already executed in this process runs a single verification job — juggle
swaps cost weights transfer + load, never a warm-plan re-run.

Remaining class-level surfaces: a class-defined ``warmup()`` method wins
outright (fully custom — the LTX two-stage synthetic);
``@endpoint(warmup=NoWarmup("reason"))`` is the recorded opt-out. Payload
dicts on the decorator are a decoration-time error.

A GPU inference class with NO warmable path and no opt-out fails at spec
construction time (discovery walk / CI), not at first request.
"""

from __future__ import annotations

import enum
import itertools
import os
import struct
import tempfile
import types as py_types
import typing
import wave
import zlib
from dataclasses import dataclass
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import msgspec

from . import dispatch
from .api.compile_axis import PayloadAxis
from .api.decorators import EndpointDecl, NoWarmup
from .api.errors import IllegalCombination
from .api.types import Asset, AudioAsset, ImageAsset, VideoAsset
import inspect
from .api.slot import resolve_slot
from .api.binding import wire_ref
from .request_context import RequestContext

if typing.TYPE_CHECKING:  # pragma: no cover
    from .registry import EndpointSpec

WARMUP_TEXT = "warmup"
_IMAGE_SIDE = 512
_AUDIO_SECONDS = 2.0
_AUDIO_RATE = 48_000
_MAX_DEPTH = 4

# factory(tmp_dir) -> field value; tmp_dir hosts any synthetic asset files.
_Factory = Callable[[str], Any]


def synthetic_png(dir_path: str) -> str:
    """Write a mid-gray RGB PNG (stdlib only) and return its path."""
    path = os.path.join(dir_path, "warmup.png")
    side = _IMAGE_SIDE
    row = b"\x00" + b"\x80" * (side * 3)  # filter 0 + gray pixels
    idat = zlib.compress(row * side, 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", idat))
        f.write(chunk(b"IEND", b""))
    return path


def synthetic_wav(dir_path: str) -> str:
    """Write a short stereo silence WAV (stdlib only) and return its path."""
    path = os.path.join(dir_path, "warmup.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(_AUDIO_RATE)
        w.writeframes(b"\x00\x00" * 2 * int(_AUDIO_RATE * _AUDIO_SECONDS))
    return path


def _image_asset(dir_path: str) -> ImageAsset:
    return ImageAsset(
        ref="boot-warmup.png", local_path=synthetic_png(dir_path),
        mime_type="image/png",
    )


def _audio_asset(dir_path: str) -> AudioAsset:
    return AudioAsset(
        ref="boot-warmup.wav", local_path=synthetic_wav(dir_path),
        mime_type="audio/wav",
    )


def _unwrap(t: Any) -> Any:
    while typing.get_origin(t) is typing.Annotated:
        t = typing.get_args(t)[0]
    return t


def _is_struct(t: Any) -> bool:
    return isinstance(t, type) and issubclass(t, msgspec.Struct)


def _field_factory(t: Any, depth: int) -> Tuple[Optional[_Factory], str]:
    """-> (factory, blocked_reason). Exactly one side is meaningful."""
    t = _unwrap(t)
    if depth > _MAX_DEPTH:
        return None, f"nesting deeper than {_MAX_DEPTH}"
    origin = typing.get_origin(t)
    if origin in (typing.Union, py_types.UnionType):
        args = typing.get_args(t)
        if type(None) in args:
            return (lambda d: None), ""
        for arm in args:
            factory, _ = _field_factory(arm, depth + 1)
            if factory is not None:
                return factory, ""
        return None, f"no synthesizable union arm in {t!r}"
    if t is str:
        return (lambda d: WARMUP_TEXT), ""
    if isinstance(t, type):
        # Asset ordering matters: check concrete media kinds before the
        # ambiguous bases.
        if issubclass(t, ImageAsset):
            return _image_asset, ""
        if issubclass(t, AudioAsset):
            return _audio_asset, ""
        if issubclass(t, VideoAsset):
            return None, "required video input is not synthesizable"
        if issubclass(t, Asset):
            return None, f"required {t.__name__} input is not synthesizable"
        if issubclass(t, enum.Enum):
            members = list(t)
            if members:
                first = members[0]
                return (lambda d: first), ""
            return None, f"enum {t.__name__} has no members"
        if issubclass(t, msgspec.Struct):
            return _struct_factory(t, depth + 1)
    if origin in (list, typing.List, Sequence, typing.Sequence, tuple, typing.Tuple):
        args = tuple(a for a in typing.get_args(t) if a is not Ellipsis)
        if len(args) == 1:
            inner, reason = _field_factory(args[0], depth + 1)
            if inner is None:
                return None, reason
            if origin in (tuple, typing.Tuple):
                return (lambda d: (inner(d),)), ""
            return (lambda d: [inner(d)]), ""
        return None, f"unsupported sequence shape {t!r}"
    return None, f"required field type {t!r} is not synthesizable"


def _present_factory(t: Any, depth: int) -> Optional[_Factory]:
    """A factory that POPULATES a field — never the `None` arm of an option.

    `_field_factory` answers `None` for any optional type, which is right for
    a neutral default and useless for repairing a one-of (th#1771).
    """
    t = _unwrap(t)
    if depth > _MAX_DEPTH:
        return None
    origin = typing.get_origin(t)
    if origin in (typing.Union, py_types.UnionType):
        for arm in typing.get_args(t):
            if arm is type(None):
                continue
            factory = _present_factory(arm, depth + 1)
            if factory is not None:
                return factory
        return None
    factory, _ = _field_factory(t, depth)
    return factory


def _struct_factory(payload_type: type, depth: int = 0) -> Tuple[Optional[_Factory], str]:
    field_factories: List[Tuple[str, _Factory]] = []
    # th#1771: the one-of repair set. A struct whose fields are ALL optional
    # but which asserts "exactly one of these is set" (the standard one-of
    # media shape) is not sparse — it is unconditionally illegal empty, and
    # required-only synthesis built exactly that.
    optional_factories: List[Tuple[str, _Factory]] = []
    for f in msgspec.structs.fields(payload_type):
        if not f.required:
            factory = _present_factory(f.type, depth)
            if factory is not None:
                optional_factories.append((f.name, factory))
            continue
        factory, reason = _field_factory(f.type, depth)
        if factory is None:
            return None, (
                f"required field {f.name!r}: {reason}"
                if reason else f"required field {f.name!r} is not synthesizable"
            )
        field_factories.append((f.name, factory))

    def build(dir_path: str) -> Any:
        required = {name: fac(dir_path) for name, fac in field_factories}
        try:
            return payload_type(**required)
        except IllegalCombination:
            # A declared sparse legal set — the plan's own signal. Never a
            # shape this repair may paper over.
            raise
        except (ValueError, TypeError):
            # Populate one optional field at a time, in declaration order,
            # until the struct accepts itself. Re-raise the ORIGINAL refusal
            # when nothing satisfies it, so the reason stays the schema's.
            for name, fac in optional_factories:
                try:
                    return payload_type(**required, **{name: fac(dir_path)})
                except IllegalCombination:
                    raise
                except (ValueError, TypeError):
                    continue
            raise

    return build, ""


def synthesize_factory(payload_type: type) -> Tuple[Optional[_Factory], str]:
    """Payload factory for one handler's input struct, or (None, reason)."""
    return _struct_factory(payload_type, 0)


def _media_value_factory(t: Any) -> Optional[_Factory]:
    """Synthesized-media factory for an OPTIONAL field's type, or None."""
    t = _unwrap(t)
    origin = typing.get_origin(t)
    if origin in (typing.Union, py_types.UnionType):
        for arm in typing.get_args(t):
            if arm is type(None):
                continue
            fac = _media_value_factory(arm)
            if fac is not None:
                return fac
        return None
    if isinstance(t, type):
        if issubclass(t, ImageAsset):
            return _image_asset
        if issubclass(t, AudioAsset):
            return _audio_asset
        return None
    if origin in (list, typing.List, Sequence, typing.Sequence, tuple, typing.Tuple):
        args = tuple(a for a in typing.get_args(t) if a is not Ellipsis)
        if len(args) == 1:
            inner = _media_value_factory(args[0])
            if inner is not None:
                if origin in (tuple, typing.Tuple):
                    return lambda d: (inner(d),)
                return lambda d: [inner(d)]
    return None


def media_variants(
    payload_type: type, base_build: _Factory,
) -> List[Tuple[str, _Factory]]:
    """gw#614: (label, factory) variants adding synthesized media to optional
    media-capable fields the base payload leaves empty — the modality an
    input-routed sibling lane (e.g. edit needing an input image) requires.
    Each variant differs from the base in EXACTLY one field, so the lane
    token / guidance / shape derivation matches a real request of that
    modality. A variant factory returns None when the base already carries
    media in its field."""
    variants: List[Tuple[str, _Factory]] = []
    for f in msgspec.structs.fields(payload_type):
        if f.required:
            continue
        fac = _media_value_factory(f.type)
        if fac is None:
            continue

        def build(d: str, _name: str = f.name, _fac: _Factory = fac) -> Any:
            payload = base_build(d)
            if getattr(payload, _name, None):
                return None
            return msgspec.structs.replace(payload, **{_name: _fac(d)})

        variants.append((f"media:{f.name}", build))
    return variants


@dataclass(frozen=True)
class WarmupJob:
    """One planned synthetic invocation: ``build(tmp_dir)`` -> payload.

    ``graph_key`` is the class-scoped graph identity this run traces (the
    dedup key across sibling functions); ``covers`` names every sibling
    function whose graph set contains this run's class (proof attribution
    is by GRAPH COVERAGE: an alias is proven once ALL of its graph classes
    proved — pgw#654/pgw#647 gap #1); ``declared`` is True when a
    ``@worker_function(warm=...)`` override applied.

    ``eager_group`` and ``shape_rank`` drive :func:`select_runs`'s
    eager-lane reduction: jobs sharing an ``eager_group`` (function +
    guidance classes + text pin) trace the SAME batch regime and differ
    only in shape; ``shape_rank`` orders them so the reduction keeps the
    largest (allocator-peak) representative."""

    spec: "EndpointSpec"
    build: _Factory
    declared: bool
    graph_key: Tuple = ()
    covers: Tuple[str, ...] = ()
    eager_group: Tuple = ()
    shape_rank: Tuple = ()


@dataclass(frozen=True)
class WarmupSkip:
    """One recorded reason a handler contributes fewer warm runs than the
    naive derivation would.

    ``illegal``/``combos`` are non-zero only for the pgw#669 sparse-legal-set
    row: they turn "we warmed fewer graphs than the axis product" from a
    silent degradation into an explicit, countable coverage claim ("29 of 120
    combinations are legal; 29 warmed")."""

    spec: "EndpointSpec"
    reason: str
    illegal: int = 0
    combos: int = 0


# Guidance-class axis fields (mirrors api.compile_axis.warm_guidance_values):
# a sibling function whose payload LACKS the field cannot be asked for CFG,
# so its graph falls in the class containing 0 — that mapping is what lets
# generate's cfg_off trace cover turbo's (pgw#647 gap #1).
_GUIDANCE_FIELDS = ("guidance_scale", "guidance", "cfg", "true_cfg_scale")

# Step-count payload fields (post-gap-#4 family vocabulary). The traced
# graph and the allocator peak are step-count INDEPENDENT (the denoise loop
# repeats one graph), so synthesized warm payloads clamp these to the
# field's declared floor — a warm run must never pay a full recipe's steps
# (the 0.61.0 ~30-min boot regression; ie#546 canary measurement).
_STEP_FIELDS = ("num_inference_steps", "steps")


def _numeric_floor(t: Any) -> Tuple[bool, int]:
    """(is_numeric_field, declared floor) for a payload field type: unwraps
    Annotated/Optional, honors ``msgspec.Meta`` ge/gt bounds, floor >= 1."""
    lo = 1
    while typing.get_origin(t) is typing.Annotated:
        args = typing.get_args(t)
        for extra in args[1:]:
            if isinstance(extra, msgspec.Meta):
                if extra.ge is not None:
                    lo = max(lo, int(extra.ge))
                if extra.gt is not None:
                    lo = max(lo, int(extra.gt) + 1)
        t = args[0]
    origin = typing.get_origin(t)
    if origin in (typing.Union, py_types.UnionType):
        for arm in typing.get_args(t):
            if arm is type(None):
                continue
            ok, arm_lo = _numeric_floor(arm)
            if ok:
                return True, max(lo, arm_lo)
        return False, lo
    if t is int or t is float:
        return True, lo
    return False, lo


def _step_clamps(
    payload_type: type, axis_fields: AbstractSet[str], overrides: Any,
) -> dict:
    """{field: floor} for step-count fields this plan clamps: known fields
    from :data:`_STEP_FIELDS` that are neither axes nor warm-overridden."""
    out: dict = {}
    for f in msgspec.structs.fields(payload_type):
        if f.name not in _STEP_FIELDS:
            continue
        if f.name in axis_fields or f.name in dict(overrides or {}):
            continue
        ok, lo = _numeric_floor(f.type)
        if ok:
            out[f.name] = lo
    return out


def _axis_combos(
    axes: Sequence[PayloadAxis],
) -> List[Tuple[Tuple[str, str, Any], ...]]:
    """Cross-product of the axes' classes: each combo is a tuple of
    ``(field, class_name, warm_value)`` rows — the derived graph set."""
    if not axes:
        return [()]
    per_axis = [
        [(a.field, n, w) for n, w in zip(a.class_names, a.warm_values)]
        for a in axes
    ]
    return [tuple(combo) for combo in itertools.product(*per_axis)]


def _union_axes(specs: Sequence["EndpointSpec"]) -> List[Tuple[str, PayloadAxis]]:
    """Ordered (field, representative axis) union across sibling specs."""
    out: List[Tuple[str, PayloadAxis]] = []
    seen: set = set()
    for s in specs:
        for a in getattr(s, "payload_axes", ()) or ():
            if a.field not in seen:
                seen.add(a.field)
                out.append((a.field, a))
    return out


def _combo_signature(
    spec: "EndpointSpec",
    combo: Tuple[Tuple[str, str, Any], ...],
    union_axes: Sequence[Tuple[str, PayloadAxis]],
) -> Tuple:
    own = {f: c for f, c, _ in combo}
    sig: List[Tuple[str, Any]] = []
    for field, axis in union_axes:
        if field in own:
            sig.append((field, own[field]))
        elif field in _GUIDANCE_FIELDS:
            cls = axis.classify(0.0)
            sig.append((field, cls if cls is not None else "absent"))
        else:
            sig.append((field, "absent"))
    sig.append(("__text_len__", getattr(spec, "text_len", None)))
    return tuple(sig)


def _job_changes(
    combo: Tuple[Tuple[str, str, Any], ...],
    overrides: Any,
    clamps: Optional[dict] = None,
) -> Dict[str, Any]:
    """The field changes one planned combo applies over the base payload:
    step clamps, then the axis warm representatives, then the declared
    per-function overrides (last wins)."""
    changes: Dict[str, Any] = dict(clamps or {})
    changes.update({f: w for f, _, w in combo})
    changes.update(dict(overrides or {}))
    return changes


def _job_factory(
    base: _Factory,
    combo: Tuple[Tuple[str, str, Any], ...],
    overrides: Any,
    clamps: Optional[dict] = None,
) -> _Factory:
    changes = _job_changes(combo, overrides, clamps)

    def build(dir_path: str) -> Any:
        payload = base(dir_path)
        if not changes:
            return payload
        return msgspec.structs.replace(payload, **changes)

    return build


def _legal_combos(
    owner: str,
    base: _Factory,
    combos: Sequence[Tuple[Tuple[str, str, Any], ...]],
    overrides: Any,
    clamps: Optional[dict],
) -> Tuple[List[Tuple[Tuple[str, str, Any], ...]], int]:
    """``(legal combos, illegal count)`` — the pgw#669 sparsity filter.

    Axes partition fields independently, so the cross-product can contain
    combinations the endpoint's payload rejects by contract (ltx-video-2.3:
    29 legal of 120). Each candidate is built ONCE here, at plan time, so an
    illegal row never reaches the warm RUN — where a raise is a load failure.

    Only :class:`~gen_worker.IllegalCombination` counts as "not legal". Any
    other exception propagates: a payload the SDK cannot synthesize, or a
    struct that rejects a warm representative for some other reason, is a
    declaration/synthesis bug and must fail at the earliest possible moment
    (discovery walk / CI) rather than degrade the coverage claim in silence.

    Every combination being illegal is itself a declaration bug — axes that
    admit no servable request — and raises.
    """
    legal: List[Tuple[Tuple[str, str, Any], ...]] = []
    illegal = 0
    first_reason = ""
    with tempfile.TemporaryDirectory(prefix="gw-warmplan-") as tmp:
        try:
            payload = base(tmp)
        except IllegalCombination as exc:
            # The BASE payload is the schema's own neutral defaults. If those
            # are not servable there is no legal row to derive from, and the
            # declaration is wrong — say so rather than surfacing one field's
            # message with no context.
            raise ValueError(
                f"{owner}: the payload's own DEFAULT values are declared "
                f"IllegalCombination ({exc}) — the neutral schema defaults "
                "must be a servable request; no warm plan can be derived "
                "from them"
            ) from exc
        for combo in combos:
            changes = _job_changes(combo, overrides, clamps)
            try:
                if changes:
                    msgspec.structs.replace(payload, **changes)
            except IllegalCombination as exc:
                illegal += 1
                if not first_reason:
                    first_reason = str(exc)
                continue
            legal.append(combo)
    if combos and not legal:
        raise ValueError(
            f"{owner}: every one of the {len(combos)} derived warm "
            f"combinations is declared IllegalCombination (e.g. "
            f"{first_reason!r}) — the CompileAxis declarations admit no "
            "servable request, so no graph set can be derived"
        )
    return legal, illegal


def _shape_rank(
    spec: "EndpointSpec", combo: Tuple[Tuple[str, str, Any], ...],
) -> Tuple:
    """Sortable preference over one function's combos, HIGHER = preferred by
    the eager reduction: per non-guidance axis, a numeric warm value ranks
    by magnitude (megapixels-style axes keep their max-area bucket), a
    non-numeric one by reverse declaration order (first declared class
    wins). Structure is identical across one spec's combos, so tuples
    compare cleanly."""
    order: dict = {}
    for a in getattr(spec, "payload_axes", ()) or ():
        for i, n in enumerate(a.class_names):
            order[(a.field, n)] = i
    rank: List[Tuple[int, float]] = []
    for field, cls_name, warm in combo:
        if field in _GUIDANCE_FIELDS:
            continue
        if isinstance(warm, (int, float)) and not isinstance(warm, bool):
            rank.append((1, float(warm)))
        else:
            rank.append((0, -float(order.get((field, cls_name), 0))))
    return tuple(rank)


def _eager_group(
    spec: "EndpointSpec", combo: Tuple[Tuple[str, str, Any], ...],
) -> Tuple:
    """Jobs sharing this key trace the same batch regime (function +
    guidance classes + text pin) and differ only in shape."""
    guid = tuple(
        (f, c) for f, c, _ in combo if f in _GUIDANCE_FIELDS
    )
    return (spec.name, guid, getattr(spec, "text_len", None))


def select_runs(
    jobs: Sequence[WarmupJob],
    *,
    tracing: bool,
    executed: AbstractSet[Tuple] = frozenset(),
) -> Tuple[List[WarmupJob], str]:
    """The subset of the derived plan one setup actually EXECUTES, with the
    mode: ``"full"`` | ``"eager"`` | ``"verify"`` (pgw#654 warm-tax fix; the
    ie#546 canary measured the full plan re-running per checkpoint
    INSTANCE: ~30-min first boots and a ~9-min tax on every juggle swap).

    - ``tracing=True`` (a compile artifact is armed or minting): the FULL
      plan — every graph class must trace into the capture / prove against
      the cell.
    - ``tracing=False`` (eager lane — nothing armed, nothing minting): one
      run per (function, guidance class), collapsed to a single shape
      representative (:func:`_shape_rank`). Eager warm cost is allocator-
      pool growth plus kernel-heuristic selection — shape-driven, not
      coverage-driven (gw#470) — and nothing is being traced, so the class
      x bucket cross-product buys nothing.
    - ``executed`` covering every selected run's ``graph_key``: ONE clamped
      verification run. Allocator pool, cuBLAS/cuDNN heuristics and
      dynamo's in-memory compiled code are PROCESS-global, and handler code
      paths were exercised when this contract first warmed — a new
      checkpoint INSTANCE of the same contract shares all of it. The single
      run proves the fresh weights compose and forward end-to-end before
      READY (and, on compiled lanes, provides the calls>0 the pgw#637
      in-memory/cache-hit proof needs). The caller passes ``executed``
      non-empty only when inheritance is sound (same contract, proven
      cells, no pending mint).
    """
    runs = list(jobs)
    if not runs:
        return [], "full"
    mode = "full"
    if not tracing:
        best: dict = {}
        for wj in runs:
            cur = best.get(wj.eager_group)
            if cur is None or wj.shape_rank > cur.shape_rank:
                best[wj.eager_group] = wj
        chosen = {id(wj) for wj in best.values()}
        runs = [wj for wj in runs if id(wj) in chosen]
        mode = "eager"
    if executed and all(wj.graph_key in executed for wj in runs):
        return [runs[0]], "verify"
    return runs, mode


def plan(
    specs: Iterable["EndpointSpec"],
    *,
    decl_warmup: Any = None,
    has_warmup_method: bool = False,
) -> Tuple[List[WarmupJob], List[WarmupSkip]]:
    """The DERIVED warm plan for the GPU inference handlers of ONE instance
    group (pgw#654): per handler, the cross-product of its axis classes'
    warm representatives over a synthesized base payload — one run per
    (function, graph class), so each alias proves its own code path.
    Per-function ``@worker_function(warm=...)`` overrides apply to
    non-axis fields."""
    eligible = [
        s for s in specs
        if s.cls is not None and s.kind == "inference" and s.needs_gpu
    ]
    if has_warmup_method or not eligible:
        return [], []
    if isinstance(decl_warmup, NoWarmup):
        return [], [
            WarmupSkip(s, f"NoWarmup: {decl_warmup.reason}") for s in eligible
        ]
    union_axes = _union_axes(eligible)
    skips: List[WarmupSkip] = []
    jobs: List[WarmupJob] = []
    for s in eligible:
        base, reason = synthesize_factory(s.payload_type)
        if base is None:
            skips.append(WarmupSkip(s, f"not auto-synthesizable: {reason}"))
            continue
        overrides = dict(getattr(s, "warm_overrides", {}) or {})
        axis_fields = {
            a.field for a in (getattr(s, "payload_axes", ()) or ())
        }
        clamps = _step_clamps(s.payload_type, axis_fields, overrides)
        axes = tuple(getattr(s, "payload_axes", ()) or ())
        combos = _axis_combos(axes)
        if axes:
            # pgw#669: drop the combinations the endpoint declares illegal
            # BEFORE they become planned runs, and record the coverage.
            combos, illegal = _legal_combos(
                s.name, base, combos, overrides, clamps)
            if illegal:
                skips.append(WarmupSkip(
                    s,
                    f"{illegal} of {illegal + len(combos)} axis combinations "
                    f"are declared IllegalCombination (inter-field contract); "
                    f"{len(combos)} legal graph classes planned",
                    illegal=illegal,
                    combos=illegal + len(combos),
                ))
        seen: set = set()
        for combo in combos:
            sig = _combo_signature(s, combo, union_axes)
            if sig in seen:
                continue
            seen.add(sig)
            jobs.append(WarmupJob(
                spec=s,
                build=_job_factory(base, combo, overrides, clamps),
                declared=bool(overrides),
                graph_key=(s.name,) + sig,
                covers=(s.name,),
                eager_group=_eager_group(s, combo),
                shape_rank=_shape_rank(s, combo),
            ))
    return jobs, skips


def validate_at_decoration(cls: type, decl: EndpointDecl) -> None:
    """Best-effort decoration-time enforcement (fails at import, the
    earliest possible moment). Unresolvable type hints defer silently to
    the authoritative walk-time check (``validate_class_warmup``)."""

    if decl.kind != "inference" or not decl.resources.gpu:
        return
    _refuse_compile_nowarmup(cls, decl)
    if callable(getattr(cls, "warmup", None)) or isinstance(decl.warmup, NoWarmup):
        return
    pairs: List[Tuple[str, type]] = []
    for attr, method in getattr(cls, "__gen_worker_handlers__", []) or []:
        try:
            hints = typing.get_type_hints(method)
        except Exception:
            return  # forward refs unresolvable here — walk time will check
        params = [p for p in inspect.signature(method).parameters if p != "self"]
        if len(params) < 2:
            return
        pt = hints.get(params[1])
        if not (isinstance(pt, type) and issubclass(pt, msgspec.Struct)):
            return  # walk time raises its own, clearer error
        pairs.append((attr, pt))
    skips: List[Tuple[str, str]] = []
    warmable = False
    for attr, pt in pairs:
        factory, reason = synthesize_factory(pt)
        if factory is None:
            skips.append((attr, f"not auto-synthesizable: {reason}"))
        else:
            warmable = True
    if pairs and not warmable:
        _raise_unwarmable(cls.__name__, skips)


def _refuse_compile_nowarmup(cls: type, decl: EndpointDecl) -> None:
    """th#959: ``compile=`` + ``NoWarmup`` is a contradiction — compile cells
    are minted and proven by warmup execution, so a NoWarmup class can never
    arm one. Live, the combination struck a release broken in 3 pods. A
    custom ``warmup()`` method resolves it (proof runs through that)."""
    if (
        decl.compile is not None
        and isinstance(decl.warmup, NoWarmup)
        and not callable(getattr(cls, "warmup", None))
    ):
        raise TypeError(
            f"@endpoint class {cls.__name__!r}: compile= with "
            f"warmup=NoWarmup({decl.warmup.reason!r}) can never mint or "
            "prove a compile cell (proof runs through warmup execution — "
            "th#959). Drop compile=, define a custom warmup() method, or "
            "make a handler warmable (warm= overrides, pgw#654)."
        )


def _raise_unwarmable(owner: str, skips: Sequence[Tuple[str, str]]) -> None:
    detail = "; ".join(f"{a}: {r}" for a, r in skips) or "no handlers"
    raise TypeError(
        f"@endpoint class {owner!r}: boot warmup is default-on for GPU "
        f"inference endpoints but no handler is warmable ({detail}). The "
        "warm plan is DERIVED (pgw#654) — make at least one handler's "
        "payload synthesizable, define a custom warmup() method, or opt "
        "out with warmup=NoWarmup(\"reason\")."
    )


def validate_class_warmup(cls: type, decl: EndpointDecl, specs: List["EndpointSpec"]) -> None:
    """Spec-construction-time enforcement: a GPU inference class must have a
    warmable path — a custom ``warmup()``, at least one derivable warm job,
    or an explicit ``NoWarmup(reason)``."""
    eligible = [
        s for s in specs
        if s.cls is not None and s.kind == "inference" and s.needs_gpu
    ]
    if not eligible:
        return
    if callable(getattr(cls, "warmup", None)):
        return
    if isinstance(decl.warmup, NoWarmup):
        return
    jobs, skips = plan(specs, decl_warmup=decl.warmup, has_warmup_method=False)
    if jobs:
        return
    _raise_unwarmable(cls.__name__, [(s.spec.attr_name, s.reason) for s in skips])


# ---------------------------------------------------------------------------
# The warm-job RequestContext — ONE construction (pgw#828)
# ---------------------------------------------------------------------------


def spec_root_slot(spec: "EndpointSpec") -> str:
    """The declared root slot name (pgw#654 gap #7): ``Slot(root=True)``,
    else ``"pipeline"``, else the single slot; ``""`` when nothing resolves
    a root."""
    for name, slot in spec.slots.items():
        if getattr(slot, "root", False):
            return name
    if "pipeline" in spec.slots:
        return "pipeline"
    if len(spec.slots) == 1:
        return next(iter(spec.slots))
    return ""


def resolved_slots_kwargs(
    spec: "EndpointSpec",
    slots: Optional[Mapping[str, "dispatch.SlotOrder"]] = None,
    adapters: Optional[Mapping[str, Tuple["dispatch.AdapterOrder", ...]]] = None,
) -> Dict[str, Any]:
    """``ctx.slots`` resolution chain (pgw#520 / pgw#516): merge each
    ``Slot``-declared slot's repo-metadata ``inference_defaults`` over its
    code fallback preset, then apply each riding lora's own
    ``inference_defaults`` as a FIELD-LEVEL override, in lora order (pgw#516
    composition rule — see ``api.slot._apply_lora_overrides``).

    Returns the ``resolved_slots=`` / ``slot_errors=`` / ``root_slot=`` kwargs
    for ``RequestContext.__init__``. A slot that fails to resolve (no metadata
    + no fallback, or no ref) is deferred to a ``ctx.slots[name]`` access
    error instead of failing the whole dispatch.

    ``slots``/``adapters`` are the NEUTRAL per-dispatch orders (pgw#904) —
    never a wire message. ``None`` is the WARM shape: no per-dispatch binding
    metadata, so every slot resolves from the spec's own declaration. That is
    what makes this reachable from the delegated mint child, which has the
    spec and no job.
    """
    if not spec.slots:
        return {
            "resolved_slots": {}, "slot_errors": {},
            "declared_slot_errors": (), "root_slot": "",
        }

    slot_orders = dict(slots or {})
    raw_defaults = {
        name: so.inference_defaults
        for name, so in slot_orders.items() if so.inference_defaults}
    lora_defaults = {
        name: tuple(a.inference_defaults for a in advs if a.inference_defaults)
        for name, advs in dict(adapters or {}).items()
    }
    # pgw#654: the resolved checkpoint's stamped objective/distilled facts
    # ride the binding; the per-function declaration is the backstop (the
    # hub gates checkpoint<->function compatibility at deploy/dispatch).
    objectives = {
        name: so.objective for name, so in slot_orders.items()}
    distilled_facts = {
        name: so.distilled for name, so in slot_orders.items()}
    distilled_statuses = {
        name: so.distilled_status for name, so in slot_orders.items()}
    resolved: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    # pgw#763/th#1288: a FIXED slot's ref is the RELEASE's own declaration, so
    # its resolution failure is version-independent origin evidence and gets a
    # typed label. `selected_by` slots stay untyped — the payload picks there.
    declared: list = []
    for name, slot in spec.slots.items():
        try:
            resolved[name] = resolve_slot(
                name, slot,
                ref=spec.models.get(name),
                defaults_cls=spec.defaults_type,
                family=spec.slot_family.get(name, ""),
                raw_metadata_json=raw_defaults.get(name, ""),
                lora_metadata_json=lora_defaults.get(name, ()),
                objective=objectives.get(name, ""),
                distilled=distilled_facts.get(name, False),
                distilled_status=distilled_statuses.get(name, ""),
                allowed_objectives=spec.objectives,
                allowed_distilled=spec.distilled,
            )
        except ValueError as exc:
            errors[name] = str(exc)
            if not getattr(slot, "selected_by", ""):
                declared.append(name)
    return {
        "resolved_slots": resolved,
        "slot_errors": errors,
        "declared_slot_errors": tuple(declared),
        "root_slot": spec_root_slot(spec),
    }


def warm_context(
    spec: "EndpointSpec", *,
    request_id: str,
    local_output_dir: str,
    execution_lane: str = "",
    config: Optional[Mapping[str, Any]] = None,
    origin: str = "",
) -> Any:
    """The ``RequestContext`` a WARM forward runs under — one construction.

    pgw#828: the delegated mint child hand-rolled its own, and the one it
    rolled had no slots at all. On a real L4 the child LOADED the pipeline in
    16.45 s (pgw#816's fix holding) and then died in the endpoint's own warm
    job at ``ctx.slots["pipeline"]`` with ``KeyError: 'pipeline'`` — so the
    dynamo route published nothing, exactly as the AOT route published
    nothing, and no mint route on the platform could produce a cell.

    That is the same shape as pgw#816 one stage later, and the same shape as
    pgw#816/#822/#825/#827: two constructions of one thing, free to drift.
    The graphs the child traces are the graphs the parent must later hit, so
    a child whose ctx resolves DIFFERENT slot defaults traces different
    shapes and the parent's proof misses — the "mint succeeded, no artifact"
    class. There is one factory now, and both sides call it.

    It lives in ``warmup`` rather than in ``executor`` on purpose: the child
    already derives its warm plan from here, and it must never import the
    executor's whole arming brain to build a context.

    ``origin`` (pgw#969) names the warm this context belongs to and is
    prefixed onto every deferred slot-resolution error, so a handler's
    ``ctx.slots[...]`` failure says WHICH warm it killed. A mint child holds
    no orchestrator session; the text it raises is its only channel, and
    ``ValueError: slot 'pipeline': no resolved model ref`` named a symptom
    and no mint.
    """

    slot_kwargs = resolved_slots_kwargs(spec, None)
    if origin:
        slot_kwargs["slot_errors"] = {
            name: f"{origin}: {why}"
            for name, why in slot_kwargs["slot_errors"].items()
        }
    ctx: Any = RequestContext(
        request_id=str(request_id),
        local_output_dir=local_output_dir,
        models={slot: wire_ref(b) for slot, b in spec.models.items()},
        **slot_kwargs,
        boot_warmup=True,
    )
    if execution_lane:
        ctx._set_execution_lane(str(execution_lane))
    if config:
        ctx._set_config(dict(config))
    return ctx
