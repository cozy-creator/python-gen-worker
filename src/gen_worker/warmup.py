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
   computable.
3. Required no-default fields synthesize neutral values by type: ``str``
   fills ``"warmup"`` (content never affects the graph; ``text_len`` pins
   the traced shape), ``ImageAsset``/``AudioAsset`` get a tiny generated
   PNG/WAV, nested structs/lists recurse. A handler whose schema cannot
   synthesize (e.g. required video input) is skipped with a logged reason.
4. ``@worker_function(warm={...}, warm_reason=...)`` overrides a NON-AXIS
   field that genuinely changes tracing (validated at walk time; needing
   it usually means the field should be an axis).

Warm RUNS are per GRAPH CLASS, not per function: the plan dedupes the
cross-product ACROSS sibling functions of one class via the class-level
axis union (pgw#647 gap #1's fix shape) — a distilled sibling with no
guidance field maps to the guidance class containing 0 (no wire guidance
== CFG off), so generate's cfg_off trace covers turbo's.

Handlers SHOULD cheapen non-graph work on ``ctx.boot_warmup`` (e.g.
``steps = 1 if ctx.boot_warmup else steps``): the allocator peak is
shape-driven and the traced graph is step-count-independent.

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
import types as py_types
import typing
import wave
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import msgspec

from .api.compile_axis import PayloadAxis
from .api.decorators import EndpointDecl, NoWarmup
from .api.types import Asset, AudioAsset, ImageAsset, VideoAsset

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


def _struct_factory(payload_type: type, depth: int = 0) -> Tuple[Optional[_Factory], str]:
    field_factories: List[Tuple[str, _Factory]] = []
    for f in msgspec.structs.fields(payload_type):
        if not f.required:
            continue
        factory, reason = _field_factory(f.type, depth)
        if factory is None:
            return None, (
                f"required field {f.name!r}: {reason}"
                if reason else f"required field {f.name!r} is not synthesizable"
            )
        field_factories.append((f.name, factory))

    def build(dir_path: str) -> Any:
        return payload_type(**{name: fac(dir_path) for name, fac in field_factories})

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
    dedup key across sibling functions); ``declared`` is True when a
    ``@worker_function(warm=...)`` override applied."""

    spec: "EndpointSpec"
    build: _Factory
    declared: bool
    graph_key: Tuple = ()


@dataclass(frozen=True)
class WarmupSkip:
    spec: "EndpointSpec"
    reason: str


# Guidance-class axis fields (mirrors api.compile_axis.warm_guidance_values):
# a sibling function whose payload LACKS the field cannot be asked for CFG,
# so its graph falls in the class containing 0 — that mapping is what lets
# generate's cfg_off trace cover turbo's (pgw#647 gap #1).
_GUIDANCE_FIELDS = ("guidance_scale", "guidance", "cfg", "true_cfg_scale")


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


def _job_factory(
    base: _Factory,
    combo: Tuple[Tuple[str, str, Any], ...],
    overrides: Any,
) -> _Factory:
    changes = {f: w for f, _, w in combo}
    changes.update(dict(overrides or {}))

    def build(dir_path: str) -> Any:
        payload = base(dir_path)
        if not changes:
            return payload
        return msgspec.structs.replace(payload, **changes)

    return build


def plan(
    specs: Iterable["EndpointSpec"],
    *,
    decl_warmup: Any = None,
    has_warmup_method: bool = False,
) -> Tuple[List[WarmupJob], List[WarmupSkip]]:
    """The DERIVED warm plan for the GPU inference handlers of ONE instance
    group (pgw#654): per handler, the cross-product of its axis classes'
    warm representatives over a synthesized base payload, deduped ACROSS
    sibling functions per graph class (class-level axis union — one trace
    per graph, generate's cfg_off covering turbo's). Per-function
    ``@worker_function(warm=...)`` overrides apply to non-axis fields."""
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
    jobs: List[WarmupJob] = []
    skips: List[WarmupSkip] = []
    seen_graphs: set = set()
    for s in eligible:
        base, reason = synthesize_factory(s.payload_type)
        if base is None:
            skips.append(WarmupSkip(s, f"not auto-synthesizable: {reason}"))
            continue
        overrides = dict(getattr(s, "warm_overrides", {}) or {})
        for combo in _axis_combos(tuple(getattr(s, "payload_axes", ()) or ())):
            sig = _combo_signature(s, combo, union_axes)
            if sig in seen_graphs:
                continue
            seen_graphs.add(sig)
            jobs.append(WarmupJob(
                spec=s,
                build=_job_factory(base, combo, overrides),
                declared=bool(overrides),
                graph_key=sig,
            ))
    return jobs, skips


def validate_at_decoration(cls: type, decl: EndpointDecl) -> None:
    """Best-effort decoration-time enforcement (fails at import, the
    earliest possible moment). Unresolvable type hints defer silently to
    the authoritative walk-time check (``validate_class_warmup``)."""
    import inspect

    if decl.kind != "inference" or not decl.resources.gpu:
        return
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
