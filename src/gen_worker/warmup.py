"""Boot-time synthetic warmup planning (gw#470).

First-call tax on a fresh worker is EAGER cost — allocator-pool growth to
the activation peak plus cuBLAS/cuDNN heuristic selection (measured 216s vs
63s warm on LTX/H100 pre-expandable-segments; 0-152s host lottery on B200).
The worker therefore runs one synthetic request per GPU inference function
after ``setup()``, BEFORE the function reports READY. Output is discarded
(never billing/outputs/CAS); a failure is a load failure (loud).

Author surface, smallest first (Paul ruling 2026-07-16):

1. DEFAULT — nothing. The payload is synthesized from the handler's typed
   msgspec schema: defaulted fields keep their defaults, required ``str``
   fields fill ``"warmup"``, required ``ImageAsset``/``AudioAsset`` fields
   get a tiny generated PNG/WAV, nested structs and lists synthesize
   recursively. A function whose schema cannot synthesize (e.g. required
   video input) is skipped with a logged reason.
2. ``@endpoint(warmup={"method": {...}})`` — declarative per-method payload
   overriding synthesis (e.g. to warm the largest preset when the schema
   default is not the allocator peak). ``{"method": None}`` skips a method.
3. A class-defined ``warmup()`` method wins outright (fully custom — the
   LTX two-stage synthetic).
4. ``@endpoint(warmup=NoWarmup("reason"))`` — class-level opt-out, reason
   recorded in code. Never an env knob.

A GPU inference class with NO warmable path and no opt-out fails at spec
construction time (discovery walk / CI), not at first request.
"""

from __future__ import annotations

import enum
import os
import struct
import types as py_types
import typing
import wave
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

import msgspec

from .api.decorators import ATTR, EndpointDecl, NoWarmup
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


@dataclass(frozen=True)
class WarmupJob:
    """One planned synthetic invocation: ``build(tmp_dir)`` -> payload."""

    spec: "EndpointSpec"
    build: _Factory
    declared: bool  # True when the payload came from `warmup=`


@dataclass(frozen=True)
class WarmupSkip:
    spec: "EndpointSpec"
    reason: str


def _declared_factory(owner: str, attr: str, payload_type: type, value: Any) -> _Factory:
    """Validate a declared warmup payload against the handler's schema NOW
    (decoration/walk time), returning a factory that rebuilds it fresh."""
    if isinstance(value, payload_type):
        builtins_value = msgspec.to_builtins(value)
    elif isinstance(value, Mapping):
        builtins_value = dict(value)
    else:
        raise TypeError(
            f"{owner}: warmup[{attr!r}] must be a dict or "
            f"{payload_type.__name__} instance, got {type(value).__name__}"
        )
    try:
        msgspec.convert(builtins_value, type=payload_type, strict=False)
    except msgspec.ValidationError as exc:
        raise TypeError(
            f"{owner}: warmup[{attr!r}] is not a valid "
            f"{payload_type.__name__}: {exc}"
        ) from exc
    return lambda d: msgspec.convert(builtins_value, type=payload_type, strict=False)


def _plan_pairs(
    owner: str,
    pairs: Sequence[Tuple[str, type]],
    decl_warmup: Any,
) -> Tuple[List[Tuple[str, _Factory, bool]], List[Tuple[str, str]]]:
    """Core planner over (attr_name, payload_type) pairs ->
    (jobs=[(attr, factory, declared)], skips=[(attr, reason)])."""
    if isinstance(decl_warmup, NoWarmup):
        return [], [(a, f"NoWarmup: {decl_warmup.reason}") for a, _ in pairs]
    declared: Mapping[str, Any] = decl_warmup if isinstance(decl_warmup, Mapping) else {}
    known = {a for a, _ in pairs}
    unknown = set(declared) - known
    if unknown:
        raise TypeError(
            f"{owner}: warmup= names unknown or non-GPU handler method(s) "
            f"{sorted(unknown)!r} (known: {sorted(known)!r})"
        )
    jobs: List[Tuple[str, _Factory, bool]] = []
    skips: List[Tuple[str, str]] = []
    for attr, payload_type in pairs:
        if attr in declared:
            value = declared[attr]
            if value is None:
                skips.append((attr, "declared skip (warmup={...: None})"))
                continue
            jobs.append((attr, _declared_factory(owner, attr, payload_type, value), True))
            continue
        factory, reason = synthesize_factory(payload_type)
        if factory is None:
            skips.append((attr, f"not auto-synthesizable: {reason}"))
        else:
            jobs.append((attr, factory, False))
    return jobs, skips


def plan(
    specs: Iterable["EndpointSpec"],
    *,
    decl_warmup: Any = None,
    has_warmup_method: bool = False,
) -> Tuple[List[WarmupJob], List[WarmupSkip]]:
    """Warmup plan for the GPU inference handlers of ONE instance group.

    Declared ``warmup=`` payloads override synthesis per method; methods the
    mapping does not name still auto-synthesize. Raises TypeError for
    declarations that name unknown methods or fail schema validation.
    """
    eligible = [
        s for s in specs
        if s.cls is not None and s.kind == "inference" and s.needs_gpu
    ]
    if has_warmup_method or not eligible:
        return [], []
    cls0 = eligible[0].cls
    assert cls0 is not None  # eligible filters cls None
    owner = cls0.__name__
    by_attr = {s.attr_name: s for s in eligible}
    raw_jobs, raw_skips = _plan_pairs(
        owner, [(s.attr_name, s.payload_type) for s in eligible], decl_warmup)
    jobs = [WarmupJob(by_attr[a], f, d) for a, f, d in raw_jobs]
    skips = [WarmupSkip(by_attr[a], r) for a, r in raw_skips]
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
    jobs, skips = _plan_pairs(cls.__name__, pairs, decl.warmup)
    # A present `warmup=` mapping is itself an explicit in-code decision
    # (per-method None = declared skip) — enforcement targets only classes
    # whose author never engaged with warmup at all.
    if pairs and not jobs and decl.warmup is None:
        _raise_unwarmable(cls.__name__, skips)


def _raise_unwarmable(owner: str, skips: Sequence[Tuple[str, str]]) -> None:
    detail = "; ".join(f"{a}: {r}" for a, r in skips) or "no handlers"
    raise TypeError(
        f"@endpoint class {owner!r}: boot warmup is default-on for GPU "
        f"inference endpoints but no handler is warmable ({detail}). Declare "
        "warmup={method: payload} with a self-contained payload, define a "
        "warmup() method, or opt out with warmup=NoWarmup(\"reason\")."
    )


def plan_for_class(cls: type) -> Tuple[List[WarmupJob], List[WarmupSkip]]:
    """Plan from a decorated class (endpoint CPU-stub tests use this)."""
    from .registry import extract_specs

    decl: Optional[EndpointDecl] = getattr(cls, ATTR, None)
    if decl is None:
        raise TypeError(f"{cls.__name__} is not an @endpoint class")
    return plan(
        extract_specs(cls),
        decl_warmup=decl.warmup,
        has_warmup_method=callable(getattr(cls, "warmup", None)),
    )


def validate_class_warmup(cls: type, decl: EndpointDecl, specs: List["EndpointSpec"]) -> None:
    """Spec-construction-time enforcement: a GPU inference class must have a
    warmable path — a custom ``warmup()``, at least one auto/declared
    warmup job, or an explicit ``NoWarmup(reason)``."""
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
    if jobs or decl.warmup is not None:
        return
    _raise_unwarmable(cls.__name__, [(s.spec.attr_name, s.reason) for s in skips])
