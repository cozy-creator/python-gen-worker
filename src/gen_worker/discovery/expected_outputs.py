"""The ``expected_outputs`` manifest block: what this function is ABOUT to
produce, before it produces it (pgw#1580).

Read off the RETURN struct's ``Annotated[..., ExpectedOutput(...)]`` markers —
not a decorator kwarg, because the fact belongs to the output field it
describes. ``ExpectedOutput`` is still exported from ``gen_worker`` and
endpoints still write it; pgw#1373 deleted the v1 collector with the rest of
``discover.py`` and the v2 row never grew a replacement, so every endpoint
promoted to v2 silently stopped telling the platform what it would return.
Measured on the live hub: ``ltx-video-2.3`` (v1) 4/4 functions carried it,
``anima``/``sd15``/``sdxl``/``dj-utils``/``music-analysis`` (v2) 0/10.

**What is lost is per-REQUEST.** ``expectedOutputsForRelease``
(``internal/orchestrator/http/expected_outputs.go``) resolves each plan against
the actual input payload and stamps ``ExpectedOutputsJSON`` onto the request
row — the count, the type and the resolved aspect ratio of what is coming — so
a consumer that renders output placeholders before generation finishes has
something to render. ``len(fn.ExpectedOutputs) == 0`` returns ``nil`` and the
whole path goes quiet.

THE SHAPE IS THE HUB'S, read off ``release.ExpectedOutputPlan`` /
``builder.ExpectedOutputPlan`` rather than guessed::

    {"field": str, "type": str, "mime_type"?: str,
     "count"?: int|str, "width"?: int|str, "height"?: int|str,
     "aspect_ratio"?: int|str}

``type`` is the hub's five-value vocabulary (``normalizeExpectedOutputType``:
image | video | audio | file | other — anything else becomes ``other``), which
is exactly :class:`~gen_worker.api.types.ExpectedOutput`'s ``media_type``
Literal. Expression values are a positive int literal or an ``input.<field>``
ref the hub resolves against the request payload.

**``duration_s`` IS DELIBERATELY NOT EMITTED.** v1 emitted it and no hub reader
has ever decoded it — it is absent from both ``ExpectedOutputPlan`` structs and
from ``expectedOutputsFromPlans`` — so a key here would be the
mirror-with-no-reader shape th#2087's fence exists to catch. The marker's own
docstring already routes settlement at the probed ``VideoAsset.duration_s``
instead. It is validated like every other expression so a wrong spelling is
still refused; it just does not travel.
"""

from __future__ import annotations

import types as py_types
import typing
from typing import Any, Dict, List, Set

import msgspec

from ..api.types import (
    AudioAsset,
    ExpectedOutput,
    ImageAsset,
    MediaAsset,
    VideoAsset,
)


def _is_msgspec_struct(t: Any) -> bool:
    return isinstance(t, type) and issubclass(t, msgspec.Struct)


def _unwrap_optional(ann: Any) -> Any:
    origin = typing.get_origin(ann)
    if origin in (typing.Union, py_types.UnionType):
        arms = [arm for arm in typing.get_args(ann) if arm is not type(None)]
        if len(arms) == 1:
            return arms[0]
    return ann


def _media_kind(ann: Any) -> str:
    """The hub's ``normalizeExpectedOutputType`` vocabulary, inferred from the
    annotated field when the author did not name a ``media_type``."""
    ann = _unwrap_optional(ann)
    origin = typing.get_origin(ann)
    if origin in (list, tuple, set, frozenset):
        args = typing.get_args(ann)
        ann = _unwrap_optional(args[0]) if args else Any
    if isinstance(ann, type):
        if issubclass(ann, ImageAsset):
            return "image"
        if issubclass(ann, VideoAsset):
            return "video"
        if issubclass(ann, AudioAsset):
            return "audio"
        if issubclass(ann, MediaAsset):
            return "file"
    return "other"


def _hints_or_refuse(struct: type, path: str) -> Any:
    """Resolved type hints, or a BUILD ERROR naming the struct.

    The v1 collector swallowed a resolution failure and fell back to
    ``__annotations__`` — which, under ``from __future__ import annotations``
    (every endpoint module in this repo), is a dict of STRINGS. A string is not
    a type, so the walk below skips it and the function emits an EMPTY plan:
    the pgw#1418 silence one field over, and indistinguishable from an endpoint
    that declared nothing. Refuse instead.
    """
    try:
        return typing.get_type_hints(struct, include_extras=True)
    except Exception as exc:
        raise ValueError(
            f"{path or struct.__name__}: cannot resolve the type hints of "
            f"{struct.__name__} ({type(exc).__name__}: {exc}) — the "
            f"expected_outputs plan would be silently EMPTY, which the hub "
            f"reads as an endpoint that promises nothing"
        ) from exc


def _payload_has_field_path(payload_type: type, ref: str) -> bool:
    """Does ``input.<path>`` name a real payload field? The hub resolves the
    ref against the request payload at submit time and silently drops a plan it
    cannot resolve, so a typo here would cost the whole row, at request time,
    with no error anywhere."""
    if not ref.startswith("input."):
        return True
    path = ref.removeprefix("input.")
    if not path:
        return False
    current: Any = payload_type
    for raw_part in path.replace("[]", "").split("."):
        part = raw_part.strip()
        if not part:
            return False
        current = _unwrap_optional(current)
        origin = typing.get_origin(current)
        if origin in (list, tuple, set, frozenset):
            args = typing.get_args(current)
            current = _unwrap_optional(args[0]) if args else Any
        if not _is_msgspec_struct(current):
            return False
        hints = _hints_or_refuse(current, part)
        if part not in hints:
            return False
        current = hints[part]
    return True


def _expression(value: Any, *, payload_type: type, field: str, key: str) -> Any:
    """One plan expression: a positive int literal, an ``input.<field>`` ref,
    or ``None`` for undeclared. Anything else is a build error."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field}: ExpectedOutput.{key} must be int, str, or None")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{field}: ExpectedOutput.{key} must be positive")
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.startswith("input.") and not _payload_has_field_path(payload_type, raw):
            raise ValueError(
                f"{field}: ExpectedOutput.{key} references unknown payload "
                f"field {raw!r}"
            )
        return raw
    raise TypeError(f"{field}: ExpectedOutput.{key} must be int, str, or None")


#: Marker attribute -> manifest key, in the hub's field order. ``duration_s``
#: is absent on purpose (module docstring); it is still validated below.
_EXPRESSIONS = (("count", "count"), ("width", "width"), ("height", "height"),
                ("aspect_ratio", "aspect_ratio"))


def expected_outputs(payload_type: type, return_type: type) -> List[Dict[str, Any]]:
    """The ``expected_outputs`` rows for one entrypoint — an empty list when
    the return struct carries no marker, and the caller omits the key then."""
    out: List[Dict[str, Any]] = []
    seen_structs: Set[type] = set()

    def walk(ann: Any, path: str) -> None:
        origin = typing.get_origin(ann)

        if origin is typing.Annotated:
            args = typing.get_args(ann)
            if not args:
                return
            base = args[0]
            markers = [m for m in args[1:] if isinstance(m, ExpectedOutput)]
            if not markers:
                walk(base, path)
                return
            marker = markers[-1]
            item: Dict[str, Any] = {
                "field": path,
                "type": marker.media_type or _media_kind(base),
            }
            for attribute, key in _EXPRESSIONS:
                value = _expression(
                    getattr(marker, attribute),
                    payload_type=payload_type, field=path, key=attribute,
                )
                if value is not None:
                    item[key] = value
            # Validated, never emitted — see the module docstring.
            _expression(
                marker.duration_s,
                payload_type=payload_type, field=path, key="duration_s",
            )
            mime = (marker.mime_type or "").strip()
            if mime:
                item["mime_type"] = mime
            out.append(item)
            return

        if origin in (typing.Union, py_types.UnionType):
            for arg in typing.get_args(ann):
                if arg is not type(None):
                    walk(arg, path)
            return

        if origin in (list, tuple, set, frozenset):
            args = typing.get_args(ann)
            if args:
                walk(args[0], f"{path}[]")
            return

        if isinstance(ann, type) and _is_msgspec_struct(ann):
            if ann in seen_structs:
                return
            seen_structs.add(ann)
            hints = _hints_or_refuse(ann, path)
            for field in getattr(ann, "__struct_fields__", ()) or ():
                if field in hints:
                    walk(hints[field], f"{path}.{field}" if path else field)
            seen_structs.discard(ann)

    walk(return_type, "")
    return out


__all__ = ["expected_outputs"]
