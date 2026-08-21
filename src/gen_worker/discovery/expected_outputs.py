"""The expected_outputs manifest block, read off the return struct's Annotated[..., ExpectedOutput(...)] markers. The row shape is the hub's (release.ExpectedOutputPlan): {field, type, mime_type?, count?, width?, height?, aspect_ratio?}, type in image|video|audio|file|other, expression values a positive int literal or an input.<field> ref. duration_s is validated but deliberately never emitted — no hub reader decodes it."""

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


_EXPRESSIONS = (("count", "count"), ("width", "width"), ("height", "height"),
                ("aspect_ratio", "aspect_ratio"))


def expected_outputs(payload_type: type, return_type: type) -> List[Dict[str, Any]]:
    """The ``expected_outputs`` rows for one entrypoint — an empty list when the return struct carries no marker, and the caller omits the key then."""
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
