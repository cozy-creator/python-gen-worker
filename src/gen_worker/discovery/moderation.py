"""The moderation manifest block: which payload paths are MEDIA and which are PROMPTS. The path grammar is the hub's (internal/jsonschema.NodeAtPath): dotted segments, [] for array items, * for an open mapping's values, rooted bare at the payload struct — the hub refuses a path that does not resolve, so a wrong spelling fails the publish."""

from __future__ import annotations

import types as py_types
import typing
from typing import Any, Dict, List, Optional, Set

import msgspec

from ..api.types import (
    Asset,
    AudioAsset,
    ImageAsset,
    PromptRole,
    Tensors,
    VideoAsset,
)


def _is_msgspec_struct(t: Any) -> bool:
    return isinstance(t, type) and issubclass(t, msgspec.Struct)


def _media_kind(t: type) -> str:
    if issubclass(t, ImageAsset):
        return "image"
    if issubclass(t, VideoAsset):
        return "video"
    if issubclass(t, AudioAsset):
        return "audio"
    return "media"



def _hints_or_refuse(struct: type, path: str) -> Any:
    try:
        return typing.get_type_hints(struct, include_extras=True)
    except Exception as exc:
        raise ValueError(
            f"{path or struct.__name__}: cannot resolve the type hints of "
            f"{struct.__name__} ({type(exc).__name__}: {exc}) — the moderation "
            f"block would be silently EMPTY, and an endpoint with no declared "
            f"media inputs cannot be served one"
        ) from exc


def _annotation_carries_asset(ann: Any, _seen: Optional[Set[type]] = None) -> bool:
    seen = _seen if _seen is not None else set()
    origin = typing.get_origin(ann)
    if origin is typing.Annotated:
        args = typing.get_args(ann)
        return bool(args) and _annotation_carries_asset(args[0], seen)
    if origin in (typing.Union, py_types.UnionType):
        return any(
            _annotation_carries_asset(arg, seen)
            for arg in typing.get_args(ann)
            if arg is not type(None)
        )
    if origin in (list, tuple, set, frozenset):
        args = typing.get_args(ann)
        return bool(args) and _annotation_carries_asset(args[0], seen)
    if origin is dict:
        args = typing.get_args(ann)
        return len(args) == 2 and _annotation_carries_asset(args[1], seen)
    if isinstance(ann, type):
        if issubclass(ann, Asset):
            return True
        if issubclass(ann, Tensors):
            return False
        if _is_msgspec_struct(ann):
            if ann in seen:
                return False
            seen.add(ann)
            hints = _hints_or_refuse(ann, ann.__name__)
            return any(
                _annotation_carries_asset(hints[field], seen)
                for field in getattr(ann, "__struct_fields__", ()) or ()
                if field in hints
            )
    return False


def payload_moderation(payload_type: type) -> Dict[str, List[Dict[str, str]]]:
    """``{"prompts": [...], "media": [...]}`` for one entrypoint's payload — keys omitted when empty, so a payload with neither emits ``{}`` and the caller omits the block entirely."""
    out: Dict[str, List[Dict[str, str]]] = {"prompts": [], "media": []}
    seen_structs: Set[type] = set()

    def walk(ann: Any, path: str) -> None:
        origin = typing.get_origin(ann)

        if origin is typing.Annotated:
            args = typing.get_args(ann)
            if not args:
                return
            base = args[0]
            roles = [m for m in args[1:] if isinstance(m, PromptRole)]
            if roles:
                if base is not str:
                    raise ValueError(
                        f"{path}: PromptRole markers must annotate str fields"
                    )
                out["prompts"].append({"field": path, "role": roles[-1].role})
                return
            walk(base, path)
            return

        if origin in (typing.Union, py_types.UnionType):
            for arg in typing.get_args(ann):
                if arg is not type(None):
                    walk(arg, path)
            return

        if origin in (set, frozenset):
            args = typing.get_args(ann)
            if args and _annotation_carries_asset(args[0]):
                raise ValueError(
                    f"{path}: Asset fields cannot ride unordered set/frozenset "
                    "containers; use list or tuple"
                )
            if args:
                walk(args[0], f"{path}[]")
            return

        if origin in (list, tuple):
            args = typing.get_args(ann)
            if args:
                walk(args[0], f"{path}[]")
            return

        if origin is dict:
            args = typing.get_args(ann)
            if len(args) == 2:
                if args[0] is not str and _annotation_carries_asset(args[1]):
                    raise ValueError(
                        f"{path}: Asset-bearing mappings require string keys"
                    )
                walk(args[1], f"{path}.*")
            return

        if isinstance(ann, type):
            if issubclass(ann, Tensors):
                return
            if issubclass(ann, Asset):
                out["media"].append({"field": path, "kind": _media_kind(ann)})
                return
            if _is_msgspec_struct(ann):
                if ann in seen_structs:
                    return
                seen_structs.add(ann)
                hints = _hints_or_refuse(ann, path)
                for field in getattr(ann, "__struct_fields__", ()) or ():
                    if field in hints:
                        walk(hints[field], f"{path}.{field}" if path else field)
                seen_structs.discard(ann)

    walk(payload_type, "")
    return {k: v for k, v in out.items() if v}


__all__ = ["payload_moderation"]
