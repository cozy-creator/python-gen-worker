"""The ``moderation`` manifest block: which payload paths are MEDIA, and
which are PROMPTS (pgw#1418).

The hub will not guess. ``inputassets``' package doc opens by refusing to treat
arbitrary string fields as media URLs, so the ONLY way it learns that
``extract_frame(payload.video)`` is a video is this block — decoded at
``release/resolver.go`` into ``PayloadModerationMetadata`` and read by
``input_asset_validation.go`` to build the stored-input manifest, and by the
prompt-moderation pipeline for the ``prompts`` half.

pgw#1373 deleted the v1 collector with the rest of ``discover.py``'s endpoint
walk and the v2 row never grew a replacement, so every v2 release shipped a
manifest with no ``moderation`` key at all: no typed media inputs, and no
prompt-moderation field declarations either — the safety half nobody noticed.

The path grammar is the hub's (``internal/jsonschema.NodeAtPath``): dotted
segments, ``[]`` for array items, ``*`` for an open mapping's values. Paths are
BARE (``video``, ``slots[].prompt``), rooted at the payload struct, because
that is what ``validateModerationSchemaPath`` navigates the input schema with —
and the hub REFUSES a path that does not resolve, so a wrong spelling here
fails the publish rather than degrading quietly.
"""

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
    """The hub's four-value ``kind`` vocabulary. A bare ``Asset``/``MediaAsset``
    is ``media`` — the widest of the four, never a guess at a narrower one."""
    if issubclass(t, ImageAsset):
        return "image"
    if issubclass(t, VideoAsset):
        return "video"
    if issubclass(t, AudioAsset):
        return "audio"
    return "media"



def _hints_or_refuse(struct: type, path: str) -> Any:
    """Resolved type hints, or a BUILD ERROR naming the struct.

    The v1 collector swallowed a resolution failure and fell back to
    ``__annotations__`` — which, under ``from __future__ import annotations``
    (every endpoint module in this repo), is a dict of STRINGS. A string is not
    a type, so the walk below skips it and the function emits an EMPTY
    moderation block: no media, no prompts, no error, and an endpoint that
    cannot serve an asset. That is the exact silence pgw#1418 cost a rented pod
    to find, one layer down. Refuse instead.
    """
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
    """True when the annotation subtree can hold an input ``Asset``."""
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
    """``{"prompts": [...], "media": [...]}`` for one entrypoint's payload —
    keys omitted when empty, so a payload with neither emits ``{}`` and the
    caller omits the block entirely.

    Two shapes are BUILD ERRORS rather than silent omissions, both because the
    input-asset manifest is ORDERED and an unordered container has no stable
    occurrence order to match it against.
    """
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
