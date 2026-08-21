"""The boot warm pass's payload: one entrypoint's schema at NEUTRAL DEFAULTS."""

from __future__ import annotations

import enum
import os
import struct
import types as py_types
import typing
import wave
import zlib
from typing import Any, Callable, List, Optional, Sequence, Tuple

import msgspec

from .api.types import Asset, AudioAsset, ImageAsset, VideoAsset

WARMUP_TEXT = "warmup"

_IMAGE_SIDE = 128
_AUDIO_SECONDS = 1.0
_AUDIO_RATE = 48_000
_MAX_DEPTH = 4

_Factory = Callable[[str], Any]


def synthetic_png(dir_path: str) -> str:
    """Write a flat mid-gray RGB PNG (stdlib only) and return its path."""
    path = os.path.join(dir_path, "boot-warmup.png")
    side = _IMAGE_SIDE
    row = b"\x00" + b"\x80" * (side * 3)
    idat = zlib.compress(row * side, 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", ihdr))
        handle.write(chunk(b"IDAT", idat))
        handle.write(chunk(b"IEND", b""))
    return path


def synthetic_wav(dir_path: str) -> str:
    """Write a short stereo silence WAV (stdlib only) and return its path."""
    path = os.path.join(dir_path, "boot-warmup.wav")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(_AUDIO_RATE)
        handle.writeframes(b"\x00\x00" * 2 * int(_AUDIO_RATE * _AUDIO_SECONDS))
    return path


def _image_asset(dir_path: str) -> ImageAsset:
    return ImageAsset(
        ref="boot-warmup.png",
        local_path=synthetic_png(dir_path),
        mime_type="image/png",
    )


def _audio_asset(dir_path: str) -> AudioAsset:
    return AudioAsset(
        ref="boot-warmup.wav",
        local_path=synthetic_wav(dir_path),
        mime_type="audio/wav",
    )


def _unwrap(annotation: Any) -> Any:
    while typing.get_origin(annotation) is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    return annotation


def _field_factory(annotation: Any, depth: int) -> Tuple[Optional[_Factory], str]:
    annotation = _unwrap(annotation)
    if depth > _MAX_DEPTH:
        return None, f"nesting deeper than {_MAX_DEPTH}"
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, py_types.UnionType):
        args = typing.get_args(annotation)
        if type(None) in args:
            return (lambda _dir: None), ""
        for arm in args:
            factory, _ = _field_factory(arm, depth + 1)
            if factory is not None:
                return factory, ""
        return None, f"no synthesizable union arm in {annotation!r}"
    if origin is typing.Literal:
        values = list(typing.get_args(annotation))
        if not values:
            return None, "empty Literal"
        first = values[0]
        return (lambda _dir: first), ""
    if annotation is str:
        return (lambda _dir: WARMUP_TEXT), ""
    if annotation is bool:
        return (lambda _dir: False), ""
    if annotation is int:
        return (lambda _dir: 0), ""
    if annotation is float:
        return (lambda _dir: 0.0), ""
    if isinstance(annotation, type):
        if issubclass(annotation, ImageAsset):
            return _image_asset, ""
        if issubclass(annotation, AudioAsset):
            return _audio_asset, ""
        if issubclass(annotation, VideoAsset):
            return None, "required video input is not synthesizable"
        if issubclass(annotation, Asset):
            return None, f"required {annotation.__name__} input is not synthesizable"
        if issubclass(annotation, enum.Enum):
            members = list(annotation)
            if not members:
                return None, f"enum {annotation.__name__} has no members"
            first_member = members[0]
            return (lambda _dir: first_member), ""
        if issubclass(annotation, msgspec.Struct):
            return _struct_factory(annotation, depth + 1)
    if origin in (list, typing.List, Sequence, typing.Sequence, tuple, typing.Tuple):
        args = tuple(a for a in typing.get_args(annotation) if a is not Ellipsis)
        if len(args) == 1:
            inner, reason = _field_factory(args[0], depth + 1)
            if inner is None:
                return None, reason
            if origin in (tuple, typing.Tuple):
                return (lambda d: (inner(d),)), ""
            return (lambda d: [inner(d)]), ""
        return None, f"unsupported sequence shape {annotation!r}"
    return None, f"required field type {annotation!r} is not synthesizable"


def _struct_factory(
    payload_type: type, depth: int = 0
) -> Tuple[Optional[_Factory], str]:
    factories: List[Tuple[str, _Factory]] = []
    try:
        struct_fields = msgspec.structs.fields(payload_type)
    except TypeError as exc:
        return None, f"{payload_type!r} is not a msgspec struct: {exc}"
    for field in struct_fields:
        if not field.required:
            continue
        factory, reason = _field_factory(field.type, depth)
        if factory is None:
            return None, (
                f"required field {field.name!r}: {reason}"
                if reason
                else f"required field {field.name!r} is not synthesizable"
            )
        factories.append((field.name, factory))

    def build(dir_path: str) -> Any:
        return payload_type(**{name: fac(dir_path) for name, fac in factories})

    return build, ""


def payload_factory(payload_type: type) -> Tuple[Optional[_Factory], str]:
    """``(factory, reason)`` for one entrypoint's payload struct."""
    return _struct_factory(payload_type, 0)


def neutral_payload(payload_type: type, dir_path: str) -> Tuple[Any, str]:
    """One payload at the schema's neutral defaults, or ``(None, reason)``."""
    factory, reason = payload_factory(payload_type)
    if factory is None:
        return None, reason
    return factory(dir_path), ""


__all__ = [
    "WARMUP_TEXT",
    "neutral_payload",
    "payload_factory",
    "synthetic_png",
    "synthetic_wav",
]
