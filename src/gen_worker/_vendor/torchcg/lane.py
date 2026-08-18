"""Execution lanes: the author-facing compile declaration (tcg#41).

A lane is the WHOLE author surface for compilation under the 2026-08-18
ship-code-as-is ruling: the author's serving code runs as-is, and a lane names
(1) the modules on the author's own objects that are compile targets, spelled
as attribute paths, (2) the tensor-layout contract that lane expects
checkpoints in -- the same contract name tensorfs negotiates checkpoints
toward -- and (3) the torch dtype the lane loads under. Architecture forks
(inpaint channels, refiner, fp8 vs bf16) are simply more lanes. No lanes means
eager forever, stated rather than implied.

The spelling is the contract file's (sdxl ``main_v2.py``, under Paul's
line review)::

    Lane("bf16", compile=("unet",), contract="plain.bf16@1",
         dtype=torch.bfloat16)

``compile`` paths resolve against the root namespace the discovery caller
supplies -- for a diffusers pipeline, its ``components`` mapping, so ``"unet"``
is ``pipe.unet`` and ``"vae.decoder"`` is ``pipe.vae.decoder``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_PATH_SEPARATOR = "."


class LaneError(ValueError):
    """An execution-lane declaration is incomplete or noncanonical."""


def _require_canonical(field: str, value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise LaneError(f"lane {field} must be a non-empty canonical string")
    return value


@dataclass(frozen=True, slots=True)
class Lane:
    """One compile lane: target module paths, checkpoint contract, dtype.

    ``compile`` holds attribute paths on the author's own objects, resolved
    against the namespace handed to discovery. The path names the module the
    author means -- the one actually CALLED (``"vae.decoder"``, not ``"vae"``
    when ``.decode`` bypasses ``__call__``) -- never a family the library
    knows. ``dtype`` is the lane's load dtype, carried for the serve host
    (``ctx.lane.dtype``); it is a derivation input by way of the author's own
    setup, never a name component.
    """

    name: str
    compile: tuple[str, ...]
    contract: str
    dtype: Any = None

    def __post_init__(self) -> None:
        _require_canonical("name", self.name)
        _require_canonical("contract", self.contract)
        if not isinstance(self.compile, tuple) or not self.compile:
            raise LaneError(
                f"lane {self.name!r} must declare at least one compile-target path"
            )
        for path in self.compile:
            _require_canonical("compile-target path", path)
            segments = path.split(_PATH_SEPARATOR)
            if any(not segment.isidentifier() for segment in segments):
                raise LaneError(
                    f"lane {self.name!r} compile target {path!r} must be a dotted "
                    f"attribute path of identifiers, e.g. 'unet' or 'vae.decoder'"
                )
        if len(set(self.compile)) != len(self.compile):
            raise LaneError(f"lane {self.name!r} declares a duplicate compile target")


def resolve_target(roots: Mapping[str, object], path: str) -> object:
    """Resolve one compile-target path against the author's root namespace.

    The first segment names a root object; the rest are attribute accesses on
    the author's own graph of objects. A miss is refused with the exact
    segment that failed, because a silently skipped target would turn a typo
    into an eager-forever lane.
    """

    segments = _require_canonical("compile-target path", path).split(_PATH_SEPARATOR)
    root_name = segments[0]
    if root_name not in roots:
        raise LaneError(
            f"compile target {path!r} names root {root_name!r}, which is not in "
            f"the author namespace {sorted(roots)!r}"
        )
    value: object = roots[root_name]
    for depth, segment in enumerate(segments[1:], start=1):
        try:
            value = getattr(value, segment)
        except AttributeError as exc:
            raise LaneError(
                f"compile target {path!r}: {segment!r} does not exist on the "
                f"object at {'.'.join(segments[:depth])!r}"
            ) from exc
    return value


__all__ = ["Lane", "LaneError", "resolve_target"]
