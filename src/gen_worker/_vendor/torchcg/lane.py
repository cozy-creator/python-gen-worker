"""Execution lanes: a lane IS a tensorfs contract reference (tcg#41).

Paul's ruling from the main_v2.py line review (2026-08-18): the named-Lane
class was under-determined -- with arbitrary names and a free-form contract
field, two lanes were functionally indistinguishable. The author surface is
now exactly::

    @endpoint(compile=("unet",), lanes=("sdxl.diffusers-bf16@1", ...))

``compile`` hoists to the endpoint level (targets do not vary by weight
format); each lane is a REGISTRY CONTRACT NAME -- ``<producer>.<format>@<major>``,
the same handle tensorfs's ``spec/v1/contracts`` library uses. Everything
else about a lane (its load dtype, its layout) is derived from the registry
entry, never typed by the author. ``LaneRef`` is that RESOLVED form: the
contract handle plus the resolution's dtype, which is what serve code reads
back as ``ctx.lane.dtype``.

Omitting ``lanes=`` means one lane -- the model type's canonical contract.
No lanes at all means eager forever, stated rather than implied.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_PATH_SEPARATOR = "."

#: ``<producer>.<format>@<major>`` -- the tensorfs contract-handle spelling.
_CONTRACT_RE = re.compile(r"[a-z0-9][a-z0-9._-]*[a-z0-9]@[1-9][0-9]*\Z")

#: ``<name>@<major>`` -- one transform pass. Versioned because a pass NAME is
#: a ``cg-graph-v1`` derivation input (tcg#52): two lanes differing only by a
#: pass are different graphs, exactly as two lanes differing by a pin are.
_PASS_RE = re.compile(r"[a-z0-9][a-z0-9-]*[a-z0-9]@[1-9][0-9]*\Z")


class LaneError(ValueError):
    """An execution-lane declaration is incomplete or noncanonical."""


def require_contract_ref(value: object) -> str:
    """Validate one lane spelling: a registry contract handle, nothing else."""

    if not isinstance(value, str) or _CONTRACT_RE.fullmatch(value) is None:
        raise LaneError(
            f"a lane is a tensorfs contract reference "
            f"('<producer>.<format>@<major>', e.g. 'sdxl.diffusers-bf16@1'); "
            f"got {value!r}"
        )
    return value


def require_pass_ref(value: object) -> str:
    """Validate one transform-pass spelling: ``<name>@<major>``, nothing else."""

    if not isinstance(value, str) or _PASS_RE.fullmatch(value) is None:
        raise LaneError(
            f"a pass name is '<name>@<major>' (e.g. 'precompute-and-free@1'); "
            f"got {value!r}"
        )
    return value


def require_passes(names: object) -> tuple[str, ...]:
    """Validate a lane's pass tuple: canonical, ordered as given, no repeats."""

    if names is None:
        return ()
    if not isinstance(names, (tuple, list)):
        raise LaneError("a lane's passes must be a tuple of pass names")
    passes = tuple(require_pass_ref(name) for name in names)
    if len(set(passes)) != len(passes):
        raise LaneError(f"a lane names a pass twice: {passes!r}")
    return passes


def require_targets(paths: object) -> tuple[str, ...]:
    """Validate the endpoint-level compile-target paths."""

    if not isinstance(paths, tuple) or not paths:
        raise LaneError(
            "compile= must be a non-empty tuple of attribute paths on the "
            "author's own objects, e.g. ('unet',) or ('unet', 'vae.decoder')"
        )
    for path in paths:
        if not isinstance(path, str) or not path or path != path.strip():
            raise LaneError("compile-target paths must be canonical strings")
        segments = path.split(_PATH_SEPARATOR)
        if any(not segment.isidentifier() for segment in segments):
            raise LaneError(
                f"compile target {path!r} must be a dotted attribute path of "
                f"identifiers, e.g. 'unet' or 'vae.decoder'"
            )
    if len(set(paths)) != len(paths):
        raise LaneError(f"compile targets repeat a path: {paths!r}")
    return paths


@dataclass(frozen=True, slots=True)
class LaneRef:
    """One RESOLVED lane: the contract handle + what resolution derived.

    The author types only the contract string; ``dtype`` is filled by the
    resolver from the registry entry (interim: the model-type pointer table,
    until the canonical per-model-type entries land in tensorfs
    ``spec/v1/contracts``). Serve code reads it back as ``ctx.lane.dtype``.

    ``passes`` is the lane's TRANSFORM binding (tcg#52): the ordered pass
    names that MAKE this lane's format. It is resolved from the registry
    entry exactly like ``dtype`` -- an fp8 lane's weights are fp8 because
    something converted them, and a precompute lane's blocks hold a side
    table because something folded them. Empty means no pass, stated.
    """

    contract: str
    dtype: Any = None
    passes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_contract_ref(self.contract)
        object.__setattr__(self, "passes", require_passes(self.passes))


def resolve_target(roots: Mapping[str, object], path: str) -> object:
    """Resolve one compile-target path against the author's root namespace.

    The first segment names a root object; the rest are attribute accesses on
    the author's own graph of objects. A miss is refused with the exact
    segment that failed, because a silently skipped target would turn a typo
    into an eager-forever lane.
    """

    if not isinstance(path, str) or not path:
        raise LaneError("compile-target path must be a non-empty string")
    segments = path.split(_PATH_SEPARATOR)
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


__all__ = [
    "LaneError",
    "LaneRef",
    "require_contract_ref",
    "require_pass_ref",
    "require_passes",
    "require_targets",
    "resolve_target",
]
