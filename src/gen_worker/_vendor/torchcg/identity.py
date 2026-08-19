from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

KEY_SCHEME = "cg-key-v1"
MAX_KEY_LENGTH = 96
_DIGEST_HEX = 56
_TOOLCHAIN_DIGEST_HEX = 16

#: The v1 identity axes, in canonical order. Public because consumers otherwise
#: re-declare it and fence the copy; a duplicate that drifts is how a correctly
#: named module computes wrong keys with nothing raising.
#:
#: These three are the whole canonical fingerprint: the toolchain block's
#: MEMBERS are the caller's to choose, not this package's. The measured
#: rationale — a 2026-08-16 pod matrix, 10 conclusive rows, 3 hosts — lives in
#: `benchmarks/host_fingerprint/README.md`. It found os_release/glibc/
#: libstdc++/c++-compiler load-breaking in both directions, python_abi and
#: triton inert on a CPU target, and torch_config_digest unstable across hosts
#: for the same wheel. Every one of those is a caller-supplied member; the
#: facts this package does own (`machine`, `host_isa_*`) went unmeasured and
#: stay fail-closed.
REQUIRED_AXES: tuple[str, ...] = ("graph", "sm", "toolchain")

#: The artifact-metadata block carrying graph-specialization facts. Public for the same
#: reason: a consumer reading a block this package no longer writes fails at
#: run time, not at import, and its fixtures keep the obsolete shape green.
GRAPH_SPECIALIZATION_BLOCK = "graph_specialization"

#: The artifact `kind` this package writes and refuses on read. Public for the
#: same reason again, and defined HERE rather than in `artifact` because this
#: module is the dependency root: `artifact` imports from it, so the constant
#: can have exactly one definition that both readers share.
ARTIFACT_KIND = "aot-inductor"

_NOT_TOOLCHAIN = frozenset(("diffusers", "transformers", "peft"))
_KEY_RE = re.compile(rf"[a-z0-9][a-z0-9._-]*-[0-9a-f]{{{_DIGEST_HEX}}}\Z")


class IdentityError(ValueError):
    """A compiled graph cannot state its complete, canonical identity."""


@dataclass(frozen=True, slots=True)
class CompiledGraphKey:
    """The three identity axes and their deterministic v1 address."""

    axes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axes, tuple) or any(
            not isinstance(axis, tuple) or len(axis) != 2 for axis in self.axes
        ):
            raise IdentityError("compiled graph key axes must be canonical name/value tuples")
        if tuple(name for name, _ in self.axes) != REQUIRED_AXES:
            raise IdentityError(
                f"compiled graph key axes must be exactly {list(REQUIRED_AXES)!r}"
            )
        for name, value in self.axes:
            if not isinstance(value, str) or not value or value != value.strip():
                raise IdentityError(f"compiled graph key requires canonical string {name!r}")
            if is_compiled_graph_key(value):
                raise IdentityError(f"{name} is a compiled-graph key, not an identity fact")

    def as_dict(self) -> dict[str, str]:
        return dict(self.axes)

    def canonical(self) -> bytes:
        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")

    @property
    def value(self) -> str:
        digest = hashlib.sha256(self.canonical()).hexdigest()[:_DIGEST_HEX]
        return f"{KEY_SCHEME}-{digest}"

    def __str__(self) -> str:
        return self.value


def is_compiled_graph_key(value: object) -> bool:
    """Return whether a boundary value has the compiled-graph key shape.

    The digest is the anchored suffix; never split on ``-`` because schemes may
    contain hyphens. The grammar refuses shape, not scheme, so a future scheme
    can reach axis-based admission without teaching every boundary its name.
    """

    return (
        isinstance(value, str)
        and len(value) <= MAX_KEY_LENGTH
        and _KEY_RE.fullmatch(value) is not None
    )


def _refuse_key_as_fact(name: str, value: str) -> None:
    if is_compiled_graph_key(value):
        raise IdentityError(f"{name} is a compiled-graph key, not an identity fact")


def from_axes(axes: Mapping[str, str]) -> CompiledGraphKey:
    unknown = sorted(set(axes) - set(REQUIRED_AXES))
    if unknown:
        raise IdentityError(
            f"unknown identity axes {unknown!r}; v1 is exactly {list(REQUIRED_AXES)!r}"
        )
    clean: dict[str, str] = {}
    for name in REQUIRED_AXES:
        raw = axes.get(name)
        if not isinstance(raw, str) or not raw or raw != raw.strip():
            raise IdentityError(f"compiled graph identity requires canonical string {name!r}")
        value = raw
        _refuse_key_as_fact(name, value)
        clean[name] = value
    return CompiledGraphKey(tuple(sorted(clean.items())))


def _facts_digest(facts: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(facts), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def toolchain_axis_digest(block: Mapping[str, Any]) -> str:
    """Restate the worker's compiler-content axis from its recorded block."""

    facts = {
        str(name): str(value)
        for name, value in block.items()
        if str(name) not in _NOT_TOOLCHAIN
    }
    return _facts_digest(facts)[:_TOOLCHAIN_DIGEST_HEX]


def from_artifact_metadata(metadata: Mapping[str, Any]) -> CompiledGraphKey:
    if metadata.get("kind") != ARTIFACT_KIND:
        raise IdentityError(
            f"only an {ARTIFACT_KIND} artifact has compiled-graph identity"
        )
    sm = metadata.get("sm")
    if not isinstance(sm, str) or not sm.strip():
        raise IdentityError("artifact records no GPU compute capability")
    graph_specialization = metadata.get(GRAPH_SPECIALIZATION_BLOCK)
    if not isinstance(graph_specialization, Mapping):
        raise IdentityError(f"compiled graph records no {GRAPH_SPECIALIZATION_BLOCK} object")
    graph = graph_specialization.get("specialization_hash")
    if not isinstance(graph, str) or not graph.strip():
        raise IdentityError("compiled graph records no graph-specialization hash")
    toolchain = metadata.get("toolchain")
    if (
        not isinstance(toolchain, Mapping)
        or not toolchain
        or any(
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or not isinstance(value, str)
            or not value
            or value != value.strip()
            for name, value in toolchain.items()
        )
    ):
        raise IdentityError("artifact records no toolchain object")
    return from_axes({"graph": graph, "sm": sm, "toolchain": toolchain_axis_digest(toolchain)})
