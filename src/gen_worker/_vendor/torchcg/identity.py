from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: v2 (tcg#80): the COMPILE POLICY joined the axes. v1 keyed (graph, sm,
#: toolchain) and was therefore blind to the compiler options the mint
#: actually ran under -- two mints differing only in
#: `use_runtime_constant_folding` produced one key and collided in the CAS,
#: measured. The scheme moves because the derivation moved; the GRAMMAR does
#: not (`contracts/compiled_graph_key_vectors.json` already admits an unseen
#: scheme by shape, th#1183).
#: v3 (tcg#83): the DECLARED INPUT LAYOUT joined the axes. v2 keyed (policy,
#: graph, sm, toolchain) and the graph axis is stride-BLIND -- the canonical
#: graph form renders `t(dtype|[shape]|device)` and never a stride -- so a
#: contiguous mint and a channels_last mint of one module produce identical
#: graph, sm, toolchain and policy axes while emitting different `.so` bytes
#: (measured: the same conv module minted twice, one wrapper with a permute
#: kernel for `conv_weight` and one without).
KEY_SCHEME = "cg-key-v3"
_DIGEST_HEX = 56
_TOOLCHAIN_DIGEST_HEX = 16

#: The exact length of a key THIS package makes: ``<scheme>-<digest>``.
_OWN_KEY_LENGTH = len(KEY_SCHEME) + len("-") + _DIGEST_HEX

#: How much longer a FOREIGN scheme's name may be than ours.
#:
#: tcg#86 (b)+(d). `MAX_KEY_LENGTH` was a bare 96 and the derivable part of it
#: was never stated: 66 of those bytes are `_OWN_KEY_LENGTH`, above, and they
#: move when the scheme or digest width moves. The remaining 30 are the whole
#: reason a length bound exists at all -- `_KEY_RE` anchors the DIGEST but
#: leaves the scheme name unbounded by design (`is_compiled_graph_key` admits
#: an unseen scheme BY SHAPE, th#1183), so without a cap a boundary would run
#: an unanchored regex over an arbitrarily long untrusted string. Thirty bytes
#: is ~3x this scheme's own name: room for a longer successor without making
#: the bound a second place a scheme rename has to be remembered.
_FOREIGN_SCHEME_ALLOWANCE = 30

#: The boundary admission bound. Derived, so the day the digest widens this
#: follows instead of silently refusing every key the package itself mints.
MAX_KEY_LENGTH = _OWN_KEY_LENGTH + _FOREIGN_SCHEME_ALLOWANCE

#: The v1 identity axes, in canonical order. Public because consumers otherwise
#: re-declare it and fence the copy; a duplicate that drifts is how a correctly
#: named module computes wrong keys with nothing raising.
#:
#: `compile_policy` (tcg#80) is the digest of the codegen-relevant compile
#: options the mint ran under (`compiler.compile_policy`). It is not a fourth
#: kind of fact: DESIGN-RULINGS addendum 4 as corrected says the key is "the
#: compiler's actual input signature", and the options ARE that signature's
#: other half -- the compile stack says which compiler, the policy says what
#: it was told to do. Leaving it out is what let a fold-on and a fold-off
#: artifact share one address.
#:
#: These are the whole canonical fingerprint: the toolchain block's
#: MEMBERS are the caller's to choose, not this package's. The measured
#: rationale — a 2026-08-16 pod matrix, 10 conclusive rows, 3 hosts — lives in
#: `benchmarks/host_fingerprint/README.md`. It found os_release/glibc/
#: libstdc++/c++-compiler load-breaking in both directions, python_abi and
#: triton inert on a CPU target, and torch_config_digest unstable across hosts
#: for the same wheel. Every one of those is a caller-supplied member; the
#: facts this package does own (`machine`, `host_isa_*`) went unmeasured and
#: stay fail-closed.
#: `declared_input_layout` (tcg#83) is the ratified layout-morphism handle the
#: mint compiled its inputs and constants against -- carried as the HANDLE
#: itself, not a digest, for the same reason the env carries versions: a
#: refusal that can say `torch.channels-last@1 != torch.contiguous@1` is worth
#: more than one that says two hashes differ.
REQUIRED_AXES: tuple[str, ...] = (
    "compile_policy",
    "declared_input_layout",
    "graph",
    "sm",
    "toolchain",
)

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


def policy_axis_digest(policy: Mapping[str, Any]) -> str:
    """Restate the compile-policy axis from one stated options block.

    The values keep their JSON types -- a policy is booleans, not the strings
    a toolchain block carries -- so `False` and `"False"` cannot digest alike.
    """

    if not policy or any(
        not isinstance(name, str) or not name or name != name.strip() for name in policy
    ):
        raise IdentityError("a compile policy is a non-empty block of named options")
    for name, value in policy.items():
        if not isinstance(value, (bool, int, str)) or isinstance(value, float):
            raise IdentityError(
                f"compile policy option {name!r} must be a bool, int or string"
            )
    return _facts_digest(policy)[:_TOOLCHAIN_DIGEST_HEX]


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
    policy = metadata.get("compile_policy")
    if not isinstance(policy, Mapping):
        raise IdentityError(
            "artifact records no compile_policy block; the compile options are "
            "half the compiler's input signature and a key without them "
            "collides across configurations (tcg#80)"
        )
    from .layout import LayoutError, require_morphism

    try:
        declared_layout = require_morphism(metadata.get("declared_input_layout")).handle
    except LayoutError as exc:
        raise IdentityError(
            f"artifact states no ratified declared_input_layout: {exc}"
        ) from exc
    return from_axes(
        {
            "compile_policy": policy_axis_digest(policy),
            "declared_input_layout": declared_layout,
            "graph": graph,
            "sm": sm,
            "toolchain": toolchain_axis_digest(toolchain),
        }
    )
