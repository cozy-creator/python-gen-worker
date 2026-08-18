"""Two-level compiled-graph identity (pgw#1368, ruling 2026-08-18).

Level 1 -- ``cg-graph-v1`` -- is a content hash of the DERIVED graph: the
canonical serialization of the traced program plus its call-ingress contract.
Code version, torch version, config and ingress are derivation INPUTS, never
name components, so two code versions that trace byte-identically produce ONE
graph, and an edit that changes the trace changes the hash by construction
(the pgw#1031 closure).

Level 2 positions artifacts under a graph: ``graph -> [env -> .so]``, where
env identity is the hash of the uv lockfile's RESOLVED CLOSURE plus the GPU
``sm``. The image digest is deliberately NOT identity: docker is a thin
wrapper over ``uv sync`` and cozy-local resolves the same lockfile in a bare
venv, so both compute the same env hash. Adoption is EXACT-ENV; compatibility
reasoning lives only in the miss-path ladder (``requirements.rank``).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .declaration import DeclarationError, _canonical_graph, _literal_digest
from .ingress import CallIngress

GRAPH_SCHEME = "cg-graph-v1"
ENV_SCHEME = "cg-env-v1"
_DIGEST_HEX = 56
_GRAPH_RE = re.compile(rf"{GRAPH_SCHEME}-[0-9a-f]{{{_DIGEST_HEX}}}\Z")
_CLOSURE_RE = re.compile(r"[0-9a-f]{64}\Z")
_SM_RE = re.compile(r"(sm_[0-9]{2,3}|cpu(-[a-z0-9_]+)?)\Z")


class GraphIdentityError(ValueError):
    """A derived graph or environment cannot state its canonical identity."""


def is_graph_hash(value: object) -> bool:
    return isinstance(value, str) and _GRAPH_RE.fullmatch(value) is not None


def graph_hash(program: object, ingress: CallIngress) -> str:
    """Content-hash one derived graph: canonical trace + ingress contract.

    The payload is the sole v1 canonical serialization of the exported program
    (graph nodes, symbol ranges, signature -- the ``declaration`` renderer,
    reused rather than re-derived), the ingress contract's canonical JSON, and
    the literal-constant digest when the trace baked literal tensor values in.
    Parameters and buffers contribute names, dtypes and shapes only: weights
    are checkpoint-side and never part of graph identity.
    """

    try:
        lines = list(_canonical_graph(program))
        literals = _literal_digest(program)
    except DeclarationError as exc:
        raise GraphIdentityError(str(exc)) from exc
    lines.append(
        "ingress "
        + json.dumps(
            ingress.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
    )
    lines.append(f"literals {literals or '-'}")
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:_DIGEST_HEX]
    return f"{GRAPH_SCHEME}-{digest}"


def closure_hash(entries: Mapping[str, str]) -> str:
    """Hash one resolved dependency closure: canonical name -> version.

    The caller states the closure -- normally the uv lockfile's resolved
    package set, and for the boot-time audit the actually-installed
    distribution set (``installed_closure``). Names are normalized the way
    package indexes compare them, so ``Foo_Bar`` and ``foo-bar`` are one
    entry rather than a phantom mismatch.
    """

    clean: dict[str, str] = {}
    for name, version in entries.items():
        if not isinstance(name, str) or not name.strip():
            raise GraphIdentityError("closure entry names must be non-empty strings")
        if not isinstance(version, str) or not version.strip():
            raise GraphIdentityError(f"closure entry {name!r} states no version")
        canonical = re.sub(r"[-_.]+", "-", name.strip().lower())
        known = clean.get(canonical)
        if known is not None and known != version.strip():
            raise GraphIdentityError(
                f"closure entry {canonical!r} appears twice with different versions "
                f"({known!r} and {version.strip()!r})"
            )
        clean[canonical] = version.strip()
    if not clean:
        raise GraphIdentityError("a resolved closure cannot be empty")
    payload = json.dumps(
        clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def installed_closure() -> dict[str, str]:
    """The distribution set actually importable from this process's env.

    This is the AUDIT side of env identity: the publish pipeline stamps the
    lockfile's resolved closure, and a booting worker restates its own
    installed set through the same ``closure_hash`` to assert env == metadata
    (``requirements.assert_exact_env``).
    """

    import importlib.metadata

    entries: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        try:
            name = distribution.name
        except KeyError:
            continue
        if not name:
            continue
        entries[name] = distribution.version
    if not entries:
        raise GraphIdentityError("no installed distributions are visible")
    return entries


@dataclass(frozen=True, slots=True)
class EnvIdentity:
    """One artifact environment: (resolved-closure hash, sm).

    ``closure`` is ``closure_hash`` output -- 64 hex characters. ``sm`` is the
    concrete compile target (``sm_89``) or a cpu capability. CUDA driver
    version is deliberately outside identity: it is a host preflight floor,
    recorded in the requirements manifest, never an axis.
    """

    closure: str
    sm: str

    def __post_init__(self) -> None:
        if not isinstance(self.closure, str) or _CLOSURE_RE.fullmatch(self.closure) is None:
            raise GraphIdentityError(
                "env closure must be the 64-hex resolved-closure hash"
            )
        if not isinstance(self.sm, str) or _SM_RE.fullmatch(self.sm) is None:
            raise GraphIdentityError(
                f"env sm must be a concrete 'sm_NN' or cpu capability, got {self.sm!r}"
            )

    @property
    def value(self) -> str:
        payload = json.dumps(
            {"closure": self.closure, "sm": self.sm},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        digest = hashlib.sha256(payload).hexdigest()[:_DIGEST_HEX]
        return f"{ENV_SCHEME}-{digest}"

    def __str__(self) -> str:
        return self.value


__all__ = [
    "ENV_SCHEME",
    "EnvIdentity",
    "GRAPH_SCHEME",
    "GraphIdentityError",
    "closure_hash",
    "graph_hash",
    "installed_closure",
    "is_graph_hash",
]
