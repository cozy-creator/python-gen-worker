"""Two-level compiled-graph identity (pgw#1368; rekeyed by pgw#1489).

Level 1 -- ``cg-graph-v1`` -- is a content hash of the DERIVED graph: the
canonical serialization of the traced program plus its call-ingress contract.
Code version, torch version, config and ingress are derivation INPUTS, never
name components, so two code versions that trace byte-identically produce ONE
graph, and an edit that changes the trace changes the hash by construction
(the pgw#1031 closure).

Level 2 positions artifacts under a graph: ``graph -> [env -> .so]``. The
artifact key is **(trace digest, sm, compile stack)** and nothing else
(Paul, 2026-08-19, DESIGN-RULINGS addendum 4 as corrected): the trace digest
is the level-1 graph, ``sm`` is the only free hardware variable, and the
COMPILE STACK is the compiler's own input signature -- torch, triton and the
nvidia libraries, whose versions change what inductor EMITS.

What died with pgw#1489, and why: the key used to be a hash of the WHOLE
resolved package set. That is a second representation of an environment the
endpoint's ``uv.lock`` already pins, and it split the artifact pool on drift
that cannot reach the compiler -- a docs extra, a pillow bump, a dev tool
(pgw#1472 measured 43-package diffs between envs that serve identically).
The stack is READ from the lock at both ends, never recomputed from the
installed set, so the two ends cannot spell the same environment differently.

Everything else that a pod must satisfy is ADMISSION METADATA checked at
adopt, never a key: the ELF-derived driver range and the measured peak-VRAM
stamp ride the requirements manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .declaration import DeclarationError, _canonical_graph, _literal_digest
from .ingress import CallIngress

GRAPH_SCHEME = "cg-graph-v1"
ENV_SCHEME = "cg-env-v2"
_DIGEST_HEX = 56
_GRAPH_RE = re.compile(rf"{GRAPH_SCHEME}-[0-9a-f]{{{_DIGEST_HEX}}}\Z")
_SM_RE = re.compile(r"(sm_[0-9]{2,3}|cpu(-[a-z0-9_]+)?)\Z")


class GraphIdentityError(ValueError):
    """A derived graph or environment cannot state its canonical identity."""


def is_graph_hash(value: object) -> bool:
    return isinstance(value, str) and _GRAPH_RE.fullmatch(value) is not None


def graph_hash(
    program: object, ingress: CallIngress, *, passes: Sequence[str] = ()
) -> str:
    """Content-hash one derived graph: canonical trace + ingress + passes.

    The payload is the sole v1 canonical serialization of the exported program
    (graph nodes, symbol ranges, signature -- the ``declaration`` renderer,
    reused rather than re-derived), the ingress contract's canonical JSON, and
    the literal-constant digest when the trace baked literal tensor values in.
    Parameters and buffers contribute names, dtypes and shapes only: weights
    are checkpoint-side and never part of graph identity.

    ``passes`` (tcg#52) are the transform passes that ran BEFORE this trace,
    in the order the lane names them. They are a derivation input exactly as
    pins are: two lanes differing only by a pass are DIFFERENT graphs. A
    transformed module usually traces differently anyway -- the pass name is
    here so that a pass which does NOT change the trace still cannot share an
    identity with the untransformed lane. The line is always written
    (``passes -`` when there are none), because "no pass" is a stated
    property of the lane, never an absent row.
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
    for name in passes:
        if not isinstance(name, str) or not name.strip():
            raise GraphIdentityError("a pass name must be a non-empty string")
    lines.append("passes " + (",".join(passes) if passes else "-"))
    lines.append(f"literals {literals or '-'}")
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:_DIGEST_HEX]
    return f"{GRAPH_SCHEME}-{digest}"


#: The distributions whose versions change what the compiler EMITS -- the
#: compile stack, and the whole env half of an artifact key (pgw#1489).
#: ``torch`` carries inductor and the AOTI runtime, ``triton`` compiles the
#: kernels, and the ``nvidia-*`` wheels ARE the CUDA toolkit the generated
#: code links against. Nothing else in a lockfile can reach codegen, so
#: nothing else is allowed to split the artifact pool.
STACK_NAMES = frozenset({"torch", "triton", "pytorch-triton"})
STACK_PREFIXES = ("nvidia-",)


def is_compile_relevant(name: str) -> bool:
    """Whether one distribution name is part of the compile stack.

    Spelling-tolerant on the SELECTION side only (``nvidia_cublas_cu12`` and
    ``nvidia-cublas-cu12`` are one library). The selected pair goes into the
    key VERBATIM: normalizing key inputs was the pgw#1472 disease, and it can
    only ever paper over two sources that should have been one.
    """

    handle = str(name).strip().lower().replace("_", "-")
    return handle in STACK_NAMES or handle.startswith(STACK_PREFIXES)


def compile_stack(entries: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """The compile-relevant subset of a stated package set, sorted by name.

    The input is normally every ``[[package]]`` row of the endpoint's
    ``uv.lock``. The output is the ~15 rows that decide what a mint produces;
    a bump anywhere else leaves every existing artifact valid, which is the
    entire point of the pgw#1489 rekey.
    """

    return require_stack(
        [(name, version) for name, version in entries.items() if is_compile_relevant(name)]
    )


def require_stack(
    entries: Sequence[tuple[str, str]] | Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    pairs = entries.items() if isinstance(entries, Mapping) else entries
    clean: dict[str, str] = {}
    for row in pairs:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise GraphIdentityError("a compile stack is [name, version] rows")
        name, version = row
        if not isinstance(name, str) or not name.strip():
            raise GraphIdentityError("compile stack names must be non-empty strings")
        if not isinstance(version, str) or not version.strip():
            raise GraphIdentityError(f"compile stack entry {name!r} states no version")
        if not is_compile_relevant(name):
            # The guard against the pgw#1489 regression: handing this the whole
            # installed set is how the pool got split on a docs extra. Select
            # with `compile_stack` and the caller cannot make that mistake.
            raise GraphIdentityError(
                f"{name.strip()!r} is not a compile-stack package; the key is "
                f"torch/triton/nvidia-* only (use `compile_stack` to select)"
            )
        known = clean.get(name.strip())
        if known is not None and known != version.strip():
            raise GraphIdentityError(
                f"compile stack entry {name.strip()!r} appears twice with different "
                f"versions ({known!r} and {version.strip()!r})"
            )
        clean[name.strip()] = version.strip()
    if not any(is_compile_relevant(name) and name.strip().lower() == "torch"
               for name in clean):
        raise GraphIdentityError(
            "a compile stack states torch: it is the compiler. Read the stack "
            "off the endpoint's uv.lock (`compile_stack`), never off a guess"
        )
    return tuple(sorted(clean.items()))


@dataclass(frozen=True, slots=True)
class EnvIdentity:
    """One artifact environment: (compile stack, sm) -- the key's env half.

    ``stack`` is ``[(name, version), ...]``, the compile-relevant rows of the
    endpoint's resolved lockfile (:func:`compile_stack`), carried as the
    VERSIONS THEMSELVES rather than as a digest of them: a refusal that can
    say ``torch 2.13.0 != 2.14.0`` is worth more than one that says two
    hashes differ, and there is nothing to look up to expand it.

    ``sm`` is the concrete compile target (``sm_89``) or a cpu capability.
    CUDA DRIVER version is deliberately outside identity -- it is a host
    preflight floor recorded in the requirements manifest (pgw#1471), like
    the measured peak-VRAM stamp. So is the ambient C++ toolchain: pinned by
    the image, not fingerprinted (pgw#1467, stated residual).
    """

    #: Declared as either shape because ``__post_init__`` NORMALIZES it --
    #: `require_stack` takes a mapping or rows and always stores the canonical
    #: tuple, so every reader sees `tuple[tuple[str, str], ...]` and every
    #: caller may hand it whichever it has. pgw ruled the same way on its own
    #: side of the boundary (pgw#1490, "a compile stack is either shape");
    #: declaring only the stored shape left torchcg's own `assert_exact_env`
    #: red under mypy against a constructor its docstring invites.
    stack: tuple[tuple[str, str], ...] | Mapping[str, str] | Sequence[tuple[str, str]]
    sm: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stack", require_stack(self.stack))
        if not isinstance(self.sm, str) or _SM_RE.fullmatch(self.sm) is None:
            raise GraphIdentityError(
                f"env sm must be a concrete 'sm_NN' or cpu capability, got {self.sm!r}"
            )

    @property
    def value(self) -> str:
        payload = json.dumps(
            {"sm": self.sm, "stack": [list(row) for row in self.stack]},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        digest = hashlib.sha256(payload).hexdigest()[:_DIGEST_HEX]
        return f"{ENV_SCHEME}-{digest}"

    def describe(self) -> str:
        """The stack in words, for the refusal and the boot line."""

        rows = dict(self.stack)
        head = [f"{name} {rows[name]}" for name in ("torch", "triton") if name in rows]
        others = len(rows) - len(head)
        if others:
            head.append(f"+{others} more")
        return f"{', '.join(head)} @ {self.sm}"

    def __str__(self) -> str:
        return self.value


__all__ = [
    "ENV_SCHEME",
    "EnvIdentity",
    "GRAPH_SCHEME",
    "GraphIdentityError",
    "STACK_NAMES",
    "STACK_PREFIXES",
    "compile_stack",
    "graph_hash",
    "is_compile_relevant",
    "is_graph_hash",
    "require_stack",
]
