"""What a compiled graph IS: one graph hash, one artifact key.

The key is the compiler's actual input signature and nothing else:

    key = hash(graph capture x env fingerprint x declared input layout x mint policy)

``graph capture`` is the level-1 content hash of the derived graph -- the
canonical serialization of the traced program plus its call ingress. Code
version, torch version and config are derivation INPUTS, never name components,
so two code versions that trace byte-identically produce ONE graph.

The other three are the environment half. ``sm`` is the only free hardware
variable. The ``env`` block says WHICH compiler and WHICH host (its members are
the caller's to choose -- the 2026-08-16 pod matrix found os_release, glibc,
libstdc++ and the C++ compiler load-breaking in both directions, and every one
of them is caller-supplied). The ``policy`` says what that compiler was TOLD to
do, and the ``layout`` says which byte arrangement it compiled its inputs
against. Leaving either out was not minimalism: it made a fold-on mint and a
fold-off mint one address, and a contiguous mint and a channels-last mint
another.

Layout handles are READ from tensorfs' ratified corpus, never authored here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from .refuse import (
    IdentityError,
    IngressError,
    LayoutCorpusError,
    LayoutError,
    LayoutUndeliverableError,
)

GRAPH_SCHEME = "cg-graph-v1"
KEY_SCHEME = "cg-key-v4"
_DIGEST_HEX = 56
_BLOCK_DIGEST_HEX = 16

_GRAPH_RE = re.compile(rf"{GRAPH_SCHEME}-[0-9a-f]{{{_DIGEST_HEX}}}\Z")
_KEY_RE = re.compile(rf"[a-z0-9][a-z0-9._-]*-[0-9a-f]{{{_DIGEST_HEX}}}\Z")
_SM_RE = re.compile(r"(sm_[0-9]{2,3}|cpu(-[a-z0-9_]+)?)\Z")

#: The exact length of a key THIS package makes, plus room for a longer
#: successor scheme. `_KEY_RE` anchors the digest and leaves the scheme name
#: unbounded by design (an unseen scheme is admitted BY SHAPE), so without a cap
#: a boundary would run an unanchored regex over untrusted input.
MAX_KEY_LENGTH = len(KEY_SCHEME) + 1 + _DIGEST_HEX + 30

#: The key's axes, in canonical order. Public because consumers otherwise
#: re-declare it and a duplicate that drifts computes wrong keys silently.
AXES: tuple[str, ...] = ("env", "graph", "layout", "policy", "sm")


def is_graph_hash(value: object) -> bool:
    return isinstance(value, str) and _GRAPH_RE.fullmatch(value) is not None


def is_artifact_key(value: object) -> bool:
    """Whether a boundary value has the artifact-key shape.

    The digest is the anchored suffix; never split on ``-``, because schemes may
    contain hyphens. Refuses shape, not scheme, so a future scheme reaches
    axis-based admission without teaching every boundary its name.
    """

    return (
        isinstance(value, str)
        and len(value) <= MAX_KEY_LENGTH
        and _KEY_RE.fullmatch(value) is not None
    )


# ---------------------------------------------------------------------------
# The compile stack -- which lockfile rows are allowed to split the pool
# ---------------------------------------------------------------------------

#: The distributions whose versions change what the compiler EMITS. ``torch``
#: carries inductor and the AOTI runtime, ``triton`` compiles the kernels, and
#: the ``nvidia-*`` wheels ARE the CUDA toolkit the generated code links
#: against. Nothing else in a lockfile can reach codegen, so nothing else is
#: allowed to split the artifact pool.
STACK_NAMES = frozenset({"torch", "triton", "pytorch-triton"})
STACK_PREFIXES = ("nvidia-",)


def is_compile_relevant(name: str) -> bool:
    """Whether one distribution name is part of the compile stack.

    Spelling-tolerant on the SELECTION side only (``nvidia_cublas_cu12`` and
    ``nvidia-cublas-cu12`` are one library). The selected pair goes into the env
    block VERBATIM: normalizing key inputs can only ever paper over two sources
    that should have been one.
    """

    handle = str(name).strip().lower().replace("_", "-")
    return handle in STACK_NAMES or handle.startswith(STACK_PREFIXES)


def compile_stack(entries: Mapping[str, str]) -> dict[str, str]:
    """The compile-relevant subset of a stated package set.

    The input is normally every ``[[package]]`` row of the endpoint's
    ``uv.lock``. The output is the ~15 rows that decide what a mint produces, so
    a bump anywhere else leaves every existing artifact valid -- which is the
    entire point of selecting rather than digesting the whole set. Handing an
    env fingerprint the WHOLE installed set is how an artifact pool gets split
    on a docs extra or a pillow bump (measured: 43-package diffs between
    environments that serve identically).

    ``torch`` is required: it is the compiler, and an env block that cannot name
    it is describing a machine that cannot mint.
    """

    clean: dict[str, str] = {}
    # Duplicates are detected on the NORMALIZED handle and stored VERBATIM.
    # Both halves matter: `nvidia_cublas_cu13` and `nvidia-cublas-cu13` are one
    # library, so admitting both would put one distribution into the env block
    # twice and digest differently than either spelling alone -- while
    # normalizing what gets STORED would paper over two sources that should
    # have been one.
    seen: dict[str, str] = {}
    for name, version in entries.items():
        if not is_compile_relevant(name):
            continue
        if not isinstance(version, str) or not version.strip():
            raise IdentityError(f"compile stack entry {name!r} states no version")
        handle = str(name).strip()
        normal = handle.lower().replace("_", "-")
        known = seen.get(normal)
        if known is not None and (known != handle or clean[known] != version.strip()):
            raise IdentityError(
                f"compile stack names {handle!r} and {known!r} are one "
                f"distribution ({clean[known]!r} and {version.strip()!r}); the "
                f"caller has two sources for one fact"
            )
        seen[normal] = handle
        clean[handle] = version.strip()
    if not any(name.strip().lower() == "torch" for name in clean):
        raise IdentityError(
            "a compile stack states torch: it is the compiler. Read the stack off "
            "the endpoint's uv.lock, never off a guess"
        )
    return clean


# ---------------------------------------------------------------------------
# Layout morphisms -- CONSUMED from tensorfs, never authored
# ---------------------------------------------------------------------------

#: Where the ratified corpus is when it is not simply beside the installed
#: tensorfs. A PATH, which is configuration -- setting it cannot change what any
#: code DOES, only which directory the one producer's records are read from.
CORPUS_ENV = "TENSORFS_SPEC_V2"

#: The memory formats THIS COMPILER can deliver. A capability of torch, not a
#: vocabulary: no handle, no rank, no permutation is stated here.
_DELIVERABLE_FORMATS = ("contiguous_format", "channels_last", "channels_last_3d")


@dataclass(frozen=True, slots=True)
class LayoutMorphism:
    """One ratified arrangement this compiler can deliver.

    ``handle`` and ``rank`` are the RECORD's. ``memory_format`` names the
    ``torch.memory_format`` proven to produce that record's permutation -- the
    pairing, not a second statement of the arrangement.
    """

    handle: str
    memory_format: str
    rank: int | None

    def format(self) -> Any:
        import torch

        return getattr(torch, self.memory_format)

    def applies_to(self, rank: int) -> bool:
        return self.rank is None or self.rank == rank

    def matches(self, tensor: Any) -> bool:
        """Whether these bytes are ALREADY in this layout -- no copy needed.

        Inductor's ``require_strides`` emits no copy when the strides already
        match, so a tree delivered in the declared layout costs zero permute
        kernels. A morphism that does not apply to a tensor's rank leaves it
        row-major: a 1-D bias under channels-last is contiguous, not an error.
        """

        import torch

        if not self.applies_to(tensor.dim()):
            return bool(tensor.is_contiguous(memory_format=torch.contiguous_format))
        return bool(tensor.is_contiguous(memory_format=self.format()))

    def describe(self, tensor: Any) -> str:
        return (
            f"shape {tuple(tensor.shape)} strides {tuple(tensor.stride())} "
            f"(declared {self.handle})"
        )


def _corpus_root(root: Path | None) -> Path | None:
    """Resolved BEFORE the cache: caching on the argument while reading the
    environment underneath makes the first call's environment the answer
    forever, which is a guard that cannot go red."""

    if root is not None:
        return root
    stated = os.environ.get(CORPUS_ENV)
    return Path(stated) if stated else None


@cache
def _identity_morphism(root: Path | None = None) -> LayoutMorphism:
    """The identity morphism, resolved WITHOUT importing torch.

    Load-bearing, not an optimization: handing an artifact back out of the store
    happens in processes that never import torch, and every artifact declares a
    layout. The identity is the corpus' unique rankless record, it permutes
    nothing, and row-major is row-major without asking torch.
    """

    from tensorfs import identity_arrangement

    try:
        arrangement = identity_arrangement(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise LayoutCorpusError(
            f"the identity layout is the corpus' unique rankless record and it "
            f"could not be read: {exc}. tensorfs' wheel deliberately carries no "
            f"corpus; vendor one beside the package, pass root=, or name one in "
            f"${CORPUS_ENV}"
        ) from exc
    return LayoutMorphism(str(arrangement.handle), "contiguous_format", None)


@cache
def _paired(root: Path | None = None) -> tuple[dict[str, LayoutMorphism], tuple[str, ...]]:
    """(deliverable morphisms, ratified-but-undeliverable handles).

    The pairing is arithmetic and runs in BOTH directions: a record whose
    permutation moves stops pairing, and a torch whose memory format changes
    stops pairing, rather than either being silently absorbed.
    """

    try:
        from tensorfs import layouts
    except ImportError as exc:  # pragma: no cover - tensorfs is a hard dependency
        raise LayoutCorpusError(
            f"the ratified layout corpus is tensorfs' (spec/v2/layouts) and "
            f"tensorfs is not importable: {exc}"
        ) from exc
    try:
        records = dict(layouts(root))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise LayoutCorpusError(
            f"no ratified layout corpus is reachable, so no layout can be "
            f"resolved: {exc}"
        ) from exc
    deliverable: dict[str, LayoutMorphism] = {}
    for arrangement in records.values():
        memory_format = _format_producing(arrangement)
        if memory_format is not None:
            deliverable[arrangement.handle] = LayoutMorphism(
                arrangement.handle, memory_format, arrangement.rank
            )
    undeliverable = tuple(sorted(h for h in records if h not in deliverable))
    return deliverable, undeliverable


def _format_producing(arrangement: Any) -> str | None:
    if arrangement.rank is None:
        return "contiguous_format" if arrangement.is_identity else None
    if arrangement.is_identity:
        return None
    wanted = tuple(arrangement.permutation)
    # A memory format PERMUTES axes; it never SPLITS one. A record that factors
    # an axis is arithmetic no memory_format can express, and comparing a
    # rank-length probe against a longer permutation would answer "no" for the
    # right reason by accident. Say it on purpose.
    if len(wanted) != arrangement.rank or len(arrangement.sub_axes) != arrangement.rank:
        return None
    for memory_format in _DELIVERABLE_FORMATS:
        if _probe_permutation(memory_format, arrangement.rank) == wanted:
            return memory_format
    return None


@cache
def _probe_permutation(memory_format: str, rank: int) -> tuple[int, ...] | None:
    """Storage order over logical axes, outermost first -- the RECORD's spelling.

    ``None`` when torch declines the format at that rank (channels_last needs
    rank 4): a NEGATIVE ANSWER, not a swallowed error.
    """

    import torch

    size = tuple(range(2, 2 + rank))
    try:
        probe = torch.empty(size, device="meta").to(
            memory_format=getattr(torch, memory_format)
        )
    except RuntimeError:
        return None
    strides = [int(v) for v in probe.stride()]
    return tuple(sorted(range(rank), key=lambda axis: -strides[axis]))


def contiguous_handle(root: Path | None = None) -> str:
    """The identity handle, READ from the corpus rather than spelled."""

    return _identity_morphism(_corpus_root(root)).handle


def require_morphism(handle: object, root: Path | None = None) -> LayoutMorphism:
    """Resolve one declared layout, or refuse naming which kind of miss it is.

    An unclassified layout is a refusal, never a coercion to row-major: a
    declaration nobody can verify would let an artifact claim a layout its
    loader silently ignores, which is the defect this axis exists to remove.
    """

    resolved = _corpus_root(root)
    # The identity first, and WITHOUT torch: it is what every artifact in the
    # fleet declares, and the cold resolve path must stay torch-free.
    identity = _identity_morphism(resolved)
    if handle == identity.handle:
        return identity
    deliverable, undeliverable = _paired(resolved)
    if isinstance(handle, str) and handle in deliverable:
        return deliverable[handle]
    if isinstance(handle, str) and handle in undeliverable:
        raise LayoutUndeliverableError(
            f"layout morphism {handle!r} IS ratified, and this compiler cannot "
            f"deliver it: no torch memory format produces its arrangement. It "
            f"reaches a tensor through the tensorfs fill path, which applies the "
            f"morphism in flight -- not by being declared here. Deliverable: "
            f"{sorted(deliverable)!r}"
        )
    raise LayoutError(
        f"declared input layout {handle!r} is not a ratified layout morphism; the "
        f"ratified catalog is {sorted([*deliverable, *undeliverable])!r}. Machines "
        f"derive along ratified morphisms and never invent one: an unknown layout "
        f"is a CANDIDATE for ratification, and the mint falls back to the identity"
    )


def catalog(root: Path | None = None) -> dict[str, LayoutMorphism]:
    return dict(_paired(_corpus_root(root))[0])


# ---------------------------------------------------------------------------
# Call ingress -- the one fact the exported program cannot encode
# ---------------------------------------------------------------------------

PathStep = int | str


def _leaf_name(param: str, path: Sequence[PathStep]) -> str:
    name = param
    for step in path:
        name = step if isinstance(step, str) else f"{name}.{step}"
    return name


def exported_input_name(param: str, path: Sequence[PathStep] = ()) -> str:
    """The placeholder spelling ``torch.export`` produces."""

    return "_".join((str(param), *(str(step) for step in path)))


@dataclass(frozen=True, slots=True)
class CallInput:
    """One tensor feed and its exact identity in the unflattened call."""

    name: str
    position: int
    param: str
    param_position: int
    path: tuple[PathStep, ...]
    exported_name: str
    dtype: str
    shape: tuple[int | str, ...]

    def __post_init__(self) -> None:
        for field, value in (
            ("name", self.name),
            ("param", self.param),
            ("exported_name", self.exported_name),
            ("dtype", self.dtype),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise IngressError("input_invalid", f"call input {field} must be canonical")
        for field, index in (
            ("position", self.position),
            ("param_position", self.param_position),
        ):
            if type(index) is not int or index < 0:
                raise IngressError("input_invalid", f"call input {field} must be non-negative")
        if not isinstance(self.path, tuple) or any(
            (type(step) is not int and not isinstance(step, str))
            or (type(step) is int and step < 0)
            or (isinstance(step, str) and (not step or step != step.strip()))
            for step in self.path
        ):
            raise IngressError("input_invalid", "call input path is not canonical")
        if self.name != _leaf_name(self.param, self.path):
            raise IngressError("input_invalid", "call input name does not restate param/path")
        if self.exported_name != exported_input_name(self.param, self.path):
            raise IngressError(
                "input_invalid", "call input exported_name does not restate param/path"
            )
        if not isinstance(self.shape, tuple) or any(
            (type(d) is not int and not isinstance(d, str))
            or (type(d) is int and d < 0)
            or (isinstance(d, str) and (not d or d != d.strip()))
            for d in self.shape
        ):
            raise IngressError("input_invalid", "call input shape is not canonical")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": self.position,
            "param": self.param,
            "param_position": self.param_position,
            "path": list(self.path),
            "exported_name": self.exported_name,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    def resolve(
        self, args: Sequence[object], kwargs: Mapping[str, object]
    ) -> tuple[bool, object | None]:
        """Pull this input's value out of a real call, or report it absent."""

        if self.param in kwargs:
            value: object = kwargs[self.param]
        elif self.param_position < len(args):
            value = args[self.param_position]
        else:
            return False, None
        for step in self.path:
            if isinstance(step, str):
                if not isinstance(value, Mapping) or step not in value:
                    return False, None
                value = value[step]
            else:
                if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
                    return False, None
                if step >= len(value):
                    return False, None
                value = value[step]
        return True, value


@dataclass(frozen=True, slots=True)
class CallIngress:
    """The closed, identity-hashed call contract consumed by graph runners."""

    parameters: tuple[str, ...]
    flat_arity: int
    inputs: tuple[CallInput, ...]
    symbols: tuple[tuple[str, tuple[int, int]], ...] = ()
    excluded_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parameters, tuple)
            or not self.parameters
            or any(
                not isinstance(n, str) or not n or n != n.strip() for n in self.parameters
            )
        ):
            raise IngressError("ingress_invalid", "parameters must be canonical strings")
        if len(self.parameters) != len(set(self.parameters)):
            raise IngressError("ingress_invalid", "parameters must be unique")
        if type(self.flat_arity) is not int or self.flat_arity <= 0:
            raise IngressError("ingress_invalid", "flat_arity must be a positive integer")
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise IngressError("ingress_invalid", "call ingress must declare tensor inputs")
        positions = tuple(row.position for row in self.inputs)
        if positions != tuple(sorted(set(positions))):
            raise IngressError("ingress_invalid", "input positions must be strictly increasing")
        if positions[-1] >= self.flat_arity:
            raise IngressError("ingress_invalid", "input position exceeds flat_arity")
        param_positions = tuple(row.param_position for row in self.inputs)
        if param_positions != tuple(sorted(param_positions)):
            raise IngressError(
                "ingress_invalid", "input param_position values must be nondecreasing"
            )
        for row in self.inputs:
            if row.param_position >= len(self.parameters):
                raise IngressError(
                    "ingress_invalid", f"input {row.name!r} param_position is out of range"
                )
            if self.parameters[row.param_position] != row.param:
                raise IngressError(
                    "ingress_invalid",
                    f"input {row.name!r} param does not match parameters[param_position]",
                )
        for attribute, values in (
            ("name", tuple(row.name for row in self.inputs)),
            ("exported_name", tuple(row.exported_name for row in self.inputs)),
        ):
            if len(values) != len(set(values)):
                raise IngressError("ingress_invalid", f"duplicate input {attribute}")
        if not isinstance(self.symbols, tuple):
            raise IngressError("ingress_invalid", "symbols must be canonical tuples")
        names = tuple(name for name, _ in self.symbols)
        if names != tuple(sorted(set(names))):
            raise IngressError("ingress_invalid", "symbols must be sorted and unique")
        for name, bounds in self.symbols:
            if not isinstance(name, str) or not name or name != name.strip():
                raise IngressError("ingress_invalid", "symbol name must be canonical")
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(type(b) is not int for b in bounds)
                or bounds[1] < bounds[0]
            ):
                raise IngressError("ingress_invalid", f"symbol {name!r} bounds are invalid")
        referenced = {
            d for row in self.inputs for d in row.shape if isinstance(d, str)
        }
        known = set(names)
        missing = sorted(referenced - known)
        if missing:
            raise IngressError("ingress_invalid", f"symbol {missing[0]!r} has no declared bounds")
        unused = sorted(known - referenced)
        if unused:
            raise IngressError("ingress_invalid", f"symbol {unused[0]!r} is unreferenced")
        if not isinstance(self.excluded_inputs, tuple) or any(
            not isinstance(n, str) or not n or n != n.strip() for n in self.excluded_inputs
        ):
            raise IngressError("ingress_invalid", "excluded_inputs must be canonical strings")
        if self.excluded_inputs != tuple(sorted(set(self.excluded_inputs))):
            raise IngressError("ingress_invalid", "excluded_inputs must be sorted and unique")
        overlap = sorted(set(self.excluded_inputs) & {row.name for row in self.inputs})
        if overlap:
            raise IngressError(
                "ingress_invalid", f"input {overlap[0]!r} is both declared and excluded"
            )

    @property
    def symbol_bounds(self) -> dict[str, tuple[int, int]]:
        return dict(self.symbols)

    def as_dict(self) -> dict[str, Any]:
        return {
            "v": 1,
            "parameters": list(self.parameters),
            "flat_arity": self.flat_arity,
            "inputs": [row.as_dict() for row in self.inputs],
            "symbols": {name: list(bounds) for name, bounds in self.symbols},
            "excluded_inputs": list(self.excluded_inputs),
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.as_dict())).hexdigest()[:32]

    @classmethod
    def decode(cls, raw: Mapping[str, Any]) -> CallIngress:
        if not isinstance(raw, Mapping) or set(raw) != {
            "v",
            "parameters",
            "flat_arity",
            "inputs",
            "symbols",
            "excluded_inputs",
        }:
            raise IngressError("ingress_invalid", "ingress document has the wrong field set")
        if raw.get("v") != 1:
            raise IngressError("ingress_invalid", f"unsupported ingress version {raw.get('v')!r}")
        rows = raw.get("inputs")
        if not isinstance(rows, list):
            raise IngressError("ingress_invalid", "ingress inputs must be a list")
        inputs = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "name",
                "position",
                "param",
                "param_position",
                "path",
                "exported_name",
                "dtype",
                "shape",
            }:
                raise IngressError("ingress_invalid", "ingress input has the wrong field set")
            inputs.append(
                CallInput(
                    name=row["name"],
                    position=row["position"],
                    param=row["param"],
                    param_position=row["param_position"],
                    path=tuple(row["path"]),
                    exported_name=row["exported_name"],
                    dtype=row["dtype"],
                    shape=tuple(row["shape"]),
                )
            )
        symbols = raw.get("symbols")
        if not isinstance(symbols, Mapping):
            raise IngressError("ingress_invalid", "ingress symbols must be an object")
        return cls(
            parameters=tuple(raw["parameters"]),
            flat_arity=raw["flat_arity"],
            inputs=tuple(inputs),
            symbols=tuple(
                sorted((str(n), (int(b[0]), int(b[1]))) for n, b in symbols.items())
            ),
            excluded_inputs=tuple(raw["excluded_inputs"]),
        )

    def bind(
        self, args: Sequence[object], kwargs: Mapping[str, object]
    ) -> dict[str, object]:
        """Every declared input's value from one real call, by name."""

        bound: dict[str, object] = {}
        for row in self.inputs:
            present, value = row.resolve(args, kwargs)
            if not present:
                raise IngressError(
                    "call_invalid", f"call carries no value for declared input {row.name!r}"
                )
            bound[row.name] = value
        return bound

    def feeds(
        self, args: Sequence[object], kwargs: Mapping[str, object]
    ) -> tuple[object, ...]:
        """The declared inputs in FLAT position order -- the runner's arg list."""

        bound = self.bind(args, kwargs)
        return tuple(bound[row.name] for row in self.inputs)


def symbol_terms(name: str) -> tuple[int, str]:
    """``'8*s1'`` -> ``(8, 's1')``; ``'s1'`` -> ``(1, 's1')``.

    A latent side a UNet accepts only in multiples of eight exports as ``8*s1``.
    The coefficient is carried by the symbol's own NAME rather than a second
    schema field, so every guard reads the stride and the ROOT from one place
    (``8*s1`` and ``16*s1`` are one degree of freedom, not two).
    """

    head, _, tail = name.partition("*")
    if tail and head.lstrip("-").isdigit():
        return int(head), tail
    return 1, name


def _flatten_call(
    param_names: Sequence[str], args: Sequence[object], kwargs: Mapping[str, object]
) -> tuple[tuple[str, int, tuple[PathStep, ...], object], ...]:
    names = tuple(param_names)
    if any(
        not isinstance(n, str) or not n or n != n.strip() for n in names
    ) or len(names) != len(set(names)):
        raise IngressError("call_invalid", "call parameter names must be canonical and unique")
    if len(args) > len(names):
        raise IngressError("call_invalid", "call has more positional values than parameters")
    values = list(args)
    for name in names[len(args) :]:
        if name not in kwargs:
            raise IngressError("call_invalid", f"call carries no value for parameter {name!r}")
        values.append(kwargs[name])
    output: list[tuple[str, int, tuple[PathStep, ...], object]] = []

    def walk(param: str, position: int, path: tuple[PathStep, ...], value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str) or not key or key != key.strip():
                    raise IngressError("call_invalid", "mapping input key is not canonical")
                walk(param, position, path + (key,), item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(param, position, path + (index,), item)
        else:
            output.append((param, position, path, value))

    for position, (param, value) in enumerate(zip(names, values, strict=True)):
        walk(param, position, (), value)
    return tuple(output)


def _finite(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IngressError("range_invalid", f"symbol range bound {value!r} is not finite") from exc


def build_call_ingress(
    program: object,
    param_names: Sequence[str],
    args: Sequence[object],
    kwargs: Mapping[str, object],
    *,
    excluded_inputs: Sequence[str] = (),
) -> CallIngress:
    """Build the sole ingress declaration from a real exported call."""

    import torch

    if not isinstance(program, torch.export.ExportedProgram):
        raise IngressError("program_invalid", "call ingress requires an ExportedProgram")
    leaves = _flatten_call(param_names, args, kwargs)
    signature = getattr(program, "graph_signature", None)
    user_inputs = tuple(getattr(signature, "user_inputs", ()) or ())
    if len(user_inputs) != len(leaves):
        raise IngressError(
            "call_invalid",
            f"export records {len(user_inputs)} flat inputs but the call has "
            f"{len(leaves)} leaves",
        )
    placeholders = {
        str(node.name): node.meta.get("val")
        for node in program.graph_module.graph.nodes
        if node.op == "placeholder"
    }
    ranges = {
        str(symbol): (_finite(getattr(i, "lower", None)), _finite(getattr(i, "upper", None)))
        for symbol, i in (getattr(program, "range_constraints", {}) or {}).items()
    }
    used: dict[str, tuple[int, int]] = {}
    rows: list[CallInput] = []
    for position, (leaf, exported) in enumerate(zip(leaves, user_inputs, strict=True)):
        param, param_position, path, _ = leaf
        # A non-str user input IS a ConstantArgument the trace baked.
        if not isinstance(exported, str):
            continue
        value = placeholders.get(exported)
        dtype = str(getattr(value, "dtype", "") or "").removeprefix("torch.")
        shape_value = getattr(value, "shape", None)
        if not dtype or shape_value is None:
            continue
        leaf_exported = exported_input_name(param, path)
        if exported != leaf_exported:
            raise IngressError(
                "call_invalid",
                f"torch.export input {exported!r} does not match call leaf {leaf_exported!r}",
            )
        shape: list[int | str] = []
        for dimension in shape_value:
            text = str(dimension)
            if text.lstrip("-").isdigit():
                shape.append(int(text))
                continue
            bounds = ranges.get(text)
            if bounds is None:
                # A derived dim: torch bounds the ROOT and spells the axis as the
                # expression, so the axis's range is the root's, scaled. Recorded
                # in the units the CALL presents, which is what guards compare.
                coefficient, root = symbol_terms(text)
                root_bounds = ranges.get(root) if coefficient > 1 else None
                if root_bounds is not None:
                    bounds = (coefficient * root_bounds[0], coefficient * root_bounds[1])
            if bounds is None:
                raise IngressError(
                    "range_invalid",
                    f"input {_leaf_name(param, path)!r} symbol {text!r} has no finite range",
                )
            used[text] = bounds
            shape.append(text)
        rows.append(
            CallInput(
                name=_leaf_name(param, path),
                position=position,
                param=param,
                param_position=param_position,
                path=path,
                exported_name=leaf_exported,
                dtype=dtype,
                shape=tuple(shape),
            )
        )
    return CallIngress(
        parameters=tuple(param_names),
        flat_arity=len(leaves),
        inputs=tuple(rows),
        symbols=tuple(sorted(used.items())),
        excluded_inputs=tuple(sorted(excluded_inputs)),
    )


# ---------------------------------------------------------------------------
# The canonical graph form -- BYTE-EXACT, the graph hash is banked against it
# ---------------------------------------------------------------------------

_CANONICAL_FORMAT = 1

#: Ops whose CALL SPELLING varies with the trace path while the call itself does
#: not (tcg#88). A direct ``torch.export`` of author code passes
#: ``_assert_tensor_metadata``'s optional dtype as a kwarg; re-exporting the same
#: program's flat graph module materializes it positionally behind two ``None``
#: placeholders. Same op, same values, two renders -- which split the graph hash
#: between a direct static export and a static bind of the symbolic parent.
#: Scoped to the named ops so no existing direct-export hash moves.
_SPELLING_NORMALIZED_SUFFIXES = ("::_assert_tensor_metadata",)


class _Symbols:
    def __init__(self) -> None:
        self._ids: dict[str, str] = {}

    def name(self, symbol: Any) -> str:
        raw = str(symbol)
        known = self._ids.get(raw)
        if known is None:
            known = f"S{len(self._ids)}"
            self._ids[raw] = known
        return known

    def known(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._ids.items(), key=lambda item: item[1]))


def _render_symbol(value: Any, symbols: _Symbols) -> str:
    # `_expr` is the symbol the EXPORT established; `.expr` folds in whatever the
    # ShapeEnv has been told since. tcg#78: an AOTI compile shares the program's
    # ShapeEnv and installs replacements there, so reading `.expr` made this
    # canonical form a function of COMPILER STATE rather than of the exported
    # program -- the same program declared twice around a compile produced two
    # witnesses. A declaration must describe its input.
    node = getattr(value, "node", None)
    expression = getattr(node, "_expr", None)
    if expression is None:
        expression = getattr(node, "expr", None)
    free = getattr(expression, "free_symbols", None) or ()
    if expression is None or not free:
        return str(expression if expression is not None else value)
    rendered = str(expression)
    for symbol in sorted(free, key=lambda item: len(str(item)), reverse=True):
        rendered = rendered.replace(str(symbol), symbols.name(symbol))
    return rendered


def _render_scalar(value: Any) -> str:
    import torch

    if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
        rendered = str(value)
        return rendered.split(":", 1)[0] if isinstance(value, torch.device) else rendered
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, (int, str, bytes)):
        return repr(value)
    if isinstance(value, slice):
        return f"slice({value.start!r},{value.stop!r},{value.step!r})"
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _render_tensor(value: Any, symbols: _Symbols, *, include_device: bool) -> str:
    shape = ",".join(_render_symbol(d, symbols) for d in value.shape)
    device = f"|{value.device.type}" if include_device else ""
    return f"t({value.dtype}|[{shape}]{device})"


def _render_value(value: Any, symbols: _Symbols) -> str:
    import torch

    if value is None:
        return "-"
    if isinstance(value, torch.Tensor):
        return _render_tensor(value, symbols, include_device=True)
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return f"sym({_render_symbol(value, symbols)})"
    if isinstance(value, (list, tuple)):
        body = ",".join(_render_value(item, symbols) for item in value)
        return f"[{body}]" if isinstance(value, list) else f"({body})"
    if isinstance(value, dict):
        body = ",".join(
            f"{key!r}:{_render_value(item, symbols)}"
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
        return "{" + body + "}"
    return _render_scalar(value)


def _render_argument(value: Any, names: dict[Any, str], symbols: _Symbols) -> str:
    import torch

    if isinstance(value, torch.fx.Node):
        return names.get(value, "%?")
    if isinstance(value, (list, tuple)):
        return "(" + ",".join(_render_argument(i, names, symbols) for i in value) + ")"
    if isinstance(value, dict):
        body = ",".join(
            f"{key!r}:{_render_argument(item, names, symbols)}"
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
        return "{" + body + "}"
    if isinstance(value, torch.Tensor):
        return _render_tensor(value, symbols, include_device=False)
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return f"sym({_render_symbol(value, symbols)})"
    return _render_scalar(value)


def _target(node: Any) -> str:
    if node.op == "placeholder":
        return ""
    if node.op in ("call_method", "get_attr", "output"):
        return str(node.target)
    module = getattr(node.target, "__module__", "")
    name = getattr(node.target, "__qualname__", "")
    return f"{module}.{name}" if module and name else str(node.target)


def _node_value(node: Any) -> tuple[bool, Any]:
    for key in ("val", "example_value", "tensor_meta"):
        if key in node.meta:
            return True, node.meta[key]
    return False, None


def _normalized_call(node: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    schema = getattr(getattr(node, "target", None), "_schema", None)
    if schema is None:
        return tuple(node.args), dict(node.kwargs)
    arguments = list(schema.arguments)
    if len(node.args) > len(arguments):
        return tuple(node.args), dict(node.kwargs)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for argument, value in zip(arguments, node.args, strict=False):
        if not argument.has_default_value():
            args.append(value)
        elif value != argument.default_value:
            kwargs[argument.name] = value
    by_name = {argument.name: argument for argument in arguments}
    for name, value in node.kwargs.items():
        argument = by_name.get(name)
        if argument is None or not argument.has_default_value():
            kwargs[name] = value
        elif value != argument.default_value:
            kwargs[name] = value
    return tuple(args), kwargs


def _graph_lines(graph: Any, symbols: _Symbols) -> list[str]:
    names = {node: f"%{index}" for index, node in enumerate(graph.nodes)}
    lines: list[str] = []
    for index, node in enumerate(graph.nodes):
        node_args, node_kwargs = tuple(node.args), dict(node.kwargs)
        if _target(node).endswith(_SPELLING_NORMALIZED_SUFFIXES):
            node_args, node_kwargs = _normalized_call(node)
        args = ",".join(_render_argument(i, names, symbols) for i in node_args)
        kwargs = ",".join(
            f"{key}={_render_argument(item, names, symbols)}"
            for key, item in sorted(node_kwargs.items())
        )
        present, raw_value = _node_value(node)
        value = _render_value(raw_value, symbols) if present else "-"
        placeholder = f"arg{index}" if node.op == "placeholder" else ""
        lines.append(
            f"node {index} {node.op} {_target(node)} {placeholder} "
            f"args=({args}) kwargs=({kwargs}) val={value}"
        )
    return lines


def _render_bound(value: Any) -> str:
    if value is None:
        return "oo"
    try:
        return str(int(value))
    except (TypeError, ValueError, OverflowError):
        return str(value).replace("+", "")


def _range_lines(ranges: Any, symbols: _Symbols) -> list[str]:
    if not ranges:
        return []
    known = dict(symbols.known())
    lines = []
    for symbol, value_range in ranges.items():
        canonical = known.get(str(symbol))
        if canonical is not None:
            lines.append(
                f"sym {canonical} range=[{_render_bound(getattr(value_range, 'lower', None))},"
                f"{_render_bound(getattr(value_range, 'upper', None))}]"
            )
    return sorted(lines)


def _signature_lines(program: Any) -> list[str]:
    signature = getattr(program, "graph_signature", None)
    lines: list[str] = []
    for direction, specs in (
        ("in", getattr(signature, "input_specs", None) or ()),
        ("out", getattr(signature, "output_specs", None) or ()),
    ):
        for index, spec in enumerate(specs):
            kind_object = getattr(spec, "kind", None)
            kind = getattr(kind_object, "name", str(kind_object or ""))
            target = getattr(spec, "target", None) or "-"
            lines.append(f"sig {direction} {index} kind={kind} target={target}")
    for attribute, direction in (("_in_spec", "in"), ("_out_spec", "out")):
        spec = getattr(program, attribute, None)
        if spec is not None:
            lines.append(f"spec {direction} {spec!s}".replace("\n", " "))
    return lines


def canonical_graph(program: object) -> tuple[str, ...]:
    """The sole canonical form of a ``torch.export.ExportedProgram``."""

    import torch

    if not isinstance(program, torch.export.ExportedProgram):
        raise IdentityError("a canonical graph requires a torch.export.ExportedProgram")
    symbols = _Symbols()
    lines = [f"v={_CANONICAL_FORMAT} ir=export"]
    lines.extend(_graph_lines(program.graph_module.graph, symbols))
    lines.extend(_range_lines(program.range_constraints, symbols))
    lines.extend(_signature_lines(program))
    return tuple(lines)


def constant_names(program: object) -> tuple[str, ...]:
    signature = getattr(program, "graph_signature", None)
    names = {
        str(name)
        for field in ("parameters", "buffers", "lifted_tensor_constants")
        for name in (getattr(signature, field, ()) or ())
    }
    names.update(str(name) for name in (getattr(program, "constants", {}) or {}))
    return tuple(sorted(names))


def literal_names(program: object) -> tuple[str, ...]:
    """Constants that are NOT state dict -- values the trace baked in."""

    signature = getattr(program, "graph_signature", None)
    state = {
        str(name)
        for field in ("parameters", "buffers")
        for name in (getattr(signature, field, ()) or ())
    }
    names = {str(n) for n in getattr(signature, "lifted_tensor_constants", ()) or ()}
    names.update(str(n) for n in (getattr(program, "constants", {}) or {}))
    return tuple(sorted(names - state))


def literal_digest(program: object) -> str:
    names = literal_names(program)
    if not names:
        return ""
    values = getattr(program, "constants", {}) or {}
    digest = hashlib.sha256()
    for name in names:
        value = values.get(name)
        if value is None:
            raise IdentityError(f"literal constant {name!r} carries no value")
        try:
            import torch

            flat = value.detach().cpu().contiguous().reshape(-1)
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(flat.view(torch.uint8).numpy().tobytes())
        except Exception as exc:
            raise IdentityError(
                f"literal constant {name!r} could not be digested: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return digest.hexdigest()[:32]


def placement(program: object) -> tuple[str, ...]:
    """Every device spelling the trace mentions. Always recorded, single
    included: a lifted constant left behind on another device is the failure
    that exports cleanly and only AOTI rejects."""

    import torch

    devices: set[str] = set()

    def note(value: Any) -> None:
        if isinstance(value, torch.device):
            devices.add(
                f"{value.type}:{value.index}" if value.index is not None else value.type
            )
        elif isinstance(value, torch.Tensor):
            note(value.device)
        elif isinstance(value, (list, tuple)):
            for item in value:
                note(item)
        elif isinstance(value, dict):
            for item in value.values():
                note(item)

    for node in program.graph_module.graph.nodes:  # type: ignore[attr-defined]
        present, value = _node_value(node)
        if present:
            note(value)
        note(node.args)
        note(node.kwargs)
    return tuple(sorted(devices))


def graph_hash(
    program: object, ingress: CallIngress, *, passes: Sequence[str] = ()
) -> str:
    """Content-hash one derived graph: canonical trace + ingress + passes.

    Parameters and buffers contribute names, dtypes and shapes only: weights are
    checkpoint-side and never part of graph identity.

    ``passes`` are the transform passes that ran BEFORE this trace, in the order
    the lane names them -- a derivation input exactly as pins are. The line is
    always written (``passes -`` when there are none), because "no pass" is a
    stated property of the lane, never an absent row.
    """

    lines = list(canonical_graph(program))
    literals = literal_digest(program)
    lines.append("ingress " + json.dumps(
        ingress.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ))
    for name in passes:
        if not isinstance(name, str) or not name.strip():
            raise IdentityError("a pass name must be a non-empty string")
    lines.append("passes " + (",".join(passes) if passes else "-"))
    lines.append(f"literals {literals or '-'}")
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:_DIGEST_HEX]
    return f"{GRAPH_SCHEME}-{digest}"


# ---------------------------------------------------------------------------
# The artifact key -- ONE address
# ---------------------------------------------------------------------------


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _block_digest(name: str, block: Mapping[str, Any], types: tuple[type, ...]) -> str:
    """Digest one named block of scalar facts, refusing anything unstatable.

    Values keep their JSON types: a policy is booleans and an env is strings, and
    ``False`` must not digest as ``"False"``.
    """

    if not block:
        raise IdentityError(f"a {name} is a non-empty block of named facts")
    for key, value in block.items():
        if not isinstance(key, str) or not key or key != key.strip():
            raise IdentityError(f"{name} fact names must be canonical strings")
        if not isinstance(value, types) or isinstance(value, float):
            allowed = "/".join(t.__name__ for t in types)
            raise IdentityError(f"{name} fact {key!r} must be {allowed}, got {value!r}")
    return hashlib.sha256(_canonical_json(dict(block))).hexdigest()[:_BLOCK_DIGEST_HEX]


@dataclass(frozen=True, slots=True)
class ArtifactKey:
    """One compiled artifact's complete address.

    ``env`` and ``policy`` are carried as the BLOCKS themselves, not as digests,
    for the reason a refusal that says ``torch 2.13.0 != 2.14.0`` is worth more
    than one that says two hashes differ -- and there is nothing to look up to
    expand it. Only ``value`` digests them.
    """

    graph: str
    sm: str
    env: Mapping[str, str]
    policy: Mapping[str, bool | int | str]
    layout: str

    def __post_init__(self) -> None:
        if not is_graph_hash(self.graph):
            raise IdentityError(
                f"the graph axis is a {GRAPH_SCHEME} hash, got {self.graph!r}"
            )
        if not isinstance(self.sm, str) or _SM_RE.fullmatch(self.sm) is None:
            raise IdentityError(
                f"sm must be a concrete 'sm_NN' or cpu capability, got {self.sm!r}"
            )
        object.__setattr__(self, "env", dict(self.env))
        object.__setattr__(self, "policy", dict(self.policy))
        _block_digest("env fingerprint", self.env, (str,))
        _block_digest("compile policy", self.policy, (bool, int, str))
        require_morphism(self.layout)

    def axes(self) -> dict[str, str]:
        return {
            "env": _block_digest("env fingerprint", self.env, (str,)),
            "graph": self.graph,
            "layout": self.layout,
            "policy": _block_digest("compile policy", self.policy, (bool, int, str)),
            "sm": self.sm,
        }

    @property
    def value(self) -> str:
        digest = hashlib.sha256(_canonical_json(self.axes())).hexdigest()[:_DIGEST_HEX]
        return f"{KEY_SCHEME}-{digest}"

    def describe(self) -> str:
        stack = ", ".join(
            f"{n} {self.env[n]}" for n in ("torch", "triton") if n in self.env
        )
        options = ", ".join(f"{n}={v!r}" for n, v in sorted(self.policy.items()))
        return f"{stack or 'env'} @ {self.sm} [{options}] in {self.layout}"

    def __str__(self) -> str:
        return self.value


def artifact_key(
    graph: str,
    *,
    sm: str,
    env: Mapping[str, str],
    policy: Mapping[str, bool | int | str],
    layout: str,
) -> ArtifactKey:
    return ArtifactKey(graph=graph, sm=sm, env=env, policy=policy, layout=layout)


__all__ = [
    "AXES",
    "ArtifactKey",
    "CORPUS_ENV",
    "CallIngress",
    "CallInput",
    "GRAPH_SCHEME",
    "KEY_SCHEME",
    "LayoutMorphism",
    "MAX_KEY_LENGTH",
    "STACK_NAMES",
    "STACK_PREFIXES",
    "artifact_key",
    "build_call_ingress",
    "canonical_graph",
    "catalog",
    "compile_stack",
    "constant_names",
    "contiguous_handle",
    "exported_input_name",
    "graph_hash",
    "is_artifact_key",
    "is_compile_relevant",
    "is_graph_hash",
    "literal_digest",
    "literal_names",
    "placement",
    "require_morphism",
    "symbol_terms",
]
