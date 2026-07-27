"""The declaration vocabulary for the AOT export contract (pgw#739).

Paul's SDK-generic rule, applied to the export-input contract: per-family
content is a DECLARATION in the endpoint spec, never worker code. This module
defines the *vocabulary*; the endpoint writes the declaration;
:mod:`gen_worker.aot_declaration` derives export inputs, dynamic-shape marks
and mint plans from it. No family name appears in the SDK.

The vocabulary, per the #739 ratification:

- **NO formula DSL.** Endpoints emit RESOLVED ROWS (:class:`GraphClass`);
  the vocabulary is named dims with ``(input, axis)`` bindings
  (:class:`Dim`). Rows are coordinates, not expressions — the endpoint's own
  legality oracle resolves them (LTX-AOT-DESIGN.md §2.3), and the SDK derives
  every bound FROM the rows. A relational axis (wan ti2v's per-token
  timestep) is a :class:`Dim` with ``relates_to``: the SDK derives its free
  range from the rows and ``torch.export``'s solver unifies the relation into
  the shape env (measured: ``31*s25*s56`` carried with 458 asserts, ie#566
  §5) — endpoint hand-math is exactly what this prevents.
- **Graph-class forks** (:class:`Fork`) bind to ``(source: pipeline|module,
  field)``, not to ``module.config``: wan's ``expand_timesteps`` is a
  ``WanPipeline`` field in ``model_index.json`` and is ABSENT from all four
  transformer configs, so a builder that only reads module configs never
  sees it (ie#566 G6).
- **Shape strategy is a per-family DECLARED choice** (#730 ratification):
  conv-bearing families declare ``static-rows`` (symbolic latent H/W turns
  off inductor's channels-last layout opt — +7.2% measured on sdxl), DiTs
  declare ``dynamic-collapse``.
- **Mint-warm canon is a per-family DECLARED fact** (``warm_changes_key``):
  sdxl measured False, z-image measured True (the rope pre-warm changes the
  graph, 4327 cold vs 4285 warmed). A family with classes but no declared
  canon is refused at mint time, not defaulted.
"""

from __future__ import annotations

from threading import RLock
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import msgspec

#: A graph-class fork arm value. Bools are the common case (LTX's
#: ``isolate_modalities``); ints/strs cover arity-style forks (z-image's CFG
#: arity ``N``).
ForkValue = Union[bool, int, str]

#: One entry of an :class:`Input` shape template: a literal size, the NAME of
#: a declared :class:`Dim` (resolved from the class row), or ``("config",
#: field)`` — read off the resolved module's own config so a family member
#: with a different width exports correctly instead of failing in the trace.
AxisSpec = Union[int, str, Tuple[str, str]]

STATIC_ROWS = "static-rows"
DYNAMIC_COLLAPSE = "dynamic-collapse"
SHAPE_STRATEGIES = (STATIC_ROWS, DYNAMIC_COLLAPSE)

_FORK_SOURCES = ("pipeline", "module")


class DeclarationError(ValueError):
    """An export declaration is malformed. Raised at declaration time (module
    import / ``Compile`` construction), never at mint time."""


def _identifier(kind: str, name: Any) -> str:
    text = str(name or "").strip()
    if not text or not text.replace(".", "_").isidentifier():
        raise DeclarationError(f"{kind} name {name!r} is not a valid identifier")
    return text


class Dim(msgspec.Struct, frozen=True):
    """A named axis of the export contract with its ingress bindings.

    ``carried_by`` is the ``(input, axis)`` binding set: every place this one
    logical extent enters the traced call. This is what replaces the
    two-literal ``DynamicDim.dim`` validation — wan's latent-spatial axis and
    LTX's ``T_v`` are nameable — and it doubles as the ``Compile`` ->
    ``aot_mint.DynamicDim`` bridge: the mint derives one
    ``aot_mint.DynamicDim(input, axis, min, max, multiple_of)`` row per
    binding.

    A :class:`Dim` deliberately carries NO ``min``/``max``: bounds derive
    from the declared :class:`GraphClass` rows (rows are coordinates), so
    "these are the shapes we serve" is stated once and the admissible range
    cannot drift from it. ``relates_to`` names the dims this axis is a
    function of (wan ti2v: ``N_tok`` relates to ``F_lat``/``H_lat``/
    ``W_lat``); the relation itself is NOT expressed — torch's solver
    unifies it from the free derived range (ie#566 §5 remedy (a) part 2).
    """

    name: str
    carried_by: Tuple[Tuple[str, int], ...]
    multiple_of: int = 1
    relates_to: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        force(self, "name", _identifier("Dim", self.name))
        raw = tuple(self.carried_by)
        if not raw:
            raise DeclarationError(
                f"Dim {self.name!r} declares no (input, axis) bindings — an "
                f"unbound dim can never reach the traced call")
        bindings: list[Tuple[str, int]] = []
        for entry in raw:
            if len(tuple(entry)) != 2:
                raise DeclarationError(
                    f"Dim {self.name!r}: binding {entry!r} is not (input, axis)")
            inp, axis = entry
            inp = str(inp or "").strip()
            if not inp:
                raise DeclarationError(
                    f"Dim {self.name!r}: binding names an empty input")
            axis = int(axis)
            if axis < 0:
                raise DeclarationError(
                    f"Dim {self.name!r}: binding {inp!r} has negative axis {axis}")
            if (inp, axis) in bindings:
                raise DeclarationError(
                    f"Dim {self.name!r} repeats binding ({inp!r}, {axis})")
            bindings.append((inp, axis))
        force(self, "carried_by", tuple(bindings))
        mult = int(self.multiple_of)
        if mult < 1:
            raise DeclarationError(
                f"Dim {self.name!r}: multiple_of must be >= 1, got {mult}")
        force(self, "multiple_of", mult)
        rel = tuple(_identifier("Dim.relates_to", n) for n in self.relates_to)
        if self.name in rel:
            raise DeclarationError(f"Dim {self.name!r} relates_to itself")
        if len(set(rel)) != len(rel):
            raise DeclarationError(f"Dim {self.name!r} repeats a relates_to name")
        force(self, "relates_to", rel)

    def as_row(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "carried_by": [list(b) for b in self.carried_by],
            "multiple_of": self.multiple_of,
            "relates_to": list(self.relates_to),
        }


class Fork(msgspec.Struct, frozen=True):
    """A declared flag that FORKS the graph class rather than varying within
    it (LTX's ``isolate_modalities``: 20,509 vs 30,877 nodes behind one
    declaration).

    ``served``/``unserved`` partition the arms: every declared
    :class:`GraphClass` must sit on a served arm, and declaring a class on an
    unserved arm is refused by name. ``source`` is the ``(pipeline|module,
    field)`` binding the mint asserts against the composed pipeline — a fork
    without a source (LTX's ``cfg``/``stg`` are call-argument facts) is
    keyed but cannot be asserted from composition. ``targets`` scopes the
    fork to a subset of ``Compile.targets`` (wan's ``use_tiling`` lives on
    the VAE, not the denoiser); empty means every target.
    """

    name: str
    served: Tuple[ForkValue, ...]
    unserved: Tuple[ForkValue, ...] = ()
    source: Optional[Tuple[str, str]] = None
    targets: Tuple[str, ...] = ()
    why: str = ""

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        force(self, "name", _identifier("Fork", self.name))
        served = tuple(self.served)
        if not served:
            raise DeclarationError(
                f"Fork {self.name!r} declares no served arm — a fork nothing "
                f"serves is not a graph class, it is dead vocabulary")
        unserved = tuple(self.unserved)
        overlap = [v for v in served if v in unserved]
        if overlap:
            raise DeclarationError(
                f"Fork {self.name!r}: arm(s) {overlap!r} declared BOTH served "
                f"and unserved")
        force(self, "served", served)
        force(self, "unserved", unserved)
        if self.source is not None:
            src = tuple(self.source)
            if len(src) != 2 or src[0] not in _FORK_SOURCES or not str(src[1]).strip():
                raise DeclarationError(
                    f"Fork {self.name!r}: source must be "
                    f"(\"pipeline\"|\"module\", field), got {self.source!r}")
            force(self, "source", (str(src[0]), str(src[1]).strip()))
        force(self, "targets", tuple(str(t).strip() for t in self.targets))
        force(self, "why", str(self.why or ""))

    def as_row(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "served": list(self.served),
            "unserved": list(self.unserved),
            "source": list(self.source) if self.source else None,
            "targets": list(self.targets),
        }


def _sorted_pairs(kind: str, owner: str, value: Any) -> Tuple[Tuple[str, Any], ...]:
    if isinstance(value, Mapping):
        items = list(value.items())
    else:
        items = [tuple(entry) for entry in value]
    out: list[Tuple[str, Any]] = []
    for entry in items:
        if len(entry) != 2:
            raise DeclarationError(f"{owner}: {kind} entry {entry!r} is not (name, value)")
        out.append((str(entry[0]).strip(), entry[1]))
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise DeclarationError(f"{owner}: {kind} repeats a name")
    return tuple(sorted(out))


class GraphClass(msgspec.Struct, frozen=True):
    """One RESOLVED coordinate a legal request can trace: dim values plus
    fork arm values. Generated by the endpoint's own legality oracle —
    pgw#669: the payload is the legality oracle for the sparse legal set —
    so the relation behind a relational axis stays in the one place that can
    be right about it, and the declaration carries only its resolved values.

    Accepts mappings at construction (``GraphClass(fork={...}, dims={...})``,
    the LTX §2.3 draft form) and normalizes to sorted pairs, so instances
    are hashable and endpoint generators can dedupe with ``dict.fromkeys``.
    """

    dims: Any
    fork: Any = ()

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        dims = _sorted_pairs("dims", "GraphClass", self.dims)
        if not dims:
            raise DeclarationError("GraphClass declares no dim values")
        for name, value in dims:
            if int(value) <= 0:
                raise DeclarationError(
                    f"GraphClass dim {name!r} has non-positive value {value!r}")
        force(self, "dims", tuple((n, int(v)) for n, v in dims))
        force(self, "fork", _sorted_pairs("fork", "GraphClass", self.fork))

    @property
    def dim_map(self) -> Dict[str, int]:
        return dict(self.dims)

    @property
    def fork_map(self) -> Dict[str, ForkValue]:
        return dict(self.fork)

    def as_row(self) -> Dict[str, Any]:
        return {"dims": dict(self.dims), "fork": dict(self.fork)}


class Input(msgspec.Struct, frozen=True):
    """One example-input template of the export call contract.

    ``name`` may be dotted for nested container kwargs (sdxl's
    ``added_cond_kwargs.text_embeds``). ``shape`` entries are
    :data:`AxisSpec`; a rank-0 tensor is ``shape=()`` with ``value``
    (sdxl's scalar timestep). ``dtype`` "" means the resolved module's own
    dtype (the resident-precision truth); anything else names an explicit
    torch dtype (wan's int64 scalar timestep, ti2v's float32 per-token one).
    ``positional`` inputs are emitted as args in declaration order — call
    convention is part of the contract, because the serve side replays the
    recorded pytree spec. ``targets`` scopes the row; empty = every target.
    """

    name: str
    shape: Tuple[AxisSpec, ...]
    dtype: str = ""
    value: Optional[float] = None
    positional: bool = False
    targets: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        force(self, "name", _identifier("Input", self.name))
        shape: list[AxisSpec] = []
        for entry in tuple(self.shape):
            if isinstance(entry, bool):
                raise DeclarationError(
                    f"Input {self.name!r}: bool is not an axis spec")
            if isinstance(entry, int):
                if entry <= 0:
                    raise DeclarationError(
                        f"Input {self.name!r}: literal axis {entry} must be positive")
                shape.append(entry)
            elif isinstance(entry, str):
                shape.append(_identifier(f"Input {self.name!r} axis", entry))
            else:
                ref = tuple(entry)
                if len(ref) != 2 or ref[0] != "config" or not str(ref[1]).strip():
                    raise DeclarationError(
                        f"Input {self.name!r}: axis spec {entry!r} is not an "
                        f"int, a dim name, or (\"config\", field)")
                shape.append(("config", str(ref[1]).strip()))
        force(self, "shape", tuple(shape))
        force(self, "dtype", str(self.dtype or "").strip())
        if self.positional and "." in self.name:
            raise DeclarationError(
                f"Input {self.name!r} is dotted (nested) and cannot be positional")
        force(self, "targets", tuple(str(t).strip() for t in self.targets))

    @property
    def top_name(self) -> str:
        return self.name.split(".", 1)[0]

    def as_row(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": [list(e) if isinstance(e, tuple) else e for e in self.shape],
            "dtype": self.dtype,
            "value": self.value,
            "positional": self.positional,
            "targets": list(self.targets),
        }


class Arg(msgspec.Struct, frozen=True):
    """A non-tensor literal argument of the export call (``return_dict=False``
    — a dataclass output is not a valid export output; the consumer
    re-wraps)."""

    name: str
    value: Union[bool, int, float, str, None]
    targets: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        force(self, "name", _identifier("Arg", self.name))
        force(self, "targets", tuple(str(t).strip() for t in self.targets))

    def as_row(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "targets": list(self.targets)}


def validate_contract(compile_decl: Any) -> None:
    """Cross-validate the #739 declaration fields on a ``Compile``.

    Called from ``Compile.__post_init__`` — a malformed declaration fails at
    endpoint import, never on a mint pod. Every refusal names the offending
    thing.
    """
    dims: Tuple[Dim, ...] = compile_decl.dims
    forks: Tuple[Fork, ...] = compile_decl.forks
    classes: Tuple[GraphClass, ...] = compile_decl.classes
    inputs: Tuple[Input, ...] = compile_decl.inputs
    args: Tuple[Arg, ...] = compile_decl.args
    targets = tuple(compile_decl.targets)

    dim_names = [d.name for d in dims]
    if len(set(dim_names)) != len(dim_names):
        raise DeclarationError("Compile.dims repeats a dim name")
    fork_names = [f.name for f in forks]
    if len(set(fork_names)) != len(fork_names):
        raise DeclarationError("Compile.forks repeats a fork name")
    shared = set(dim_names) & set(fork_names)
    if shared:
        raise DeclarationError(
            f"name(s) {sorted(shared)!r} declared as BOTH a dim and a fork")

    for d in dims:
        unknown = sorted(set(d.relates_to) - set(dim_names))
        if unknown:
            raise DeclarationError(
                f"Dim {d.name!r} relates_to undeclared dim(s) {unknown!r}")

    for f in forks:
        bad = sorted(set(f.targets) - set(targets))
        if bad:
            raise DeclarationError(
                f"Fork {f.name!r} names target(s) {bad!r} not in "
                f"Compile.targets {list(targets)!r}")
    scoped: Tuple[Union[Input, Arg], ...] = (*inputs, *args)
    for row in scoped:
        bad = sorted(set(row.targets) - set(targets))
        if bad:
            raise DeclarationError(
                f"{type(row).__name__} {row.name!r} names target(s) {bad!r} "
                f"not in Compile.targets {list(targets)!r}")

    if classes and not dims:
        raise DeclarationError(
            "Compile.classes declared without Compile.dims — rows are "
            "coordinates over named dims")

    served_by_fork = {f.name: set(f.served) for f in forks}
    unserved_by_fork = {f.name: set(f.unserved) for f in forks}
    seen: set = set()
    for i, cls in enumerate(classes):
        if cls in seen:
            raise DeclarationError(f"Compile.classes repeats row #{i}: {cls.as_row()!r}")
        seen.add(cls)
        missing = sorted(set(dim_names) - set(cls.dim_map))
        if missing:
            raise DeclarationError(
                f"graph class #{i} omits declared dim(s) {missing!r}")
        unknown = sorted(set(cls.dim_map) - set(dim_names))
        if unknown:
            raise DeclarationError(
                f"graph class #{i} carries undeclared dim(s) {unknown!r}")
        # The omitted-fork refusal (#739 red test): a coordinate that varies
        # on a flag the vocabulary does not declare would be silently
        # exported into one class; a declared flag a coordinate does not
        # state is the same defect read the other way.
        missing_f = sorted(set(fork_names) - set(cls.fork_map))
        if missing_f:
            raise DeclarationError(
                f"graph class #{i} omits declared fork(s) {missing_f!r} — "
                f"every class states every fork arm, or two graph classes "
                f"hide behind one declaration")
        unknown_f = sorted(set(cls.fork_map) - set(fork_names))
        if unknown_f:
            raise DeclarationError(
                f"graph class #{i} carries fork(s) {unknown_f!r} that "
                f"Compile.forks does not declare — declare the forking flag "
                f"by name rather than silently exporting into one class")
        for name, value in cls.fork:
            if value in unserved_by_fork.get(name, set()):
                raise DeclarationError(
                    f"graph class #{i} sits on UNSERVED arm {name}={value!r}")
            if value not in served_by_fork.get(name, set()):
                raise DeclarationError(
                    f"graph class #{i}: fork {name}={value!r} is not a "
                    f"declared arm (served: {sorted(map(repr, served_by_fork.get(name, set())))})")

    strategy = str(compile_decl.shape_strategy or "")
    if strategy and strategy not in SHAPE_STRATEGIES:
        raise DeclarationError(
            f"Compile.shape_strategy must be one of {SHAPE_STRATEGIES!r} "
            f"(#730: conv-bearing families declare {STATIC_ROWS!r}, DiTs "
            f"{DYNAMIC_COLLAPSE!r}), got {strategy!r}")

    # Rows are coordinates: with classes declared, a hand-written range on a
    # named dim is exactly the endpoint hand-math #739 exists to prevent.
    for dd in compile_decl.dynamic:
        if dd.dim in ("batch", "sequence"):
            continue
        if dd.dim not in dim_names:
            raise DeclarationError(
                f"Compile.dynamic names {dd.dim!r}, which is neither "
                f"\"batch\"/\"sequence\" nor a declared Dim "
                f"(declared: {dim_names!r})")
        if classes:
            raise DeclarationError(
                f"Compile.dynamic hand-ranges dim {dd.dim!r} while classes "
                f"are declared — bounds derive from the class rows "
                f"(rows are coordinates, #739); delete the hand range")

    # Input rows must be unambiguous per target.
    for target in targets or ("",):
        names: set[str] = set()
        for inp in inputs:
            if inp.targets and target not in inp.targets:
                continue
            if inp.name in names:
                raise DeclarationError(
                    f"Input {inp.name!r} is declared twice for target {target!r}")
            names.add(inp.name)

    # A binding may belong to any target's inputs; refuse only when NO
    # declared Input row knows the name at all.
    if inputs:
        known = {i.name for i in inputs} | {i.top_name for i in inputs}
        for d in dims:
            for bound, _axis in d.carried_by:
                if bound not in known and bound.split(".", 1)[0] not in known:
                    raise DeclarationError(
                        f"Dim {d.name!r} binds input {bound!r}, which no "
                        f"declared Input row names")


# ---------------------------------------------------------------------------
# The export-declaration registry (the #740 pattern: vocabulary here,
# registrations in the endpoint, derivation in gen_worker.aot_declaration).
# ---------------------------------------------------------------------------

_lock = RLock()
_declared: Dict[str, Any] = {}


def register_export_declaration(compile_decl: Any, *, replace: bool = False) -> Any:
    """Register one family's export declaration (a ``Compile`` carrying
    classes). Idempotent for an identical declaration; refuses a conflicting
    one by name."""
    family = str(getattr(compile_decl, "family", "") or "").strip()
    if not family:
        raise DeclarationError(
            "cannot register an export declaration with no Compile.family")
    if not getattr(compile_decl, "classes", ()):
        raise DeclarationError(
            f"export declaration for {family!r} carries no graph classes — "
            f"there is nothing to derive from")
    with _lock:
        existing = _declared.get(family)
        if existing is not None and existing != compile_decl and not replace:
            raise DeclarationError(
                f"family {family!r} already has a DIFFERENT export "
                f"declaration registered; pass replace=True only if you own both")
        _declared[family] = compile_decl
    return compile_decl


def export_declaration(family: str) -> Optional[Any]:
    with _lock:
        return _declared.get(str(family or "").strip())


def registered_export_families() -> Tuple[str, ...]:
    with _lock:
        return tuple(sorted(_declared))


def reset_export_declarations() -> None:
    """Drop every registration. Tests only."""
    with _lock:
        _declared.clear()


__all__ = [
    "Arg",
    "AxisSpec",
    "DYNAMIC_COLLAPSE",
    "DeclarationError",
    "Dim",
    "Fork",
    "ForkValue",
    "GraphClass",
    "Input",
    "SHAPE_STRATEGIES",
    "STATIC_ROWS",
    "export_declaration",
    "register_export_declaration",
    "registered_export_families",
    "reset_export_declarations",
    "validate_contract",
]
