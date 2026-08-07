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

import logging
from threading import RLock
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import msgspec
import importlib

_logger = logging.getLogger(__name__)

#: A graph-class fork arm value. Bools are the common case (LTX's
#: ``isolate_modalities``); ints/strs cover arity-style forks (z-image's CFG
#: arity ``N``).
ForkValue = Union[bool, int, str]

#: One entry of an :class:`Input` shape template: a literal size, the NAME of
#: a declared :class:`Dim` (resolved from the class row), or ``("config",
#: field)`` — read off the resolved module's own config so a family member
#: with a different width exports correctly instead of failing in the trace.
AxisSpec = Union[int, str, Tuple[str, str]]

#: pgw#853/ie#591: WHY an unserved fork arm is closed. Five kinds, because a
#: fleet audit found `unserved` was recording five materially different
#: guarantees as one word — and only ONE of them means "unreachable".
#:
#: * ``absent_path``      — the code that reaches the arm is never imported or
#:                          instantiated (flux2's ``kv_cache``: KV caching
#:                          lives in a pipeline class the endpoint does not
#:                          import). The guarantee the word implies.
#: * ``unpassed_arg``     — reachable code; the endpoint simply never passes
#:                          the argument that arms it (ltx's
#:                          ``isolate_modalities``, ``stg``). ONE code change
#:                          away.
#: * ``default_value``    — fully plumbed end to end; only the VALUE keeps it
#:                          closed (ltx's ``cfg``: guidance flows all the way
#:                          to the pipeline and is 1.0). Needs no code change
#:                          at all — a recipe number does it.
#: * ``checkpoint_config`` — decided by the resolved checkpoint's own config,
#:                          not by this repo (wan's ``expand_timesteps``).
#: * ``eager_by_choice``  — reachable AND served; excluded from the COMPILED
#:                          set on purpose (qwen's edit lane). The opposite of
#:                          unreachable, and it used to be spelled the same.
FORK_REASONS: Tuple[str, ...] = (
    "absent_path",
    "unpassed_arg",
    "default_value",
    "checkpoint_config",
    "eager_by_choice",
)

#: The two kinds a single payload field or recipe number can flip.
WEAK_FORK_REASONS: Tuple[str, ...] = ("unpassed_arg", "default_value")

#: Max nesting depth of an ``Arg.template``. qwen-image's img_shapes is 3
#: (list -> list -> tuple); anything deeper is a boundary-shrink problem.
_TEMPLATE_MAX_DEPTH = 4

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


def _axis_spec(entry: Any, what: str) -> AxisSpec:
    """Validate and canonicalise one :data:`AxisSpec`.

    Lifted out of ``Input.__post_init__`` (pgw#853) so the SAME three-way
    rule — literal / declared-dim name / ``("config", field)`` — governs
    every place the vocabulary lets a number come from the class row:
    tensor axes, ``Input.repeat`` container arity, and the leaves of
    ``Arg.template``. One rule, one error sentence, one resolver
    (``aot_declaration._resolve_axis``) downstream.
    """
    if isinstance(entry, bool):
        raise DeclarationError(f"{what}: bool is not an axis spec")
    if isinstance(entry, int):
        if entry <= 0:
            raise DeclarationError(f"{what}: literal {entry} must be positive")
        return entry
    if isinstance(entry, str):
        return _identifier(what, entry)
    try:
        ref = tuple(entry)
    except TypeError:
        ref = ()
    if len(ref) != 2 or ref[0] != "config" or not str(ref[1]).strip():
        raise DeclarationError(
            f"{what}: {entry!r} is not an int, a dim name, or "
            f"(\"config\", field)")
    return ("config", str(ref[1]).strip())


def _canon_template(node: Any, what: str, _depth: int = 0) -> Any:
    """Canonicalise a nested :data:`AxisSpec` template to tuples.

    Lists and tuples nest; anything else is a leaf and must be an
    ``AxisSpec``. Tuples (not lists) so the structure stays hashable and a
    ``Compile`` remains comparable by value — which registration idempotence
    depends on.
    """
    if _depth > _TEMPLATE_MAX_DEPTH:
        raise DeclarationError(
            f"{what}: nested more than {_TEMPLATE_MAX_DEPTH} deep — a traced "
            f"call argument that deep is a boundary problem, not a "
            f"declaration one")
    if isinstance(node, (list, tuple)):
        if not node:
            raise DeclarationError(f"{what}: empty container in the template")
        return tuple(_canon_template(x, what, _depth + 1) for x in node)
    return _axis_spec(node, what)


def _template_rows(node: Any) -> Any:
    """A JSON-able mirror of a canonical template (lists, not tuples), used
    only for ``as_row``/mint-request serialisation."""
    if isinstance(node, tuple) and len(node) == 2 and node[0] == "config":
        return ["config", node[1]]
    if isinstance(node, tuple):
        return [_template_rows(x) for x in node]
    return node


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
    #: Prose, for humans: the evidence and its citations. KEPT alongside
    #: ``reason`` rather than replaced — the failure this pair exists to fix
    #: was prose doing a MACHINE's job, not prose existing.
    why: str = ""
    #: One of :data:`FORK_REASONS`: the KIND of guarantee that closes the
    #: unserved arm, for machines. ``None`` means the declaration has not
    #: answered the question yet.
    #:
    #: OPTIONAL BY DESIGN, FOR NOW. Every fleet declaration predates this
    #: field, so requiring it would break them all at import. **Phase 2 makes
    #: it required whenever ``unserved`` is non-empty** — an un-annotated
    #: unserved arm is a declaration that has not said how strong its own
    #: guarantee is, which is exactly the state this field exists to end.
    reason: Optional[str] = None

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
        if self.reason is not None:
            reason = str(self.reason).strip()
            if reason not in FORK_REASONS:
                raise DeclarationError(
                    f"Fork {self.name!r}: reason must be one of "
                    f"{list(FORK_REASONS)!r}, got {self.reason!r}")
            if not unserved and reason != "eager_by_choice":
                raise DeclarationError(
                    f"Fork {self.name!r} declares reason={reason!r} with no "
                    f"unserved arm — a reason explains what keeps an arm "
                    f"CLOSED, and this fork closes nothing")
            force(self, "reason", reason)

    def as_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "name": self.name,
            "served": list(self.served),
            "unserved": list(self.unserved),
            "source": list(self.source) if self.source else None,
            "targets": list(self.targets),
        }
        # pgw#846: OMITTED when absent, so every declaration written before
        # this field existed serialises byte-identically.
        if self.reason is not None:
            row["reason"] = self.reason
        return row

    @property
    def weak(self) -> bool:
        """Is this arm one payload field or one recipe number from opening?

        ``True`` for ``unpassed_arg`` and ``default_value``. ``None`` reads as
        FALSE deliberately — an unanswered question is not evidence of a weak
        guarantee, and phase 2 (making ``reason`` required) is what removes
        the ambiguity rather than a guess here.
        """
        return bool(self.unserved) and self.reason in WEAK_FORK_REASONS


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

    ``targets`` scopes the row, exactly as it does on :class:`Input`,
    :class:`Arg` and :class:`Fork`; empty means every target (pgw#967). A
    cell's targets are unrelated modules with unrelated call contracts —
    sdxl's UNet traces 9 aspect rows x 2 CFG arms, its VAE decoder traces
    the 9 aspect rows at batch 1 (CFG has collapsed by decode time), and its
    two text encoders trace ONE row each (77 tokens, batch 1, both
    aspect- and CFG-invariant). Without scoping, ``mint_plans`` hands every
    target the whole class table, so declaring a text encoder alongside the
    denoiser would mint 18 identical text-encoder graphs under 18 different
    entry names and pay for each. That is why the SDK's own
    ``("transformer", "vae.decode")`` default has never been exercised by a
    fleet family: it was not expressible, only payable.
    """

    dims: Any
    fork: Any = ()
    targets: Tuple[str, ...] = ()

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
        force(self, "targets", tuple(str(t).strip() for t in self.targets))

    @property
    def dim_map(self) -> Dict[str, int]:
        return dict(self.dims)

    @property
    def fork_map(self) -> Dict[str, ForkValue]:
        return dict(self.fork)

    def serves(self, target: str) -> bool:
        """Whether this row is one of ``target``'s coordinates."""
        return not self.targets or str(target) in self.targets

    def as_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"dims": dict(self.dims), "fork": dict(self.fork)}
        # pgw#846 discipline: an absent field is OMITTED, so every
        # declaration written before this vocabulary existed serialises
        # byte-identically and no published cell re-keys on the SDK change
        # alone (`Compile.contract_axes` feeds the cell key's contract
        # digest). A declaration that ADOPTS scoping re-keys, correctly: it
        # is a different class set.
        if self.targets:
            row["targets"] = list(self.targets)
        return row


class Input(msgspec.Struct, frozen=True):
    """One example-input template of the export call contract.

    ``name`` may be dotted for nested container arguments (sdxl's
    ``added_cond_kwargs.text_embeds`` — the container dict is built and fed
    as ONE argument). ``shape`` entries are :data:`AxisSpec`; a rank-0
    tensor is ``shape=()`` with ``value`` (sdxl's scalar timestep).
    ``dtype`` "" means the resolved module's own dtype (the
    resident-precision truth); anything else names an explicit torch dtype
    (wan's int64 scalar timestep, ti2v's float32 per-token one).

    There is NO args/kwargs choice to declare: all-positional example feeds
    are a MINT OBLIGATION (pod 9, pgw#723 residuals — an AOTI package's call
    convention mirrors the traced args/kwargs split, and the serve marshal
    is positional, so a kwarg-traced package silently revokes to eager on
    first call). The SDK binds each named row to its slot in the traced
    signature and feeds everything positionally; declaring a name the
    target takes only keyword-only is refused at mint time by name.
    ``targets`` scopes the row; empty = every target.
    """

    name: str
    shape: Tuple[AxisSpec, ...]
    dtype: str = ""
    value: Optional[float] = None
    targets: Tuple[str, ...] = ()
    #: pgw#853: the parameter is a LIST of this many tensors, each of
    #: ``shape`` — an :data:`AxisSpec`, so the arity may be a literal, a
    #: declared :class:`Dim` resolved from the class row, or ``("config",
    #: field)``. ``None`` (the default) means a single tensor: every
    #: declaration written before this field existed is unchanged, by
    #: construction. See :data:`REPEAT_RATIONALE`.
    repeat: Optional[AxisSpec] = None

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        force(self, "name", _identifier("Input", self.name))
        if self.repeat is not None:
            force(self, "repeat",
                  _axis_spec(self.repeat, f"Input {self.name!r} repeat"))
        force(self, "shape", tuple(
            _axis_spec(entry, f"Input {self.name!r} axis spec")
            for entry in tuple(self.shape)))
        force(self, "dtype", str(self.dtype or "").strip())
        force(self, "targets", tuple(str(t).strip() for t in self.targets))

    @property
    def top_name(self) -> str:
        return self.name.split(".", 1)[0]

    def as_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "name": self.name,
            "shape": [list(e) if isinstance(e, tuple) else e for e in self.shape],
            "dtype": self.dtype,
            "value": self.value,
            "targets": list(self.targets),
        }
        # pgw#846: an absent field is OMITTED, so every declaration written
        # before this vocabulary existed serialises byte-identically.
        if self.repeat is not None:
            row["repeat"] = (list(self.repeat)
                             if isinstance(self.repeat, tuple) else self.repeat)
        return row


class Arg(msgspec.Struct, frozen=True):
    """A non-tensor argument of the export call.

    Two forms, and the second is pgw#853's:

    * ``value=`` — a LITERAL, declaration-global (``return_dict=False`` — a
      dataclass output is not a valid export output; the consumer re-wraps).
    * ``template=`` — a **row-derived structure**. A nested list/tuple whose
      leaves are :data:`AxisSpec` (literal, declared :class:`Dim` name, or
      ``("config", field)``), resolved to python ints against the SELECTED
      CLASS ROW. ``repeat=`` then makes the value a LIST OF THAT MANY COPIES
      of the resolved template, so it reads as the ``x * n`` the call site
      itself writes: qwen's ``[[(1, H_pat, W_pat)]] * B`` is
      ``template=[(1, "H_pat", "W_pat")], repeat="B"``.

    ``template`` exists because a real family's traced signature takes one:
    qwen-image's ``img_shapes`` is ``[[(1, H_pat, W_pat)]] * B`` — literally
    the class row restated as python ints. ``value`` could not express it on
    two counts, both structural: its type is a scalar union, and an ``Arg``
    is declaration-GLOBAL while this value is PER-ROW. That gap is
    qwen-image's blocker B1, and it is the whole reason the family's
    declaration carries no ``args`` rows at all.

    Exactly one of ``value``/``template`` may be given. A specializing
    python scalar that does NOT vary by row stays a ``value``: the point of
    the vocabulary is that a declaration states which of the two a number is.
    """

    name: str
    value: Union[bool, int, float, str, None] = None
    targets: Tuple[str, ...] = ()
    #: Nested list/tuple of :data:`AxisSpec` leaves, resolved per class row.
    template: Optional[Any] = None
    #: Wrap the resolved ``template`` in a list of this arity (an
    #: :data:`AxisSpec`, so it too may come from the row).
    repeat: Optional[AxisSpec] = None

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        force(self, "name", _identifier("Arg", self.name))
        force(self, "targets", tuple(str(t).strip() for t in self.targets))
        if self.template is not None:
            if self.value is not None:
                raise DeclarationError(
                    f"Arg {self.name!r} declares both value= and template= — "
                    f"a literal and a row-derived structure are different "
                    f"claims about where the number comes from; state one")
            force(self, "template",
                  _canon_template(self.template, f"Arg {self.name!r} template"))
        elif self.repeat is not None:
            raise DeclarationError(
                f"Arg {self.name!r} declares repeat= without template= — "
                f"there is no structure to repeat")
        if self.repeat is not None:
            force(self, "repeat",
                  _axis_spec(self.repeat, f"Arg {self.name!r} repeat"))

    def as_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "name": self.name, "value": self.value,
            "targets": list(self.targets),
        }
        # pgw#846: absent fields are OMITTED, so every declaration written
        # before this vocabulary existed serialises byte-identically.
        if self.template is not None:
            row["template"] = _template_rows(self.template)
        if self.repeat is not None:
            row["repeat"] = (list(self.repeat)
                             if isinstance(self.repeat, tuple) else self.repeat)
        return row


def _carrier_names(
    inputs: Tuple[Input, ...], args: Tuple[Arg, ...], targets: Sequence[str],
) -> set:
    """Every name a dim may be carried by, for the given target scope."""
    names: set = set()
    for target in targets:
        for inp in inputs:
            if not inp.targets or target in inp.targets:
                names.add(inp.name)
                names.add(inp.top_name)
        for arg in args:
            if arg.template is not None and (not arg.targets or target in arg.targets):
                names.add(arg.name)
    return names


def dims_for_targets(
    dims: Tuple[Dim, ...],
    inputs: Tuple[Input, ...],
    args: Tuple[Arg, ...],
    targets: Sequence[str],
) -> set:
    """The dim NAMES a target-scoped graph class must state (pgw#967).

    A dim belongs to a target when one of its ``carried_by`` bindings names
    an :class:`Input` (or a templated :class:`Arg`) that target declares.
    That relation is already written down — deriving it beats adding a
    second place for an endpoint to say the same thing and drift.

    Unscoped rows, and any declaration carrying no input templates, keep the
    original rule: every row states every declared dim.
    """
    all_names = {d.name for d in dims}
    if not targets or not inputs:
        return all_names
    known = _carrier_names(inputs, args, targets)
    scoped: set = set()
    for d in dims:
        for bound, _axis in d.carried_by:
            if bound in known or bound.split(".", 1)[0] in known:
                scoped.add(d.name)
                break
    return scoped


def forks_for_targets(forks: Tuple[Fork, ...], targets: Sequence[str]) -> set:
    """The fork NAMES a target-scoped graph class must state (pgw#967)."""
    if not targets:
        return {f.name for f in forks}
    want = set(targets)
    return {f.name for f in forks if not f.targets or (set(f.targets) & want)}


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
    for i, cls in enumerate(classes):
        bad = sorted(set(cls.targets) - set(targets))
        if bad:
            raise DeclarationError(
                f"graph class #{i} names target(s) {bad!r} not in "
                f"Compile.targets {list(targets)!r}")

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
        # pgw#967: a row scoped to a target states that TARGET's dims and
        # forks. An unscoped row states all of them — the original rule,
        # unchanged for every declaration written before scoping existed.
        want_dims = dims_for_targets(dims, inputs, args, cls.targets)
        want_forks = forks_for_targets(forks, cls.targets)
        missing = sorted(want_dims - set(cls.dim_map))
        if missing:
            raise DeclarationError(
                f"graph class #{i} omits declared dim(s) {missing!r}")
        unknown = sorted(set(cls.dim_map) - set(dim_names))
        if unknown:
            raise DeclarationError(
                f"graph class #{i} carries undeclared dim(s) {unknown!r}")
        extra = sorted(set(cls.dim_map) - want_dims)
        if extra:
            raise DeclarationError(
                f"graph class #{i} is scoped to target(s) {list(cls.targets)!r} "
                f"and carries dim(s) {extra!r} that no input those targets "
                f"declare can carry — a coordinate the traced call cannot "
                f"receive is not a coordinate")
        # The omitted-fork refusal (#739 red test): a coordinate that varies
        # on a flag the vocabulary does not declare would be silently
        # exported into one class; a declared flag a coordinate does not
        # state is the same defect read the other way.
        missing_f = sorted(want_forks - set(cls.fork_map))
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
        extra_f = sorted(set(cls.fork_map) - want_forks - set(unknown_f))
        if extra_f:
            raise DeclarationError(
                f"graph class #{i} is scoped to target(s) {list(cls.targets)!r} "
                f"and carries fork(s) {extra_f!r} that Compile.forks scopes "
                f"to other targets — a fork the target does not take splits "
                f"its entries on nothing")
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
    # declared row knows the name at all.
    #
    # pgw#853: a TEMPLATED Arg is a legal carrier. qwen-image's H_pat/W_pat
    # genuinely enter the traced call at tuple positions 1 and 2 of
    # `img_shapes[b][0] = (frames, height, width)` — the endpoint's own
    # comment says the (name, index) pair is accurate and that "the
    # vocabulary means 'tensor axis' by it is exactly blocker B1". So the
    # vocabulary now means either, and `aot_declaration` keeps them apart
    # where it matters: an Arg-carried extent is a PYTHON INT, which
    # specializes the graph and can never become a torch symbol.
    if inputs:
        known = {i.name for i in inputs} | {i.top_name for i in inputs}
        templated = {a.name for a in args if a.template is not None}
        for d in dims:
            for bound, _axis in d.carried_by:
                head = bound.split(".", 1)[0]
                if bound in known or head in known:
                    continue
                if bound in templated or head in templated:
                    continue
                raise DeclarationError(
                    f"Dim {d.name!r} binds {bound!r}, which no declared "
                    f"Input row names and no declared Arg templates")


# ---------------------------------------------------------------------------
# The export-declaration registry (the #740 pattern: vocabulary here,
# registrations in the endpoint, derivation in gen_worker.aot_declaration).
# ---------------------------------------------------------------------------

_lock = RLock()
_declared: Dict[str, Any] = {}
#: (module, "ExcType: message") for every declaration import that FAILED.
#: pgw#996: the swallow is correct at boot and a lie at build time — a pod
#: cannot tell "this endpoint declares no AOT" (intake, legitimate) from "the
#: declaration module blew up" (broken image), because both present as
#: `export_declaration(family) is None`. The build gate reads this and refuses.
_import_failures: List[Tuple[str, str]] = []


class _Thunk:
    """A registered declaration that has not been built yet (pgw#853).

    A family whose declaration REFUSES TO MINT (open blockers, an unresolved
    config width, a shape it cannot legally export) used to express that by
    raising at module scope — which made registering it equivalent to taking
    the endpoint down at boot. A refusal to mint and a refusal to import are
    different events with different blast radii; conflating them is the bug.
    The thunk moves the evaluation to :func:`export_declaration`, i.e. to the
    moment the MINT asks, where ``fleet_cells.mint_recipe`` already turns a
    refusal into a typed ``self_mint_skipped`` carrying the blocker text.
    """

    __slots__ = ("family", "factory", "_built")

    def __init__(self, family: str, factory: Any) -> None:
        self.family = family
        self.factory = factory
        self._built: Any = None

    def build(self) -> Any:
        """Evaluate (once on success) and validate. Lets the factory's own
        exception out — the blocker text IS the diagnostic."""
        if self._built is not None:
            return self._built
        decl = self.factory()
        got = str(getattr(decl, "family", "") or "").strip()
        if got != self.family:
            raise DeclarationError(
                f"export declaration thunk registered for family "
                f"{self.family!r} built a Compile for {got!r}")
        if not getattr(decl, "classes", ()):
            raise DeclarationError(
                f"export declaration for {self.family!r} carries no graph "
                f"classes — there is nothing to derive from")
        self._built = decl
        return decl

    def _identity(self) -> Tuple[Any, ...]:
        """SOURCE identity, deliberately not object identity.

        A ``Compile`` re-registered from a second execution of the same file
        is VALUE-equal (msgspec Struct), so registration was idempotent. Two
        thunks are two distinct function objects, so a naive ``==`` made it
        NOT idempotent — and that regression was not academic: the endpoint
        walker (``discovery/walk.py``) imports EVERY submodule of an endpoint
        package, so a declaration module reachable under two module names
        registered twice, the second raised ``DeclarationError``, and the
        walk died with ``EndpointImportError``. I.e. the compile feature broke
        serving again, through a different door. Measured on qwen-image.

        Same function, same source file => same declaration, whatever module
        name it was loaded under. A genuinely different declaration still
        conflicts by name.
        """
        fn = self.factory
        code = getattr(fn, "__code__", None)
        if code is None:  # a callable object, not a function
            return (self.family, id(fn))
        return (self.family, getattr(fn, "__qualname__", ""),
                getattr(code, "co_filename", ""))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _Thunk) and other._identity() == self._identity()

    def __hash__(self) -> int:
        return hash(self._identity())


def register_export_declaration(
    compile_decl: Any, *, family: Optional[str] = None, replace: bool = False,
) -> Any:
    """Register one family's export declaration.

    ``compile_decl`` is either a ``Compile`` carrying classes, or a ZERO-ARG
    CALLABLE returning one (pgw#853), in which case ``family=`` is required
    because there is nothing to read the name off yet. Idempotent for an
    identical declaration; refuses a conflicting one by name.

    Prefer the callable form for any family that can REFUSE — the refusal
    then reaches the mint gate as a typed event instead of killing the import
    that registers it.
    """
    if callable(compile_decl) and not hasattr(compile_decl, "classes"):
        fam = str(family or "").strip()
        if not fam:
            raise DeclarationError(
                "a callable export declaration must be registered with an "
                "explicit family= (the Compile does not exist yet)")
        entry: Any = _Thunk(fam, compile_decl)
    else:
        fam = str(getattr(compile_decl, "family", "") or "").strip()
        if not fam:
            raise DeclarationError(
                "cannot register an export declaration with no Compile.family")
        if family is not None and str(family).strip() != fam:
            raise DeclarationError(
                f"family= {family!r} does not match Compile.family {fam!r}")
        if not getattr(compile_decl, "classes", ()):
            raise DeclarationError(
                f"export declaration for {fam!r} carries no graph classes — "
                f"there is nothing to derive from")
        entry = compile_decl
    with _lock:
        existing = _declared.get(fam)
        if existing is not None and existing != entry and not replace:
            raise DeclarationError(
                f"family {fam!r} already has a DIFFERENT export "
                f"declaration registered; pass replace=True only if you own both")
        _declared[fam] = entry
    return compile_decl


def export_declaration(family: str) -> Optional[Any]:
    """The family's ``Compile``, or ``None`` if none is registered.

    pgw#853: a THUNK registration is evaluated HERE, and its exception is let
    out HERE — this is "the mint asking", and every caller on the mint path
    turns that into a typed refusal that keeps the blocker text. Callers that
    only need to know whether a declaration EXISTS must use
    :func:`has_export_declaration`, which never evaluates anything.
    """
    with _lock:
        entry = _declared.get(str(family or "").strip())
    return entry.build() if isinstance(entry, _Thunk) else entry


def weak_arms(decl: Any) -> Tuple[Any, ...]:
    """Forks whose unserved arm is closed by a VALUE rather than by absent code.

    i.e. the arms that are ONE payload field or ONE recipe number away from
    serving a graph class the declaration says is never traced. Under AOT
    that is a hard refusal (a minted cell has no entry for it); under dynamo
    it is a guard miss that merely looks like an unexplained slowdown.

    This function is the point of :data:`FORK_REASONS`. The same question was
    answered once, by hand, across three repos and two vendored pipelines, and
    the answer was three forks — a day of work that nobody would repeat. It is
    now a call.
    """
    return tuple(f for f in getattr(decl, "forks", ()) if getattr(f, "weak", False))


def weak_arms_by_family() -> Dict[str, Tuple[Any, ...]]:
    """:func:`weak_arms` across every registered family, for a fleet sweep.

    Skips families whose declaration REFUSES (a blocked family is not a
    silent one — its refusal is the pgw#853 typed event, and it should not
    take a fleet query down).
    """
    out: Dict[str, Tuple[Any, ...]] = {}
    for family in registered_export_families():
        try:
            decl = export_declaration(family)
        except Exception:  # noqa: BLE001 — a blocked family is not a query error
            continue
        if decl is None:
            continue
        arms = weak_arms(decl)
        if arms:
            out[family] = arms
    return out


def registered_entry(family: str) -> Optional[Any]:
    """The RAW registry entry — a ``Compile`` or an unevaluated thunk.

    For callers that need registration IDENTITY (has this exact declaration
    already been registered?) without triggering a blocked family's refusal.
    """
    with _lock:
        return _declared.get(str(family or "").strip())


def has_export_declaration(family: str) -> bool:
    """Whether a declaration is registered — never evaluates a thunk, so a
    blocked family still reads as DECLARED (it is; it just refuses to mint)."""
    with _lock:
        return str(family or "").strip() in _declared


def registered_export_families() -> Tuple[str, ...]:
    with _lock:
        return tuple(sorted(_declared))


def reset_export_declarations() -> None:
    """Drop every registration. Tests only."""
    with _lock:
        _declared.clear()
        _import_failures.clear()


def declaration_import_failures() -> Tuple[Tuple[str, str], ...]:
    """Every declaration module whose import raised, as (module, exception).

    :func:`import_export_declaration` deliberately swallows — an endpoint must
    come up. The build gate (pgw#996) is where that swallow stops being
    correct: an image whose declaration module cannot import ships an endpoint
    that declares AOT and mints nothing, and on a pod that is indistinguishable
    from an endpoint that never declared one.
    """
    with _lock:
        return tuple(_import_failures)


def import_export_declaration(
    module: str, *, package: Optional[str] = None,
) -> bool:
    """Import a module whose only job is to register a declaration, and NEVER
    let its failure past this call (pgw#853, backstop (3)).

    ``register_export_declaration``'s thunk form fixes declarations that opt
    into it; any other module-scope work in a declaration file can still
    throw. A compile feature must never be able to break serving, so this
    reports the failure as a typed ``aot_declaration_import_failed`` activity
    event carrying the exception and returns ``False`` — the pod serves
    eager, uncompiled, and says why.

    Returns ``True`` when the module imported.
    """

    try:
        importlib.import_module(module, package=package)
        return True
    except BaseException as exc:  # noqa: BLE001 — serving outranks compiling
        with _lock:
            _import_failures.append(
                (str(module), f"{type(exc).__name__}: {exc}"))
        detail = (
            f"export declaration module {module!r} raised at import "
            f"({type(exc).__name__}: {exc}) — this endpoint serves EAGER and "
            f"mints nothing for its family; the AOT lane is off until the "
            f"import is fixed. Registration must never be able to take an "
            f"endpoint down (pgw#853)"
        )
        try:
            from ..activity import emit_event

            emit_event("aot_declaration_import_failed", detail,
                       phase="declaration_import")
        except BaseException:  # noqa: BLE001 — reporting must not raise either
            pass
        _logger.warning("%s", detail, exc_info=True)
        return False


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
    "FORK_REASONS",
    "WEAK_FORK_REASONS",
    "declaration_import_failures",
    "export_declaration",
    "has_export_declaration",
    "registered_entry",
    "import_export_declaration",
    "register_export_declaration",
    "registered_export_families",
    "reset_export_declarations",
    "validate_contract",
    "weak_arms",
    "weak_arms_by_family",
]
