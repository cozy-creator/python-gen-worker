"""The CONSTRUCTION CENSUS: what a module IS, as data (pgw#1647).

Four times on rented hardware the same sentence was paid for — *the meta
skeleton is built from the config alone, so the ``from_pretrained`` machinery
never runs*. pgw#1626 lost ``tie_weights()``; pgw#1638 lost
``HfQuantizer.preprocess_model`` and then ``model.eval()``; pgw#1644 lost the
whole-module ``.to(device)`` and cost $0.89 eight milliseconds into a forward.
Each was fixed as a symptom, and each fix left the next member undetectable,
because the only fence that could have decided any of them offline —
``engine.py``'s "nothing survives on meta" — walked the CHECKPOINT CONTAINER.
A container names the tensors a checkpoint carries. It cannot name a tensor the
CODE creates: a tie, a quantizer's scale grid, a RoPE ``inv_freq``. Those are
exactly the four defects.

So the module's tensor identity stops being re-derived at every moment and
becomes DATA. The census is the complete answer to "what does this component
build" — every parameter and every buffer, persistent and non-persistent, with
shape and dtype; the tied alias groups by object identity; the module classes
the config's quantizer swapped in and the tensors that swap owns; and whether
the thing came up in eval mode. It is computed at RELEASE BUILD from the ONE
prepare seam (:mod:`.skeleton`), rides the release document, and is REPLAYED by
the serve-time fence rather than re-derived there.

**Five invariants, and I5 is why this closes the class.** I1 ties, I2 quantizer,
I3 serve mode, I4 placement, I5 totality. Totality is set equality in BOTH
directions: a name the module has that the census does not is as much a refusal
as one the census has and the module lacks. That is the arm no per-symptom fix
could ever have: a FIFTH ``from_pretrained`` side effect nobody has thought of
yet shows up as an unknown name and becomes a $0 publish refusal instead of the
next rental.

The same predicate runs at three moments and there is only one copy of it:
release build (refuse the release), the CPU-only conformance suite (catch
image-bump drift between releases), and serve after the fill (catch store
corruption and fill defects — the jurisdiction that cannot be delegated
upstream, because only serve has seen the bytes).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING, Any, Dict, FrozenSet, Iterable, List, Mapping, Optional,
    Sequence, Tuple,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from .skeleton import Quantization

#: The census document's wire kind. Versioned because the hub stores it
#: verbatim and forwards it to workers (th#2281) without interpreting a word of
#: torch semantics.
CENSUS_KIND = "gen-worker.construction-census@1"

I1_TIES = "I1_TIES"
I2_QUANTIZER = "I2_QUANTIZER"
I3_SERVE_MODE = "I3_SERVE_MODE"
I4_PLACEMENT = "I4_PLACEMENT"
I5_TOTALITY = "I5_TOTALITY"

PARAMETER = "parameter"
BUFFER = "buffer"


class CensusError(RuntimeError):
    """A census could not be computed or read."""


class CensusMismatch(CensusError):
    """A module does not match the census that says what it must be.

    Carries the INVARIANT and the first offending tensor as fields, not only
    inside a sentence: the serve-side refusal is recorded against the RELEASE
    (th#2281) and the recorder must not have to parse prose to know which of the
    five walls it hit.
    """

    def __init__(
        self,
        invariant: str,
        component: str,
        tensor: str,
        message: str,
        *,
        where: str = "",
    ) -> None:
        self.invariant = invariant
        self.component = component
        self.tensor = tensor
        self.where = where
        head = f"{invariant} ({where})" if where else invariant
        super().__init__(
            f"{head}: component {component!r}"
            + (f", tensor {tensor!r}" if tensor else "")
            + f" — {message}"
        )


@dataclass(frozen=True, slots=True)
class TensorRow:
    """One tensor a component's construction creates.

    ``persistent`` is False for exactly the class pgw#1644 died on: a buffer
    ``__init__`` computes from config and ``state_dict`` does not carry, so no
    container ever names it and no container-walking check can see it.

    ``rule_owned`` marks a tensor a quantizer's swapped module owns. Its dtype
    is the RULE's, not the lane's, and ``finish_quantized``'s
    ``postprocess_model`` may legitimately rewrite it after the fill (a
    ``scale_fmt="ue8m0"`` tree turns F32 scale grids into the exponent dtype the
    kernels read). So the dtype of a rule-owned row is RECORDED and not
    asserted; its name and shape are asserted like every other row.
    """

    name: str
    kind: str
    shape: Tuple[int, ...]
    dtype: str
    persistent: bool = True
    rule_owned: bool = False

    def as_document(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }
        if not self.persistent:
            row["persistent"] = False
        if self.rule_owned:
            row["rule_owned"] = True
        return row

    @classmethod
    def from_document(cls, row: Mapping[str, Any]) -> "TensorRow":
        try:
            return cls(
                name=str(row["name"]),
                kind=str(row["kind"]),
                shape=tuple(int(dim) for dim in row["shape"]),
                dtype=str(row["dtype"]),
                persistent=bool(row.get("persistent", True)),
                rule_owned=bool(row.get("rule_owned", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CensusError(f"census tensor row {row!r} is unreadable: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ComponentCensus:
    """What ONE component of a checkpoint tree builds."""

    component: str
    module_class: str
    tensors: Tuple[TensorRow, ...]
    #: Alias groups by object identity, source name first, each group sorted
    #: after its source. A checkpoint OMITS every alias, so these are precisely
    #: the names the store cannot fill (pgw#1626).
    ties: Tuple[Tuple[str, ...], ...] = ()
    #: The cozy quant rule the component's config declared, "" for none.
    quant_rule: str = ""
    #: ``submodule prefix -> class name`` for every module the quantizer swapped.
    quant_modules: Tuple[Tuple[str, str], ...] = ()
    #: True when EVERY submodule is in eval mode (pgw#1638's third member).
    eval_mode: bool = True

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(row.name for row in self.tensors)

    def by_name(self) -> Dict[str, TensorRow]:
        return {row.name: row for row in self.tensors}

    def as_document(self) -> Dict[str, Any]:
        document: Dict[str, Any] = {
            "component": self.component,
            "class": self.module_class,
            "eval_mode": self.eval_mode,
            "tensors": [row.as_document() for row in self.tensors],
        }
        if self.ties:
            document["ties"] = [list(group) for group in self.ties]
        if self.quant_rule:
            document["quant_rule"] = self.quant_rule
            document["quant_modules"] = [
                [prefix, name] for prefix, name in self.quant_modules
            ]
        return document

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ComponentCensus":
        try:
            return cls(
                component=str(document["component"]),
                module_class=str(document["class"]),
                tensors=tuple(
                    TensorRow.from_document(row) for row in document["tensors"]
                ),
                ties=tuple(
                    tuple(str(name) for name in group)
                    for group in document.get("ties", ())
                ),
                quant_rule=str(document.get("quant_rule", "")),
                quant_modules=tuple(
                    (str(pair[0]), str(pair[1]))
                    for pair in document.get("quant_modules", ())
                ),
                eval_mode=bool(document["eval_mode"]),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise CensusError(
                f"census component document is unreadable: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class Census:
    """Every weight-bearing component of one tree, under one lane."""

    components: Tuple[ComponentCensus, ...]

    def by_component(self) -> Dict[str, ComponentCensus]:
        return {row.component: row for row in self.components}

    def as_document(self) -> Dict[str, Any]:
        return {
            "kind": CENSUS_KIND,
            "components": [row.as_document() for row in self.components],
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "Census":
        kind = str(document.get("kind", ""))
        if kind != CENSUS_KIND:
            raise CensusError(
                f"census document declares kind {kind!r}, not {CENSUS_KIND!r}; "
                f"this worker cannot read it and will not guess at a schema it "
                f"does not know"
            )
        rows = document.get("components")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise CensusError(
                f"census document carries no component list ({rows!r})"
            )
        return cls(
            components=tuple(
                ComponentCensus.from_document(row) for row in rows
            )
        )

    def canonical(self) -> bytes:
        return json.dumps(
            self.as_document(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical()).hexdigest()

    @property
    def tensor_count(self) -> int:
        return sum(len(row.tensors) for row in self.components)


# ── taking a census ─────────────────────────────────────────────────────────


def _non_persistent(module: "torch.nn.Module") -> FrozenSet[str]:
    """Qualified names of every NON-PERSISTENT buffer under ``module``.

    Read from ``_non_persistent_buffers_set`` rather than by diffing
    ``state_dict()``: a state-dict diff would also report a buffer a subclass
    removed in ``state_dict``, and would materialize a dict of every tensor in a
    66 GiB module to answer a question about a handful of names.
    """
    found: List[str] = []
    for prefix, sub in module.named_modules():
        head = f"{prefix}." if prefix else ""
        for leaf in getattr(sub, "_non_persistent_buffers_set", ()) or ():
            found.append(head + str(leaf))
    return frozenset(found)


def _dtype_name(dtype: Any) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _tie_groups(module: "torch.nn.Module") -> Tuple[Tuple[str, ...], ...]:
    """Alias groups by OBJECT IDENTITY over the module as it stands.

    Read, never repair: taking a census must not mutate the thing it measures,
    or the fence would fix the defect it is supposed to name. The prepare seam
    reties; this records what the retie produced.

    Identity, never ``_tied_weights_keys``. That attribute lists names a class
    MIGHT tie, and whether the tie is live is a config question
    (``tie_word_embeddings``) — reading the attribute as an answer scores a
    class that declares ``lm_head.weight`` and does not tie it as a defect on a
    checkpoint that carries the tensor perfectly well (pgw#1633's finding).
    """
    first: Dict[int, str] = {}
    groups: Dict[str, List[str]] = {}
    for name, tensor in module.named_parameters(remove_duplicate=False):
        source = first.get(id(tensor))
        if source is None:
            first[id(tensor)] = name
            continue
        groups.setdefault(source, []).append(name)
    return tuple(
        (source, *sorted(aliases))
        for source, aliases in sorted(groups.items())
    )


def take_component(
    component: str,
    module: "torch.nn.Module",
    quantization: Optional["Quantization"] = None,
) -> ComponentCensus:
    """The census of one built component. Reads; allocates nothing; mutates nothing."""
    rule_owned = frozenset(quantization.tensors) if quantization is not None else frozenset()
    volatile = _non_persistent(module)
    rows: List[TensorRow] = []
    for name, tensor in module.named_parameters(remove_duplicate=False):
        rows.append(
            TensorRow(
                name=name, kind=PARAMETER,
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=_dtype_name(tensor.dtype),
                rule_owned=name in rule_owned,
            )
        )
    for name, buffer in module.named_buffers(remove_duplicate=False):
        rows.append(
            TensorRow(
                name=name, kind=BUFFER,
                shape=tuple(int(dim) for dim in buffer.shape),
                dtype=_dtype_name(buffer.dtype),
                persistent=name not in volatile,
                rule_owned=name in rule_owned,
            )
        )
    rows.sort(key=lambda row: (row.name, row.kind))

    swapped: List[Tuple[str, str]] = []
    if quantization is not None:
        owners = {name.rsplit(".", 1)[0] for name in quantization.tensors if "." in name}
        for prefix in sorted(owners):
            try:
                sub = module.get_submodule(prefix)
            except AttributeError:  # pragma: no cover — defensive
                continue
            swapped.append((prefix, type(sub).__name__))

    return ComponentCensus(
        component=component,
        module_class=type(module).__name__,
        tensors=tuple(rows),
        ties=_tie_groups(module),
        quant_rule=quantization.rule if quantization is not None else "",
        quant_modules=tuple(swapped),
        eval_mode=all(not sub.training for sub in module.modules()),
    )


def take(
    modules: Mapping[str, Any],
    quantized: Mapping[str, "Quantization"] = {},
) -> Census:
    """The census of every weight-bearing component of one built tree."""
    return Census(
        components=tuple(
            take_component(name, modules[name], quantized.get(name))
            for name in sorted(modules)
        )
    )


def for_tree(checkpoint_dir: Any, *, compute_dtype: Any = None) -> Census:
    """The census of a checkpoint tree, from CONFIGS ALONE.

    No bytes, no card, no download: parameters come up on meta and buffers are
    computed from config, so a 66 GiB DiT costs the same as a tiny one. This is
    what the release build calls (pgw#1370's derive seam) and what the
    conformance suite calls, and it goes through :func:`~.skeleton.build_modules`
    — the same reader production serves with — so the census can never be a
    statement about a second parser.
    """
    from pathlib import Path

    from . import skeleton as _skeleton

    built = _skeleton.build_modules(Path(checkpoint_dir), compute_dtype=compute_dtype)
    return built.census()


# ── replaying it ────────────────────────────────────────────────────────────


def _first(names: Iterable[str]) -> str:
    ordered = sorted(names)
    return ordered[0] if ordered else ""


def verify(expected: Census, actual: Census, *, where: str = "") -> None:
    """Replay ``expected`` against ``actual``. Raises :class:`CensusMismatch`.

    I1, I2, I3 and I5. Placement (I4) is not a census-to-census question — it is
    a module-to-device one — and lives in :func:`verify_placement`.
    """
    want = expected.by_component()
    have = actual.by_component()

    absent = set(want) - set(have)
    if absent:
        name = _first(absent)
        raise CensusMismatch(
            I5_TOTALITY, name, "", where=where,
            message=(
                f"the census declares this component and the built tree has no "
                f"such module. The tree builds {sorted(have)} where the release "
                f"says {sorted(want)}"
            ),
        )
    unknown = set(have) - set(want)
    if unknown:
        name = _first(unknown)
        raise CensusMismatch(
            I5_TOTALITY, name, "", where=where,
            message=(
                f"the built tree carries a component the census never declared. "
                f"Set equality is asserted in BOTH directions on purpose: a "
                f"component nobody wrote down is the shape every member of this "
                f"defect family arrived in. The tree builds {sorted(have)} where "
                f"the release says {sorted(want)}"
            ),
        )

    for name in sorted(want):
        _verify_component(want[name], have[name], where=where)


def _verify_component(
    expected: ComponentCensus, actual: ComponentCensus, *, where: str
) -> None:
    """The four census-to-census invariants, CAUSE BEFORE SYMPTOM.

    The order is the whole point of pgw#1626's post-mortem: that refusal blamed
    the CHECKPOINT for a defect in the loader, and a reader acted on it. A
    quantizer that did not swap presents as 357 names with no home; a tie that
    did not happen presents as an unfilled alias. If the name-set check ran
    first it would always name the symptom, so I2 and I1 — the construction
    facts — are asked before I5.
    """
    component = expected.component

    if expected.module_class != actual.module_class:
        raise CensusMismatch(
            I5_TOTALITY, component, "", where=where,
            message=(
                f"the census says this component is a {expected.module_class}, "
                f"the built tree made a {actual.module_class}. The image that "
                f"built the release and the image serving it do not agree about "
                f"what this class is"
            ),
        )

    # ── I2: the quantizer's swap ──
    if expected.quant_rule != actual.quant_rule:
        wanted = sorted({name for _prefix, name in expected.quant_modules})
        got = sorted({name for _prefix, name in actual.quant_modules})
        raise CensusMismatch(
            I2_QUANTIZER, component, "", where=where,
            message=(
                f"the census says the config's quantizer is "
                f"{expected.quant_rule or '(none)'} and the built module ran "
                f"{actual.quant_rule or '(none)'}. The census expects "
                f"{len(expected.quant_modules)} swapped module(s) of "
                f"{wanted or ['(none)']} and the built module has "
                f"{len(actual.quant_modules)} of {got or ['(none)']}; every "
                f"scale tensor those modules own would name nothing at load "
                f"(pgw#1638)"
            ),
        )
    if expected.quant_modules != actual.quant_modules:
        want_modules = dict(expected.quant_modules)
        have_modules = dict(actual.quant_modules)
        offender = _first(
            {
                prefix for prefix in set(want_modules) | set(have_modules)
                if want_modules.get(prefix) != have_modules.get(prefix)
            }
        )
        raise CensusMismatch(
            I2_QUANTIZER, component, offender, where=where,
            message=(
                f"the quantizer's module swap does not replay: the census says "
                f"{want_modules.get(offender, '(no such module)')!r}, the built "
                f"module is {have_modules.get(offender, '(no such module)')!r}. "
                f"A plain `nn.Linear` where the rule wants its own class is "
                f"pgw#1638 — every scale tensor beside it names nothing"
            ),
        )

    # ── I1: the ties ──
    if expected.ties != actual.ties:
        want_alias = {a: g[0] for g in expected.ties for a in g[1:]}
        have_alias = {a: g[0] for g in actual.ties for a in g[1:]}
        offender = _first(
            {
                name for name in set(want_alias) | set(have_alias)
                if want_alias.get(name) != have_alias.get(name)
            }
        )
        raise CensusMismatch(
            I1_TIES, component, offender, where=where,
            message=(
                f"the census ties this name to "
                f"{want_alias.get(offender, '(nothing)')!r} and the built module "
                f"ties it to {have_alias.get(offender, '(nothing)')!r}. A tie "
                f"that is a COPY doubles the resident bytes and serves whichever "
                f"of the two the stream wrote last; a tie that is MISSING leaves "
                f"the alias unfilled, because a checkpoint omits every alias it "
                f"ties (pgw#1626)"
            ),
        )

    # ── I3: serve mode ──
    if not actual.eval_mode:
        raise CensusMismatch(
            I3_SERVE_MODE, component, "", where=where,
            message=(
                "the built module is in TRAIN mode. Both `from_pretrained` "
                "implementations end with `model.eval()` — transformers says why "
                "in its own source: \"Set model in evaluation mode to deactivate "
                "Dropout modules by default\". Serving this randomizes the "
                "conditioning of every request and reports no error anywhere "
                "(pgw#1638's third member: 44/44 fleet components, once)"
            ),
        )
    if not expected.eval_mode:
        raise CensusMismatch(
            I3_SERVE_MODE, component, "", where=where,
            message=(
                "the CENSUS records this component as built in train mode, so "
                "the release itself was derived from a module `model.eval()` "
                "never reached. The release is the defect, not this load"
            ),
        )

    # ── I5: set equality, both directions ──
    want = expected.by_name()
    have = actual.by_name()
    missing = set(want) - set(have)
    if missing:
        name = _first(missing)
        row = want[name]
        tied = any(name in group[1:] for group in expected.ties)
        raise CensusMismatch(
            I5_TOTALITY, component, name, where=where,
            message=(
                f"the census declares a {row.kind} of {row.shape} {row.dtype} "
                f"under this name and the built module has nothing there"
                + (
                    f" — and the census records it as an ALIAS of "
                    f"{next(g[0] for g in expected.ties if name in g[1:])!r}, so "
                    f"the tie that creates it did not happen (pgw#1626)"
                    if tied else ""
                )
                + f". {len(missing)} name(s) are missing in total"
            ),
        )
    unknown = set(have) - set(want)
    if unknown:
        name = _first(unknown)
        row = have[name]
        raise CensusMismatch(
            I5_TOTALITY, component, name, where=where,
            message=(
                f"the built module carries a {row.kind} of {row.shape} "
                f"{row.dtype} that the census never declared. This is the FIFTH "
                f"member arm: a construction side effect nobody has written down "
                f"is refused here, at $0, instead of being discovered on a "
                f"rented card. {len(unknown)} unknown name(s) in total"
            ),
        )

    for name in sorted(want):
        _verify_row(component, want[name], have[name], where=where)

def _verify_row(
    component: str, expected: TensorRow, actual: TensorRow, *, where: str
) -> None:
    if expected.kind != actual.kind:
        raise CensusMismatch(
            I5_TOTALITY, component, expected.name, where=where,
            message=(
                f"the census calls this a {expected.kind} and the built module "
                f"registers it as a {actual.kind}"
            ),
        )
    if expected.shape != actual.shape:
        raise CensusMismatch(
            I5_TOTALITY, component, expected.name, where=where,
            message=(
                f"the census says {expected.shape} and the built module has "
                f"{actual.shape}"
            ),
        )
    if expected.persistent != actual.persistent:
        raise CensusMismatch(
            I5_TOTALITY, component, expected.name, where=where,
            message=(
                f"the census records this buffer as "
                f"{'persistent' if expected.persistent else 'NON-persistent'} and "
                f"the built module registers it as "
                f"{'persistent' if actual.persistent else 'NON-persistent'}. That "
                f"decides whether any container ever names it — the exact blind "
                f"spot pgw#1644 died in"
            ),
        )
    if expected.rule_owned or actual.rule_owned:
        # A rule-owned tensor's dtype is the RULE's, and `postprocess_model`
        # legitimately rewrites it after the fill (`scale_fmt="ue8m0"` turns an
        # F32 scale grid into the exponent dtype the kernels read). Its name and
        # shape are asserted like everything else; its dtype is recorded.
        return
    if expected.dtype != actual.dtype:
        raise CensusMismatch(
            I5_TOTALITY, component, expected.name, where=where,
            message=(
                f"the census says {expected.dtype} and the built module holds "
                f"{actual.dtype}. Mixing dtypes inside one forward is what this "
                f"catches; a tensor that reached the module by a path the engine "
                f"did not install is how it happens"
            ),
        )


# ── I4: placement — the module walk, never the container walk ───────────────


def _same_device(held: "torch.device", target: "torch.device") -> bool:
    """Index-tolerant device comparison.

    ``torch.device("cuda") != torch.device("cuda", 0)`` is True, and the stream
    lands tensors on ``cuda:0`` while callers pass a bare ``"cuda"``. A ``!=``
    comparison would make this fence refuse every healthy load on the commonest
    spelling of the device it checks, and a fence that fires on correct input is
    worse than the defect it guards (pgw#1644).
    """
    if held.type != target.type:
        return False
    if held.index is None or target.index is None:
        return True
    return held.index == target.index


def on_meta(module: "torch.nn.Module") -> Tuple[str, ...]:
    """Every parameter or buffer of ``module`` still on ``meta``.

    The MODULE walk, both halves of it. A container walk answers a different
    question — "was this name in the checkpoint" — and answering the placement
    question with it is what left four defects to be found on cards.
    """
    left: List[str] = []
    for name, tensor in module.named_parameters(remove_duplicate=False):
        if tensor.device.type == "meta":
            left.append(name)
    for name, buffer in module.named_buffers(remove_duplicate=False):
        if buffer.device.type == "meta":
            left.append(name)
    return tuple(sorted(left))


def place(module: "torch.nn.Module", target: Any) -> int:
    """Move every non-meta tensor of ``module`` onto ``target``. Returns the count.

    Walked from the MODULE — ``named_parameters()`` + ``named_buffers()`` — and
    never from the checkpoint container or the pipeline's component registry.
    Walking the container is precisely what missed pgw#1644, and reading
    ``pipeline.components`` is what made pgw#1454's sweep a silent no-op for
    every component of a ``ModularPipeline`` (which is neither an ``nn.Module``
    nor a carrier of ``components``).

    Meta is left alone deliberately: a tensor still on meta was never FILLED,
    which is :func:`verify_placement`'s refusal to make, and moving it here
    would turn a missing tensor into a garbage one.

    A PARAMETER moves as ``param.data = param.data.to(device)`` — the
    ``Parameter`` OBJECT survives, which is what keeps a tie a tie. Two names
    holding one ``Parameter`` still hold one afterwards, and the second visit is
    a no-op because the first already moved it. Rebinding ``_parameters[leaf]``
    would silently split every alias into a private copy: double the resident
    bytes, serving whichever half the stream wrote last — pgw#1626's failure
    wearing a placement fix's clothes.

    A BUFFER is rebound in ``_buffers``, because nothing ties buffers and
    ``.data`` assignment does not survive every tensor kind. Buffers that ARE
    one object stay one object through the identity memo.
    """
    import torch

    device = torch.device(target)
    if device.type == "meta":
        return 0
    moved = 0
    rehomed: Dict[int, Any] = {}
    for prefix, sub in module.named_modules():
        del prefix
        for leaf, held in list(sub._parameters.items()):
            del leaf
            if held is None or held.device.type == "meta":
                continue
            if _same_device(held.device, device):
                continue
            held.data = held.data.to(device)
            moved += 1
        for leaf, held in list(sub._buffers.items()):
            if held is None or held.device.type == "meta":
                continue
            if _same_device(held.device, device):
                continue
            replacement = rehomed.get(id(held))
            if replacement is None:
                replacement = held.to(device)
                rehomed[id(held)] = replacement
            sub._buffers[leaf] = replacement
            moved += 1
    return moved


def verify_placement(
    component: str,
    module: "torch.nn.Module",
    target: Any,
    census: Optional[ComponentCensus] = None,
    *,
    where: str = "",
) -> None:
    """I4: after the fill and the sweep, every tensor is ON the target.

    Two refusals, in the order that names the cause rather than the symptom.
    A tensor still on ``meta`` was never filled — that is the fill's failure and
    it is reported as such, with the tie diagnosis pgw#1626 needed. A tensor on
    some OTHER device landed and landed wrong — pgw#1644, where 146 floats on
    the CPU under an all-CUDA model surfaced as ``mat1 is on cpu`` inside
    ``diffusers`` eight milliseconds into a forward.
    """
    import torch

    device = torch.device(target)
    if device.type == "meta":
        return

    # Alias names from the census FIRST — it is measured, by object identity,
    # and it is what the release says. `_tied_weights_keys` is the fallback:
    # the class's own DECLARATION of names it might tie, which still answers
    # when the census itself was taken from a module whose retie was broken.
    aliases = (
        {name for group in census.ties for name in group[1:]}
        if census is not None else set()
    )
    aliases |= {str(name) for name in (getattr(module, "_tied_weights_keys", None) or ())}
    module_class = type(module).__name__
    held: List[Tuple[str, str, "torch.device"]] = []
    for name, tensor in module.named_parameters(remove_duplicate=False):
        held.append((name, PARAMETER, tensor.device))
    for name, buffer in module.named_buffers(remove_duplicate=False):
        held.append((name, BUFFER, buffer.device))

    stranded = sorted(name for name, _kind, dev in held if dev.type == "meta")
    if stranded:
        first = stranded[0]
        tied = [name for name in stranded if name in aliases]
        raise CensusMismatch(
            I4_PLACEMENT, component, first, where=where,
            message=(
                f"({module_class}) {len(stranded)} tensor(s) were never filled "
                f"and are STILL ON META after every container was streamed — "
                f"{', '.join(stranded[:8])}"
                + (" …" if len(stranded) > 8 else "")
                + ". Weight tying was re-established first (pgw#1626), so an "
                "untied alias is not the cause"
                + (
                    f"; {', '.join(tied[:4])} "
                    f"{'is a name' if len(tied) == 1 else 'are names'} "
                    f"{module_class} TIES to another parameter, so the "
                    f"parameter it aliases went unfilled too"
                    if tied else ""
                )
                + ". The name is the LOADER's to place; the checkpoint is not "
                "on trial here"
            ),
        )

    stray = sorted(
        (name, str(dev)) for name, _kind, dev in held
        if not _same_device(dev, device)
    )
    if stray:
        first, where_it_is = stray[0]
        shown = ", ".join(f"{name} on {dev}" for name, dev in stray[:8])
        raise CensusMismatch(
            I4_PLACEMENT, component, first, where=where,
            message=(
                f"{len(stray)} tensor(s) are on a device other than {device} "
                f"after the fill and the placement sweep — {shown}"
                + (" …" if len(stray) > 8 else "")
                + f". A tensor the container never named is still this engine's "
                f"to place; the first one is on {where_it_is}, and serving it "
                f"fails at the first matmul with a device mismatch raised inside "
                f"a library, naming nothing (pgw#1644)"
            ),
        )


def fence(
    modules: Mapping[str, Any],
    quantized: Mapping[str, "Quantization"],
    expected: Census,
    *,
    target: Any = None,
    where: str = "",
) -> Census:
    """The whole predicate, in one call. THE serve-time fence, and the CI one.

    I1/I2/I3/I5 come out of the census replay; I4 out of the module-to-device
    walk. Returns the census actually taken, so a caller that wants to report it
    does not take a second one.
    """
    actual = take(modules, quantized)
    verify(expected, actual, where=where)
    if target is not None:
        rows = actual.by_component()
        for name in sorted(modules):
            verify_placement(
                name, modules[name], target, rows.get(name), where=where)
    return actual


__all__ = [
    "BUFFER",
    "CENSUS_KIND",
    "Census",
    "CensusError",
    "CensusMismatch",
    "ComponentCensus",
    "I1_TIES",
    "I2_QUANTIZER",
    "I3_SERVE_MODE",
    "I4_PLACEMENT",
    "I5_TOTALITY",
    "PARAMETER",
    "TensorRow",
    "fence",
    "for_tree",
    "on_meta",
    "place",
    "take",
    "take_component",
    "verify",
    "verify_placement",
]
