"""§1.30 declaration 2+3, on the tensor-layout contract **v2** vocabulary: a
decoder declares WHICH QUANT RULE it decodes, beside the code that decodes it.

A v2 LAYOUT is ``quant(topology)`` — a TOPOLOGY (a finite ``{key -> logical
shape}`` map, dtype-free, extracted mechanically from a reference checkpoint's
headers) composed with a QUANT RULE (a per-tensor transformation whose
convention facts — nibble order, swizzle, scale rank and span — ARE its
identity). Identity is the digest pair; the wire rendering is th#1809's
``"<topology>+<quant>"`` (:meth:`LayoutId.render`, shared with the hub's
``tensorfs.LayoutID.String`` — a spelling drift is a CAS-address fork).

**pgw#1621 deleted the v1 vocabulary this module used to carry.** The six
``KNOWN_CONTRACTS`` handle strings and the five DECODE DIMENSIONS
(``elements``/``scales``/``key_topologies``/``file_layouts``/``bakes``) are
gone, because the axes they enumerated are exactly the convention facts a v2
quant rule carries as identity. A decoder now names the RULE and the axes come
with it — one fact, one producer, and the ``q4_k`` / ``float4_e2m1fn_x2``
spelling traps become inexpressible rather than guarded.

**THE SM FLOOR IS A PROPERTY OF THE RULE** (Paul, 2026-08-18: "the sm_x compute
floor should fall out of the contract itself"). ``DTYPE_MIN_SM`` and
``capability_floor_for_dtype`` are deleted with the rest: a table keyed on a
dtype's SPELLING silently loses the floor when a lane is spelled differently,
and the two zeros it returned — "measured: no floor" and "never heard of it" —
were indistinguishable. :func:`capability_floor_for_rule` reads
``capability_floor_sm`` off the vendored rule document, which is the ONE
producer of that number in this repo.

**NOTHING HERE COMPUTES A LAYOUT AND NOTHING HERE DECIDES AN ADMIT.**
``quant(topology)`` is evaluated by the Go engine and by nothing else — there is
deliberately no pyo3 binding for the evaluator or the decision (tensorfs
``spec/v2/README.md``), and the v2 cut deleted ``Verdict``/``Conversion``/
``Mismatch`` from the Rust extension's typed surface. v1 shipped three copies of
one matching rule and the copies were free to disagree about an admit. This
module READS identity facts out of the vendored corpus; the verdict is the
hub's and arrives binding-carried.

**No exclusion marker exists, deliberately** (A4 corollary).
Exclusions are DERIVED from declared traits — ``composes_lora`` crossed with a
function's ``lora_bucket`` is computable at build — or they do not exist.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, TypeVar

import msgspec

from .._vendor.tensorfs import layout2
from .execution_lanes import known_execution_lane_bodies

# ── The VENDORED v2 CORPUS ───────────────────────────────────────────────────
#
# `layout2` resolves the corpus by walking its own parents for a `spec/v2`
# directory, and the vendored package carries one at
# `_vendor/tensorfs/spec/v2`. The root is named here anyway rather than left to
# that walk: an unreadable corpus must REFUSE (it does — `_root` raises
# `FileNotFoundError`), and naming the path makes the refusal say which tree it
# looked in. The three readers below are `lru_cache`d because the corpus is
# 2.6 MB of JSON and every lane declaration in the image would otherwise re-read
# it at class-definition time. They hand back a `MappingProxyType` because a
# `lru_cache` over a mutable result is one caller's `.pop()` away from a corpus
# that silently disagrees with the files on disk for the rest of the process.

_SPEC_V2 = Path(layout2.__file__).resolve().parent / "spec" / "v2"


@functools.lru_cache(maxsize=1)
def quant_rules() -> Mapping[str, layout2.Quant]:
    """handle -> the rule's IDENTITY FACTS, from the vendored corpus.

    Identity only: declared dtype, capability floor, conventions, lossiness,
    the dequant identity and Go's canonical digest. The eligibility predicate
    and the emission shapes each rule document also carries belong to the
    EVALUATOR, which is Go's, and reading them here would be the first half of
    growing a second one.
    """
    return MappingProxyType(dict(layout2.rules(_SPEC_V2)))


@functools.lru_cache(maxsize=1)
def topologies() -> Mapping[str, Mapping[str, Mapping[str, tuple[int, ...]]]]:
    """handle -> component -> key -> logical shape. SHAPES ONLY — a topology
    carries no dtype, because the dtype is the quant rule's half of the pair."""
    return MappingProxyType(dict(layout2.topologies(_SPEC_V2)))


@functools.lru_cache(maxsize=1)
def display_names() -> Mapping[str, str]:
    """``"<topology>+<quant>"`` -> the v1 lane-handle spelling it used to be
    known by (``sdxl.diffusers-bf16@1``).

    **A DISPLAY NAME IS NEVER PARSED AND GATES NOTHING.** It exists so an
    operator reading a refusal recognises the lane they declared last week, and
    so this cut's re-key had a mechanical source. Deriving a quant handle by
    string surgery on one of these is a defect waiting to happen — `musicgen`'s
    pair is `plain.f16@1` while its display name says `-fp16@1`.
    """
    raw = json.loads((_SPEC_V2 / "display-names.json").read_text(encoding="utf-8"))
    return MappingProxyType(dict(raw.get("names") or {}))


def known_quant_rules() -> tuple[str, ...]:
    """Every quant-rule handle the vendored corpus carries, sorted."""
    return tuple(sorted(quant_rules()))


def known_topologies() -> tuple[str, ...]:
    """Every v2 topology handle the vendored corpus carries, sorted."""
    return tuple(sorted(topologies()))


def capability_floor_for_rule(quant: object) -> int:
    """The ``min_sm`` a lane's QUANT RULE implies.

    ``0`` means the rule states no floor (fp32 runs anywhere) — and it is a
    MEASURED zero, written in the rule document, not the "never heard of it"
    zero the deleted dtype table could not distinguish from it. A handle the
    corpus does not carry REFUSES rather than answering 0: a floor invented for
    a rule nobody shipped is a placement claim with nothing behind it, and the
    permissive direction is the one that puts an nvfp4 lane on Ampere.
    """
    handle = str(quant or "").strip()
    rule = quant_rules().get(handle)
    if rule is None:
        raise LayoutDeclarationError(
            f"quant rule {handle!r} is not in the vendored v2 corpus, so its "
            f"capability floor is unknowable. Known: "
            f"{', '.join(known_quant_rules())}. A floor answered for an "
            f"unknown rule would be a placement claim with nothing behind it."
        )
    return int(rule.capability_floor_sm)


def rule_dtype(quant: object) -> str:
    """The torch spelling the rule DECLARES (``float8_e4m3fn``) — what a loader
    wants, and what the lane SERVES. Refuses an unknown handle, same reason."""
    handle = str(quant or "").strip()
    rule = quant_rules().get(handle)
    if rule is None:
        raise LayoutDeclarationError(
            f"quant rule {handle!r} is not in the vendored v2 corpus. Known: "
            f"{', '.join(known_quant_rules())}"
        )
    return str(rule.declared_dtype)

#: The handle grammar, transcribed from tensorfs' `isTFM1ContractName`
#: (`tfm1.go:217`) + `ParseHandle` (`v2doc.go:72`), which is the one authority.
#:
#: THE PRODUCER SEGMENT ADMITS A HYPHEN, AND THIS REGEX DID NOT. It was
#: `^([a-z0-9]+)\.…$`, which is correct for every v1 QUANT handle the fleet ever
#: had (`plain.bf16`, `cozy.fp8-rowwise` — hyphens only ever fell in the FORMAT
#: segment) and wrong for half the v2 TOPOLOGY corpus: `flux2-klein.diffusers@1`,
#: `hidream-o1.diffusers@1`, `minimax-h3.diffusers@1`, `z-image.diffusers@1`,
#: `ltx2-upsampler.diffusers@1`, `sdxl-inpainting.diffusers@1` and
#: `qwen3-6-35b-a3b.transformers@1` all refused as "not a handle". Five migrated
#: endpoints could not have declared a lane at all.
#:
#: This is the SAME relaxation tensorfs#121 made and the vendored v1 port had to
#: follow ("the port refused `hidream-o1.diffusers-bf16` until it did"); the
#: tensorlayout-side regex simply never had a hyphenated producer to trip over
#: until the topology axis carried real model names. A LEADING hyphen still
#: refuses, on both segments, exactly as upstream's does.
_HANDLE_RE = re.compile(
    r"^([a-z0-9][a-z0-9-]*)\.([a-z0-9][a-z0-9._-]*)@([1-9][0-9]*)$")

# ── The five DECODE DIMENSIONS are DELETED (pgw#1621) ────────────────────────
#
# `elements`, `scales`, `key_topologies`, `file_layouts` and `bakes` enumerated
# "which of a byte format's legal shapes does this decoder read". Every one of
# them is a CONVENTION FACT, and a v2 quant rule carries its convention facts as
# IDENTITY: `cozy.nvfp4-flat@1` and `bfl.nvfp4-preswizzled@1` are two rules and
# two digests precisely because one is LOW-nibble with flat scales and the other
# is HIGH-nibble pre-swizzled (reading one as the other measured LPIPS 1.11).
# Under v1 that difference had to be spelled out on a side axis beside a shared
# handle; under v2 it IS the handle.
#
# So a decoder names the rule, and the axes come with it. Two consequences worth
# writing down because they were live defects:
#
#   * the `q4_k` trap (tensorfs#130) is gone — a GGUF lane's floor is whatever
#     its rule document says, not whatever a dtype-spelling table happened to
#     know;
#   * the `float4_e2m1fn_x2` FLOOR-LOSING spelling is gone with it. There is no
#     dtype string to mis-spell any more: `capability_floor_for_rule` takes a
#     rule handle, and a handle the corpus does not carry REFUSES.
#
# The `key_topologies` axis in particular is now the TOPOLOGY half of the pair,
# which is where it belonged: `minimax-h3.diffusers@1` (638 keys, split
# to_q/to_k/to_v) and `minimax-h3.native@1` (fused qkv_proj) are two topologies
# related by a ratified morphism, not one handle with a side note.


class QuantRuleDecoder(msgspec.Struct, frozen=True, kw_only=True):
    """One decoder's declaration: the QUANT RULE whose bytes it reads, and the
    lane BODIES the decoded units execute as.

    The execution axis (eager/compiled) is NOT declared here — the platform
    owns it, and the derivation crosses these bodies with the lane table's own
    execution support.

    There is no `decodes=` any more. What a decoder reads WITHIN a format was
    the v1 question; a v2 rule has no "within" — its conventions are its
    identity, so naming the rule is the whole declaration.
    """

    rule: str
    decoder: str
    serves: tuple[str, ...]
    composes_lora: bool
    why: str = ""


MARKER = "__cozy_tensor_layout_quant_rules__"

F = TypeVar("F", bound=Callable[..., Any])


def _validate(dec: QuantRuleDecoder) -> None:
    if not _HANDLE_RE.match(dec.rule):
        raise ValueError(
            f"{dec.decoder}: {dec.rule!r} is not a quant-rule handle "
            "(want ns.name@N)"
        )
    if dec.rule not in quant_rules():
        raise ValueError(
            f"{dec.decoder}: quant rule {dec.rule!r} is not in the vendored v2 "
            f"corpus. Rules are RATIFIED DOCUMENTS, one per byte FORMAT ever: "
            f"author it in tensorfs `spec/v2/rules/` — eligibility predicate, "
            f"emission shapes, conventions, the dequant identity — and "
            f"re-vendor per `_vendor/VENDORED.toml`. Known: "
            f"{', '.join(known_quant_rules())}. If this decoder reads real "
            f"bytes no ratified rule names, say so with "
            f"`@unregistered_decode_path(reason=...)` rather than inventing a "
            f"handle: a rule that matches nothing is the failure mode, not the "
            f"fix."
        )
    if not dec.serves:
        raise ValueError(
            f"{dec.decoder}: serves= is empty; a decoder that executes no "
            "lane body declares nothing"
        )
    bodies = set(known_execution_lane_bodies())
    for token in dec.serves:
        if token not in bodies:
            raise ValueError(
                f"{dec.decoder}: serves= token {token!r} is not a known lane "
                f"body; valid: {sorted(bodies)}"
            )
    if len(set(dec.serves)) != len(dec.serves):
        raise ValueError(f"{dec.decoder}: serves= repeats a lane body")


def implements_quant_rule(
    *,
    rule: str,
    serves: Iterable[str],
    composes_lora: bool,
    why: str = "",
) -> Callable[[F], F]:
    """Mark a decode entrypoint as reading the bytes of ``rule``.

    ``decodes=`` is GONE (pgw#1621) rather than made optional. It existed
    because a v1 handle named a byte FORMAT and said nothing about which of
    that format's legal shapes a decoder read — so a declaration that named a
    handle and stopped was incomplete. A v2 rule has no legal shapes to choose
    between: its conventions ARE its identity, so naming the rule is the whole
    declaration and there is nothing left to write by omission.

    Stackable: one function may read several rules (``decode_linear`` reads
    both the flat and the pre-swizzled nvfp4 packagings).
    """

    def deco(fn: F) -> F:
        where = f"{fn.__module__}:{fn.__qualname__}"
        dec = QuantRuleDecoder(
            rule=rule,
            decoder=where,
            serves=tuple(serves),
            composes_lora=bool(composes_lora),
            why=why,
        )
        _validate(dec)
        prior: tuple[QuantRuleDecoder, ...] = getattr(fn, MARKER, ())
        for existing in prior:
            if existing.rule == dec.rule and existing != dec:
                raise ValueError(
                    f"{dec.decoder}: conflicting declarations for {dec.rule}"
                )
        setattr(fn, MARKER, prior + (dec,))
        return fn

    return deco


def quant_rule_decoders_of(obj: Any) -> tuple[QuantRuleDecoder, ...]:
    """The declarations carried by one object, or ``()``."""
    marked = getattr(obj, MARKER, ())
    if not isinstance(marked, tuple):
        return ()
    return tuple(d for d in marked if isinstance(d, QuantRuleDecoder))


class UnregisteredDecodePath(msgspec.Struct, frozen=True, kw_only=True):
    """A decoder that reads real bytes NO ratified quant rule covers.

    It satisfies no gate and is never intersected with anything — a decode-set
    entry needs a handle, and a handle needs a hub-side descriptor (A2). It is
    recorded because the alternative is a source comment, which no refusal can
    read: when `resolve()` refuses a variant these bytes belong to, the answer
    "this image decodes them but the platform has no contract for them" is the
    remedy, and it is a different remedy from "ship a different image".
    """

    decoder: str
    reason: str


MARKER_UNREGISTERED = "__cozy_unregistered_decode_path__"


def unregistered_decode_path(*, reason: str) -> Callable[[F], F]:
    """Mark a decode entrypoint whose bytes no registered contract names."""

    def deco(fn: F) -> F:
        if not reason.strip():
            raise ValueError(
                f"{fn.__module__}:{fn.__qualname__}: an unregistered decode "
                "path without a reason is an untraceable gap")
        setattr(fn, MARKER_UNREGISTERED, UnregisteredDecodePath(
            decoder=f"{fn.__module__}:{fn.__qualname__}",
            reason=reason.strip(),
        ))
        return fn

    return deco


def unregistered_decode_path_of(obj: Any) -> tuple[UnregisteredDecodePath, ...]:
    """The unregistered-path record carried by one object, or ``()``."""
    marked = getattr(obj, MARKER_UNREGISTERED, None)
    return (marked,) if isinstance(marked, UnregisteredDecodePath) else ()


# ── §1.33: the DEMAND side of the same vocabulary ─────────────────────────────
#
# `@implements_quant_rule` above is the SUPPLY-adjacent census — "which decoders
# does this IMAGE contain". The DEMAND is what `Slot(layouts=...)` declares:
# "what does this slot's code need in order to run". Two facts, one vocabulary,
# so the handle grammar and the registration refusal are shared verbatim rather
# than transcribed twice.
#
# The SDK emits HANDLES and never digests. The hub resolves a handle to its
# descriptor digest at MANIFEST INGEST against its own registry and stores
# both; a handle its registry does not know fails the manifest there, not at
# rebind. The QUANT half of that vocabulary is no longer a transcription: it is
# the vendored v2 rule corpus, shipped with this code and digest-pinned, so
# "allowed to be stale" no longer applies to it. The LOADER-SHAPE topology half
# below still is a transcription, and still says so.

LAYOUT_KEY_ANY_COMPONENT = "*"

AXIS_TOPOLOGY = "topology"
AXIS_QUANT = "quant"
LAYOUT_AXES: tuple[str, ...] = (AXIS_TOPOLOGY, AXIS_QUANT)

LAYOUT_AXIS_ANY = "any"

# ⚠️ THIS IS THE LOADER-SHAPE AXIS, NOT THE v2 TOPOLOGY AXIS. Two different
# questions share the word "topology" and confusing them is a production
# outage, so read this before touching anything below. The wording mirrors
# tensorhub's own `internal/tensorlayout/topology_th1809.go` warning block
# (th#2250) deliberately — one hazard, one sentence, on both sides of the wire.
#
#   HERE (`KNOWN_TOPOLOGY_CONTRACTS`)   "which LOADER entry point reads this
#                                       tree?" Derived from the manifest's PATH
#                                       SET. Values: `diffusers.multifile@1`,
#                                       `gguf.native@1`, `comfy.splitfiles@1`.
#                                       Hub-side it projects onto
#                                       `checkpoints.file_layout`, a LIVE
#                                       SERVING INPUT (th#2094 took
#                                       `tensorhub/sdxl` off the air through
#                                       it) and the vocabulary
#                                       `endpoint_release_slot_layout_demand`
#                                       rows are declared in.
#   v2 TOPOLOGY (`known_topologies()`)  "WHAT TENSORS, at what SHAPES?" A finite
#                                       map extracted mechanically from a
#                                       reference checkpoint's HEADERS. Values:
#                                       `sdxl.diffusers@1`, `anima.net@1`. It is
#                                       what BIND ADMISSION matches on, and it is
#                                       the first half of a `lanes=` key.
#
# th#2250 measured this and took SEPARATE COLUMNS rather than widening one
# (`checkpoints.layout_topology` beside `checkpoints.tensor_topology_contract`),
# because writing v2 handles into the older column would blank every
# `file_layout` projection on the next publish and refuse every release whose
# declared demand names a loader-shape handle. Retiring the loader-shape
# registry is a SEPARATE cut with its own blast radius — `file_layout` gets its
# own path-set derivation and the demand rows are re-keyed — and pgw#1621
# UNBLOCKS it (nothing here decides an admission any more) rather than
# performing it.
#
# Transcribed from code we already run: tensorhub's
# `catalog/layout_contract.go` (library_name x file_layout) and jobs'
# `conversion/comfyui.py` `_SPLIT_COMPONENT_MAP`. Allowed to be stale, CHECKED
# at the hub's manifest ingest, never authoritative.
TOPOLOGY_DIFFUSERS_MULTIFILE = "diffusers.multifile@1"
TOPOLOGY_DIFFUSERS_SINGLEFILE = "diffusers.singlefile@1"
TOPOLOGY_TRANSFORMERS_NATIVE = "transformers.native@1"
TOPOLOGY_PEFT_ADAPTER = "peft.adapter@1"
TOPOLOGY_GGUF_NATIVE = "gguf.native@1"
TOPOLOGY_COMFY_SPLITFILES = "comfy.splitfiles@1"

KNOWN_TOPOLOGY_CONTRACTS: tuple[str, ...] = (
    TOPOLOGY_DIFFUSERS_MULTIFILE,
    TOPOLOGY_DIFFUSERS_SINGLEFILE,
    TOPOLOGY_TRANSFORMERS_NATIVE,
    TOPOLOGY_PEFT_ADAPTER,
    TOPOLOGY_GGUF_NATIVE,
    TOPOLOGY_COMFY_SPLITFILES,
)


class LayoutDeclarationError(ValueError):
    """A `Slot(layouts=...)` declaration the SDK refuses where it is written."""


def known_contracts(axis: str) -> tuple[str, ...]:
    """The handles of ONE axis. Unknown axis is a refusal, not an empty tuple —
    an empty answer would read as "nothing is registered".

    The QUANT axis is now the vendored v2 RULE corpus (authoritative, shipped
    with the code) rather than a transcription of the hub's registry. The
    TOPOLOGY axis here is still the LOADER-SHAPE vocabulary — see the warning
    block above; `known_topologies()` is the v2 one and they are not
    interchangeable.
    """
    if axis == AXIS_QUANT:
        return known_quant_rules()
    if axis == AXIS_TOPOLOGY:
        return KNOWN_TOPOLOGY_CONTRACTS
    raise LayoutDeclarationError(
        f"unknown layout axis {axis!r}; the axes are {list(LAYOUT_AXES)}")


def validate_layout_handle(
    handle: object, *, where: str, axis: str = AXIS_QUANT,
) -> str:
    """One declared handle on ONE axis, normalized, or a refusal.

    `Slot(layouts=...)` declares the QUANT axis alone, and still does after
    the v2 cut. A slot's demand is "what can my code EXECUTE" — a kernel
    question, which is exactly what a quant rule answers; the topology half is
    a fact about which tensors a checkpoint has, which the slot does not
    constrain. The SDK-internal converter registry names the topology axis
    explicitly (`axis=AXIS_TOPOLOGY`), on the LOADER-SHAPE vocabulary.

    A LANE key is the other thing entirely: it is the full `(topology, quant)`
    stamp pair and is parsed by :func:`parse_lane_stamp`.
    """
    if not isinstance(handle, str):
        raise LayoutDeclarationError(
            f"{where}: layout handle must be a string, got "
            f"{type(handle).__name__}"
        )
    text = handle.strip()
    known = known_contracts(axis)
    if "+" in text:
        raise LayoutDeclarationError(
            f"{where}: {text!r} names a <topology>+<quant> STAMP PAIR, which is "
            "what a LANE is keyed by. A slot's demand is a kernel question — "
            "what can this code execute — so it declares the quant handle "
            "alone. Write the pair in `lanes=`, not here."
        )
    if not _HANDLE_RE.match(text):
        raise LayoutDeclarationError(
            f"{where}: {text!r} is not a contract handle (want ns.name@N)"
        )
    if text not in known:
        if axis == AXIS_QUANT:
            raise LayoutDeclarationError(
                f"{where}: quant rule {text!r} is not in the vendored v2 "
                f"corpus. A rule is a RATIFIED DOCUMENT — author it in "
                f"tensorfs `spec/v2/rules/` and re-vendor per "
                f"`_vendor/VENDORED.toml`; a slot cannot demand a rule no "
                f"document describes. Known: {', '.join(known)}"
            )
        raise LayoutDeclarationError(
            f"{where}: {axis} contract {text!r} is not registered. "
            "The loader-shape topology vocabulary is CODE (th#1580 A2): "
            "register it in tensorhub's internal/tensorlayout with a "
            f"descriptor and a probe set first. Known: {', '.join(known)}"
        )
    return text


def normalize_layout_demand(
    layouts: object, *, where: str,
) -> dict[str, tuple[str, ...]]:
    """`Slot(layouts=...)` -> `{component_path: accepted handle SET}`."""
    if not isinstance(layouts, dict):
        raise LayoutDeclarationError(
            f"{where}: layouts= must be a mapping of component path -> ordered "
            f"handles, got {type(layouts).__name__}"
        )
    if not layouts:
        raise LayoutDeclarationError(
            f"{where}: layouts={{}} declares nothing. Omit layouts= to leave "
            "this slot UNDECLARED; an empty declaration is neither 'accepts "
            "everything' nor 'accepts nothing' and the platform will not "
            "guess which."
        )
    out: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_value in layouts.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise LayoutDeclarationError(
                f"{where}: layouts= key {raw_key!r} must be a non-empty "
                f"component path or {LAYOUT_KEY_ANY_COMPONENT!r}"
            )
        key = raw_key.strip()
        if isinstance(raw_value, (str, bytes)) or not isinstance(
                raw_value, (tuple, list)):
            raise LayoutDeclarationError(
                f"{where}: layouts[{key!r}] must be a tuple of handles — the "
                f"SET this component accepts, whose order carries no "
                f"preference — got {type(raw_value).__name__}"
            )
        if not raw_value:
            raise LayoutDeclarationError(
                f"{where}: layouts[{key!r}] is empty. A component that "
                "accepts no layout cannot be bound at all; omit the key to "
                "fall back to the whole-tree declaration, or omit layouts= "
                "entirely to leave the slot UNDECLARED."
            )
        handles: list[str] = []
        for item in raw_value:
            handle = validate_layout_handle(
                item, where=f"{where}: layouts[{key!r}]")
            if handle in handles:
                raise LayoutDeclarationError(
                    f"{where}: layouts[{key!r}] repeats {handle!r}; a set "
                    "states each member once"
                )
            handles.append(handle)
        out[key] = tuple(sorted(handles))
    return out


class LayoutId(msgspec.Struct, frozen=True, kw_only=True):
    """One side of the two-sided vocabulary, as a PAIR of axes."""

    topology: str | None = None
    quant: str | None = None

    def render(self) -> str:
        return f"{self.topology or ''}+{self.quant or ''}"

    def axis(self, name: str) -> str | None:
        if name == AXIS_TOPOLOGY:
            return self.topology
        if name == AXIS_QUANT:
            return self.quant
        raise LayoutDeclarationError(
            f"unknown layout axis {name!r}; the axes are {list(LAYOUT_AXES)}")

    def with_axis(self, name: str, value: str | None) -> "LayoutId":
        if name == AXIS_TOPOLOGY:
            return LayoutId(topology=value, quant=self.quant)
        if name == AXIS_QUANT:
            return LayoutId(topology=self.topology, quant=value)
        raise LayoutDeclarationError(
            f"unknown layout axis {name!r}; the axes are {list(LAYOUT_AXES)}")


def parse_layout_id(text: object, *, where: str) -> LayoutId:
    """`"<topology>+<quant>"`, or a bare handle meaning the QUANT axis alone."""
    if isinstance(text, LayoutId):
        return text
    if not isinstance(text, str) or not text.strip():
        raise LayoutDeclarationError(
            f"{where}: a layout id is '<topology>+<quant>' or a bare quant "
            f"handle, got {text!r}")
    raw = text.strip()
    if "+" not in raw:
        return LayoutId(quant=_axis_member(raw, axis=AXIS_QUANT, where=where))
    topology, _, quant = raw.partition("+")
    return LayoutId(
        topology=_axis_member(topology, axis=AXIS_TOPOLOGY, where=where),
        quant=_axis_member(quant, axis=AXIS_QUANT, where=where),
    )


def _axis_member(text: str, *, axis: str, where: str) -> str | None:
    value = text.strip()
    if not value:
        return None
    if value == LAYOUT_AXIS_ANY:
        return LAYOUT_AXIS_ANY
    if "+" in value:
        raise LayoutDeclarationError(
            f"{where}: {value!r} carries a second '+' on the {axis} axis")
    if not _HANDLE_RE.match(value):
        raise LayoutDeclarationError(
            f"{where}: {value!r} is not a contract handle (want ns.name@N)")
    if value not in known_contracts(axis):
        raise LayoutDeclarationError(
            f"{where}: {axis} contract {value!r} is not registered. "
            "Contracts are CODE (th#1580 A2): register it in tensorhub's "
            "internal/tensorlayout with a descriptor and a probe set before "
            f"anything may name it. Known: {', '.join(known_contracts(axis))}")
    return value


# ── The LANE key: a v2 STAMP PAIR ────────────────────────────────────────────
#
# `lanes = {(topology, quant): lane(request=…)}`. Both halves are handles into
# the vendored v2 corpus, both are REQUIRED, and neither is inferred from the
# other. This is the whole of pgw#1621's re-key on the declaration surface.
#
# WHY A PAIR AND NOT ONE STRING. A whole-string compare cannot express
# "topology differs, quant matches", which is the DERIVABLE rung the hub's
# admission is built on. `LayoutId` is compared FIELD-WISE for exactly that
# reason and the rendered form exists only to travel.
#
# WHY NOT THE OLD SPELLING. `sdxl.diffusers-bf16@1` survives as a DISPLAY name
# and nothing else — see :func:`display_names`. There is no alias resolution
# here on purpose: a spelling that resolves is a spelling that spreads, and the
# hub's own bridge for the un-re-keyed fleet (`tensorstamp.LaneID`'s second arm)
# exists to be DELETED once this re-key has reached every release. A worker that
# also accepted the old spelling would keep that arm alive forever.


def parse_lane_stamp(key: object, *, where: str) -> "LayoutId":
    """One `lanes=` key -> the `(topology, quant)` stamp pair, or a refusal.

    Accepts the author's spelling — a two-tuple ``("sdxl.diffusers@1",
    "plain.bf16@1")`` — and the rendered wire form ``"sdxl.diffusers@1+
    plain.bf16@1"``, because that is what a document read back off the wire
    carries. Both produce the same :class:`LayoutId`; there is one identity.

    Every refusal names the corpus, because "not registered" with no list is
    the message that sends an author to grep.
    """
    if isinstance(key, LayoutId):
        topology_raw: object = key.topology
        quant_raw: object = key.quant
    elif isinstance(key, str):
        text = key.strip()
        if "+" not in text:
            raise LayoutDeclarationError(
                f"{where}: {text!r} is a bare handle, and a lane is named by a "
                f"PAIR. Write the tuple — `lanes={{(\"<topology>@N\", "
                f"\"<quant>@N\"): lane(request=…)}}`. If this is an old v1 lane "
                f"handle, it is a DISPLAY name now and names no pair: "
                f"{_display_hint(text)}"
            )
        topology_raw, _, quant_raw = text.partition("+")
    elif isinstance(key, (tuple, list)):
        if len(key) != 2:
            raise LayoutDeclarationError(
                f"{where}: a lane key is exactly (topology, quant); got "
                f"{len(key)} element(s)")
        topology_raw, quant_raw = key[0], key[1]
    else:
        raise LayoutDeclarationError(
            f"{where}: a lane key is a (topology, quant) pair of handles, got "
            f"{type(key).__name__}. The tensorfs Contract OBJECT that used to "
            f"key this mapping is deleted with the v1 corpus (pgw#1621)."
        )

    topology = _stamp_half(topology_raw, axis=AXIS_TOPOLOGY, where=where)
    quant = _stamp_half(quant_raw, axis=AXIS_QUANT, where=where)
    return LayoutId(topology=topology, quant=quant)


def _stamp_half(value: object, *, axis: str, where: str) -> str:
    if not isinstance(value, str):
        raise LayoutDeclarationError(
            f"{where}: the {axis} half of a lane stamp must be a handle "
            f"string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise LayoutDeclarationError(
            f"{where}: the {axis} half of a lane stamp is empty. Both halves "
            f"are REQUIRED — a pair with a blank half is not a stamp, and "
            f"there is no 'any' rung on a lane declaration.")
    if not _HANDLE_RE.match(text):
        raise LayoutDeclarationError(
            f"{where}: {text!r} is not a handle (want ns.name@N)")
    known = known_topologies() if axis == AXIS_TOPOLOGY else known_quant_rules()
    if text not in known:
        extra = ""
        if axis == AXIS_TOPOLOGY:
            extra = (
                " A v2 topology is EXTRACTED MECHANICALLY from a reference "
                "checkpoint's headers — it is never hand-authored — so the fix "
                "is to bank the headers in tensorfs `spec/v2/headers/`, add the "
                "`CORPUS.tsv` row, rebuild, and re-vendor.")
        raise LayoutDeclarationError(
            f"{where}: {axis} {text!r} is not in the vendored v2 corpus. "
            f"Known {axis}: {', '.join(known)}.{extra}")
    return text


def _display_hint(text: str) -> str:
    """The reverse display-name lookup, for a refusal message ONLY.

    This is the one place an old v1 spelling is looked at, it produces PROSE,
    and it can only ever appear inside an exception. It resolves nothing: the
    author still has to write the pair.
    """
    for pair, name in display_names().items():
        if name == text:
            return f"{text!r} used to name {pair!r} — declare that pair"
    return f"{text!r} matches no ratified display name either"


class UndeclaredSlotLayoutError(LayoutDeclarationError):
    """A model slot that declares neither a consumed contract nor a reason."""


def normalize_layout_undeclarable(reason: object, *, where: str) -> str:
    """`Slot(layouts_undeclarable=...)` -> the reason, or a refusal."""
    if not isinstance(reason, str):
        raise LayoutDeclarationError(
            f"{where}: layouts_undeclarable= must be a string reason, got "
            f"{type(reason).__name__}")
    text = reason.strip()
    if not text:
        raise LayoutDeclarationError(
            f"{where}: layouts_undeclarable= needs a REASON. An empty escape "
            "is the silence A19 removed — say which bytes this slot holds and "
            "why no registered handle names them.")
    return text


def undeclared_slot_refusal(*, function: str, slot: str) -> str:
    """The one sentence both the SDK and the manifest gate speak."""
    return (
        f"function {function!r}: model slot {slot!r} declares no consumed "
        "tensor-layout contract. Every model slot states what its code can "
        "execute — A19 is a hard cut, so ABSENT is a refusal, never the "
        "UNDECLARED tri-state:\n"
        "    Slot(Pipe, layouts={\"*\": (\"plain.bf16@1\",)})\n"
        f"Ratified quant rules: {', '.join(known_quant_rules())}.\n"
        "If no ratified rule names this slot's bytes (a tokenizer tree with no "
        "tensors, a GGUF block quant, a compressed-tensors checkpoint), declare "
        "that explicitly and say why:\n"
        "    Slot(Pipe, layouts_undeclarable=\"gguf: no ratified v2 quant rule "
        "describes this block-quant packaging\")"
    )


KNOWN_REQUIREMENT_TERMS: tuple[str, ...] = (
    "min_sm", "min_vram_gb", "min_host_ram_gb", "min_cuda", "min_torch")

REQUIREMENT_LEVELS: tuple[str, ...] = ("minimum", "recommended")

_UNBUILT_TERMS: tuple[str, ...] = ("kernels",)

_SM_TERM_RE = re.compile(r"^sm([1-9][0-9]{1,2})\+$")
_VRAM_TERM_RE = re.compile(r"^vram([0-9]+(?:\.[0-9]+)?)g$")
_RAM_TERM_RE = re.compile(r"^ram([0-9]+(?:\.[0-9]+)?)g$")
_CUDA_TERM_RE = re.compile(r"^cuda([0-9]+(?:\.[0-9]+)*)\+$")
_TORCH_TERM_RE = re.compile(r"^torch([0-9]+(?:\.[0-9]+)*)\+$")

_TERM_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("min_sm", _SM_TERM_RE),
    ("min_vram_gb", _VRAM_TERM_RE),
    ("min_host_ram_gb", _RAM_TERM_RE),
    ("min_cuda", _CUDA_TERM_RE),
    ("min_torch", _TORCH_TERM_RE),
)

_COMPACT_EXAMPLES = "'sm100+', 'vram80g', 'ram64g', 'cuda12.8+', 'torch2.9+'"


class RequirementTerms(msgspec.Struct, frozen=True, kw_only=True,
                       omit_defaults=True):
    """The term bag — ONE machine axis per field, at ONE level."""

    min_sm: int = 0
    min_vram_gb: float = 0.0
    min_host_ram_gb: float = 0.0
    min_cuda: str = ""
    min_torch: str = ""

    def declared(self) -> bool:
        return bool(self.declared_terms())

    def declared_terms(self) -> dict[str, Any]:
        """Only the axes this bag actually states, in vocabulary order."""
        out: dict[str, Any] = {}
        for term in KNOWN_REQUIREMENT_TERMS:
            value = getattr(self, term)
            if value:
                out[term] = value
        return out

    def render(self) -> str:
        """The compact spelling, so one requirement has one text form."""
        parts: list[str] = []
        if self.min_sm:
            parts.append(f"sm{self.min_sm}+")
        if self.min_vram_gb:
            parts.append(f"vram{_render_gb(self.min_vram_gb)}g")
        if self.min_host_ram_gb:
            parts.append(f"ram{_render_gb(self.min_host_ram_gb)}g")
        if self.min_cuda:
            parts.append(f"cuda{self.min_cuda}+")
        if self.min_torch:
            parts.append(f"torch{self.min_torch}+")
        return ", ".join(parts)


# ── The compute-capability floor: DELETED HERE, READ FROM THE RULE ───────────
#
# `DTYPE_MIN_SM`, `FLOOR_LOSING_SPELLINGS` and `capability_floor_for_dtype`
# stood here. All three are gone (pgw#1621), and this note is what replaces
# them, because the reasoning was right and only the KEY was wrong.
#
# Paul, 2026-08-18, verbatim: *"the sm_x compute floor should fall out of the
# contract itself, rather than being a separate annotation. Only the VRAM
# requirement needs a separate annotation, because it's not clear, based on the
# contract, how much VRAM is needed."* That ruling is unchanged and now holds
# LITERALLY: `capability_floor_sm` is a field ON the ratified rule document,
# and :func:`capability_floor_for_rule` is the one reader.
#
# What the dtype-keyed table could not do, and why it had to go:
#
#   * IT COULD NOT TELL ITS OWN TWO ZEROS APART. "measured: this needs no
#     special silicon" and "never heard of this spelling" were the same return
#     value, and the second is silent and permissive — it prices a lane by
#     ACCIDENT. `capability_floor_for_rule` REFUSES an unknown handle instead.
#   * IT WAS KEYED ON A SPELLING, so it needed a `FLOOR_LOSING_SPELLINGS` guard
#     to stop `float4_e2m1fn_x2` (the packed-pair container type, which EXISTS
#     in torch where `float4_e2m1fn` does not) from resolving to floor 0 and
#     putting an nvfp4 lane on Ampere. There is no dtype string to mis-spell any
#     more; the handle is `cozy.nvfp4-flat@1` and it carries 100.
#   * IT NEEDED A ROW PER ggml BLOCK QUANT (`q4_k`, `q5_k`, …) whose honest
#     answer was 0, written out explicitly only so an absent row could not be
#     mistaken for a measured one. A GGUF rule states its own 0.
#
# The numbers still mean what they meant — the capability at which the
# arithmetic exists in hardware, in tensorhub's bare spelling (sm_89 -> 89):
# sm70 fp16 tensor cores (Volta), sm75 int8 (Turing), sm80 bf16 + int4
# (Ampere), sm89 fp8 e4m3/e5m2 (Ada), sm100 nvfp4 / mx-fp4 (Blackwell). They
# now live in tensorfs `spec/v2/rules/*.json`, one per rule, and this repo has
# no second copy.
#
# STILL NOT A KERNEL LIST, and it must never grow into one (Paul, same ruling):
# a specific attention kernel (sage2, flash-*) is a SERVE-RECIPE fallback arm
# with its own degrade path, not a placement floor.



class LayoutRequirements(msgspec.Struct, frozen=True, kw_only=True,
                         omit_defaults=True):
    """What EXECUTING one declared contract needs of the machine, at both levels of one vocabulary."""

    minimum: Any = None
    recommended: Any = None

    def declared(self) -> bool:
        return bool(self.min_terms().declared()
                    or self.recommended_terms().declared())

    def min_terms(self) -> RequirementTerms:
        return self.minimum if isinstance(
            self.minimum, RequirementTerms) else RequirementTerms()

    def recommended_terms(self) -> RequirementTerms:
        return self.recommended if isinstance(
            self.recommended, RequirementTerms) else RequirementTerms()

    def render(self) -> str:
        """The MINIMUM's compact spelling."""
        return self.min_terms().render()

    def manifest_row(self) -> dict[str, Any]:
        """Only DECLARED axes reach the manifest, per term AND per level — an undeclared axis must not arrive at the hub as a zero it could read as a floor of none."""
        row: dict[str, Any] = dict(self.min_terms().declared_terms())
        recommended = self.recommended_terms().declared_terms()
        if recommended:
            row["recommended"] = recommended
        return row


def _render_gb(value: float) -> str:
    return f"{value:g}"


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def term_meets(term: str, candidate: Any, floor: Any) -> bool:
    """Is `candidate` at least `floor` for this term? One comparator per KIND, chosen by name — never one per term."""
    if term in ("min_cuda", "min_torch"):
        return _version_tuple(str(candidate)) >= _version_tuple(str(floor))
    return float(candidate) >= float(floor)


def _min_sm_value(value: object, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LayoutDeclarationError(
            f"{where}: min_sm must be an int compute-capability code "
            f"(89, 90, 100, ...), got {type(value).__name__}")
    if value <= 0 or value > 999:
        raise LayoutDeclarationError(
            f"{where}: min_sm={value} is not a compute-capability code. Use "
            "tensorhub's spelling — sm_89 is 89, sm_100 is 100. There is no "
            "'no floor' value: omit the requirement instead, which leaves the "
            "axis UNDECLARED and unevaluated.")
    return value


def _gb_value(term: str, value: object, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LayoutDeclarationError(
            f"{where}: {term} must be a number of GB, got "
            f"{type(value).__name__}")
    out = float(value)
    if not out > 0 or out != out or out in (float("inf"), float("-inf")):
        raise LayoutDeclarationError(
            f"{where}: {term}={value!r} is not a floor. There is no 'no "
            "floor' value: omit the term, which leaves the axis UNDECLARED "
            "and unevaluated.")
    return out


def _version_value(term: str, value: object, *, where: str) -> str:
    text = str(value).strip() if isinstance(value, (str, int, float)) else ""
    if isinstance(value, bool) or not text:
        raise LayoutDeclarationError(
            f"{where}: {term} must be a dotted version ('12.8', '2.9'), got "
            f"{value!r}")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", text):
        raise LayoutDeclarationError(
            f"{where}: {term}={value!r} is not a dotted version. Write the "
            "version the runtime reports ('12.8', '2.9'), with no operator "
            "and no suffix.")
    return text


def _term_value(term: str, value: object, *, where: str) -> Any:
    if term == "min_sm":
        return _min_sm_value(value, where=where)
    if term in ("min_vram_gb", "min_host_ram_gb"):
        return _gb_value(term, value, where=where)
    return _version_value(term, value, where=where)


def _unknown_term_refusal(named: list[str], *, where: str) -> str:
    unbuilt = [t for t in named
               if t in _UNBUILT_TERMS or t.removeprefix("min_") in _UNBUILT_TERMS]
    tail = ""
    if unbuilt:
        tail = (f" {unbuilt} is named in the ruling and deliberately NOT "
                "built: there is no runtime kernel-capability probe in this "
                "worker, so it would be a floor with no fact behind it.")
    return (f"{where}: unknown requirement term(s) {named}. This SDK "
            f"understands {list(KNOWN_REQUIREMENT_TERMS)}; an ignored "
            f"requirement is one that silently does not hold.{tail}")


def parse_requirement_terms(
    value: object, *, where: str,
) -> RequirementTerms:
    """One LEVEL's term bag, from the compact list, a mapping, or the struct."""
    if isinstance(value, RequirementTerms):
        value = value.declared_terms()
    if isinstance(value, dict):
        named = sorted(str(k) for k in value)
        unknown = [t for t in named if t not in KNOWN_REQUIREMENT_TERMS]
        if unknown:
            raise LayoutDeclarationError(
                _unknown_term_refusal(unknown, where=where))
        if not named:
            raise LayoutDeclarationError(
                f"{where}: requirement mapping declares no axis; omit the "
                "entry rather than declaring an empty one.")
        return RequirementTerms(**{
            term: _term_value(term, value[term], where=where)
            for term in named})
    if not isinstance(value, str):
        raise LayoutDeclarationError(
            f"{where}: a requirement is the compact form 'sm100+', a "
            f"LayoutRequirements, a RequirementTerms, or a mapping — got "
            f"{type(value).__name__}")
    terms = [t.strip() for t in value.split(",")]
    if not any(terms) or any(not t for t in terms):
        raise LayoutDeclarationError(
            f"{where}: {value!r} is not a requirement. The compact form is a "
            f"comma-separated term list, e.g. {_COMPACT_EXAMPLES}.")
    parsed: dict[str, Any] = {}
    for term in terms:
        for name, pattern in _TERM_PATTERNS:
            match = pattern.match(term)
            if match is None:
                continue
            if name in parsed:
                raise LayoutDeclarationError(
                    f"{where}: {value!r} states {name} twice")
            parsed[name] = _term_value(name, (
                int(match.group(1)) if name == "min_sm"
                else float(match.group(1)) if name.endswith("_gb")
                else match.group(1)), where=where)
            break
        else:
            raise LayoutDeclarationError(
                f"{where}: unknown requirement term {term!r}. The terms this "
                f"SDK understands are {_COMPACT_EXAMPLES}; kernel "
                "requirements are named in the ruling but not built, and an "
                "ignored requirement is one that silently does not hold.")
    return RequirementTerms(**parsed)


def parse_layout_requirements(
    value: object, *, where: str,
) -> LayoutRequirements:
    """Dual form at two levels: the compact term list IS the minimum."""
    minimum: object = None
    recommended: object = None
    if isinstance(value, LayoutRequirements):
        minimum, recommended = value.minimum, value.recommended
        if minimum is None and recommended is None:
            raise LayoutDeclarationError(
                f"{where}: LayoutRequirements() declares no axis. A "
                "requirement that requires nothing is not a declaration — "
                "omit the entry.")
    elif isinstance(value, dict) and any(k in REQUIREMENT_LEVELS for k in value):
        stray = sorted(str(k) for k in value if k not in REQUIREMENT_LEVELS)
        if stray:
            raise LayoutDeclarationError(
                f"{where}: {stray} sit beside {list(REQUIREMENT_LEVELS)} in "
                "one mapping. A bare term list IS the minimum; naming a level "
                "means naming every term under one.")
        minimum = value.get("minimum")
        recommended = value.get("recommended")
    else:
        minimum = value

    min_terms = (RequirementTerms() if minimum is None
                 else parse_requirement_terms(
                     minimum, where=f"{where}: minimum"))
    rec_terms = (RequirementTerms() if recommended is None
                 else parse_requirement_terms(
                     recommended, where=f"{where}: recommended"))
    if not min_terms.declared() and not rec_terms.declared():
        raise LayoutDeclarationError(
            f"{where}: the requirement declares no axis at either level; omit "
            "the entry rather than declaring an empty one.")

    if min_terms.min_host_ram_gb:
        raise LayoutDeclarationError(
            f"{where}: min_host_ram_gb is declarable as RECOMMENDED only "
            "(Paul, 2026-07-11: RunPod GPU pods cannot select or guarantee "
            "host RAM, so a minimum is unenforceable theater). Move it: "
            "recommended='ram64g'.")

    floor = min_terms.declared_terms()
    for term, value_ in rec_terms.declared_terms().items():
        if term in floor and not term_meets(term, value_, floor[term]):
            raise LayoutDeclarationError(
                f"{where}: recommended {term}={value_} is below minimum "
                f"{term}={floor[term]}. A recommendation below the floor is a "
                "contradiction, not a preference.")
    return LayoutRequirements(
        minimum=min_terms if min_terms.declared() else None,
        recommended=rec_terms if rec_terms.declared() else None)


def normalize_layout_requirements(
    requirements: object, *, where: str, accepted: Iterable[str],
) -> dict[str, LayoutRequirements]:
    """`Slot(layout_requirements=...)` -> `{handle: LayoutRequirements}`."""
    if not isinstance(requirements, dict):
        raise LayoutDeclarationError(
            f"{where}: layout_requirements= must be a mapping of contract "
            f"handle -> requirement, got {type(requirements).__name__}")
    if not requirements:
        raise LayoutDeclarationError(
            f"{where}: layout_requirements={{}} declares nothing. Omit it to "
            "leave every axis UNDECLARED; an empty mapping is not a statement "
            "that this slot's contracts run anywhere.")
    accepted_set = set(accepted)
    out: dict[str, LayoutRequirements] = {}
    for raw_handle, raw_value in requirements.items():
        handle = validate_layout_handle(
            raw_handle, where=f"{where}: layout_requirements")
        if handle not in accepted_set:
            raise LayoutDeclarationError(
                f"{where}: layout_requirements[{handle!r}] guards a contract "
                f"this slot does not accept. Its declared set is "
                f"{sorted(accepted_set)} — add the handle to layouts= or drop "
                "the requirement; a requirement over nothing is never checked."
            )
        out[handle] = parse_layout_requirements(
            raw_value, where=f"{where}: layout_requirements[{handle!r}]")
    return out
