"""§1.30 declaration 2+3: a decoder declares WHICH TENSOR-LAYOUT CONTRACT it
implements, beside the code that implements it.

The tensor-layout contract is how tensors exist ON DISK — byte packing, scale
layout, swizzle, key-naming convention, file topology — named by a descriptor
handle ``<producer>.<format>@<major>``. Its sibling, the tensor-binding contract
(``docs/endpoint-authoring.md``), is how a tensor is ADDRESSED at load: bound by
name, or baked as a literal.

The endpoint half of §1.30's compatibility intersection is DERIVED at image
build (``gen_worker.discovery.execution_lanes``) from these markers. A
hand-maintained capability list would be a second trusted string, so there is no
list: the declaration is a property of the decoder function, and it ships or
fails to ship with it.

The vocabulary lives in tensorhub's ``internal/tensorlayout`` (A2: contracts
are CODE). This module carries only the handles a decoder may name and refuses
anything else; it does not re-specify the descriptors.

**No exclusion marker exists, deliberately** (A4 corollary).
Exclusions are DERIVED from declared traits — ``composes_lora`` crossed with a
function's ``lora_bucket`` is computable at build — or they do not exist.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, TypeVar

import msgspec

from .execution_lanes import known_execution_lane_bodies
from .file_layout import KNOWN_FILE_LAYOUTS

# The registered handles, transcribed from tensorhub's seeded registry. A
# decoder naming anything else fails the BUILD: an unregistered contract is
# OPAQUE (A2), and a decoder cannot unilaterally register one.
CONTRACT_PLAIN_BF16 = "plain.bf16@1"
CONTRACT_COZY_FP8_ROWWISE = "cozy.fp8-rowwise@1"
CONTRACT_NUNCHAKU_V1 = "nunchaku.v1@1"
CONTRACT_COZY_SVDQ_NVFP4_LR8 = "cozy.svdq-nvfp4-lr8@1"
CONTRACT_BFL_NVFP4_PRESWIZZLED = "bfl.nvfp4-preswizzled@1"
# transformers' FineGrainedFP8 / DeepSeek-style 128x128 block scales.
# NOT cozy.fp8-rowwise@1 — same element type and activation scheme, different
# scale leaf, rank and span (`models/hf_fp8_blockwise.py`).
CONTRACT_HF_FP8_BLOCKWISE = "hf.fp8-blockwise@1"

KNOWN_CONTRACTS: tuple[str, ...] = (
    CONTRACT_PLAIN_BF16,
    CONTRACT_COZY_FP8_ROWWISE,
    CONTRACT_NUNCHAKU_V1,
    CONTRACT_COZY_SVDQ_NVFP4_LR8,
    CONTRACT_BFL_NVFP4_PRESWIZZLED,
    CONTRACT_HF_FP8_BLOCKWISE,
)

_HANDLE_RE = re.compile(r"^([a-z0-9]+)\.([a-z0-9][a-z0-9._-]*)@([1-9][0-9]*)$")

# ── The DECODE DIMENSIONS (pgw#1245; th#1937 ratifies the vocabulary) ────────
#
# A handle names a byte FORMAT. It does not say which of that format's legal
# shapes a given decoder reads, and the shapes are exactly where decoders
# branch: `cozy.fp8-rowwise@1` is one handle whether or not the tree carries
# the optional `input_scale` leaf, and a decoder that ignores it serves
# different bytes than one that consumes it. So a declaration carries five
# axes, and th#1938's `resolve()` intersects a variant's derived contract
# against them rather than against the handle alone.
#
# Registered tokens only, on every axis. An unregistered token fails the
# BUILD for the same reason an unregistered handle does: a decoder that can
# mint its own vocabulary is back to being a trusted string.

# Element encoding of the quantized weights the decoder reads.
ELEMENT_BF16 = "bf16"
ELEMENT_FP16 = "fp16"
ELEMENT_FP32 = "fp32"
ELEMENT_FP8_E4M3 = "fp8_e4m3"
ELEMENT_NVFP4 = "nvfp4"
ELEMENT_INT4 = "int4"
KNOWN_ELEMENTS: tuple[str, ...] = (
    ELEMENT_BF16, ELEMENT_FP16, ELEMENT_FP32,
    ELEMENT_FP8_E4M3, ELEMENT_NVFP4, ELEMENT_INT4,
)

# Scale granularity. `none` is EXPLICIT: a dense decoder states that it reads
# no scale tensors, which is a fact, where an empty axis would be a silence.
SCALE_NONE = "none"
SCALE_PER_TENSOR = "per_tensor"
SCALE_PER_CHANNEL_OUT = "per_channel_out"
SCALE_STATIC_ACTIVATION = "static_activation"
SCALE_BLOCK_128X128 = "block_128x128"
SCALE_GROUP_16 = "group_16"
KNOWN_SCALES: tuple[str, ...] = (
    SCALE_NONE, SCALE_PER_TENSOR, SCALE_PER_CHANNEL_OUT,
    SCALE_STATIC_ACTIVATION, SCALE_BLOCK_128X128, SCALE_GROUP_16,
)

# FILE LAYOUT — which on-disk shape the decoder's ENTRY POINT can read.
# The tokens are th#1937's ruling and are IMPORTED from `models/file_layout.py`
# (pgw#1252), never transcribed: that module is the same one `convert/` publishes
# through, so the load side and the publish side cannot drift into two
# spellings of one axis. `KNOWN_FILE_LAYOUTS` is `multi-file` | `single-file`.
#
# `pre_sharded` / `shard_axis` are likewise publish-side dimensions. No decoder
# in this image branches on an SP-sharded tree and none can detect one, so the
# decode-set states nothing about them rather than carrying a claim no code
# backs: a pre-sharded variant must be refused by th#1938 against th#1937's
# derived `pre_sharded`, not by a declaration here.

# Structural bakes the decoder CONSUMES — tensor-set membership facts, not
# element facts. RULED tensor-set names (th#1937): a bake token is a registered
# tensor set, so this axis and the publish side name one thing once.
#
# `modulation_table` / `modulation_baked` are deliberately ABSENT: th#1937
# keeps that rule UNREGISTERED until te#195's file format exists, because
# "inventing a pattern that matches nothing is the failure mode, not the fix".
BAKE_ADALN_PROJECTIONS = "adaln_projections"
BAKE_LOW_RANK_BRANCH = "low_rank_branch"
KNOWN_BAKES: tuple[str, ...] = (
    BAKE_ADALN_PROJECTIONS,
    BAKE_LOW_RANK_BRANCH,
)

# KEY TOPOLOGY — which tensor-KEY convention the decoder's model class can
# ingest. **RULED VOCABULARY (th#1937 lane, 2026-08-14): these exact strings,
# no aliases.**
#
# This axis exists because of a measured failure (te#185 second stop):
# DiffSynth's MiniMaxH3DiT accepts the MINIMAX-NATIVE key set (535 keys, fused
# `blocks.N.attn.qkv_proj`) and every minimax-h3 artifact we hold is the
# DIFFUSERS repackaging (638 keys, split `transformer_blocks.N.attn.to_q/
# to_k/to_v`) — ONE key in common. It surfaced as `Cannot detect the model
# type` from an md5-over-key:shape lookup deep inside a detection helper,
# after a 71 GB fetch onto a rented 4xH100.
#
# **File topology cannot see it**: both are multi-file safetensors trees, so
# `file_layout` classifies them identically. Nor can the quant contract: both
# would be `plain.bf16@1`. The key convention is a third fact and it needs its
# own axis or the decode-set is a lie that dies at load.
#: `…attn.to_q|to_k|to_v` — the diffusers repackaging.
KEYS_DIFFUSERS_SPLIT_QKV = "diffusers.split-qkv@1"
#: `…attn.qkv_proj|to_qkv` — the upstream/native fused set.
KEYS_NATIVE_FUSED_QKV = "native.fused-qkv@1"
#: `*layers.N.self_attn.q_proj` / `*layers.N.attention.self.query`. RENAMED
#: from the provisional `transformers.native` by th#1937: the FILE-topology
#: registry already spends `transformers.native@1` on a different axis, and
#: one string meaning two things on two dimensions is the confusion this ends.
KEYS_TRANSFORMERS_SPLIT_QKV = "transformers.split-qkv@1"
KNOWN_KEY_TOPOLOGIES: tuple[str, ...] = (
    KEYS_DIFFUSERS_SPLIT_QKV,
    KEYS_NATIVE_FUSED_QKV,
    KEYS_TRANSFORMERS_SPLIT_QKV,
)
# The provisional `contract.native` is DECLINED (th#1937): "the descriptor
# fixes the keys, no repackaging exists" is a fact about the QUANT CONTRACT,
# and the `quant` dimension already carries it BY HANDLE. A decoder whose
# handle fixes its keys declares `key_topologies=()` — an axis it does not
# constrain, rather than a synonym for its own handle.

DECODE_AXES: tuple[str, ...] = (
    "elements", "scales", "key_topologies", "file_layouts", "bakes",
)


class DecodeDimensions(msgspec.Struct, frozen=True, kw_only=True):
    """What one decoder reads WITHIN a contract handle.

    Every axis is REQUIRED — there is no default, so a declaration cannot be
    written by omission. `elements` and `scales` must be non-empty: a decoder
    that states nothing there has declared nothing. `key_topologies`,
    `file_layouts` and `bakes` MAY be empty, and empty is a statement rather
    than a silence —
    "this decoder does not constrain the axis" (the tri-state's UNDECLARED
    rung), which is the honest answer for a decoder whose quant handle already
    fixes its keys, and the answer that makes a baked variant refuse.
    """

    elements: tuple[str, ...]
    scales: tuple[str, ...]
    key_topologies: tuple[str, ...]
    file_layouts: tuple[str, ...]
    bakes: tuple[str, ...]


def _axis(values: object, *, known: tuple[str, ...], axis: str,
          where: str, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(
            values, (tuple, list)):
        raise ValueError(
            f"{where}: decodes.{axis} must be a tuple of tokens, got "
            f"{type(values).__name__}")
    out: list[str] = []
    for item in values:
        if not isinstance(item, str) or item.strip() not in known:
            raise ValueError(
                f"{where}: decodes.{axis} token {item!r} is not registered; "
                f"valid: {', '.join(known)}")
        token = item.strip()
        if token in out:
            raise ValueError(
                f"{where}: decodes.{axis} repeats {token!r}; a set states "
                "each member once")
        out.append(token)
    if not out and not allow_empty:
        raise ValueError(
            f"{where}: decodes.{axis} is empty. A decoder that states nothing "
            f"on {axis} has declared nothing — the handle alone is what this "
            "mechanism replaces.")
    # Canonical order: the declaration is a SET, and two authors writing the
    # same set in different orders must produce the same derived bytes or the
    # image's decode-set digest is not deterministic.
    return tuple(sorted(out))


def _validate_dimensions(dims: object, *, where: str) -> DecodeDimensions:
    if not isinstance(dims, DecodeDimensions):
        raise ValueError(
            f"{where}: decodes= must be a DecodeDimensions, got "
            f"{type(dims).__name__}")
    return DecodeDimensions(
        elements=_axis(dims.elements, known=KNOWN_ELEMENTS,
                       axis="elements", where=where),
        scales=_axis(dims.scales, known=KNOWN_SCALES,
                     axis="scales", where=where),
        key_topologies=_axis(dims.key_topologies,
                             known=KNOWN_KEY_TOPOLOGIES,
                             axis="key_topologies", where=where,
                             allow_empty=True),
        file_layouts=_axis(dims.file_layouts,
                           known=tuple(sorted(KNOWN_FILE_LAYOUTS)),
                           axis="file_layouts", where=where,
                           allow_empty=True),
        bakes=_axis(dims.bakes, known=KNOWN_BAKES, axis="bakes",
                    where=where, allow_empty=True),
    )


class ContractDecoder(msgspec.Struct, frozen=True, kw_only=True):
    """One decoder's declaration: the contract it decodes, WHICH SHAPES of it
    it decodes, and the lane BODIES the decoded units execute as. The
    execution axis (eager/compiled) is NOT declared here — the platform owns
    it, and the derivation crosses these bodies with the lane table's own
    execution support."""

    contract: str
    decoder: str  # "module:qualname" — the function carrying the marker
    serves: tuple[str, ...]  # lane body tokens, e.g. "svdq-fp4-w4a4"
    composes_lora: bool
    decodes: DecodeDimensions
    why: str = ""


# The declaration lives ON THE DECODER OBJECT, not in a module-level registry.
# That is the point: it cannot be present without the decoder, cannot survive
# the decoder being removed, and cannot be assembled anywhere else.
MARKER = "__cozy_tensor_layout_contracts__"

F = TypeVar("F", bound=Callable[..., Any])


def _validate(dec: ContractDecoder) -> None:
    if not _HANDLE_RE.match(dec.contract):
        raise ValueError(
            f"{dec.decoder}: {dec.contract!r} is not a contract handle "
            "(want ns.name@N)"
        )
    if dec.contract not in KNOWN_CONTRACTS:
        raise ValueError(
            f"{dec.decoder}: contract {dec.contract!r} is not registered. "
            "Contracts are CODE (th#1580 A2): register it in tensorhub's "
            "internal/tensorlayout with a descriptor and a probe set "
            "before a decoder may claim it."
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


def implements_contract(
    *,
    contract: str,
    serves: Iterable[str],
    composes_lora: bool,
    decodes: DecodeDimensions,
    why: str = "",
) -> Callable[[F], F]:
    """Mark a decode entrypoint as implementing ``contract``.

    ``decodes`` is REQUIRED and has no default: a declaration that names a
    handle and stops is the incomplete declaration pgw#1245 exists to remove,
    and a default would let one be written by omission.

    Stackable: one function may implement several contracts (``decode_linear``
    implements both the nunchaku layout and the quantized-branch major).
    """

    def deco(fn: F) -> F:
        where = f"{fn.__module__}:{fn.__qualname__}"
        dec = ContractDecoder(
            contract=contract,
            decoder=where,
            serves=tuple(serves),
            composes_lora=bool(composes_lora),
            decodes=_validate_dimensions(decodes, where=where),
            why=why,
        )
        _validate(dec)
        prior: tuple[ContractDecoder, ...] = getattr(fn, MARKER, ())
        for existing in prior:
            if existing.contract == dec.contract and existing != dec:
                raise ValueError(
                    f"{dec.decoder}: conflicting declarations for {dec.contract}"
                )
        setattr(fn, MARKER, prior + (dec,))
        return fn

    return deco


def contract_decoders_of(obj: Any) -> tuple[ContractDecoder, ...]:
    """The declarations carried by one object, or ``()``."""
    marked = getattr(obj, MARKER, ())
    if not isinstance(marked, tuple):
        return ()
    return tuple(d for d in marked if isinstance(d, ContractDecoder))


class UnregisteredDecodePath(msgspec.Struct, frozen=True, kw_only=True):
    """A decoder that reads real bytes NO registered contract covers.

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
# `@implements_contract` above is the SUPPLY-adjacent census — "which decoders
# does this IMAGE contain". The DEMAND is what `Slot(layouts=...)` declares:
# "what does this slot's code need in order to run". Two facts, one vocabulary,
# so the handle grammar and the registration refusal are shared verbatim rather
# than transcribed twice.
#
# The SDK emits HANDLES and never digests: descriptors are Go, in tensorhub
# (A2). The hub resolves handle -> `Contract.Digest()` at MANIFEST
# INGEST against its own registry and stores both; a handle its registry does
# not know fails the manifest there, not at rebind. So `KNOWN_CONTRACTS` is
# honestly what it is — a transcription that is allowed to be stale and is
# CHECKED, never authoritative.

#: The every-component key: this slot's demand for every component that has no
#: more specific declaration. The map is one level deep, keyed by component.
LAYOUT_KEY_ANY_COMPONENT = "*"

# ── The TWO AXES (§1.33) ─────────────────────────────────────────────────────
#
# TOPOLOGY is SHALLOW — keys/nesting/file layout, "essentially just different
# keys". It constrains the LOADER, and a conversion between two topologies is
# pure renaming, bit-lossless. QUANT is DEEP — "what the weights ARE, which
# requires different runtime code/kernels". It constrains the endpoint's
# KERNELS.
#
# They are two registries with two digests, compared FIELD-WISE and never as
# one string: a composite handle cannot express "topology differs, quant
# matches ⇒ CONVERTIBLE", which is the whole ladder.
AXIS_TOPOLOGY = "topology"
AXIS_QUANT = "quant"
LAYOUT_AXES: tuple[str, ...] = (AXIS_TOPOLOGY, AXIS_QUANT)

#: An axis a demand declares itself AGNOSTIC on. A declaration, never an
#: inference: an axis nobody stated is UNDECLARED (``None``) and is not
#: evaluated, which is a different fact.
LAYOUT_AXIS_ANY = "any"

# The topology axis, transcribed from code we already run: tensorhub's
# `catalog/layout_contract.go` (library_name x file_layout) and
# training-endpoints' `conversion/comfyui.py`
# `_SPLIT_COMPONENT_MAP`. Same transcription posture as KNOWN_CONTRACTS above —
# allowed to be stale, CHECKED at the hub's manifest ingest, never authoritative.
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
    """The transcribed handles of ONE axis. Unknown axis is a refusal, not an
    empty tuple — an empty answer would read as "nothing is registered"."""
    if axis == AXIS_QUANT:
        return KNOWN_CONTRACTS
    if axis == AXIS_TOPOLOGY:
        return KNOWN_TOPOLOGY_CONTRACTS
    raise LayoutDeclarationError(
        f"unknown layout axis {axis!r}; the axes are {list(LAYOUT_AXES)}")


def validate_layout_handle(
    handle: object, *, where: str, axis: str = AXIS_QUANT,
) -> str:
    """One declared handle on ONE axis, normalized, or a refusal.

    `Slot(layouts=...)` declares the QUANT axis alone: §1.33's rendered
    `"<topology>+<quant>"` pair needs the hub's topology REGISTRY to resolve
    against, and until that exists a composite in the manifest is half a pair
    stored as if it were exact. The SDK-internal converter registry names the
    topology axis explicitly (`axis=AXIS_TOPOLOGY`) — that vocabulary is real
    here and inert hub-side.
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
            f"{where}: {text!r} names a <topology>+<quant> pair. The topology "
            "axis has no registry yet (th#1809 T3) — declare the quant handle "
            "alone until it does; a pair the hub cannot resolve field-wise is "
            "not exact."
        )
    if not _HANDLE_RE.match(text):
        raise LayoutDeclarationError(
            f"{where}: {text!r} is not a contract handle (want ns.name@N)"
        )
    if text not in known:
        raise LayoutDeclarationError(
            f"{where}: {axis} contract {text!r} is not registered. "
            "Contracts are CODE (th#1580 A2): register it in tensorhub's "
            "internal/tensorlayout with a descriptor and a probe set "
            f"before a slot may demand it. Known: {', '.join(known)}"
        )
    return text


def normalize_layout_demand(
    layouts: object, *, where: str,
) -> dict[str, tuple[str, ...]]:
    """`Slot(layouts=...)` -> `{component_path: accepted handle SET}`.

    **The set is a compatibility FILTER; its order carries NO preference**
    (§1.33 point 2). Preference has exactly ONE authority — the author-configured
    ordered ladder of (GPU, lane) pairs — and "one filter, one order, never two
    orderings that can disagree" is the property that keeps it that way. So the
    handles are returned in CANONICAL order, not as written: two authors who
    declare the same set in different orders state the SAME demand, and no
    downstream reader can recover a preference from a position that never
    carried one.

    Component-path keys are validated against the DERIVED component tree
    separately, at decoration time, by the registry — this function owns only
    the shape and the vocabulary.
    """
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
    """One side of the two-sided vocabulary, as a PAIR of axes.

    Rendered `"<topology>+<quant>"` and compared FIELD-WISE — never as one
    string, because a whole-string compare cannot express "topology differs,
    quant matches", which is the CONVERTIBLE rung.

    Each axis is tri-state: a handle (declared), :data:`LAYOUT_AXIS_ANY`
    (declared agnostic), or ``None`` (UNDECLARED — not evaluated, and the
    verdict says so rather than guessing).
    """

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
    """`"<topology>+<quant>"`, or a bare handle meaning the QUANT axis alone.

    A bare handle is quant because that is what `Slot(layouts=...)` publishes
    (there is no topology registry yet), so one spelling has one meaning across
    the wire and this parser.
    """
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
    """One axis of a layout id: a handle, the explicit agnostic, or UNDECLARED."""
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


# ── A19: the declaration is MANDATORY, and absence is not the tri-state ──────
#
# `layouts=None` used to mean UNDECLARED — a legitimate third rung that made
# the hub's gate fall back to the image-wide decoder census. Measured
# fleet-wide, that rung was not a considered choice anywhere: it was the
# default, so "this slot has no opinion" and "nobody wrote the line" were one
# state and no refusal could tell them apart.
#
# A19 (Paul, 2026-08-15) cuts the default away. A model slot states what it
# consumes, or it states — with a reason — that no registered handle describes
# its bytes. Both are DECLARATIONS. What no longer exists is silence.
#
# The escape is not a loophole and not a default: it carries a reason string,
# it is refused when blank, and it is mutually exclusive with `layouts=`. It
# exists because real slots hold bytes the registry genuinely does not name —
# a tokenizer tree with no tensors at all, a GGUF quant axis th#1809 T3 has
# not registered, a vLLM compressed-tensors checkpoint with no descriptor.
# Inventing a handle for those would be the failure mode, not the fix.

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
        f"    Slot(Pipe, layouts={{\"*\": (\"{CONTRACT_PLAIN_BF16}\",)}})\n"
        f"Registered quant handles: {', '.join(KNOWN_CONTRACTS)}.\n"
        "If no registered handle names this slot's bytes (a tokenizer tree "
        "with no tensors, a GGUF quant axis, a compressed-tensors checkpoint), "
        "declare that explicitly and say why:\n"
        "    Slot(Pipe, layouts_undeclarable=\"gguf: the quant axis has no "
        "registered handle (th#1809 T3)\")"
    )


# ── The REQUIREMENTS axis (Paul, 2026-08-15) ────────────────────────────────
#
# "Certain contracts can only be executed efficiently if their
# hardware-requirements are met... other requirements might be kernels or torch
# versions."
#
# A contract handle describes BYTES AT REST. Executing those bytes is a
# separate fact, and it belongs to the code that executes them rather than to
# the artifact — which is exactly where tensorhub already put it:
# `contractspec.DecodeEntry.MinSM`, "the card floor this loader's kernels
# need. 0 = no floor. It is an EXECUTION fact and lives here rather than on the
# artifact contract, which describes bytes at rest only." So this axis speaks
# that field's name and semantics verbatim; it does not invent a second
# vocabulary for one fact.
#
# The requirement is PER (slot, handle), not per handle globally: one contract
# has different floors in different code. `cozy.fp8-rowwise@1` is sm89 through
# `_scaled_mm` per-tensor and sm90 rowwise (`models/w8a8.py`), and the 4-bit
# contracts need sm100 (`models/w4a4.py: W4A4_MIN_SM = 100`). A global table
# would have to pick one and be wrong for the other.
#
# NO DEFAULTS, and the same rule as every other axis: a declared requirement is
# checked, an undeclared one is not evaluated at all. `0`/absent is not "no
# floor asserted by the author" dressed as "runs anywhere" — it is the axis
# nobody answered.
#
# EXTENSIBLE, NOT BUILT: the compact grammar is a comma-separated term list so
# a kernel or torch-version term can be added without re-spelling every
# declaration. Only the compute-capability term exists today, and an unknown
# term is REFUSED by name rather than ignored — an ignored requirement is a
# requirement that silently does not hold.

#: The requirement terms this SDK understands. Growing this tuple is the whole
#: cost of adding an axis; nothing else parses a term.
KNOWN_REQUIREMENT_TERMS: tuple[str, ...] = ("min_sm",)

_SM_TERM_RE = re.compile(r"^sm([1-9][0-9]{1,2})\+$")


class LayoutRequirements(msgspec.Struct, frozen=True, kw_only=True):
    """What EXECUTING one declared contract needs of the machine.

    ``min_sm`` is the compute-capability floor, in tensorhub's own
    two-or-three digit spelling (``sm_89`` -> 89, ``sm_100`` -> 100), and is
    the value `contractspec.DecodeEntry.MinSM` compares a card against. ``0``
    means the axis was not declared and is therefore not evaluated.
    """

    min_sm: int = 0

    def declared(self) -> bool:
        return self.min_sm > 0

    def render(self) -> str:
        """The compact spelling, so one requirement has one text form."""
        return f"sm{self.min_sm}+" if self.min_sm else ""

    def manifest_row(self) -> dict[str, int]:
        """Only DECLARED axes reach the manifest — an undeclared axis must not
        arrive at the hub as a zero it could read as a floor of none."""
        return {"min_sm": self.min_sm} if self.min_sm else {}


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


def parse_layout_requirements(
    value: object, *, where: str,
) -> LayoutRequirements:
    """Dual form: the compact term list, or the structured object.

    ``"sm100+"`` and ``LayoutRequirements(min_sm=100)`` are the same
    declaration; the compact form is what an author writes and what
    :meth:`LayoutRequirements.render` produces, so a round trip is stable.
    """
    if isinstance(value, LayoutRequirements):
        if not value.declared():
            raise LayoutDeclarationError(
                f"{where}: LayoutRequirements() declares no axis. A "
                "requirement that requires nothing is not a declaration — "
                "omit the entry.")
        return LayoutRequirements(min_sm=_min_sm_value(
            value.min_sm, where=where))
    if isinstance(value, dict):
        unknown = sorted(set(value) - set(KNOWN_REQUIREMENT_TERMS))
        if unknown:
            raise LayoutDeclarationError(
                f"{where}: unknown requirement term(s) {unknown}. This SDK "
                f"understands {list(KNOWN_REQUIREMENT_TERMS)}; kernel and "
                "torch-version requirements are named in the ruling but not "
                "built, and an ignored requirement is one that silently does "
                "not hold.")
        if "min_sm" not in value:
            raise LayoutDeclarationError(
                f"{where}: requirement mapping declares no axis; omit the "
                "entry rather than declaring an empty one.")
        return LayoutRequirements(
            min_sm=_min_sm_value(value["min_sm"], where=where))
    if not isinstance(value, str):
        raise LayoutDeclarationError(
            f"{where}: a requirement is the compact form 'sm100+', a "
            f"LayoutRequirements, or a mapping — got {type(value).__name__}")
    terms = [t.strip() for t in value.split(",")]
    if not any(terms) or any(not t for t in terms):
        raise LayoutDeclarationError(
            f"{where}: {value!r} is not a requirement. The compact form is a "
            "comma-separated term list, e.g. 'sm100+'.")
    min_sm = 0
    for term in terms:
        match = _SM_TERM_RE.match(term)
        if match is None:
            raise LayoutDeclarationError(
                f"{where}: unknown requirement term {term!r}. The only term "
                "this SDK understands is the compute-capability floor "
                "('sm89+', 'sm90+', 'sm100+'); kernel and torch-version "
                "requirements are named in the ruling but not built, and an "
                "ignored requirement is one that silently does not hold.")
        if min_sm:
            raise LayoutDeclarationError(
                f"{where}: {value!r} states the compute-capability floor "
                "twice")
        min_sm = _min_sm_value(int(match.group(1)), where=where)
    return LayoutRequirements(min_sm=min_sm)


def normalize_layout_requirements(
    requirements: object, *, where: str, accepted: Iterable[str],
) -> dict[str, LayoutRequirements]:
    """`Slot(layout_requirements=...)` -> `{handle: LayoutRequirements}`.

    Keyed by HANDLE, not by component path: the floor is a property of the
    code that decodes that contract, and the same decoder serves every
    component the slot accepts it for.

    A key naming a handle this slot does not accept is a REFUSAL rather than a
    dead entry — it is a requirement guarding nothing, and the shapes that
    guard nothing are the ones this whole row exists to remove.
    """
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
