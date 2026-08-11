"""th#1580 A4 (§1.30 declaration 2+3): a decoder declares WHICH TENSOR-LAYOUT
CONTRACT it implements, beside the code that implements it.

The tensor-layout contract (th#1721; was "the artifact contract") is how
tensors exist ON DISK — byte packing, scale layout, swizzle, key-naming
convention, file topology — named by a descriptor handle
``<producer>.<format>@<major>``. Its sibling, the tensor-binding contract
(pgw#857, ``docs/endpoint-authoring.md``), is how a tensor is ADDRESSED at
load: bound by name, or baked as a literal.

The endpoint half of §1.30's compatibility intersection is DERIVED at image
build (``gen_worker.discovery.execution_lanes``) from these markers. A
hand-maintained capability list is a second trusted string — the defect class
th#1580 exists to remove — so there is no list: the declaration is a property
of the decoder function, and it ships or fails to ship with it.

The vocabulary lives in tensorhub's ``internal/tensorlayout`` (A2: contracts
are CODE). This module carries only the handles a decoder may name and refuses
anything else; it does not re-specify the descriptors.

**No exclusion marker exists, deliberately** (A4 corollary, Paul 2026-08-04).
Exclusions are DERIVED from declared traits — ``composes_lora`` crossed with a
function's ``lora_bucket`` is computable at build — or they do not exist.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, TypeVar

import msgspec

from .execution_lanes import known_execution_lane_bodies

# The registered handles, transcribed from tensorhub's seeded registry. A
# decoder naming anything else fails the BUILD: an unregistered contract is
# OPAQUE (A2), and a decoder cannot unilaterally register one.
CONTRACT_PLAIN_BF16 = "plain.bf16@1"
CONTRACT_COZY_FP8_ROWWISE = "cozy.fp8-rowwise@1"
CONTRACT_NUNCHAKU_V1 = "nunchaku.v1@1"
CONTRACT_COZY_SVDQ_NVFP4_LR8 = "cozy.svdq-nvfp4-lr8@1"
CONTRACT_BFL_NVFP4_PRESWIZZLED = "bfl.nvfp4-preswizzled@1"
# th#1803: transformers' FineGrainedFP8 / DeepSeek-style 128x128 block scales.
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


class ContractDecoder(msgspec.Struct, frozen=True, kw_only=True):
    """One decoder's declaration: the contract it decodes, and the lane BODIES
    the decoded units execute as. The execution axis (eager/compiled) is NOT
    declared here — the platform owns it, and the derivation crosses these
    bodies with the lane table's own execution support."""

    contract: str
    decoder: str  # "module:qualname" — the function carrying the marker
    serves: tuple[str, ...]  # lane body tokens, e.g. "svdq-fp4-w4a4"
    composes_lora: bool
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
    why: str = "",
) -> Callable[[F], F]:
    """Mark a decode entrypoint as implementing ``contract``.

    Stackable: one function may implement several contracts (``decode_linear``
    implements both the nunchaku layout and te#148's quantized-branch major).
    """

    def deco(fn: F) -> F:
        dec = ContractDecoder(
            contract=contract,
            decoder=f"{fn.__module__}:{fn.__qualname__}",
            serves=tuple(serves),
            composes_lora=bool(composes_lora),
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


# ── §1.33 / pgw#1143: the DEMAND side of the same vocabulary ──────────────────
#
# `@implements_contract` above is the SUPPLY-adjacent census — "which decoders
# does this IMAGE contain". The DEMAND is what `Slot(layouts=...)` declares:
# "what does this slot's code need in order to run". Two facts, one vocabulary,
# so the handle grammar and the registration refusal are shared verbatim rather
# than transcribed twice.
#
# The SDK emits HANDLES and never digests: descriptors are Go, in tensorhub
# (th#1580 A2). The hub resolves handle -> `Contract.Digest()` at MANIFEST
# INGEST against its own registry and stores both; a handle its registry does
# not know fails the manifest there, not at rebind. So `KNOWN_CONTRACTS` is
# honestly what it is — a transcription that is allowed to be stale and is
# CHECKED, never authoritative.

#: The whole-tree key: this slot's demand for every component that has no
#: more specific declaration.
LAYOUT_KEY_ANY_COMPONENT = "*"


class LayoutDeclarationError(ValueError):
    """A `Slot(layouts=...)` declaration the SDK refuses where it is written."""


def validate_layout_handle(handle: object, *, where: str) -> str:
    """One declared handle, normalized, or a decoration-time refusal.

    The quant axis only. §1.33's rendered `"<topology>+<quant>"` pair needs a
    topology REGISTRY to compare against (th#1809 T3); until that exists a
    composite is a handle naming a vocabulary the platform cannot resolve, and
    exact-or-refused means refusing it rather than storing half a pair.
    """
    if not isinstance(handle, str):
        raise LayoutDeclarationError(
            f"{where}: layout handle must be a string, got "
            f"{type(handle).__name__}"
        )
    text = handle.strip()
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
    if text not in KNOWN_CONTRACTS:
        raise LayoutDeclarationError(
            f"{where}: contract {text!r} is not registered. "
            "Contracts are CODE (th#1580 A2): register it in tensorhub's "
            "internal/tensorlayout with a descriptor and a probe set "
            f"before a slot may demand it. Known: {', '.join(KNOWN_CONTRACTS)}"
        )
    return text


def normalize_layout_demand(
    layouts: object, *, where: str,
) -> dict[str, tuple[str, ...]]:
    """`Slot(layouts=...)` -> `{component_path: ordered handles}`.

    Ordering IS preference (§1.33 point 2), so the tuple is kept as written
    and never sorted. Component-path keys are validated against the DERIVED
    component tree separately, at decoration time, by the registry — this
    function owns only the shape and the vocabulary.
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
                f"{where}: layouts[{key!r}] must be an ORDERED tuple of "
                f"handles (order is preference), got "
                f"{type(raw_value).__name__}"
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
                    f"{where}: layouts[{key!r}] repeats {handle!r}; the "
                    "second position is unreachable"
                )
            handles.append(handle)
        out[key] = tuple(handles)
    return out
