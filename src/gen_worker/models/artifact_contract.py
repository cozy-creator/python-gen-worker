"""th#1580 A4 (§1.30 declaration 2+3): a decoder declares WHICH ARTIFACT
CONTRACT it implements, beside the code that implements it.

The endpoint half of §1.30's compatibility intersection is DERIVED at image
build (``gen_worker.discovery.execution_lanes``) from these markers. A
hand-maintained capability list is a second trusted string — the defect class
th#1580 exists to remove — so there is no list: the declaration is a property
of the decoder function, and it ships or fails to ship with it.

The vocabulary lives in tensorhub's ``internal/artifactcontract`` (A2:
contracts are CODE). This module carries only the handles a decoder may name
and refuses anything else; it does not re-specify the descriptors.

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

KNOWN_CONTRACTS: tuple[str, ...] = (
    CONTRACT_PLAIN_BF16,
    CONTRACT_COZY_FP8_ROWWISE,
    CONTRACT_NUNCHAKU_V1,
    CONTRACT_COZY_SVDQ_NVFP4_LR8,
    CONTRACT_BFL_NVFP4_PRESWIZZLED,
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
MARKER = "__cozy_artifact_contracts__"

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
            "internal/artifactcontract with a descriptor and a probe set "
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
