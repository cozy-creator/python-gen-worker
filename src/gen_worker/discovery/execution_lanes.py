"""The endpoint-side execution-lane declaration: the image's DECODE-SET
(``discovery.decode_set``) crossed with the runtime's execution options.

Two steps, neither of them a list anybody maintains:

1. take the derived decode-set — the ``@implements_contract`` markers that
   survived the import walk, with the dimensions each decoder reads;
2. cross each declared lane BODY with the execution options the runtime table
   says that body supports (``models.execution_lanes``) — the platform owns
   eager/compiled, so a decoder never declares it.

Subtract nothing. Exclusions are derived per FUNCTION from declared traits
(A4 corollary: no exclusion marker in v1).

ONE census feeds both blocks: what bytes the image can READ is the decode-set
(th#1938's third intersection) and what lanes it can RUN is this one. Two
facts, two renders, one import walk — never two censuses that can disagree.
"""

from __future__ import annotations

from typing import Any, Dict

import msgspec

from gen_worker.discovery.decode_set import (
    DEFAULT_DECODER_PACKAGES,
    DecodeSet,
    ExcludedDecoderModule,
    derive_decode_set,
)
from gen_worker.models.execution_lanes import (
    execution_lane_body_id,
    known_execution_lanes,
    parse_execution_lane,
)

# Stamped into the manifest and into the hub's release row. It names the
# MECHANISM, so an older image that predates it emits no block at all and is
# UNPROVEN hub-side — distinct from an image that derived an empty set.
DERIVATION = "gen_worker.discovery.execution_lanes@1"

__all__ = [
    "DERIVATION",
    "DEFAULT_DECODER_PACKAGES",
    "DerivedContract",
    "DerivedExecutionLanes",
    "ExcludedDecoderModule",
    "derive_execution_lanes",
    "execution_lanes_for_function",
    "manifest_block",
]


class DerivedContract(msgspec.Struct, frozen=True, kw_only=True):
    contract: str
    decoder: str
    execution_lanes: tuple[str, ...]
    composes_lora: bool


class DerivedExecutionLanes(msgspec.Struct, frozen=True, kw_only=True):
    derivation: str
    execution_lanes: tuple[str, ...]
    contracts: tuple[DerivedContract, ...]
    excluded_modules: tuple[ExcludedDecoderModule, ...]
    decode_set: DecodeSet


def _rank() -> dict[str, int]:
    return {lane: i for i, lane in enumerate(known_execution_lanes())}


def _lanes_for_body(body: str) -> tuple[str, ...]:
    """Every concrete lane id the runtime supports for one lane body. This is
    the cross with the execution options: the lane table is authoritative and
    the decoder never names eager/compiled."""
    out = []
    for lane_id in known_execution_lanes():
        if execution_lane_body_id(parse_execution_lane(lane_id)) == body:
            out.append(lane_id)
    return tuple(out)


def _derived_contract(entry: Any) -> DerivedContract:
    rank = _rank()
    lanes: list[str] = []
    for body in entry.serves:
        for lane in _lanes_for_body(body):
            if lane not in lanes:
                lanes.append(lane)
    lanes.sort(key=lambda lane: rank[lane])
    return DerivedContract(
        contract=entry.contract,
        decoder=entry.decoder,
        execution_lanes=tuple(lanes),
        composes_lora=entry.composes_lora,
    )


def derive_execution_lanes(
    packages: tuple[str, ...] = DEFAULT_DECODER_PACKAGES,
    decode_set: DecodeSet | None = None,
) -> DerivedExecutionLanes:
    """Cross the image's decode-set with the runtime's execution options.

    ``decode_set`` lets a caller that already derived one (the manifest build)
    pass it in rather than paying a second import walk for the same answer.
    """
    ds = decode_set if decode_set is not None else derive_decode_set(packages)
    contracts = tuple(_derived_contract(entry) for entry in ds.entries)
    rank = _rank()
    union: list[str] = []
    for contract in contracts:
        for lane in contract.execution_lanes:
            if lane not in union:
                union.append(lane)
    union.sort(key=lambda lane: rank[lane])
    return DerivedExecutionLanes(
        derivation=DERIVATION,
        execution_lanes=tuple(union),
        contracts=contracts,
        excluded_modules=ds.excluded_modules,
        decode_set=ds,
    )


def execution_lanes_for_function(
    derived: DerivedExecutionLanes,
) -> tuple[str, ...]:
    """The image's derived lane set, ranked, for one function.

    pgw#1599 (sweep, routed from the pgw#1548 archaeology lane): the
    ``lora_bucket`` parameter and the per-function EXCLUSION it computed are
    DELETED. ``lora_bucket`` was a DEAD AXIS — nothing in the tree ever set
    it non-zero (the sole production caller passed a literal 0), so the
    "function needs runtime adapter composition" narrowing could never fire,
    and its only effect was to make an always-empty exclusions list look like
    a live instrument. Two sibling sites carry the same dead axis and die in
    their OWN changes, deliberately not smuggled in here: torchcg still
    HASHES it into graph identity (``declaration.py``; a dead axis inside an
    identity hash is a live source of phantom cache misses — its own torchcg
    PR + vendor bump), and tensorhub still accepts it on the wire
    (``manifest_contract.go``).
    """
    rank = _rank()
    keep: list[str] = []
    for contract in derived.contracts:
        for lane in contract.execution_lanes:
            if lane not in keep:
                keep.append(lane)
    keep.sort(key=lambda lane: rank[lane])
    return tuple(keep)


def manifest_block(derived: DerivedExecutionLanes) -> Dict[str, Any]:
    """The ``[execution_lanes]`` block as it lands in endpoint.lock."""
    return {
        "derivation": derived.derivation,
        "lanes": list(derived.execution_lanes),
        "contracts": [
            {
                "contract": c.contract,
                "decoder": c.decoder,
                "lanes": list(c.execution_lanes),
                "composes_lora": c.composes_lora,
            }
            for c in derived.contracts
        ],
        "excluded_modules": [
            {"module": m.module, "reason": m.reason}
            for m in derived.excluded_modules
        ],
    }
