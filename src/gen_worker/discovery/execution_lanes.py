"""The endpoint-side execution-lane declaration: the image's DECODE-SET (``discovery.decode_set``) crossed with the runtime's execution options."""

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
    #: The QUANT RULE handle this decoder reads (pgw#1621). The field and the
    #: wire key below still say `contract` because that is what tensorhub's
    #: `manifestExecutionLanesBlock` reads and validates with
    #: `tensorlayout.Lookup`; the rename is the hub-side follow-up.
    rule: str
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
        rule=entry.rule,
        decoder=entry.decoder,
        execution_lanes=tuple(lanes),
        composes_lora=entry.composes_lora,
    )


def derive_execution_lanes(
    packages: tuple[str, ...] = DEFAULT_DECODER_PACKAGES,
    decode_set: DecodeSet | None = None,
) -> DerivedExecutionLanes:
    """Cross the image's decode-set with the runtime's execution options."""
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
    """The image's derived lane set, ranked, for one function."""
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
                "contract": c.rule,
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
