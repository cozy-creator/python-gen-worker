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
    "FunctionExclusion",
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


class FunctionExclusion(msgspec.Struct, frozen=True, kw_only=True):
    execution_lane: str
    reason: str


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
    *,
    lora_bucket: int = 0,
) -> tuple[tuple[str, ...], tuple[FunctionExclusion, ...]]:
    """The image's derived set narrowed by one function's DECLARED traits.

    The only trait that narrows anything today is ``lora_bucket``: a function
    that takes runtime adapters cannot run on a lane whose decoder has no
    adapter branch (``w8a8_lora`` is branch-capable on exactly three lanes),
    and that is computable here — which is why A4's corollary refuses an
    exclusion marker. Nothing about owner PREFERENCE lives here; §1.31 layer 2
    owns ordering.
    """
    rank = _rank()
    keep: list[str] = []
    exclusions: dict[str, FunctionExclusion] = {}
    for contract in derived.contracts:
        for lane in contract.execution_lanes:
            if lora_bucket > 0 and not contract.composes_lora:
                exclusions.setdefault(
                    lane,
                    FunctionExclusion(
                        execution_lane=lane,
                        reason=(
                            f"lora_bucket={lora_bucket} needs runtime adapter "
                            f"composition; decoder {contract.decoder} for "
                            f"{contract.contract} has no adapter branch"
                        ),
                    ),
                )
                continue
            if lane not in keep:
                keep.append(lane)
    # A lane another contract serves WITH composition is not excluded.
    for lane in keep:
        exclusions.pop(lane, None)
    keep.sort(key=lambda lane: rank[lane])
    ordered = tuple(
        exclusions[lane] for lane in sorted(exclusions, key=lambda x: rank[x])
    )
    return tuple(keep), ordered


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
