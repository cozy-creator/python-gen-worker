"""th#1580 A4: the endpoint-side execution-lane declaration, DERIVED at image
build from the decoders actually present in the image.

Three steps, none of them a list anybody maintains:

1. import every module of the decoder packages and collect the
   ``@implements_contract`` markers that survived the import;
2. cross each declared lane BODY with the execution options the runtime table
   says that body supports (``models.execution_lanes``) — the platform owns
   eager/compiled, so a decoder never declares it;
3. subtract nothing. Exclusions are derived per FUNCTION from declared traits
   (A4 corollary: no exclusion marker in v1).

A module that fails to import is EXCLUDED WITH ITS REASON, never silently
dropped: an image whose svdq decoder cannot import must not claim the svdq
lane, and it must be able to say why it does not.
"""

from __future__ import annotations

import importlib
import pkgutil

import msgspec

from gen_worker.models.tensor_layout_contract import (
    ContractDecoder,
    contract_decoders_of,
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

DEFAULT_DECODER_PACKAGES: tuple[str, ...] = ("gen_worker.models",)


class ExcludedDecoderModule(msgspec.Struct, frozen=True, kw_only=True):
    module: str
    reason: str


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


def _collect(
    packages: tuple[str, ...],
) -> tuple[tuple[ContractDecoder, ...], tuple[ExcludedDecoderModule, ...]]:
    """Import every decoder module and harvest the markers that survived.

    Import is the whole test: a module that raises contributes NO declaration
    and one excluded-with-reason record. There is no third state where a
    decoder is claimed but absent.
    """
    declared: dict[tuple[str, str], ContractDecoder] = {}
    excluded: list[ExcludedDecoderModule] = []

    def harvest(module: object) -> None:
        for attr in sorted(dir(module)):
            for dec in contract_decoders_of(getattr(module, attr, None)):
                declared[(dec.contract, dec.decoder)] = dec

    for package in packages:
        try:
            pkg = importlib.import_module(package)
        except Exception as exc:
            excluded.append(
                ExcludedDecoderModule(
                    module=package, reason=f"{type(exc).__name__}: {exc}"
                )
            )
            continue
        harvest(pkg)
        paths = getattr(pkg, "__path__", None)
        if not paths:
            continue
        for info in sorted(pkgutil.iter_modules(paths), key=lambda m: m.name):
            name = f"{package}.{info.name}"
            try:
                harvest(importlib.import_module(name))
            except Exception as exc:
                excluded.append(
                    ExcludedDecoderModule(
                        module=name, reason=f"{type(exc).__name__}: {exc}"
                    )
                )
    return tuple(declared[k] for k in sorted(declared)), tuple(excluded)


def _derived_contract(dec: ContractDecoder) -> DerivedContract:
    rank = _rank()
    lanes: list[str] = []
    for body in dec.serves:
        for lane in _lanes_for_body(body):
            if lane not in lanes:
                lanes.append(lane)
    lanes.sort(key=lambda lane: rank[lane])
    return DerivedContract(
        contract=dec.contract,
        decoder=dec.decoder,
        execution_lanes=tuple(lanes),
        composes_lora=dec.composes_lora,
    )


def derive_execution_lanes(
    packages: tuple[str, ...] = DEFAULT_DECODER_PACKAGES,
) -> DerivedExecutionLanes:
    """Enumerate the contracts this image's decoders implement, crossed with
    the runtime's execution options."""
    declared, excluded = _collect(packages)
    contracts = tuple(_derived_contract(dec) for dec in declared)
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
        excluded_modules=excluded,
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


def manifest_block(derived: DerivedExecutionLanes) -> dict:
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
