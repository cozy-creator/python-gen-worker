"""The release compiled-graph document: discovery's output, the store's row.

Stamped ONCE per release at publish time (pgw#1367 clause 3): per (lane,
graph) the cg-graph-v1 content hash, the observed ingress contract, and the
module-path provenance; plus the COMPILE STACK of the env the trace ran in
(pgw#1489: torch/triton/nvidia versions, read off the endpoint's uv.lock).
Serving pods ADOPT this document -- they never derive.

An endpoint with no lanes stamps an EXPLICIT empty document: eager-permanent
is stated, never implied by an absent row.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .graph_identity import (
    EnvIdentity,
    GraphIdentityError,
    is_graph_hash,
    require_stack,
)
from .ingress import CallIngress, IngressError
from .lane import LaneError, require_contract_ref, require_passes, require_targets

DOCUMENT_FORMAT = 2
_DOCUMENT_FIELDS = frozenset(("v", "stack", "lanes"))
_LANE_FIELDS = frozenset(
    ("contract", "targets", "graphs", "unobserved_targets", "passes")
)
_GRAPH_FIELDS = frozenset(("graph", "target", "ingress", "program"))


class DocumentError(ValueError):
    """A compiled-graph document is incomplete or noncanonical."""


@dataclass(frozen=True, slots=True)
class GraphRecord:
    """One discovered graph: content hash, provenance, ingress contract."""

    graph: str
    target: str
    ingress: CallIngress
    #: CAS digest of the SERIALIZED ExportedProgram for this graph specialization
    #: (Paul, 2026-08-20: the derive stores the whole traced graph, not just
    #: its hash -- "we only ever need to run trace() once" now holds
    #: literally, and the runtime miner downloads this blob and runs inductor
    #: instead of re-tracing author code). **The NAME is the contract with
    #: the mint lane**, which reads it as `PROGRAM_DIGEST_FIELD = "program"`
    #: (pgw#962) and raises `MissingProgramDigest` on an empty one. Empty
    #: only when the producer stored no blob. Portability is fenced by the
    #: document's own compile stack: an ExportedProgram is torch-coupled, and
    #: the stack pins torch.
    program: str = ""

    def __post_init__(self) -> None:
        if not is_graph_hash(self.graph):
            raise DocumentError(f"graph must be a cg-graph-v1 hash, got {self.graph!r}")
        if not isinstance(self.target, str) or not self.target.strip():
            raise DocumentError("graph record must name its target module path")
        if not isinstance(self.ingress, CallIngress):
            raise DocumentError("graph record must carry its CallIngress contract")
        if not isinstance(self.program, str):
            raise DocumentError("graph program digest must be a string")

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph,
            "target": self.target,
            "ingress": self.ingress.as_dict(),
            "program": self.program,
        }


@dataclass(frozen=True, slots=True)
class LaneGraphs:
    """One lane's observed graph set, unobserved targets stated.

    A lane IS its contract reference (Paul's main_v2 review ruling): the
    handle is the row key, and there is no separate name.

    ``passes`` (tcg#52) restates the transform passes that ran before these
    graphs were derived. It is not decoration: it is a ``cg-graph-v1``
    derivation input, so a serving boot whose ran-pass set differs from this
    row is adopting graphs for a module it does not have -- ``AdoptSession``
    refuses that at construction.
    """

    contract: str
    targets: tuple[str, ...]
    graphs: tuple[GraphRecord, ...]
    unobserved_targets: tuple[str, ...] = ()
    passes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            require_contract_ref(self.contract)
            require_targets(self.targets)
            object.__setattr__(self, "passes", require_passes(self.passes))
        except LaneError as exc:
            raise DocumentError(f"lane graphs restate an invalid lane: {exc}") from exc
        # pgw#1384: GRAPH ORDER IS SEMANTIC, not canonical. The serving
        # lane's hole list inherits document order and the miner mints holes
        # in that order, so the PRODUCER states the order -- the derive puts
        # the default-parameter graph specializations (what an all-defaults payload
        # exercises) FIRST. Determinism still holds: the producer's
        # enumeration is deterministic, so identical inputs still yield
        # identical bytes. Only duplicates are refused here.
        ordered = tuple(self.graphs)
        if len({record.graph for record in ordered}) != len(ordered):
            # Identical traces dedup by construction; a duplicate row here is
            # a producer bug, not a second graph.
            raise DocumentError(f"lane {self.contract!r} repeats a graph hash")
        for record in ordered:
            if record.target not in self.targets:
                raise DocumentError(
                    f"lane {self.contract!r} records graph for undeclared target "
                    f"{record.target!r}"
                )
        unobserved = tuple(sorted(set(self.unobserved_targets)))
        observed = {record.target for record in ordered}
        for path in unobserved:
            if path not in self.targets:
                raise DocumentError(
                    f"lane {self.contract!r} states unobserved target {path!r} it never declared"
                )
            if path in observed:
                raise DocumentError(
                    f"lane {self.contract!r} states {path!r} both observed and unobserved"
                )
        stated = observed | set(unobserved)
        missing = sorted(set(self.targets) - stated)
        if missing:
            raise DocumentError(
                f"lane {self.contract!r} says nothing about declared target {missing[0]!r}; "
                f"every target is either observed or stated unobserved"
            )
        object.__setattr__(self, "graphs", ordered)
        object.__setattr__(self, "unobserved_targets", unobserved)

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "targets": list(self.targets),
            "graphs": [record.as_dict() for record in self.graphs],
            "unobserved_targets": list(self.unobserved_targets),
            "passes": list(self.passes),
        }


@dataclass(frozen=True, slots=True)
class GraphSetDocument:
    """The static release metadata: every lane's graphs, plus the trace env.

    ``stack`` is the COMPILE STACK the derive ran under -- the torch/triton/
    nvidia rows of the endpoint's resolved lockfile, verbatim. Joined with a
    concrete sm it is the artifact env identity (:meth:`env`), and it is the
    whole env half of the key: no other package can reach codegen (pgw#1489).
    An empty ``lanes`` tuple is the explicit eager-permanent marker.
    """

    stack: tuple[tuple[str, str], ...]
    lanes: tuple[LaneGraphs, ...] = field(default=())

    def __post_init__(self) -> None:
        try:
            # Validated by the ONE definition of a stack, so a document and an
            # artifact position can never disagree about what a stack is.
            object.__setattr__(self, "stack", require_stack(self.stack))
        except GraphIdentityError as exc:
            raise DocumentError(f"document states an invalid compile stack: {exc}") from exc
        ordered = tuple(sorted(self.lanes, key=lambda lane: lane.contract))
        if len({lane.contract for lane in ordered}) != len(ordered):
            raise DocumentError("document repeats a lane contract")
        object.__setattr__(self, "lanes", ordered)

    def env(self, sm: str) -> EnvIdentity:
        """This document's env identity on a card: (compile stack, sm)."""

        try:
            return EnvIdentity(stack=self.stack, sm=sm)
        except GraphIdentityError as exc:
            raise DocumentError(str(exc)) from exc

    @property
    def eager_permanent(self) -> bool:
        return not self.lanes

    def as_dict(self) -> dict[str, Any]:
        return {
            "v": DOCUMENT_FORMAT,
            "stack": [list(row) for row in self.stack],
            "lanes": [lane.as_dict() for lane in self.lanes],
        }

    def encode(self) -> bytes:
        """The sole byte serialization; equality of documents is equality of bytes."""

        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")

    def digest(self) -> str:
        return hashlib.sha256(self.encode()).hexdigest()

    @classmethod
    def decode(cls, raw: bytes | Mapping[str, Any]) -> GraphSetDocument:
        if isinstance(raw, (bytes, bytearray)):
            try:
                parsed: Any = json.loads(raw.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DocumentError(f"document bytes are not canonical JSON: {exc}") from exc
        else:
            parsed = raw
        if not isinstance(parsed, Mapping) or set(parsed) != _DOCUMENT_FIELDS:
            raise DocumentError(
                f"document fields must be exactly {sorted(_DOCUMENT_FIELDS)!r}"
            )
        if parsed.get("v") != DOCUMENT_FORMAT:
            raise DocumentError(f"document v must be {DOCUMENT_FORMAT}")
        raw_lanes = parsed.get("lanes")
        if not isinstance(raw_lanes, list):
            raise DocumentError("document lanes must be an array")
        lanes: list[LaneGraphs] = []
        for raw_lane in raw_lanes:
            if not isinstance(raw_lane, Mapping) or set(raw_lane) != _LANE_FIELDS:
                raise DocumentError(
                    f"lane fields must be exactly {sorted(_LANE_FIELDS)!r}"
                )
            raw_graphs = raw_lane.get("graphs")
            raw_targets = raw_lane.get("targets")
            raw_unobserved = raw_lane.get("unobserved_targets")
            raw_passes = raw_lane.get("passes")
            if (
                not isinstance(raw_graphs, list)
                or not isinstance(raw_targets, list)
                or not isinstance(raw_unobserved, list)
                or not isinstance(raw_passes, list)
            ):
                raise DocumentError("lane arrays are malformed")
            records: list[GraphRecord] = []
            for raw_record in raw_graphs:
                if not isinstance(raw_record, Mapping) or set(raw_record) != _GRAPH_FIELDS:
                    raise DocumentError(
                        f"graph fields must be exactly {sorted(_GRAPH_FIELDS)!r}"
                    )
                raw_ingress = raw_record.get("ingress")
                if not isinstance(raw_ingress, Mapping):
                    raise DocumentError("graph ingress must be an object")
                try:
                    ingress = CallIngress.decode(raw_ingress)
                except IngressError as exc:
                    raise DocumentError(f"graph ingress is invalid: {exc}") from exc
                records.append(
                    GraphRecord(
                        graph=raw_record["graph"],
                        target=raw_record["target"],
                        ingress=ingress,
                        program=raw_record["program"],
                    )
                )
            lanes.append(
                LaneGraphs(
                    contract=raw_lane["contract"],
                    targets=tuple(raw_targets),
                    graphs=tuple(records),
                    unobserved_targets=tuple(raw_unobserved),
                    passes=tuple(raw_passes),
                )
            )
        raw_stack = parsed.get("stack")
        if not isinstance(raw_stack, list) or any(
            not isinstance(row, list) or len(row) != 2 for row in raw_stack
        ):
            raise DocumentError("document stack must be [name, version] rows")
        try:
            return cls(
                stack=tuple((row[0], row[1]) for row in raw_stack), lanes=tuple(lanes)
            )
        except GraphIdentityError as exc:  # pragma: no cover - stack re-validation
            raise DocumentError(str(exc)) from exc


__all__ = [
    "DOCUMENT_FORMAT",
    "DocumentError",
    "GraphRecord",
    "GraphSetDocument",
    "LaneGraphs",
]
