"""Diagnostic worker functions shipped by the SDK (pgw#674).

The swap-latency harness (:mod:`gen_worker.benchmarks.swap_latency`) needs
a first-class delivery path onto serving-class pods — ie#546 recorded that
no such path existed (no sshd on pods). This module gives it one: an
ordinary ``@endpoint`` class an endpoint project re-exports, so the harness
is dispatched through the NORMAL request path — the same payload path
th#1198's admin benchmark runs use — and its rows come back as the job
result.

Usage (in an endpoint project, e.g. inference-endpoints)::

    from gen_worker.diagnostics import SwapLatencyDiagnostics  # noqa: F401

Discovery picks the class up like any other endpoint; publish it as its
own diagnostics release. Bind ``checkpoint``/``to`` at deploy time or per
request via ``selected_by`` — the trees come out of the hub CAS through
the worker's ordinary snapshot materialization, so the benchmark measures
the exact tiers production serving uses. Invoker policy (who may call it)
is hub catalog data, not SDK code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import msgspec

from .api.decorators import NoWarmup, Resources, endpoint, worker_function
from .api.slot import Slot
from .benchmarks import swap_latency as bench
from .request_context import RequestContext

__all__ = [
    "SwapLatencyDiagnostics",
    "SwapLatencyInput",
    "SwapLatencyOutput",
    "SwapLatencyRow",
]


class SwapLatencyInput(msgspec.Struct, frozen=True):
    """One benchmark request.

    ``checkpoint``/``to`` drive the slot picks (``selected_by``); the hub
    overlays the allowed-value enum. ``cases`` defaults to everything that
    needs only tree A — pass ``("swap",)`` with a distinct ``to`` pick for
    the component-diff swap case.
    """

    checkpoint: str
    to: str = ""
    cases: Tuple[str, ...] = ("load", "demote", "stage", "overlap")
    overlap_gb: float = 4.0

    def __post_init__(self) -> None:
        unknown = sorted(set(self.cases) - set(bench.ALL_CASES))
        if unknown:
            raise ValueError(
                f"unknown cases {unknown}; known: {list(bench.ALL_CASES)}")
        if not self.cases:
            raise ValueError("cases must not be empty")


class SwapLatencyRow(msgspec.Struct, frozen=True):
    case: str
    label: str
    seconds: float
    bytes: int
    gib: float
    gib_per_s: float
    extra_json: str = ""


class SwapLatencyOutput(msgspec.Struct, frozen=True):
    rows: List[SwapLatencyRow]
    # Component-first plan A -> B (content-address diff) — always computed
    # when both trees are bound, even without the measured swap case.
    differing_components: Tuple[str, ...] = ()
    shared_components: Tuple[str, ...] = ()


def _to_row(row: bench.Row) -> SwapLatencyRow:
    d = row.as_dict()
    return SwapLatencyRow(
        case=str(d["case"]),
        label=str(d["label"]),
        seconds=float(d["seconds"]),
        bytes=int(d["bytes"]),
        gib=float(d["gib"]),
        gib_per_s=float(d["gib_per_s"]),
        extra_json=json.dumps(row.extra, sort_keys=True) if row.extra else "",
    )


@endpoint(
    models={
        "checkpoint": Slot(str, selected_by="checkpoint", root=True),
        "to": Slot(str, selected_by="to"),
    },
    resources=Resources(gpu=True),
    warmup=NoWarmup(
        "diagnostic endpoint: the handler IS a multi-minute benchmark; a "
        "synthesized boot warmup would run it against unbound trees"
    ),
)
class SwapLatencyDiagnostics:
    """Pod-side physical swap benchmark as an ordinary worker function."""

    def setup(self, checkpoint: str, to: str) -> None:
        self._checkpoint = Path(checkpoint)
        self._to = Path(to) if to else None

    @worker_function()
    def swap_latency(
        self, ctx: RequestContext, payload: SwapLatencyInput,
    ) -> SwapLatencyOutput:
        to: Optional[Path] = self._to
        if to is not None and to == self._checkpoint:
            to_effective: Optional[Path] = None
        else:
            to_effective = to
        cases = tuple(payload.cases)
        if "swap" in cases and to_effective is None:
            raise ValueError(
                "the swap case needs a 'to' pick distinct from 'checkpoint'")
        rows = bench.run_cases(
            cases,
            checkpoint=self._checkpoint,
            to=to_effective,
            overlap_gb=payload.overlap_gb,
            echo=True,  # rows also land in the pod log for scrapes
        )
        differing: Tuple[str, ...] = ()
        shared: Tuple[str, ...] = ()
        if to_effective is not None:
            diff, same = bench.swap_plan(self._checkpoint, to_effective)
            differing, shared = tuple(diff), tuple(same)
        return SwapLatencyOutput(
            rows=[_to_row(r) for r in rows],
            differing_components=differing,
            shared_components=shared,
        )
