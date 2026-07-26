"""Diagnostic worker functions shipped by the SDK (pgw#674).

The swap-latency harness (:mod:`gen_worker.benchmarks.swap_latency`) needs
a first-class delivery path onto serving-class pods — ie#546 recorded that
no such path existed (no sshd on pods). This module gives it one: an
ordinary ``@endpoint`` class an endpoint project adopts, so the harness is
dispatched through the NORMAL request path — the same payload path
th#1198's admin benchmark runs use — and its rows come back as the job
result.

Usage (in an endpoint project, e.g. inference-endpoints)::

    from gen_worker.diagnostics import SwapLatencyDiagnostics

    class SwapLatency(SwapLatencyDiagnostics):
        '''This project's swap-latency diagnostics endpoint.'''

SUBCLASS it — do not merely re-export it. Discovery walks a package and
skips decorated objects whose ``__module__`` lies outside it (a bare
``from ... import X`` re-export is a third-party object by that rule and is
dropped — loudly, since pgw#689). A subclass is declared IN the endpoint
package, inherits the decorator's declaration, and is picked up like any
other endpoint. Publish it as its own diagnostics release.

Bind ``checkpoint``/``to`` at deploy time — the trees come out of the hub
CAS through the worker's ordinary snapshot materialization, so the
benchmark measures the exact tiers production serving uses. Invoker policy
(who may call it) is hub catalog data, not SDK code.

Failure contract (pgw#689 defect 3): this handler NEVER raises for a
benchmark-domain failure. A diagnostic that cannot measure has produced a
measurement RESULT — ``status="failed"`` naming the cause — not evidence
that the release is broken. Raising here fed the hub's fast-failure breaker
and tripped ``model_load_failure_streak`` on a live release, which recycled
a warm pod that had just spent 26 minutes minting. Sibling functions serve
tenant traffic on that same pod; this one never does.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import msgspec

from .api.decorators import NoWarmup, Resources, endpoint, worker_function
from .api.slot import Slot
from .benchmarks import swap_latency as bench
from .request_context import RequestContext

logger = logging.getLogger(__name__)

__all__ = [
    "SwapLatencyDiagnostics",
    "SwapLatencyInput",
    "SwapLatencyOutput",
    "SwapLatencyRow",
    "failure_outcome",
]


class SwapLatencyInput(msgspec.Struct, frozen=True):
    """One benchmark request.

    ``checkpoint``/``to`` drive the slot picks (``selected_by``); the hub
    overlays the allowed-value enum. Both default to ``""`` — the hub reads
    an empty pick as "no value supplied" and uses the DEPLOY binding, which
    is what this handler reads anyway (``setup()`` owns both trees). The
    defaults are load-bearing, not cosmetic: a family-less ``Slot(str)``
    cannot carry a ``curated`` policy (the hub cannot derive an architecture
    family for it), a ``fixed`` slot rejects every supplied value, and a
    REQUIRED msgspec field cannot be omitted — so without the defaults no
    payload exists that the hub accepts (pgw#689). Dropping them re-breaks
    dispatch; making a family-less string slot curatable hub-side is the
    alternative fix.

    ``cases`` defaults to everything that needs only tree A — pass
    ``("swap",)`` with a distinct ``to`` binding for the component-diff swap
    case.
    """

    checkpoint: str = ""
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
    # The measurement's own outcome (pgw#689 defect 3): "ok" | "refused"
    # (this host must not run weight benchmarks — off-pod / no CUDA) |
    # "failed" (it ran and could not measure). Never a raised fatal: a
    # broken diagnostic is not a broken release.
    status: str = "ok"
    error: str = ""
    traceback_text: str = ""


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


def failure_outcome(exc: BaseException) -> SwapLatencyOutput:
    """A failed measurement, as data. Loud in the pod log, benign on the
    wire — the job SUCCEEDS, so no release-health signal is emitted."""
    status = "refused" if isinstance(exc, bench.OffPodError) else "failed"
    logger.error(
        "swap-latency diagnostics %s — %s: %s (diagnostic outcome only; the "
        "release's serving functions are unaffected)",
        status, type(exc).__name__, exc, exc_info=exc,
    )
    return SwapLatencyOutput(
        rows=[],
        status=status,
        error=f"{type(exc).__name__}: {exc}",
        traceback_text="".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)),
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
    """Pod-side physical swap benchmark as an ordinary worker function.

    Subclass this inside an endpoint package to publish it — a bare
    re-export is skipped by discovery (see the module docstring)."""

    def setup(self, checkpoint: str, to: str) -> None:
        self._checkpoint = Path(checkpoint)
        self._to = Path(to) if to else None

    @worker_function()
    def swap_latency(
        self, ctx: RequestContext, payload: SwapLatencyInput,
    ) -> SwapLatencyOutput:
        try:
            return self._measure(payload)
        except Exception as exc:  # noqa: BLE001 — the failure contract above
            return failure_outcome(exc)

    def _measure(self, payload: SwapLatencyInput) -> SwapLatencyOutput:
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
