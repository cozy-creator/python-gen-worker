"""Running a ``@job``: the run-once harness and the dispatch contract.

A job is submitted, not invoked. One dispatch is one function run: decode the
payload against the published schema, execute the body once, encode the result
struct, exit. **Nothing survives in-process between jobs by contract** — there
is no ``setup()``, no instance table and no cache here, because the lifecycle
is a FRESH CHILD per job on the existing parent-stream/child-CUDA split
(:mod:`gen_worker.procsplit`): the parent holds the hub stream, the child holds
CUDA and dies with the job. Disk caches (volume snapshots, HF caches) survive
and are the held-open-pod win; process state deliberately does not.

:class:`JobDispatch` / :class:`JobOutcome` are the wheel's dispatch contract —
the shape th#2050's scheduler head fills in and reads back. They are
transport-neutral on purpose: ``RunJob.function_name`` already routes by name
and ``JobResult`` is already the result envelope, so v1 needs no new proto
message, and anything the hub adds later lands on these two structs rather than
on a second execution path.

**Liveness is POSITION ADVANCE, never pulse and never duration** (th#1908).
:class:`ProgressWatch` is the guard for pgw#1287's class — a transfer loop that
reports nothing for its whole duration. A phase that does not advance inside
its budget cancels the context and, critically, **can never bank as a
success**: :func:`execute_job` raises
:class:`~gen_worker.api.errors.JobProgressStalledError` even if the body
returns normally afterwards. The hub stays the kill authority (deadline and
pod termination are hub-issued, th#2050); what this side owes is a position
the hub can read and a refusal to call a stalled run finished.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import msgspec

from .api.errors import JobProgressStalledError
from .registry import JobSpec

logger = logging.getLogger(__name__)

#: How long a phase may report the same position before the run is judged
#: stalled. Generous by design: this is not a latency bar, it is the boundary
#: past which "working" and "wedged" are indistinguishable from the outside.
DEFAULT_PHASE_BUDGET_S = 600.0

#: How often the watch samples the position. Sampling cost is a dict read.
_WATCH_POLL_S = 1.0


@dataclass(frozen=True)
class JobDispatch:
    """One submission handed to this worker.

    ``deadline_s`` is RECORDED, not enforced here. The wall cap is hub-issued
    through the provider API (external kill authority — a pod-side timer is
    the poweroff-no-ops anti-pattern), so arming a second timer in the child
    would put the same guarantee in two homes and let them disagree.
    """

    job_name: str
    payload: bytes
    job_id: str = ""
    attempt: int = 0
    deadline_s: Optional[float] = None
    org: str = ""
    invoker_id: str = ""
    capability_token: str = ""
    #: Per-phase progress budget for this run. The submitter may tighten it;
    #: it is never unbounded, because "unbounded" is what made pgw#1287's
    #: silent download indistinguishable from a healthy one.
    phase_budget_s: float = DEFAULT_PHASE_BUDGET_S


@dataclass(frozen=True)
class JobOutcome:
    """The result of one run-once execution.

    ``recycle_child`` is always True and is the run-once lifecycle stated as
    data: whoever drives this dispatch must not reuse the process for a second
    job. It is a field rather than a comment so the caller either honours it or
    visibly ignores it.
    """

    job_name: str
    job_id: str
    status: str                      # "succeeded" | "failed"
    result: bytes = b""
    failure: str = ""
    error_type: str = ""
    recycle_child: bool = True


class ProgressWatch:
    """Watch one context's monotonic position and judge a phase that stops.

    Not a heartbeat: a beat proves a thread is alive, which is exactly what a
    wedged transfer loop also proves. This reads the POSITION, so the only way
    to look healthy is to actually move.
    """

    def __init__(
        self,
        ctx: Any,
        *,
        budget_s: float = DEFAULT_PHASE_BUDGET_S,
        poll_s: Optional[float] = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ctx = ctx
        self._budget_s = float(budget_s)
        # A poll coarser than the budget is a guard that cannot fire: the body
        # finishes before the first sample and the stall is never seen. Sample
        # several times per budget, always.
        self._poll_s = (
            float(poll_s) if poll_s is not None
            else max(0.01, min(_WATCH_POLL_S, self._budget_s / 4.0))
        )
        self._now = now
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stalled: Optional[JobProgressStalledError] = None

    def __enter__(self) -> "ProgressWatch":
        if self._budget_s > 0:
            self._thread = threading.Thread(
                target=self._loop, name="job-progress-watch", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._poll_s * 3)

    def check(self) -> None:
        """Raise if a phase stalled. Called after the body returns, so a body
        that ignores cancellation still cannot report success."""
        if self.stalled is not None:
            raise self.stalled

    def _positions(self) -> Mapping[str, float]:
        return dict(getattr(self._ctx, "_progress_positions", {}) or {})

    def _loop(self) -> None:
        last_seen: Mapping[str, float] = self._positions()
        last_move = self._now()
        while not self._stop.wait(self._poll_s):
            current = self._positions()
            if current != last_seen:
                last_seen = current
                last_move = self._now()
                continue
            if self._now() - last_move < self._budget_s:
                continue
            phase = max(current, key=lambda k: current[k]) if current else ""
            self.stalled = JobProgressStalledError(
                phase, self._budget_s, current.get(phase) if current else None)
            logger.error("job progress stalled: %s", self.stalled)
            cancel = getattr(self._ctx, "_cancel", None)
            if callable(cancel):
                cancel()
            return


def decode_payload(spec: JobSpec, payload: bytes) -> Any:
    """Decode a dispatch payload against the job's PUBLISHED schema.

    The schema is the one the manifest published, so the struct the body
    receives is exactly the shape the submitter was validated against — the
    hub's ``job_payload_invalid`` refusal and this decode read the same
    declaration.
    """
    decoder: msgspec.msgpack.Decoder[Any] = msgspec.msgpack.Decoder(
        spec.payload_type)
    return decoder.decode(payload)


def _encode_result(spec: JobSpec, value: Any) -> bytes:
    if not isinstance(value, spec.output_type):
        raise TypeError(
            f"job {spec.name!r} returned {type(value).__name__}, but its "
            f"declared result type is {spec.output_type.__name__}"
        )
    return msgspec.msgpack.encode(value)


def _resolve(jobs: Mapping[str, JobSpec], name: str) -> JobSpec:
    spec = jobs.get(name)
    if spec is None:
        raise KeyError(
            f"unknown job {name!r} (this release declares: "
            f"{sorted(jobs) or 'none'})"
        )
    return spec


def _outcome(spec: JobSpec, dispatch: JobDispatch, result: bytes) -> JobOutcome:
    return JobOutcome(
        job_name=spec.name, job_id=dispatch.job_id,
        status="succeeded", result=result,
    )


def _failure(spec: JobSpec, dispatch: JobDispatch, exc: BaseException) -> JobOutcome:
    return JobOutcome(
        job_name=spec.name, job_id=dispatch.job_id, status="failed",
        failure=str(exc), error_type=type(exc).__name__,
    )


def _stamp_declaration(spec: JobSpec, ctx: Any) -> None:
    """Project the job's OWN ``publishes`` declaration onto its context.

    The ``JobSpec`` is the one home for the declaration — it comes from the
    decorator and rides the manifest row the hub mints the write grant off.
    The context flag is a PROJECTION of it, so ``execute_job`` sets it rather
    than trusting whoever built the context to have remembered: a dispatch
    head that forgets would hand a job a perfectly valid capability token and
    then have the SDK refuse its own publish, which reads as a broken feature
    rather than as the wiring bug it is. One home, stamped at the one place
    that holds both halves.

    The opposite direction is a REFUSAL, not a silent downgrade: a caller
    claiming publish authority for a job whose release never declared it is
    asserting an authority the hub did not mint a grant for, and quietly
    clearing the flag would hide that.
    """
    if bool(getattr(ctx, "_publishes", False)) and not spec.publishes:
        raise ValueError(
            f"job {spec.name!r}: the context was built with publishes=True but "
            "the job declares publishes=False. The declaration is the release's, "
            "not the caller's — the hub minted no write grant for this job, so "
            "the claim could only fail later and further from its cause."
        )
    ctx._publishes = bool(spec.publishes)


def execute_job(
    dispatch: JobDispatch,
    *,
    jobs: Mapping[str, JobSpec],
    ctx: Any,
    reraise: bool = False,
) -> JobOutcome:
    """Run ONE job to completion and return its outcome.

    The whole lifecycle, in order: resolve the name against this release's
    jobs, decode the payload against the published schema, arm the progress
    watch, call the body ONCE, encode the result, judge the watch. A body that
    raises is a TERMINAL failure — a deterministic re-run is money burnt — and
    the typed exception rides ``JobOutcome.error_type`` so the scheduler's
    infra-vs-body split is made from a type, not from prose.

    ``reraise=True`` propagates instead of banking a failed outcome; the CLI
    uses it so an author sees the real traceback.
    """
    spec = _resolve(jobs, dispatch.job_name)
    _stamp_declaration(spec, ctx)
    payload = decode_payload(spec, dispatch.payload)
    try:
        with ProgressWatch(ctx, budget_s=dispatch.phase_budget_s) as watch:
            value = spec.method(ctx, payload)
            if spec.is_async:
                value = asyncio.run(value)
            watch.check()
        return _outcome(spec, dispatch, _encode_result(spec, value))
    except BaseException as exc:
        if reraise:
            raise
        logger.exception("job %s failed", spec.name)
        return _failure(spec, dispatch, exc)


__all__ = [
    "DEFAULT_PHASE_BUDGET_S",
    "JobDispatch",
    "JobOutcome",
    "ProgressWatch",
    "decode_payload",
    "execute_job",
]
