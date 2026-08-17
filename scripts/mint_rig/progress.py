"""Waiting on PROGRESS, never on a clock.

The rule (Paul, standing): detect stuck work by progress toward a goal, never by
elapsed time. Raising a flaky timeout is not a fix. podguard's header states the
same rule for pod termination — *"Neither layer uses a fixed lifetime. Both kill
on liveness + progress-staleness"* — and this module is that rule for the
control-plane side of the same pod.

A :class:`Gate` polls a caller-supplied probe. Each poll returns an
:class:`Observation`: whether the goal is REACHED, whether the work FAILED, and
a *progress token* — any value whose change proves something moved. The gate
ends in exactly one of four ways:

  reached   the goal predicate said so
  failed    the probe saw a failure marker (a traceback, a refusal)
  stuck     the progress token did not change for `stall_ticks` consecutive
            polls — the ONLY negative verdict, and it names the token that
            stopped moving
  tripped   the spend rail (see :mod:`mint_rig.rail`) reached its dollar cap

There is no "timed out". A pod pulling a 3 GB image for forty minutes is
progressing (its REST state and its layer counters move); a pod whose token has
been byte-identical for twenty consecutive polls is stuck at minute two.

WHAT MAKES A GOOD TOKEN. It must advance for the SLOWEST legitimate work the
gate covers. For a boot that is the pod's REST record (status, ip, port
mappings, last status change). For a compile it is the log's byte length plus
its final line plus the artifact tree's byte size — an inductor pass that has
printed nothing for ten minutes is still writing objects into its cache, and a
token that reads only the log would call that stuck. Conversely a retry loop
that reprints the same line forever produces a CONSTANT token, which is exactly
the frozen-worker case podguard's `UpdatedAt` note describes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Observation:
    """One poll of a gate's probe."""

    #: The goal predicate. When true the gate returns immediately.
    reached: bool = False
    #: A terminal failure the probe recognised (a traceback, a refusal marker).
    failed: bool = False
    #: Anything whose change proves work moved. Compared with `!=`.
    token: object = None
    #: One line for the operator's console. Never load-bearing.
    note: str = ""


class Stuck(RuntimeError):
    """A gate's progress token stopped advancing."""

    def __init__(self, stage: str, token: object, ticks: int, note: str) -> None:
        super().__init__(
            f"pgw#1347 {stage}: no progress for {ticks} consecutive observations; "
            f"progress token has been {token!r} throughout. Last: {note}"
        )
        self.stage, self.token, self.ticks, self.note = stage, token, ticks, note


class Failed(RuntimeError):
    """A gate's probe recognised a terminal failure."""

    def __init__(self, stage: str, note: str) -> None:
        super().__init__(f"pgw#1347 {stage}: failed — {note}")
        self.stage, self.note = stage, note


@dataclass
class Gate:
    """Poll `probe` until it reaches its goal or its progress token freezes."""

    stage: str
    probe: Callable[[], Observation]
    #: Consecutive unchanged-token observations that mean STUCK. Not a duration:
    #: multiply by `tick_s` only when reading the console, never when reasoning
    #: about correctness.
    #:
    #: **Zero or less DISABLES the staleness rule**, for the gates whose phase
    #: genuinely emits no progress signal (pod bring-up — see
    #: :meth:`mint_rig.rail.Rail.check_sub`). Such a gate must be given a
    #: `rail_check`, or it would have no bound at all; :meth:`wait` refuses
    #: otherwise rather than looping forever.
    stall_ticks: int = 12
    tick_s: float = 15.0
    #: Called with every observation, for the console and for the row's trail.
    on_tick: Callable[[int, Observation], None] | None = None
    #: Raise if the spend rail has tripped. Injected so the gate never imports
    #: the rail and the rail never learns about gates.
    rail_check: Callable[[str], None] | None = None
    sleep: Callable[[float], None] = field(default=time.sleep)

    def wait(self) -> Observation:
        if self.stall_ticks <= 0 and self.rail_check is None:
            raise ValueError(
                f"pgw#1347 {self.stage}: a gate with the staleness rule disabled must carry a "
                "rail_check — otherwise it has no bound at all, which is worse than a timeout."
            )
        stale = 0
        last: object = _UNSET
        seen = 0
        note = "<no observation yet>"
        while True:
            if self.rail_check is not None:
                self.rail_check(self.stage)
            observation = self.probe()
            seen += 1
            note = observation.note or note
            if self.on_tick is not None:
                self.on_tick(seen, observation)
            if observation.reached:
                return observation
            if observation.failed:
                raise Failed(self.stage, observation.note)
            if last is _UNSET or observation.token != last:
                stale, last = 0, observation.token
            else:
                stale += 1
                if self.stall_ticks > 0 and stale >= self.stall_ticks:
                    raise Stuck(self.stage, last, stale, note)
            self.sleep(self.tick_s)


class _Unset:
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<unset>"


_UNSET = _Unset()
