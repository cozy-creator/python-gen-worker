"""The spend rail — dollars, declared per invocation, with no default.

A rig that can start without a stated budget will eventually run without one.
So :class:`Rail` has no default cap: `Rail()` is a TypeError and
``Rail(max_usd=0)`` is refused. The caller states dollars because dollars are
what the operator actually has an opinion about; the rig converts them to a
deadline using the pod's OBSERVED hourly rate, which it only learns after the
create call answers.

**Why this is not the magic timeout the policy forbids.** A magic timeout is a
number picked to be "long enough" for work whose progress nobody is watching.
This is the opposite: progress is watched by :mod:`mint_rig.progress`, and the
rail is a hard ceiling on SPEND that exists so a stuck-detector bug cannot cost
unbounded money. It is derived, never guessed — change the card and the wall
changes with it — and tripping it is a recorded verdict (`rail_tripped`) rather
than a silent abort.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class RailTripped(RuntimeError):
    """The projected spend reached the caller's declared cap."""


@dataclass
class Rail:
    """A per-invocation dollar cap and the arithmetic that enforces it."""

    max_usd: float
    #: Fraction of the cap at which the rig stops STARTING new work. A rental
    #: torn down at exactly 100% has no budget left for the teardown round-trip
    #: or the artifact fetch, and an artifact left on a deleted pod is the run
    #: wasted for the sake of the last two cents.
    start_headroom: float = 0.85
    rate_per_hr: float = 0.0
    started_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.max_usd > 0:
            raise ValueError(
                "pgw#1347: a mint-rig invocation must declare a spend rail in "
                f"dollars; got max_usd={self.max_usd!r}. There is no default — "
                "a default budget is the number nobody chose."
            )
        if not 0.0 < self.start_headroom <= 1.0:
            raise ValueError(f"start_headroom must be in (0, 1]; got {self.start_headroom!r}")

    def observe_rate(self, rate_per_hr: float) -> None:
        """Record the rate the provider actually charged for this pod.

        Asked-vs-observed matters here as much as it does for the GPU model: a
        rail computed against a catalogue price would under-count a rental that
        came in at a different spot rate.
        """
        self.rate_per_hr = float(rate_per_hr)

    def clock_started(self, at: float | None = None) -> None:
        """Start the billed clock. Called when the create call answers."""
        self.started_at = time.time() if at is None else at

    def spent_usd(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        return self.rate_per_hr * max(0.0, now - self.started_at) / 3600.0

    def remaining_usd(self, now: float | None = None) -> float:
        return self.max_usd - self.spent_usd(now)

    @property
    def wall_seconds(self) -> float:
        """The cap expressed as seconds at the observed rate.

        Zero rate (a create response that carried no `costPerHr`) yields
        ``inf``: the rig must not invent a rate, and a missing rate is recorded
        on the row rather than turned into a fabricated deadline.
        """
        if self.rate_per_hr <= 0:
            return float("inf")
        return self.max_usd / self.rate_per_hr * 3600.0

    def may_start(self, stage: str, now: float | None = None) -> None:
        """Refuse to begin `stage` when there is no headroom left to finish it."""
        spent = self.spent_usd(now)
        if spent >= self.max_usd * self.start_headroom:
            raise RailTripped(
                f"pgw#1347 spend rail: ${spent:.2f} of ${self.max_usd:.2f} spent "
                f"({self.start_headroom:.0%} headroom exhausted) — refusing to start "
                f"{stage!r}. Tearing down."
            )

    def check(self, stage: str, now: float | None = None) -> None:
        """Trip when the cap itself is reached, mid-stage."""
        spent = self.spent_usd(now)
        if spent >= self.max_usd:
            raise RailTripped(
                f"pgw#1347 spend rail: ${spent:.2f} reached the declared "
                f"${self.max_usd:.2f} cap during {stage!r}. Tearing down."
            )

    def check_sub(self, stage: str, fraction: float, now: float | None = None) -> None:
        """A named sub-cap, as a fraction of the same declared rail.

        WHERE THIS IS THE ONLY HONEST BOUND. :mod:`mint_rig.progress` stops work
        that stopped progressing — but that rule needs a progress signal, and
        POD BRING-UP HAS NONE. Measured 2026-08-17 against `rest.runpod.io/v1`:
        a pod reports `desiredStatus: RUNNING` from the instant it is rented and
        exposes `publicIp`/`portMappings` only when its container actually
        starts, so a three-gigabyte image pull and a wedged host produce
        byte-identical records for as long as either lasts. Staleness cannot
        separate them, and a tick count that pretended to would be exactly the
        magic timeout this package refuses.

        So the bring-up bound is MONEY: a declared fraction of the operator's own
        rail. It is derived (the same 15% is four minutes on a 4090 and eight on
        a cheap A4000), it is stated, and tripping it is a `railed` verdict with
        the stage named — not a silent retry. Once the pod answers, real progress
        markers exist and :class:`~mint_rig.progress.Gate` takes over.
        """
        spent = self.spent_usd(now)
        cap = self.max_usd * fraction
        if spent >= cap:
            raise RailTripped(
                f"pgw#1347 {stage} budget: ${spent:.3f} spent against a "
                f"${cap:.3f} sub-cap ({fraction:.0%} of the ${self.max_usd:.2f} rail). "
                "Bring-up has no progress signal to be stuck on, so money is the "
                "bound. Tearing down."
            )
