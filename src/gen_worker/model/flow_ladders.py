"""The two flow-match ladder parameters the SDK's scheduler does not read yet.

pgw#1346 B3a, and this module exists for ONE measured reason:
``Qwen/Qwen-Image``'s published ``scheduler/scheduler_config.json`` declares
``shift_terminal: 0.02`` and ``time_shift_type: "exponential"``, and
:class:`~gen_worker.model.scheduler.FlowMatchEulerDiscrete` reads neither. A
declaration that carried ``shift_terminal`` in its block while the math ignored
it would be the worst of both: the digest would say the ladder was stretched
and every request would walk an unstretched one.

**Why a separate module and not two fields on the scheduler.** The scheduler
SET mechanism — ``SchedulerKind``, ``IMPLEMENTED``, and codegen's single
``scheduler()`` method — is being reshaped concurrently by pgw#1346's K10 lane,
and the SET is where this belongs once it exists. Nothing here touches that
surface: this is ladder ARITHMETIC over the same declared block, built from the
generated binding's own ``SCHEDULER_PARAMETERS``, so folding it into
``FlowMatchEulerDiscrete`` when K10 lands is a move, not a rewrite.

**One definition of the shared math, deliberately.** The unshifted ladder and
the two shifts already exist and are already differenced against diffusers
(pgw#1331). This class DELEGATES to them and adds only the terminal stretch, so
there is no second copy of the schedule to drift.

**What the B3a families measured about "explicit sigmas".** The W2 batch plan
scoped B3 expecting few-step and DMD lanes to need literal sigma ladders. They
do not. All three endpoints hand ``set_timesteps`` the SAME raw ladder
``linspace(1.0, 1/steps, steps)`` that ``schedule()`` already synthesizes —
qwen-image and z-image spell it exactly, and ERNIE spells
``linspace(1.0, 0.0, steps + 1)[:-1]``, which is the same points. So a distilled
lane differs by its step COUNT and its shift, never by a table of numbers, and
no explicit-sigma vocabulary is owed by these three families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Self

from .errors import ModelError, ModelRefusal
from .scheduler import FlowMatchEulerDiscrete, Schedule, SchedulerBlock

#: The two time-shift spellings ``FlowMatchEulerDiscreteScheduler`` accepts.
#: ``linear`` is READ and REFUSED rather than silently treated as exponential:
#: no checkpoint in this fleet publishes it, so implementing it would be math
#: with nothing measuring it — the same posture ``scheduler.py`` takes toward
#: the karras/exponential/beta sigma conversions.
TIME_SHIFT_TYPES: Final[tuple[str, ...]] = ("exponential", "linear")


def _terminal(block: SchedulerBlock) -> float | None:
    """Read ``shift_terminal``, which is legitimately ABSENT on most families.

    Absent is not zero: a stretch to 0.0 is the identity only when the ladder
    already ends there, and every family that omits the key ends at ``1/steps``
    shifted — so defaulting it to a number would change every ladder that never
    asked for one.
    """

    if "shift_terminal" not in block:
        return None
    value = block["shift_terminal"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(
            ModelRefusal.SCHEDULER_INVALID,
            f"scheduler parameter 'shift_terminal' must be a real number, got {value!r}",
        )
    terminal = float(value)
    if not 0.0 <= terminal < 1.0:
        raise ModelError(
            ModelRefusal.SCHEDULER_INVALID,
            f"shift_terminal must lie in [0, 1), got {terminal}",
        )
    return terminal


def _time_shift_type(block: SchedulerBlock) -> str:
    value = block.get("time_shift_type", "exponential")
    if not isinstance(value, str) or value not in TIME_SHIFT_TYPES:
        raise ModelError(
            ModelRefusal.SCHEDULER_INVALID,
            f"scheduler parameter 'time_shift_type' must be one of "
            f"{list(TIME_SHIFT_TYPES)!r}, got {value!r}",
        )
    return value


@dataclass(frozen=True, slots=True)
class FlowMatchLadder:
    """A flow-match schedule that also honours ``shift_terminal``.

    ``base`` carries every parameter the SDK's scheduler already parses and
    validates; this class adds the two the published configs carry and it does
    not. Composition rather than a subclass, because the point is that these
    two belong ON the base class as soon as the scheduler SET can carry them.
    """

    base: FlowMatchEulerDiscrete
    time_shift_type: str = "exponential"
    shift_terminal: float | None = None

    def __post_init__(self) -> None:
        if self.time_shift_type not in TIME_SHIFT_TYPES:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"time_shift_type must be one of {list(TIME_SHIFT_TYPES)!r}, "
                f"got {self.time_shift_type!r}",
            )
        if self.time_shift_type == "linear":
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                "linear time shifting is not implemented: no checkpoint this fleet "
                "serves publishes it, and an unmeasured second shift is how a ladder "
                "silently stops matching the pipeline it replaces",
            )
        if self.shift_terminal is not None and not 0.0 <= self.shift_terminal < 1.0:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"shift_terminal must lie in [0, 1), got {self.shift_terminal}",
            )

    @classmethod
    def from_block(cls, block: SchedulerBlock) -> Self:
        """Build one from a declaration's scheduler parameter block.

        The block is the generated binding's ``SCHEDULER_PARAMETERS``, so this
        reads the same document the export digest carries — a family cannot
        stretch its ladder without re-keying its artifacts.
        """

        return cls(
            base=FlowMatchEulerDiscrete.from_block(block),
            time_shift_type=_time_shift_type(block),
            shift_terminal=_terminal(block),
        )

    def mu(self, image_seq_len: int) -> float:
        """The shift exponent for one sequence length; the base's own."""

        return self.base.mu(image_seq_len)

    def stretch(self, sigmas: tuple[float, ...]) -> tuple[float, ...]:
        """Stretch a ladder so its last EVALUATED sigma is ``shift_terminal``.

        ``diffusers``' ``stretch_shift_to_terminal``, restated in plain floats:
        ``1 - (1 - t) / ((1 - t[-1]) / (1 - shift_terminal))``. It is not a
        clamp of the final point — every sigma moves, because the whole ladder
        is rescaled in ``1 - sigma`` space so the walk still starts at 1.0.

        Applied to the EVALUATED sigmas only. The terminal 0.0 that
        :class:`~gen_worker.model.scheduler.Schedule` appends is the point the
        last step LANDS on and is not one of them — stretching it would make
        the ladder stop short of the clean sample and quietly under-denoise
        every image.
        """

        if self.shift_terminal is None:
            return sigmas
        span = 1.0 - sigmas[-1]
        if span == 0.0:
            # A ONE-STEP ladder is the single sigma 1.0, so there is no span to
            # rescale and the transform is undefined. The reference does not
            # refuse it — it divides 0 by 0 and returns a NaN ladder, which
            # renders nothing and says nothing. Refused here with the reason,
            # because the one place the fleet reaches a one-step Qwen-Image
            # walk is BOOT WARM-UP, where the ladder is irrelevant (the pass
            # exists to trace a graph and its output is discarded) and where a
            # NaN would be indistinguishable from a real numerical failure.
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                "a terminal-stretched ladder needs at least two steps: a one-step "
                "ladder is the single sigma 1.0 and has no span to rescale",
            )
        scale = span / (1.0 - self.shift_terminal)
        return tuple(1.0 - (1.0 - sigma) / scale for sigma in sigmas)

    def schedule(self, steps: int, *, image_seq_len: int = 0) -> Schedule:
        """The resolved sigma ladder for one request.

        ``image_seq_len`` is the PACKED token count and is required under
        dynamic shifting, exactly as the base class requires it: defaulting it
        would serve every resolution the schedule of whichever one the default
        happened to name.
        """

        resolved = self.base.schedule(steps, image_seq_len=image_seq_len)
        if self.shift_terminal is None:
            return resolved
        return Schedule(
            sigmas=(*self.stretch(resolved.sigmas[:-1]), 0.0),
            num_train_timesteps=resolved.num_train_timesteps,
        )


__all__ = [
    "TIME_SHIFT_TYPES",
    "FlowMatchLadder",
]
