"""HiDream-O1's FLASH schedule, as bare typed math (pgw#1346 B3b).

Its own module rather than a third class inside
:mod:`gen_worker.model.scheduler`, because it is not a member of that module's
closed vocabulary: :class:`~gen_worker.model.scheduler.SchedulerKind` names the
kinds a catalog declaration may DECLARE, and nothing declares this one yet. Why
not is worth stating precisely, because "not implemented" and "not declarable"
are different problems and only one of them is here:

* HiDream-O1's catalog entry is an EAGER ``ModelSpec`` (see
  :mod:`gen_worker.model.catalog.hidream_o1`), and the eager tier carries no
  ``Scheduler`` block at all — so there is nothing for a new kind to attach to;
* and the endpoint reaches THREE schedulers, chosen per request from the
  resolved recipe and the reference-image count. ``GraphModelSpec.scheduler`` is
  ONE ``Scheduler`` and codegen emits ONE ``scheduler()`` method, so declaring
  any one of the three would be declaring the wrong one two thirds of the time.
  That is pgw#1346 **K10** exactly — the sampler is a TUNED value and the
  declaration is single-valued — and K10's set surface is deliberately NOT
  touched here. Adding a ``SchedulerKind`` now would be adding vocabulary no
  declaration can use.

So this module is the MATH, landed and tested, ready for the declaration that
K10 unblocks. Nothing else in the SDK imports it yet.

**What the flash schedule is.** HiDream-O1's dev lane walks a hand-authored
28-entry integer ladder — not a formula, and not a function of the step count.
That is the single most surprising fact about this family and it is load-bearing
for the endpoint: ``num_inference_steps`` is decorative on the dev lane, which is
why the handler pins it to the resolved recipe's value rather than the caller's.
Both dev-lane scheduler classes (the "flash" one and DiffSynth's
``FlowMatchScheduler`` under ``special_case="dev"``) return this identical table,
so there is one ladder here and not two.

**Where it departs from flow-match Euler**, which is the reason it needs its own
step at all: after the Euler x0-estimate it RE-NOISES on the linear path,

    denoised = x - v * sigma
    x        = sigma_next * noise * scale[i] + (1 - sigma_next) * denoised

with ``noise`` clipped to a multiple of its own standard deviation. It is
therefore STOCHASTIC — a sampler, not a solver — and the noise is a caller-owned
input here rather than something this module draws. Same argument
``initial_latents`` makes for CPU-seeding: a receipt's seed has to mean the same
thing on two pods, and a scheduler that reaches for a global RNG makes that
impossible.

**The measurement instrument is pgw#1346 B2's, and it is nearly vacuous here —
deliberately.** B2's carry-forward is: never bound a torch-derived reference in
ULP, because torch's own CPU kernels disagree with each other. This ladder has no
torch-derived reference to bound: it is an integer table divided by 1000, so the
sigmas are exactly representable ratios computed in IEEE double and the table is
reproducible on every machine by construction. The tests assert that rather than
a tolerance — the same inversion B2 landed for ``EulerDiscrete``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from .errors import ModelError, ModelRefusal
from .scheduler import Schedule

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor


#: HiDream-O1's dev ladder, verbatim. Twenty-eight integer timesteps in the
#: model's own 0..1000 units, descending, ending at 8 rather than 0 — the clean
#: sample is reached by the terminal zero the schedule appends, not by the table.
FLASH_TIMESTEPS: Final[tuple[int, ...]] = (
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
    707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
)

#: The units the table is expressed in. A sigma is a timestep over this.
NUM_TRAIN_TIMESTEPS: Final = 1000


@dataclass(frozen=True, slots=True)
class HiDreamO1Flash:
    """The flash sampler: a fixed ladder, a re-noising step, and a noise clip.

    ``noise_scale_start``/``noise_scale_end`` are the endpoints of a LINEAR
    ramp across the ladder's 28 entries. The hidream-o1 endpoint passes the same
    resolved ``noise_scale`` for both, so in production the ramp is a constant
    vector — but the two fields are kept distinct because the upstream sampler's
    are, and collapsing them would make a recipe that ramps unexpressible.

    ``noise_clip_std`` clips each step's noise to that many multiples of its own
    measured standard deviation. Zero disables the clip, which is the upstream
    convention rather than a sentinel invented here.
    """

    noise_scale_start: float = 7.5
    noise_scale_end: float = 7.5
    noise_clip_std: float = 2.5

    def __post_init__(self) -> None:
        for name in ("noise_scale_start", "noise_scale_end", "noise_clip_std"):
            value = float(getattr(self, name))
            if value != value or value in (float("inf"), float("-inf")) or value < 0.0:
                raise ModelError(
                    ModelRefusal.SCHEDULER_INVALID,
                    f"{name} must be a finite non-negative number, got {value!r}",
                )

    @property
    def timesteps(self) -> tuple[int, ...]:
        """The ladder, which is a CONSTANT of this sampler.

        Exposed as a property rather than taking a step count, because taking
        one would imply it could honour it. It cannot: the upstream sampler's
        ``set_timesteps`` accepts ``num_inference_steps`` and ignores it.
        """

        return FLASH_TIMESTEPS

    @property
    def noise_scales(self) -> tuple[float, ...]:
        """The per-step noise amplitude: a linear ramp over the 28 entries."""

        span = len(FLASH_TIMESTEPS) - 1
        step = (self.noise_scale_end - self.noise_scale_start) / span
        return tuple(self.noise_scale_start + step * index for index in range(span + 1))

    def schedule(self, steps: int = len(FLASH_TIMESTEPS)) -> Schedule:
        """The resolved ladder. ``steps`` is CHECKED, never honoured.

        A caller asking for a different count is asking for something this
        sampler cannot do, and answering with 28 steps silently is how a
        request's declared cost stops describing its actual work. So it refuses
        and says what the ladder is — the same posture the endpoint takes when
        it pins the dev lane's step count and tells the caller it did.
        """

        if steps != len(FLASH_TIMESTEPS):
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"the flash ladder is a fixed {len(FLASH_TIMESTEPS)}-entry table and cannot "
                f"be resolved at {steps} steps; it is authored, not derived from a count",
            )
        sigmas = tuple(
            timestep / NUM_TRAIN_TIMESTEPS for timestep in FLASH_TIMESTEPS
        )
        return Schedule(sigmas=(*sigmas, 0.0), num_train_timesteps=NUM_TRAIN_TIMESTEPS)

    def clip_noise(self, noise: Tensor) -> Tensor:
        """Clip one step's noise to ``noise_clip_std`` of its own deviation.

        Measured from the tensor itself rather than assumed to be 1.0, which
        matters: the noise this sampler re-injects is drawn at the sample's own
        dtype, and a bf16 draw's realised deviation is not exactly one.
        """

        if self.noise_clip_std <= 0.0:
            return noise
        bound = self.noise_clip_std * float(noise.std())
        return noise.clamp(min=-bound, max=bound)

    def step(
        self,
        schedule: Schedule,
        index: int,
        model_output: Tensor,
        sample: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """One flash step: Euler to the x0 estimate, then re-noise.

        ``noise`` is supplied by the caller and clipped here. Written with
        tensor OPERATORS only, so this module imports no array library — the
        same discipline the rest of the scheduler surface keeps, because an
        adopt-only serve role holds it.
        """

        if not 0 <= index < len(schedule):
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"step {index} is outside this schedule's {len(schedule)} steps",
            )
        sigma = schedule.sigmas[index]
        sigma_next = schedule.sigmas[index + 1]
        denoised = sample - sigma * model_output
        scaled = self.noise_scales[index] * self.clip_noise(noise)
        return sigma_next * scaled + (1.0 - sigma_next) * denoised


__all__ = [
    "FLASH_TIMESTEPS",
    "NUM_TRAIN_TIMESTEPS",
    "HiDreamO1Flash",
]
