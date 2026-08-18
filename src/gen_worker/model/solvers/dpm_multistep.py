"""``dpmsolver_multistep`` — DPM-Solver++ (2M), as bare typed math.

The fleet's most-selected sampler by a wide margin: ``dpmpp_2m_karras`` is
sd15's own default (``sd15_serve.Sd15Tuned.scheduler``) and
``dpmpp_2m_sde_karras`` is STAMPED on two live sdxl catalog entries
(``cyberrealistic-pony``, ``cyberrealistic-xl``, 30 steps). What "2M" means is
fixed by ``gen_worker.view.SAMPLERS`` and not by a family: second order,
multistep, ``final_sigmas_type="zero"``; the ``_karras`` suffix changes the
LADDER and nothing else, and the ``_sde`` prefix changes the ALGORITHM and
nothing else. Those two independent switches are why this module keeps the
ladder in :mod:`.ladders` instead of inlining it.

**The multistep state is the new thing here, and it is the reproducibility
risk.** ``euler_discrete`` is memoryless: step *i* reads only sigma *i* and
sigma *i+1*. A 2M step reads the PREVIOUS step's converted model output, so a
loop carries state, and diffusers carries it as mutable attributes on the
scheduler object (``self.model_outputs``, ``self.lower_order_nums``,
``self._step_index``). That shape is why a diffusers scheduler cannot be shared
across concurrent requests, and it is also why "does this reproduce" stops being
a question about one step.

This module's answer is that the state is a VALUE, not an attribute:
:class:`MultistepHistory` is frozen, :meth:`DpmSolverSchedule.begin` is the only
way to make the initial one, and :meth:`DpmSolverSchedule.step` returns the next
one alongside the sample. So:

* **initialization is total and constant** — ``begin()`` takes no arguments,
  reads no clock, no device and no environment, and produces an EMPTY history.
  Two pods therefore start identical by construction rather than by discipline;
* **there is no counter to share** — the step index is an argument, so two
  concurrent requests on one worker cannot desynchronize each other. The frozen
  ``Schedule`` classes in this package are shaped this way for the same reason,
  and multistep is where it stops being a stylistic preference;
* **the recursion is closed over reproducible inputs** — the history holds only
  tensors the caller's own denoiser produced, and every scalar this module folds
  into them comes from a ladder that is byte-stable across CPU kernels
  (:mod:`.precision`). A multistep recursion AMPLIFIES nothing it is not fed:
  measured, the loop propagates a ladder perturbation at gain ~1 (see
  ``tests/test_dit_solvers_pgw1346.py``), the same conditioning pgw#1346 B2
  measured for Euler.

**No array library.** The sample and the model output are tensor OPERANDS;
every scalar is narrowed to float32 before it meets one, because torch takes a
Python float at its full double value and a divisor right to 17 digits instead
of 7 makes every element of the result differ (B2, measured: 131072 of 131072).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Self

from ..errors import ModelError, ModelRefusal
from . import ladders
from .block import SchedulerBlock, choice, count, flag, only, real, refuse
from .precision import f32, round_half_even, sigma_table, truncate_to_int

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor


def _log(value: float) -> float:
    """float32 ``log``, with torch's answer at zero rather than Python's.

    ``math.log(0.0)`` raises; ``torch.log`` of a float32 zero is ``-inf``, and
    the terminal sigma of every reachable configuration IS zero. The infinity is
    load-bearing, not an edge case to clamp away: it makes ``h`` infinite, which
    makes ``exp(-h)`` exactly zero, which makes the final first-order update
    return the predicted clean sample exactly.
    """

    return -math.inf if value <= 0.0 else f32(math.log(value))


@dataclass(frozen=True, slots=True)
class MultistepHistory:
    """One request's multistep state, as an immutable VALUE.

    ``outputs`` holds the converted model outputs, OLDEST FIRST and at most
    ``solver_order`` of them — the same window diffusers keeps in
    ``self.model_outputs``, minus the ``None`` padding, because an empty tuple
    says "no history" without a sentinel that every reader has to test.

    ``taken`` counts steps already folded in, capped at ``solver_order``. It is
    diffusers' ``lower_order_nums`` and it exists because the first step of a
    second-order method has no previous output to difference against, so it must
    run first-order — the multistep "warm-up".
    """

    outputs: tuple[Tensor, ...] = ()
    taken: int = 0


@dataclass(frozen=True, slots=True)
class DpmSolverSchedule:
    """One request's resolved DPM-Solver++ ladder, and the step that walks it.

    ``sigmas`` has ``len(timesteps) + 1`` entries and terminates at the final
    sigma, which is ``0.0`` for every configuration this fleet reaches.

    ``init_noise_sigma`` is ``1.0`` and not the top of the ladder: DPM-Solver++
    starts from unit-variance noise and folds the scale into its first update,
    where ``euler_discrete`` starts from ``sigma_max``-scaled noise. Getting
    this backwards produces a washed-out or a saturated image with no error, so
    it is carried explicitly rather than derived by whoever writes the loop.
    """

    sigmas: tuple[float, ...]
    timesteps: tuple[float, ...]
    num_train_timesteps: int
    prediction_type: str
    algorithm_type: str
    solver_order: int
    solver_type: str
    lower_order_final: bool
    euler_at_final: bool
    final_sigmas_type: str
    #: Read sigmas as rectified-flow (``alpha = 1 - sigma``) rather than
    #: variance-exploding. A property of the LADDER, so it travels with it.
    flow: bool = False
    init_noise_sigma: float = 1.0

    def __post_init__(self) -> None:
        if len(self.sigmas) < 2:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                "a schedule needs at least one step, so at least two sigmas",
            )
        if len(self.timesteps) != len(self.sigmas) - 1:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"{len(self.timesteps)} timesteps do not walk a "
                f"{len(self.sigmas) - 1}-step sigma ladder",
            )

    def __len__(self) -> int:
        return len(self.sigmas) - 1

    def begin(self) -> MultistepHistory:
        """The initial multistep state. Constant, and the ONLY constructor.

        Takes nothing and reads nothing, which is the whole reproducibility
        argument for a stateful solver: there is no seed, no device and no
        wall-clock anywhere in the recursion's initial condition, so two pods
        given the same ladder and the same denoiser outputs walk the identical
        sequence of samples.
        """

        return MultistepHistory()

    # ---------------------------------------------------------------- sigmas

    def _checked(self, index: int) -> int:
        if not 0 <= index < len(self):
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"step {index} is outside this schedule's {len(self)} steps",
            )
        return index

    def _alpha_sigma(self, sigma: float) -> tuple[float, float]:
        """``(alpha_t, sigma_t)`` for one rung, at the reference's precision.

        The two readings of a sigma, and they are not interchangeable: a
        rectified flow has ``alpha = 1 - sigma`` (they sum to one), a
        variance-exploding diffusion has ``alpha = 1/sqrt(sigma^2+1)`` (they
        square to one). The same ladder read the wrong way is a different model.
        """

        if self.flow:
            return f32(1.0 - sigma), f32(sigma)
        alpha = f32(1.0 / f32(math.sqrt(f32(f32(sigma * sigma) + 1.0))))
        return alpha, f32(sigma * alpha)

    def _lambda(self, sigma: float) -> float:
        """The log-SNR half-step ``log(alpha_t) - log(sigma_t)``."""

        alpha, sigma_t = self._alpha_sigma(sigma)
        return f32(_log(alpha) - _log(sigma_t))

    # ------------------------------------------------------------ the output

    def convert(self, index: int, model_output: Tensor, sample: Tensor) -> Tensor:
        """The model's output as the CLEAN-SAMPLE prediction the solver needs.

        DPM-Solver++ discretizes an integral of the data-prediction model, so
        every objective is converted to ``x0`` here and the update rules below
        never see an objective. This is also the only place ``sample`` enters
        the conversion, which is why it is a separate method: a caller doing
        classifier-free guidance combines the raw model outputs BEFORE this
        point, never after.
        """

        sigma = self.sigmas[self._checked(index)]
        if self.prediction_type == "flow_prediction":
            return sample - f32(sigma) * model_output
        alpha, sigma_t = self._alpha_sigma(sigma)
        if self.prediction_type == "epsilon":
            return (sample - sigma_t * model_output) / alpha
        return f32(alpha) * sample - f32(sigma_t) * model_output

    # -------------------------------------------------------------- the step

    def step(
        self,
        index: int,
        model_output: Tensor,
        sample: Tensor,
        history: MultistepHistory,
        *,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, MultistepHistory]:
        """One multistep update. Returns the next sample AND the next history.

        The order selection is diffusers' verbatim and every clause is reachable
        from a real recipe, so none of them is dead:

        * the FIRST step is always first-order — there is nothing to difference;
        * the LAST step is first-order whenever the ladder terminates at zero,
          which is every configuration this fleet reaches (``final_sigmas_type``
          is ``"zero"`` in ``view.SAMPLERS``' own definition of ``dpmpp_2m``).
          So a 4-step distilled recipe runs first-order, second, second,
          first-order — treating "2M" as "always second order" would silently
          change every short render.

        Upstream's ``lower_order_second`` clause is deliberately absent: it only
        ever selects between the SECOND- and THIRD-order updates, and third order
        is refused here, so reproducing it would be a branch that cannot be
        taken. Checked against the reference rather than assumed — the
        differential test walks every reachable step count.

        ``noise`` is REQUIRED by the ``sde-`` algorithms and refused as absent
        rather than defaulted: this module cannot make a random tensor (it
        imports no array library) and inventing a zero would turn a stochastic
        sampler into a deterministic one that still called itself SDE.
        """

        self._checked(index)
        steps = len(self)
        stochastic = self.algorithm_type == "sde-dpmsolver++"
        if stochastic and noise is None:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"algorithm_type {self.algorithm_type!r} integrates a stochastic "
                "differential equation and needs one noise tensor per step; the caller "
                "owns the generator, so pass noise=",
            )
        if not stochastic and noise is not None:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"algorithm_type {self.algorithm_type!r} is deterministic and would "
                "IGNORE the noise it was given; a caller passing one has the wrong sampler",
            )

        last = (index == steps - 1) and (
            self.euler_at_final
            or (self.lower_order_final and steps < 15)
            or self.final_sigmas_type == "zero"
        )

        converted = self.convert(index, model_output, sample)
        outputs = (*history.outputs, converted)[-self.solver_order :]

        if self.solver_order == 1 or history.taken < 1 or last:
            stepped = self._first_order(index, converted, sample, noise)
        else:
            stepped = self._second_order(index, outputs, sample, noise)
        return stepped, MultistepHistory(
            outputs=outputs, taken=min(history.taken + 1, self.solver_order)
        )

    def _first_order(
        self, index: int, model_output: Tensor, sample: Tensor, noise: Tensor | None
    ) -> Tensor:
        """DPM-Solver++(1), which is DDIM with the data-prediction reading."""

        alpha_t, sigma_t = self._alpha_sigma(self.sigmas[index + 1])
        alpha_s, sigma_s = self._alpha_sigma(self.sigmas[index])
        h = f32(self._lambda(self.sigmas[index + 1]) - self._lambda(self.sigmas[index]))
        decay = f32(math.exp(f32(-h))) if h != math.inf else 0.0

        if self.algorithm_type == "dpmsolver++":
            return f32(sigma_t / sigma_s) * sample - f32(alpha_t * f32(decay - 1.0)) * model_output
        assert noise is not None  # refused above
        decay2 = f32(math.exp(f32(-2.0 * h))) if h != math.inf else 0.0
        return (
            f32(f32(sigma_t / sigma_s) * decay) * sample
            + f32(alpha_t * f32(1.0 - decay2)) * model_output
            + f32(sigma_t * f32(math.sqrt(f32(1.0 - decay2)))) * noise
        )

    def _second_order(
        self,
        index: int,
        outputs: tuple[Tensor, ...],
        sample: Tensor,
        noise: Tensor | None,
    ) -> Tensor:
        """DPM-Solver++(2M): the previous output enters as a finite difference.

        ``solver_type`` picks between the midpoint and Heun forms of the same
        second-order correction. ``view.SAMPLERS`` never sets it, so every
        sampler this fleet SELECTS is ``midpoint`` (diffusers' default) — but a
        checkpoint's own ``scheduler_config.json`` can carry ``heun``, and it
        rides through ``view.clone_scheduler``'s ``{**base.config, **overrides}``
        merge untouched. Both are implemented and both are measured for that
        reason: the declared-block path is not the only way this field arrives.
        """

        alpha_t, sigma_t = self._alpha_sigma(self.sigmas[index + 1])
        _alpha_s0, sigma_s0 = self._alpha_sigma(self.sigmas[index])
        h = f32(self._lambda(self.sigmas[index + 1]) - self._lambda(self.sigmas[index]))
        h_0 = f32(self._lambda(self.sigmas[index]) - self._lambda(self.sigmas[index - 1]))
        ratio = f32(h_0 / h)

        newest, previous = outputs[-1], outputs[-2]
        difference = f32(1.0 / ratio) * (newest - previous)
        decay = f32(math.exp(f32(-h))) if h != math.inf else 0.0

        if self.algorithm_type == "dpmsolver++":
            base = f32(sigma_t / sigma_s0) * sample - f32(alpha_t * f32(decay - 1.0)) * newest
            if self.solver_type == "midpoint":
                return base - f32(0.5 * f32(alpha_t * f32(decay - 1.0))) * difference
            return base + f32(alpha_t * f32(f32(f32(decay - 1.0) / h) + 1.0)) * difference

        assert noise is not None  # refused above
        decay2 = f32(math.exp(f32(-2.0 * h))) if h != math.inf else 0.0
        base = (
            f32(f32(sigma_t / sigma_s0) * decay) * sample
            + f32(alpha_t * f32(1.0 - decay2)) * newest
            + f32(sigma_t * f32(math.sqrt(f32(1.0 - decay2)))) * noise
        )
        if self.solver_type == "midpoint":
            return base + f32(0.5 * f32(alpha_t * f32(1.0 - decay2))) * difference
        return base + f32(alpha_t * f32(f32(f32(1.0 - decay2) / f32(-2.0 * h)) + 1.0)) * difference


@dataclass(frozen=True, slots=True)
class DPMSolverMultistep:
    """The declared ``dpmsolver_multistep`` scheduler, as bare typed math.

    Every field is a DECLARED family fact read out of the recipe's scheduler
    block; the defaults are ``diffusers``' class defaults, which no Stable
    Diffusion was trained on — a family that wants SD's trained noise schedule
    declares ``beta_start``/``beta_end``/``beta_schedule`` explicitly, exactly as
    it must for ``euler_discrete``.

    **What is implemented is what an endpoint can reach**, enumerated from
    ``gen_worker.view.SAMPLERS`` and the two endpoints that select from it:

    * ``dpmpp_2m`` — ``solver_order=2``, ``final_sigmas_type="zero"`` (sd15/sd2);
    * ``dpmpp_2m_karras`` — the above on a Karras ladder (sd15's DEFAULT, sdxl);
    * ``dpmpp_2m_sde_karras`` — ``algorithm_type="sde-dpmsolver++"`` on a Karras
      ladder (sdxl, and STAMPED on two live catalog entries).

    ``dpmpp_2m_sde`` exists in the sampler table and is reachable from NOTHING —
    no payload enum, no schema enum, no stamped recipe — but it is one boolean
    away from ``dpmpp_2m_sde_karras`` and therefore costs nothing to support.

    ``solver_order=3`` and the deprecated noise-prediction ``algorithm_type``s
    (``dpmsolver``, ``sde-dpmsolver``, both slated for removal in diffusers 1.0)
    are REFUSED rather than approximated: no recipe reaches them, so implementing
    them would be math with nothing measuring it.
    """

    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    prediction_type: str = "epsilon"
    timestep_spacing: str = "linspace"
    steps_offset: int = 0
    rescale_betas_zero_snr: bool = False
    algorithm_type: str = "dpmsolver++"
    solver_type: str = "midpoint"
    solver_order: int = 2
    lower_order_final: bool = True
    euler_at_final: bool = False
    final_sigmas_type: str = "zero"
    use_karras_sigmas: bool = False
    use_exponential_sigmas: bool = False
    use_flow_sigmas: bool = False
    flow_shift: float = 1.0

    #: Every parameter this kind reads. A block naming anything else is a
    #: declaration that says one thing and schedules another (pgw#1346 K10).
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        "algorithm_type",
        "beta_end",
        "beta_schedule",
        "beta_start",
        "euler_at_final",
        "final_sigmas_type",
        "flow_shift",
        "lower_order_final",
        "num_train_timesteps",
        "prediction_type",
        "rescale_betas_zero_snr",
        "solver_order",
        "solver_type",
        "steps_offset",
        "timestep_spacing",
        "use_exponential_sigmas",
        "use_flow_sigmas",
        "use_karras_sigmas",
    )

    BETA_SCHEDULES: ClassVar[tuple[str, ...]] = ("linear", "scaled_linear")
    PREDICTION_TYPES: ClassVar[tuple[str, ...]] = (
        "epsilon",
        "v_prediction",
        "flow_prediction",
    )
    SPACINGS: ClassVar[tuple[str, ...]] = ("linspace", "leading", "trailing")
    FINAL_SIGMAS: ClassVar[tuple[str, ...]] = ("zero", "sigma_min")
    ALGORITHMS: ClassVar[tuple[str, ...]] = ("dpmsolver++", "sde-dpmsolver++")
    SOLVER_TYPES: ClassVar[tuple[str, ...]] = ("midpoint", "heun")

    def __post_init__(self) -> None:
        if self.num_train_timesteps < 1:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"num_train_timesteps must be positive, got {self.num_train_timesteps}",
            )
        if not 0.0 < self.beta_start <= self.beta_end < 1.0:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                "the beta schedule must ascend inside (0, 1); got "
                f"beta_start={self.beta_start} beta_end={self.beta_end}",
            )
        for name, value, allowed in (
            ("beta_schedule", self.beta_schedule, self.BETA_SCHEDULES),
            ("prediction_type", self.prediction_type, self.PREDICTION_TYPES),
            ("timestep_spacing", self.timestep_spacing, self.SPACINGS),
            ("final_sigmas_type", self.final_sigmas_type, self.FINAL_SIGMAS),
            ("algorithm_type", self.algorithm_type, self.ALGORITHMS),
            ("solver_type", self.solver_type, self.SOLVER_TYPES),
        ):
            if value not in allowed:
                raise _refuse_kind(name, allowed, value)
        if self.solver_order not in (1, 2):
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"solver_order {self.solver_order} is not implemented: every sampler this "
                "fleet can reach is second order ('2M' IS the order), so a third-order "
                "update would be arithmetic nothing measures",
            )
        if sum((self.use_karras_sigmas, self.use_exponential_sigmas, self.use_flow_sigmas)) > 1:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                "karras, exponential and flow ladders are mutually exclusive; a block "
                "naming two has no resolvable schedule",
            )
        if self.use_flow_sigmas and self.flow_shift <= 0.0:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID, f"flow_shift must be positive, got {self.flow_shift}"
            )

    @classmethod
    def from_block(cls, block: SchedulerBlock) -> Self:
        """Build one from a declaration's scheduler parameter block."""

        only(block, cls.PARAMETERS)
        return cls(
            num_train_timesteps=count(block, "num_train_timesteps", 1000),
            beta_start=real(block, "beta_start", 0.0001),
            beta_end=real(block, "beta_end", 0.02),
            beta_schedule=choice(block, "beta_schedule", "linear", cls.BETA_SCHEDULES),
            prediction_type=choice(block, "prediction_type", "epsilon", cls.PREDICTION_TYPES),
            timestep_spacing=choice(block, "timestep_spacing", "linspace", cls.SPACINGS),
            steps_offset=count(block, "steps_offset", 0),
            rescale_betas_zero_snr=flag(block, "rescale_betas_zero_snr", False),
            algorithm_type=choice(block, "algorithm_type", "dpmsolver++", cls.ALGORITHMS),
            solver_type=choice(block, "solver_type", "midpoint", cls.SOLVER_TYPES),
            solver_order=count(block, "solver_order", 2),
            lower_order_final=flag(block, "lower_order_final", True),
            euler_at_final=flag(block, "euler_at_final", False),
            final_sigmas_type=choice(block, "final_sigmas_type", "zero", cls.FINAL_SIGMAS),
            use_karras_sigmas=flag(block, "use_karras_sigmas", False),
            use_exponential_sigmas=flag(block, "use_exponential_sigmas", False),
            use_flow_sigmas=flag(block, "use_flow_sigmas", False),
            flow_shift=real(block, "flow_shift", 1.0),
        )

    def objective(self, prediction_type: str) -> Self:
        """This scheduler with the checkpoint's stamped objective applied.

        The same pairing ``EulerDiscrete.objective`` makes and for the same
        reason: ``gen_worker.view`` turns on ``rescale_betas_zero_snr`` for every
        v-prediction checkpoint, so the two paths must not be able to disagree
        about what "v_prediction" means.
        """

        if prediction_type not in self.PREDICTION_TYPES:
            raise _refuse_kind("prediction_type", self.PREDICTION_TYPES, prediction_type)
        rescale = True if prediction_type == "v_prediction" else self.rescale_betas_zero_snr
        if prediction_type == self.prediction_type and rescale == self.rescale_betas_zero_snr:
            return self
        return replace(self, prediction_type=prediction_type, rescale_betas_zero_snr=rescale)

    def schedule(self, steps: int) -> DpmSolverSchedule:
        """The resolved ladder for one request.

        No ``image_seq_len``: none of the three ladders consults the resolution.
        ``FlowMatchEulerDiscrete`` does, which is exactly why this is stated
        rather than left to be inferred from a missing parameter.
        """

        if steps < 1:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID, f"a schedule needs at least one step, got {steps}"
            )

        if self.use_flow_sigmas:
            if self.final_sigmas_type != "zero":
                raise ModelError(
                    ModelRefusal.SCHEDULER_INVALID,
                    "a flow ladder with final_sigmas_type='sigma_min' has no coherent "
                    "terminal sigma — upstream reads it off a BETA table that a flow "
                    "config's betas do not describe. Nothing reaches this combination",
                )
            raw = ladders.flow_sigmas(steps, self.num_train_timesteps, self.flow_shift)
            timesteps = ladders.flow_timesteps(raw, self.num_train_timesteps)
            terminal = 0.0
        else:
            table = sigma_table(
                self.num_train_timesteps,
                self.beta_start,
                self.beta_end,
                self.beta_schedule,
                self.rescale_betas_zero_snr,
            )
            # The terminal sigma is the table's SMALLEST entry under
            # `sigma_min`, whichever ladder was walked — upstream resolves it
            # once, after the branch, rather than off the ladder's own last rung.
            terminal = 0.0 if self.final_sigmas_type == "zero" else table[0]
            if self.use_karras_sigmas or self.use_exponential_sigmas:
                logs = ladders.log_table(table)
                if self.use_karras_sigmas:
                    raw = ladders.karras_sigmas(table[0], table[-1], steps)
                    # ROUNDED, half to even: the Karras ladder synthesizes
                    # sigmas that are not table entries, so the timestep the
                    # model is conditioned on is read back out by interpolation
                    # and then snapped.
                    timesteps = tuple(
                        round_half_even(ladders.sigma_to_t(sigma, logs)) for sigma in raw
                    )
                else:
                    raw = ladders.exponential_sigmas(table[0], table[-1], steps)
                    # TRUNCATED, not rounded — upstream omits the `.round()` on
                    # this branch alone and the int64 cast truncates. The
                    # asymmetry is upstream's; reproducing it is the point.
                    timesteps = tuple(
                        truncate_to_int(ladders.sigma_to_t(sigma, logs)) for sigma in raw
                    )
            else:
                timesteps = ladders.discrete_timesteps(
                    self.timestep_spacing, steps, self.num_train_timesteps, self.steps_offset
                )
                raw = ladders.interpolate_table(table, timesteps)

        resolved = tuple(f32(sigma) for sigma in raw)
        return DpmSolverSchedule(
            sigmas=(*resolved, f32(terminal)),
            timesteps=timesteps,
            num_train_timesteps=self.num_train_timesteps,
            prediction_type=self.prediction_type,
            algorithm_type=self.algorithm_type,
            solver_order=self.solver_order,
            solver_type=self.solver_type,
            lower_order_final=self.lower_order_final,
            euler_at_final=self.euler_at_final,
            final_sigmas_type=self.final_sigmas_type,
            flow=self.use_flow_sigmas or self.prediction_type == "flow_prediction",
        )


def _refuse_kind(name: str, allowed: tuple[str, ...], value: object) -> ModelError:
    return refuse(name, f"one of {list(allowed)!r}", value)


__all__ = [
    "DPMSolverMultistep",
    "DpmSolverSchedule",
    "MultistepHistory",
]
