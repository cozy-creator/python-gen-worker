"""``unipc_multistep`` — UniPC (predictor/corrector), as bare typed math.

The solver the video fleet actually renders on. wan-2.2's own mirrors ship
``UniPCMultistepScheduler`` with ``use_flow_sigmas=True`` and serve every
undistilled lane on it, and substituting flow-match Euler for it is a MEASURED
-81% frame-40 sharpness at 40 steps (wan-2.2 README) — this is not a stylistic
sampler choice, it is the checkpoint's trained solver. sd15's and sd2's recipe
vocabularies also admit a plain ``unipc`` on the diffusion ladder.

**Two ladders, one update rule.** UniPC is the only solver in this package that
is reached on BOTH a trained-beta ladder (sd15's ``unipc``) and a rectified-flow
ladder (wan, hidream), and the difference is one declared boolean plus one
reading of what a sigma means (:meth:`UniPcSchedule._alpha_sigma`). Sharing the
update rule between them is the point: the alternative is two implementations of
one paper that drift.

**The state is bigger than DPM's, and that is the interesting part.** A
predictor/corrector method carries the previous SAMPLE as well as the previous
model outputs, because the corrector re-solves the step it just took using
information that only arrived afterwards. diffusers keeps four mutable
attributes for this (``model_outputs``, ``timestep_list``, ``last_sample``,
``this_order``) plus two counters. Here they are one frozen value,
:class:`UniPcHistory`, threaded through :meth:`UniPcSchedule.step`:

* **initialization is total and constant** — ``begin()`` takes nothing, so the
  recursion's initial condition cannot differ between two pods;
* **the corrector is disabled at step 0 by the STATE, not by a counter** — an
  empty ``last_sample`` is the whole condition, so a loop that restarts mid-way
  cannot accidentally correct against a sample from a different request;
* **the order warm-up is explicit** — ``order`` records what the predictor
  actually used, because the corrector must re-solve at the SAME order and
  reading it off the step index instead is how the two silently disagree on
  short ladders.

**No array library.** The 2x2 solve the second-order corrector needs is written
in closed form rather than handed to a linear-algebra routine — partly because
this module imports none, and partly because ``torch.linalg.solve`` on a 2x2
float32 system is LAPACK, which is exactly the class of
implementation-defined primitive pgw#1346 B2 established cannot be bit-matched.
Two lines of algebra are more accurate AND deterministic.
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
    """float32 ``log``, with torch's ``-inf`` at zero rather than Python's raise."""

    return -math.inf if value <= 0.0 else f32(math.log(value))


def _expm1(value: float) -> float:
    """float32 ``expm1``. Upstream spells it ``torch.expm1``; it is not
    ``exp(x) - 1``, and at the small ``h`` of a 50-step ladder the difference is
    the entire significand."""

    return f32(math.expm1(value))


@dataclass(frozen=True, slots=True)
class UniPcHistory:
    """One request's predictor/corrector state, as an immutable VALUE.

    ``outputs`` are the converted model outputs, OLDEST FIRST, at most
    ``solver_order`` of them. ``last_sample`` is the sample the previous
    predictor step started from — ``None`` before the first step, which is
    exactly the condition that disables the corrector. ``order`` is the order the
    previous predictor step actually ran at, and ``taken`` is the multistep
    warm-up counter.
    """

    outputs: tuple[Tensor, ...] = ()
    last_sample: Tensor | None = None
    order: int = 0
    taken: int = 0


@dataclass(frozen=True, slots=True)
class UniPcSchedule:
    """One request's resolved UniPC ladder, and the step that walks it."""

    sigmas: tuple[float, ...]
    timesteps: tuple[float, ...]
    num_train_timesteps: int
    prediction_type: str
    solver_order: int
    solver_type: str
    predict_x0: bool
    lower_order_final: bool
    #: Step indices whose CORRECTOR is skipped. Not declarable — a scheduler
    #: block holds finite JSON scalars, not lists — and empty on every mirror in
    #: the fleet. Carried so the field is not silently absent from the model.
    disable_corrector: tuple[int, ...] = ()
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

    def begin(self) -> UniPcHistory:
        """The initial predictor/corrector state. Constant, and the ONLY one."""

        return UniPcHistory()

    # ---------------------------------------------------------------- sigmas

    def _checked(self, index: int) -> int:
        if not 0 <= index < len(self):
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"step {index} is outside this schedule's {len(self)} steps",
            )
        return index

    def _alpha_sigma(self, sigma: float) -> tuple[float, float]:
        """``(alpha_t, sigma_t)`` — flow sums to one, diffusion squares to one."""

        if self.flow:
            return f32(1.0 - sigma), f32(sigma)
        alpha = f32(1.0 / f32(math.sqrt(f32(f32(sigma * sigma) + 1.0))))
        return alpha, f32(sigma * alpha)

    def _lambda(self, sigma: float) -> float:
        alpha, sigma_t = self._alpha_sigma(sigma)
        return f32(_log(alpha) - _log(sigma_t))

    # ------------------------------------------------------------ the output

    def convert(self, index: int, model_output: Tensor, sample: Tensor) -> Tensor:
        """The model output in the form UniPC integrates.

        ``predict_x0`` (true on every mirror in the fleet) integrates the
        DATA-prediction model, so every objective converts to ``x0``; false
        integrates the noise-prediction model and converts to epsilon. The two
        are not a preference — the update rules below read the choice too, and
        mixing them produces a plausible, wrong image.
        """

        sigma = self.sigmas[self._checked(index)]
        alpha, sigma_t = self._alpha_sigma(sigma)
        if self.predict_x0:
            if self.prediction_type == "flow_prediction":
                return sample - f32(sigma) * model_output
            if self.prediction_type == "epsilon":
                return (sample - sigma_t * model_output) / alpha
            return f32(alpha) * sample - f32(sigma_t) * model_output
        if self.prediction_type == "epsilon":
            return model_output
        if self.prediction_type == "v_prediction":
            return f32(alpha) * model_output + f32(sigma_t) * sample
        raise ModelError(
            ModelRefusal.SCHEDULER_INVALID,
            f"prediction_type {self.prediction_type!r} has no noise-prediction reading; "
            "it is only defined under predict_x0",
        )

    # ------------------------------------------------------ the coefficients

    def _phi(self, h: float) -> tuple[float, float, float]:
        """``(hh, h_phi_1, B_h)`` — the exponential-integrator coefficients.

        ``hh`` is ``-h`` under ``predict_x0`` and ``h`` otherwise; ``B_h`` is
        ``hh`` for ``bh1`` and ``expm1(hh)`` for ``bh2``. Every mirror in the
        fleet ships ``bh2``, which upstream recommends for step counts at or
        above 10 — and wan's distilled lanes run FOUR, so ``bh1`` is the
        documented better choice there and is implemented for the day a mirror
        ships it.
        """

        hh = f32(-h) if self.predict_x0 else h
        h_phi_1 = _expm1(hh)
        b_h = hh if self.solver_type == "bh1" else _expm1(hh)
        return hh, h_phi_1, b_h

    def _weights(self, hh: float, h_phi_1: float, b_h: float, order: int) -> tuple[float, ...]:
        """``b`` — the right-hand side of the order-conditions system."""

        h_phi_k = f32(f32(h_phi_1 / hh) - 1.0)
        factorial = 1
        weights: list[float] = []
        for index in range(1, order + 1):
            weights.append(f32(f32(h_phi_k * factorial) / b_h))
            factorial *= index + 1
            h_phi_k = f32(f32(h_phi_k / hh) - f32(1.0 / factorial))
        return tuple(weights)

    @staticmethod
    def _solve2(ratio: float, weights: tuple[float, ...]) -> tuple[float, float]:
        """Solve ``[[1, 1], [ratio, 1]] @ rho = weights`` in closed form.

        The 2x2 system upstream hands to ``torch.linalg.solve``. Written out
        because a 2x2 LAPACK call is an implementation-defined primitive and this
        is two lines of exact algebra — the same argument ``math.sqrt`` over
        ``pow`` makes one module down.
        """

        first = (weights[1] - weights[0]) / (ratio - 1.0)
        return f32(first), f32(weights[0] - first)

    # -------------------------------------------------------------- the step

    def step(
        self,
        index: int,
        model_output: Tensor,
        sample: Tensor,
        history: UniPcHistory,
    ) -> tuple[Tensor, UniPcHistory]:
        """One predictor/corrector update. Returns the sample AND the history.

        The order of operations is upstream's and it matters: the CORRECTOR runs
        FIRST, re-solving the previous step with this step's model output, and
        only then does the predictor take the new step from the corrected
        sample. A loop that corrected afterwards would be a different method
        with the same name.
        """

        self._checked(index)
        converted = self.convert(index, model_output, sample)

        corrected = sample
        if index > 0 and index - 1 not in self.disable_corrector and history.last_sample is not None:
            corrected = self._correct(index, converted, history)

        outputs = (*history.outputs, converted)[-self.solver_order :]
        if self.lower_order_final:
            order = min(self.solver_order, len(self) - index)
        else:
            order = self.solver_order
        order = min(order, history.taken + 1)

        predicted = self._predict(index, outputs, corrected, order)
        return predicted, UniPcHistory(
            outputs=outputs,
            last_sample=corrected,
            order=order,
            taken=min(history.taken + 1, self.solver_order),
        )

    def _predict(
        self, index: int, outputs: tuple[Tensor, ...], sample: Tensor, order: int
    ) -> Tensor:
        """UniP: take the step, using ``order`` of the accumulated outputs."""

        newest = outputs[-1]
        alpha_t, sigma_t = self._alpha_sigma(self.sigmas[index + 1])
        alpha_s0, sigma_s0 = self._alpha_sigma(self.sigmas[index])
        h = f32(self._lambda(self.sigmas[index + 1]) - self._lambda(self.sigmas[index]))
        hh, h_phi_1, b_h = self._phi(h)

        residual: Tensor | None = None
        if order > 1:
            # Order 2 uses upstream's SIMPLIFIED weight of 1/2 rather than the
            # solve — stated because the corrector at the same order does NOT,
            # and assuming they match is a silent second-order error.
            previous = outputs[-2]
            rk = f32(
                f32(self._lambda(self.sigmas[index - 1]) - self._lambda(self.sigmas[index])) / h
            )
            # Two SEPARATE float32 operations upstream — the difference is
            # divided by `rk` and then weighted. Folding `0.5/rk` into one
            # scalar happens to be exact here (0.5 only moves an exponent), and
            # it is still written out, because the DPM sibling folds the
            # reciprocal and this one does not: the asymmetry is upstream's.
            residual = ((previous - newest) / f32(rk)) * 0.5

        if self.predict_x0:
            base = f32(sigma_t / sigma_s0) * sample - f32(alpha_t * h_phi_1) * newest
            if residual is None:
                return base
            return base - f32(alpha_t * b_h) * residual
        base = f32(alpha_t / alpha_s0) * sample - f32(sigma_t * h_phi_1) * newest
        if residual is None:
            return base
        return base - f32(sigma_t * b_h) * residual

    def _correct(self, index: int, this_output: Tensor, history: UniPcHistory) -> Tensor:
        """UniC: re-solve the PREVIOUS step now that its endpoint is known."""

        assert history.last_sample is not None  # guarded by the caller
        order = history.order
        newest = history.outputs[-1]
        alpha_t, sigma_t = self._alpha_sigma(self.sigmas[index])
        alpha_s0, sigma_s0 = self._alpha_sigma(self.sigmas[index - 1])
        h = f32(self._lambda(self.sigmas[index]) - self._lambda(self.sigmas[index - 1]))
        hh, h_phi_1, b_h = self._phi(h)

        residual: Tensor | None = None
        if order > 1:
            previous = history.outputs[-2]
            rk = f32(
                f32(self._lambda(self.sigmas[index - 2]) - self._lambda(self.sigmas[index - 1])) / h
            )
            weights = self._weights(hh, h_phi_1, b_h, order)
            leading, trailing = self._solve2(rk, weights)
            residual = ((previous - newest) / f32(rk)) * leading
            final = trailing
        else:
            # Order 1: upstream's simplified 1/2, not a solve.
            final = 0.5

        correction = f32(final) * (this_output - newest)
        if residual is not None:
            correction = residual + correction
        if self.predict_x0:
            base = f32(sigma_t / sigma_s0) * history.last_sample - f32(alpha_t * h_phi_1) * newest
            return base - f32(alpha_t * b_h) * correction
        base = f32(alpha_t / alpha_s0) * history.last_sample - f32(sigma_t * h_phi_1) * newest
        return base - f32(sigma_t * b_h) * correction


@dataclass(frozen=True, slots=True)
class UniPCMultistep:
    """The declared ``unipc_multistep`` scheduler, as bare typed math.

    **What is implemented is what an endpoint can reach**, enumerated from the
    endpoints rather than from the paper:

    * **the flow lane** — ``use_flow_sigmas=True``,
      ``prediction_type="flow_prediction"``, ``solver_order=2``,
      ``solver_type="bh2"``, ``predict_x0=True``, ``lower_order_final=True``,
      ``final_sigmas_type="zero"``, ``timestep_spacing="linspace"``. This is
      wan-2.2's mirrors verbatim, at ``flow_shift`` 12.0 (the curated T2V value),
      5.0 (the TI2V-5B mirror) and 3.0 (the A14B mirror / I2V), over 1, 4, 8, 40
      and 50 steps;
    * **the diffusion lane** — the same solver on the trained beta ladder, which
      is what sd15's and sd2's ``unipc`` recipe name selects (SD betas,
      ``leading`` spacing, ``steps_offset=1``).

    ``solver_order=3`` is REFUSED: every mirror ships 2, and a third-order
    predictor needs a 3x3 solve that nothing would measure.
    """

    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    prediction_type: str = "epsilon"
    timestep_spacing: str = "linspace"
    steps_offset: int = 0
    rescale_betas_zero_snr: bool = False
    solver_order: int = 2
    solver_type: str = "bh2"
    predict_x0: bool = True
    lower_order_final: bool = True
    final_sigmas_type: str = "zero"
    use_karras_sigmas: bool = False
    use_exponential_sigmas: bool = False
    use_flow_sigmas: bool = False
    flow_shift: float = 1.0

    #: Every parameter this kind reads. A block naming anything else is a
    #: declaration that says one thing and schedules another (pgw#1346 K10).
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        "beta_end",
        "beta_schedule",
        "beta_start",
        "final_sigmas_type",
        "flow_shift",
        "lower_order_final",
        "num_train_timesteps",
        "predict_x0",
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
    SOLVER_TYPES: ClassVar[tuple[str, ...]] = ("bh1", "bh2")

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
            ("solver_type", self.solver_type, self.SOLVER_TYPES),
        ):
            if value not in allowed:
                raise refuse(name, f"one of {list(allowed)!r}", value)
        if self.solver_order not in (1, 2):
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"solver_order {self.solver_order} is not implemented: every UniPC mirror "
                "this fleet serves ships solver_order=2, and a third-order predictor "
                "would be arithmetic nothing measures",
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
        if not self.predict_x0 and self.prediction_type == "flow_prediction":
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                "a flow-prediction checkpoint has no noise-prediction reading; "
                "predict_x0=False cannot integrate it",
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
            solver_order=count(block, "solver_order", 2),
            solver_type=choice(block, "solver_type", "bh2", cls.SOLVER_TYPES),
            predict_x0=flag(block, "predict_x0", True),
            lower_order_final=flag(block, "lower_order_final", True),
            final_sigmas_type=choice(block, "final_sigmas_type", "zero", cls.FINAL_SIGMAS),
            use_karras_sigmas=flag(block, "use_karras_sigmas", False),
            use_exponential_sigmas=flag(block, "use_exponential_sigmas", False),
            use_flow_sigmas=flag(block, "use_flow_sigmas", False),
            flow_shift=real(block, "flow_shift", 1.0),
        )

    def objective(self, prediction_type: str) -> Self:
        """This scheduler with the checkpoint's stamped objective applied."""

        if prediction_type not in self.PREDICTION_TYPES:
            raise refuse(
                "prediction_type", f"one of {list(self.PREDICTION_TYPES)!r}", prediction_type
            )
        rescale = True if prediction_type == "v_prediction" else self.rescale_betas_zero_snr
        if prediction_type == self.prediction_type and rescale == self.rescale_betas_zero_snr:
            return self
        return replace(self, prediction_type=prediction_type, rescale_betas_zero_snr=rescale)

    def shifted(self, flow_shift: float) -> Self:
        """This scheduler at another ``flow_shift``.

        wan-2.2 re-shifts per REQUEST (``scheduling.reshift``) — the served value
        is a curated 12.0 where the A14B mirror ships 3.0 — and it does so by
        rebuilding the scheduler from its own config so ``use_flow_sigmas`` and
        ``prediction_type`` survive. This is that operation with the survival
        guaranteed by the type rather than by remembering.
        """

        if not self.use_flow_sigmas:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                "flow_shift only moves a FLOW ladder; this scheduler walks the trained "
                "beta table, where the field is ignored",
            )
        return replace(self, flow_shift=flow_shift)

    def schedule(self, steps: int) -> UniPcSchedule:
        """The resolved ladder for one request."""

        if steps < 1:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID, f"a schedule needs at least one step, got {steps}"
            )

        if self.use_flow_sigmas:
            raw = ladders.flow_sigmas(steps, self.num_train_timesteps, self.flow_shift)
            timesteps = ladders.flow_timesteps(raw, self.num_train_timesteps)
            terminal = 0.0 if self.final_sigmas_type == "zero" else raw[-1]
        else:
            table = sigma_table(
                self.num_train_timesteps,
                self.beta_start,
                self.beta_end,
                self.beta_schedule,
                self.rescale_betas_zero_snr,
            )
            if self.use_karras_sigmas or self.use_exponential_sigmas:
                logs = ladders.log_table(table)
                if self.use_karras_sigmas:
                    raw = ladders.karras_sigmas(table[0], table[-1], steps)
                    timesteps = tuple(
                        round_half_even(ladders.sigma_to_t(sigma, logs)) for sigma in raw
                    )
                else:
                    raw = ladders.exponential_sigmas(table[0], table[-1], steps)
                    timesteps = tuple(
                        truncate_to_int(ladders.sigma_to_t(sigma, logs)) for sigma in raw
                    )
                # UniPC reads the terminal sigma off the LADDER's own last rung
                # here, where the plain branch below reads it off the TABLE.
                # DPMSolverMultistep resolves it once for every branch. The
                # asymmetry is upstream's; it is only observable under
                # `final_sigmas_type="sigma_min"`, which nothing reaches.
                terminal = 0.0 if self.final_sigmas_type == "zero" else raw[-1]
            else:
                timesteps = ladders.discrete_timesteps(
                    self.timestep_spacing, steps, self.num_train_timesteps, self.steps_offset
                )
                raw = ladders.interpolate_table(table, timesteps)
                terminal = 0.0 if self.final_sigmas_type == "zero" else table[0]

        resolved = tuple(f32(sigma) for sigma in raw)
        return UniPcSchedule(
            sigmas=(*resolved, f32(terminal)),
            timesteps=timesteps,
            num_train_timesteps=self.num_train_timesteps,
            prediction_type=self.prediction_type,
            solver_order=self.solver_order,
            solver_type=self.solver_type,
            predict_x0=self.predict_x0,
            lower_order_final=self.lower_order_final,
            flow=self.use_flow_sigmas or self.prediction_type == "flow_prediction",
        )


__all__ = [
    "UniPCMultistep",
    "UniPcHistory",
    "UniPcSchedule",
]
