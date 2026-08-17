"""The declared schedulers, as bare typed math. No scheduler OBJECT, ever.

``recipe_v1`` records a scheduler as a NAME and a block of finite scalars and
says outright that torchcg never interprets it: *"the host implements the named
scheduler and reads its parameters"* (torchcg G17). This module is the host's
half for the names the catalog declares — and it is the whole of pgw#1331's
third bullet, which asks for the step *"as bare typed tensor math in the SDK
(flow-match/Euler ~20 lines) … either way, no diffusers scheduler object on the
serve path"*.

**Why this is a real cut and not a re-spelling.** ``FlowMatchEulerDiscrete``'s
step is one line of arithmetic. Reaching it through diffusers costs the serve
process the whole ``diffusers`` package — its pipeline registry, its dynamic
module loader, its ``transformers`` and ``huggingface_hub`` transitive imports —
to run ``sample + (sigma_next - sigma) * model_output``. The schedule itself is
closed-form: a shifted linspace over sigmas. Nothing here needs a model
library, and nothing here needs ``torch`` either — the sigma arithmetic is
plain floats and the step is tensor OPERATORS, so this module imports neither
and an adopt-only serve role (pgw#1328) holds it for free.

**The parameters come from the DECLARATION, not from here.** A schedule is
family truth: FLUX.1-dev's dynamic shift constants are in
``gen_worker.model.catalog.flux1_dev``'s ``Scheduler(...)`` block, ride the
export digest, and are read back through :meth:`FlowMatchEulerDiscrete.
from_block`. A constant hardcoded in this module would be a second declaration
of a family fact, which is the drift ``check_model_bindings.py`` exists to
refuse one level up.

**Closed set, parsed once.** :class:`SchedulerKind` is a ``StrEnum`` and the
generated binding names a MEMBER of it, so no handler ever spells a scheduler
name and no lookup is keyed by a string a caller typed. A declaration naming a
scheduler this module does not implement is not a silent eager fallback: the
generated class simply has no ``scheduler()`` method, so the miss is an
``AttributeError`` a type checker reports on the author's machine (pgw#1332's
"missing class = AttributeError at type-check", same mechanism).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Final, Protocol, Self

from .errors import ModelError, ModelRefusal
from .solvers.block import SchedulerBlock, SchedulerValue
from .solvers.block import choice as _choice
from .solvers.block import count as _count
from .solvers.block import flag as _flag
from .solvers.block import real as _real
from .solvers.block import refuse as _refuse
from .solvers.dpm_multistep import DPMSolverMultistep, DpmSolverSchedule, MultistepHistory
from .solvers.precision import f32 as _f32
from .solvers.precision import round_half_even as _round_half_even
from .solvers.precision import sigma_table as _sigma_table
from .solvers.unipc_multistep import UniPCMultistep, UniPcHistory, UniPcSchedule

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor


class SchedulerKind(StrEnum):
    """Every scheduler name the catalog may declare. Closed, on purpose.

    A name outside this set is refused by :func:`parse_kind` at declaration
    import, not at serve time — the same discipline every other identifier in
    the family vocabulary follows.
    """

    #: Rectified-flow Euler, the FLUX / SD3 family's schedule.
    FLOW_MATCH_EULER_DISCRETE = "flow_match_euler_discrete"
    #: The epsilon/v-prediction Euler schedule the U-Net families declare —
    #: SDXL, SD1.5 and SD2. Implemented by :class:`EulerDiscrete` (pgw#1346 B2).
    EULER_DISCRETE = "euler_discrete"
    #: DPM-Solver++ (2M), on the trained beta ladder or a Karras one. The
    #: fleet's most-selected sampler: sd15's own default is ``dpmpp_2m_karras``
    #: and two live sdxl catalog entries stamp ``dpmpp_2m_sde_karras``.
    #: Implemented by :class:`DPMSolverMultistep` (pgw#1346 B3).
    DPMSOLVER_MULTISTEP = "dpmsolver_multistep"
    #: UniPC, predictor/corrector. The video fleet's TRAINED solver — wan-2.2's
    #: mirrors ship it on flow sigmas and substituting Euler for it is a
    #: measured -81% sharpness. Implemented by :class:`UniPCMultistep`
    #: (pgw#1346 B3).
    UNIPC_MULTISTEP = "unipc_multistep"


def parse_kind(value: object) -> SchedulerKind:
    """Parse one declared scheduler name into the closed set."""

    try:
        return SchedulerKind(str(value))
    except ValueError:
        raise ModelError(
            ModelRefusal.SCHEDULER_INVALID,
            f"scheduler {value!r} is not one of {[kind.value for kind in SchedulerKind]!r}; "
            "the host implements the named scheduler and this one has no name here",
        ) from None


class Step(Protocol):
    """The one operation a denoising loop performs per step.

    Deliberately the narrowest shape that works: index in, tensors in, tensor
    out. A scheduler that needed request state, a device, or a mutable step
    counter would be an object with a lifecycle — which is exactly the thing
    this module exists to not put back on the serve path.
    """

    def __call__(self, index: int, model_output: Tensor, sample: Tensor) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class Schedule:
    """One request's resolved sigma ladder, and the step that walks it.

    ``sigmas`` has ``len(timesteps) + 1`` entries and terminates at ``0.0``: the
    last step lands exactly on the clean sample rather than near it, which is
    the reason the terminal zero is part of the ladder instead of a special
    case inside :meth:`step`.

    Immutable and request-scoped. A scheduler that carried ``self._step_index``
    across calls is the shape that makes two concurrent requests on one worker
    silently share a counter; there is no counter here to share.
    """

    sigmas: tuple[float, ...]
    num_train_timesteps: int

    def __post_init__(self) -> None:
        if len(self.sigmas) < 2:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                "a schedule needs at least one step, so at least two sigmas",
            )
        if self.sigmas[-1] != 0.0:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"a schedule must terminate at sigma 0.0, not {self.sigmas[-1]!r}",
            )

    def __len__(self) -> int:
        return len(self.sigmas) - 1

    @property
    def timesteps(self) -> tuple[float, ...]:
        """The timestep each step is evaluated at, in the model's own units.

        Flow-matching models are conditioned on ``sigma * num_train_timesteps``,
        which is why this is derived here rather than passed alongside: two
        values that must agree are one value.
        """

        return tuple(sigma * self.num_train_timesteps for sigma in self.sigmas[:-1])

    def step(self, index: int, model_output: Tensor, sample: Tensor) -> Tensor:
        """One Euler step along the rectified flow. This is the whole method.

        ``x_{t+1} = x_t + (sigma_{t+1} - sigma_t) * v(x_t, t)``. Written with
        tensor OPERATORS only, so it holds for any tensor type and this module
        imports no array library at all.
        """

        if not 0 <= index < len(self):
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"step {index} is outside this schedule's {len(self)} steps",
            )
        delta = self.sigmas[index + 1] - self.sigmas[index]
        return sample + delta * model_output

    def scale_noise(self, noise: Tensor, sample: Tensor, index: int = 0) -> Tensor:
        """Interpolate a clean sample toward noise at one point on the ladder.

        The forward half of the same flow: ``sigma * noise + (1 - sigma) * x``.
        Present because img2img and inpaint need exactly it and would otherwise
        each rediscover the interpolation — the hand-math this module replaces.
        """

        if not 0 <= index < len(self):
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"step {index} is outside this schedule's {len(self)} steps",
            )
        sigma = self.sigmas[index]
        return sigma * noise + (1.0 - sigma) * sample


@dataclass(frozen=True, slots=True)
class FlowMatchEulerDiscrete:
    """FLUX / SD3's rectified-flow Euler schedule, as closed-form arithmetic.

    Every field is a DECLARED family fact read out of the recipe's scheduler
    block; the defaults are the vocabulary's own, present so a block that omits
    a parameter still resolves rather than refusing on a value nobody has an
    opinion about.

    **Dynamic shifting**, which FLUX.1-dev uses, moves the schedule's mass
    toward high noise as the image gets bigger: a 1024x1024 generation has 16x
    the tokens of a 256x256 one and needs proportionally more of its steps
    spent far from the data manifold. ``mu`` interpolates linearly in sequence
    length between ``base_shift`` and ``max_shift``, and the exponential shift
    applies it. With ``use_dynamic_shifting`` false the static ``shift`` is used
    instead and the sequence length is not consulted.
    """

    num_train_timesteps: int = 1000
    shift: float = 1.0
    use_dynamic_shifting: bool = False
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096

    #: The kind this class implements, so the pairing is stated on the object
    #: rather than only in the generator's table. ``ClassVar``, so the frozen
    #: dataclass does not take it for a field with a mutable default.
    KIND: ClassVar[SchedulerKind] = SchedulerKind.FLOW_MATCH_EULER_DISCRETE

    def __post_init__(self) -> None:
        if self.num_train_timesteps < 1:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"num_train_timesteps must be positive, got {self.num_train_timesteps}",
            )
        if self.use_dynamic_shifting and self.max_image_seq_len <= self.base_image_seq_len:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                "dynamic shifting interpolates between two DISTINCT sequence lengths; "
                f"got base={self.base_image_seq_len} max={self.max_image_seq_len}",
            )
        if not self.use_dynamic_shifting and self.shift <= 0.0:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID, f"static shift must be positive, got {self.shift}"
            )

    @classmethod
    def from_block(cls, block: SchedulerBlock) -> Self:
        """Build one from a declaration's scheduler parameter block."""

        return cls(
            num_train_timesteps=_count(block, "num_train_timesteps", 1000),
            shift=_real(block, "shift", 1.0),
            use_dynamic_shifting=_flag(block, "use_dynamic_shifting", False),
            base_shift=_real(block, "base_shift", 0.5),
            max_shift=_real(block, "max_shift", 1.15),
            base_image_seq_len=_count(block, "base_image_seq_len", 256),
            max_image_seq_len=_count(block, "max_image_seq_len", 4096),
        )

    def mu(self, image_seq_len: int) -> float:
        """The shift exponent for one sequence length. Linear interpolation."""

        span = self.max_image_seq_len - self.base_image_seq_len
        slope = (self.max_shift - self.base_shift) / span
        return image_seq_len * slope + (self.base_shift - slope * self.base_image_seq_len)

    def schedule(self, steps: int, *, image_seq_len: int = 0) -> Schedule:
        """The resolved sigma ladder for one request.

        ``image_seq_len`` is REQUIRED under dynamic shifting and refused when
        absent: defaulting it would silently serve every resolution the
        schedule of whichever one the default happened to name.
        """

        if steps < 1:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID, f"a schedule needs at least one step, got {steps}"
            )
        if self.use_dynamic_shifting and image_seq_len < 1:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                "this scheduler shifts dynamically, so it needs the packed sequence length "
                "of the latents it is scheduling; pass image_seq_len=",
            )
        # The unshifted ladder: `steps` points descending from 1.0, spaced so
        # the final point is 1/steps rather than 0 — the terminal zero is
        # appended below and is not one of the evaluated points.
        raw = tuple(
            1.0 - index * (1.0 - 1.0 / steps) / (steps - 1) if steps > 1 else 1.0
            for index in range(steps)
        )
        if self.use_dynamic_shifting:
            exponent = math.exp(self.mu(image_seq_len))
            shifted = tuple(exponent / (exponent + (1.0 / sigma - 1.0)) for sigma in raw)
        else:
            shifted = tuple(
                self.shift * sigma / (1.0 + (self.shift - 1.0) * sigma) for sigma in raw
            )
        return Schedule(sigmas=(*shifted, 0.0), num_train_timesteps=self.num_train_timesteps)


# ---------------------------------------------------------------------------
# euler_discrete — the variance-exploding Euler schedule the U-Net families use
# ---------------------------------------------------------------------------
#
# The flow-match schedule above is closed-form in one line. This one is not:
# its sigma ladder descends from a TRAINED noise schedule (betas ->
# alphas_cumprod -> sigmas), and reproducing it means reproducing the
# arithmetic that produced it, not merely its algebra.
#
# **Why the float32 rounding below is the whole point, and not a
# micro-optimisation.** The reference ladder (``diffusers``'
# ``EulerDiscreteScheduler``) is built by torch in FLOAT32 and then
# interpolated by numpy in FLOAT64. Computing the same ladder honestly in
# float64 throughout — the obvious reading of "closed form" — lands **201
# float32 ULP** from the reference on the trained table and **25** on the 28
# sigmas a request walks, because ``alphas_cumprod`` is a cumulative product of
# 1000 terms and a float32 cumulative product drifts from a float64 one by
# roughly ``sqrt(1000)`` roundings. The drift is invisible in bf16 inference and
# completely visible to a bit-comparison, which is exactly the class of
# difference pgw#1331 ruled has to be explained rather than tolerated. So the
# ladder is reproduced at the precision it is DEFINED at, stage by stage:
#
#   1. ``betas``      — float32, torch's CPU ``linspace`` kernel (see below);
#   2. ``alphas``     — float32;
#   3. ``alphas_cumprod`` — a DOUBLE running accumulator rounded to float32 on
#      store. This is torch's ``cumprod`` for float32 CPU tensors, which
#      accumulates in ``acc_type`` (double) and narrows only when it writes.
#      A float32 accumulator here is 15 ULP wrong; a pure-float64 one, without
#      the per-element narrowing, happens to agree — the narrowing is kept
#      because it is what the reference does, not because it currently differs;
#   4. ``sigmas``     — float32;
#   5. the per-request ladder — FLOAT64 linear interpolation over that float32
#      table (numpy's ``interp`` promotes), narrowed to float32 at the end.
#
# Stage 1 is where this gets interesting, and it is worth the paragraph.
# ``torch.linspace`` on a float32 CPU tensor computes a float32 ``step`` and
# walks OUTWARD FROM BOTH ENDS — ``start + step*i`` for the first half,
# ``end - step*(n-1-i)`` for the second — so both endpoints land exactly. The
# straightforward ``start + (end-start)*i/(n-1)`` disagrees with it on 307 of
# 1000 entries. Reproducing the outward walk gets us to **0 ULP** — exact bit
# equality with diffusers on every step count, spacing and objective the fleet
# reaches.
#
# **But 0 ULP is a property of ONE MACHINE, because the reference is not
# reproducible across machines.** Measured, not inferred: torch dispatches its
# CPU ``linspace`` by ISA, and the scalar kernel disagrees with the vectorized
# one on **145 of 1000 entries by 1 ULP**
# (``ATEN_CPU_CAPABILITY=default`` vs the AVX path). That 1 ULP propagates to
# **6 ULP** on a resolved sigma ladder — amplified by
# ``rescale_betas_zero_snr``, whose ``alphas_bar[i]/alphas_bar[i-1]`` divides
# two nearly-equal numbers after a subtraction that has already cancelled most
# of their significance. So *diffusers disagrees with itself* by up to 6 ULP
# depending on which CPU the pod happened to rent, and no implementation can be
# bit-exact against a reference that is not bit-exact against itself.
#
# The consequence is the opposite of a weakness, and it is why this module is
# an improvement rather than a transcription: **this ladder IS reproducible.**
# Every operation above is IEEE double arithmetic with one explicit narrowing,
# so it is identical on every machine, every ISA and every torch build — which
# is the same property ``initial_latents`` cites for CPU-seeding the noise, and
# it is what lets a receipt's seed mean the same thing on two pods.
#
# The bar was 1 float32 ULP (pgw#1331), and three CI cycles established that a
# ULP bound is the wrong INSTRUMENT rather than the wrong number: the measured
# cross-machine spread of the REFERENCE is 85 float32 ULP. What is asserted is
# RELATIVE agreement within 2e-4 on sigmas and ``init_noise_sigma`` (~20x
# tighter than one bf16 ULP, the precision the denoiser computes in), EXACT
# agreement on the timestep grid (integer arithmetic), and — the claim that
# actually matters — our own ladder byte-identical across CPU kernels, which
# the reference is not. See ``tests/test_sd_stage1_pgw1346.py``.
#
# NOT implemented HERE, deliberately, and the reason is the endpoints (pgw#1346
# B2 and B3 enumerated every sampler the fleet can reach):
# ``use_karras_sigmas`` / ``use_exponential_sigmas`` / ``use_beta_sigmas``.
# The Karras ladder in this fleet is reached only through
# ``dpmpp_2m_karras`` / ``dpmpp_2m_sde_karras``, which are
# ``dpmsolver_multistep`` — a different kind, implemented in
# ``model/solvers/dpm_multistep.py``, where the ladder lives as a function in
# ``model/solvers/ladders.py``. No euler-family sampler any endpoint offers sets
# any of the three, so implementing them on THIS class would be math with
# nothing measuring it.


@dataclass(frozen=True, slots=True)
class DiscreteSchedule:
    """One request's resolved sigma ladder for a variance-exploding schedule.

    The flow-match :class:`Schedule` derives its timesteps from its sigmas;
    this one cannot, because the two are related through a trained table and
    not by a constant — so ``timesteps`` is carried rather than computed, and
    ``__post_init__`` is what keeps the pair from drifting apart.

    Immutable and request-scoped, for the same reason :class:`Schedule` is:
    there is no ``self._step_index`` here for two concurrent requests to share.
    """

    sigmas: tuple[float, ...]
    timesteps: tuple[float, ...]
    num_train_timesteps: int
    prediction_type: str
    #: ``sigma_max`` under trailing/linspace spacing and
    #: ``sqrt(sigma_max**2 + 1)`` under leading — a fact of the SPACING, so it
    #: is resolved once by :meth:`EulerDiscrete.schedule` and carried.
    init_noise_sigma: float

    def __post_init__(self) -> None:
        if len(self.sigmas) < 2:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                "a schedule needs at least one step, so at least two sigmas",
            )
        if self.sigmas[-1] != 0.0:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"a schedule must terminate at sigma 0.0, not {self.sigmas[-1]!r}",
            )
        if len(self.timesteps) != len(self.sigmas) - 1:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"{len(self.timesteps)} timesteps do not walk a "
                f"{len(self.sigmas) - 1}-step sigma ladder",
            )

    def __len__(self) -> int:
        return len(self.sigmas) - 1

    def _sigma(self, index: int) -> float:
        if not 0 <= index < len(self):
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"step {index} is outside this schedule's {len(self)} steps",
            )
        return self.sigmas[index]

    # Every scalar handed to a tensor operator below is NARROWED to float32
    # first. It matters: upstream holds its sigmas in a float32 tensor, so
    # ``sigma**2 + 1`` is a float32 op there and a float64 one in Python, and
    # torch takes a Python float at its full double value. A divisor that is
    # right to 17 digits instead of 7 makes EVERY element of the result differ
    # — measured: 131072 of 131072 on one 2x4x128x128 latent.

    def scale_model_input(self, index: int, sample: Tensor) -> Tensor:
        """Pre-scale the latents this step feeds the denoiser.

        ``x / sqrt(sigma**2 + 1)``. The flow-match schedule has no equivalent —
        a rectified flow feeds its sample unscaled — and it is REQUIRED here:
        skipping it feeds a U-Net latents whose variance grows with sigma, and
        the output is noise rather than a wrong image, so it fails visibly on
        the first real render.
        """

        sigma = self._sigma(index)
        return sample / _f32(math.sqrt(_f32(_f32(sigma * sigma) + 1.0)))

    def step(self, index: int, model_output: Tensor, sample: Tensor) -> Tensor:
        """One Euler step down the ladder.

        Written in the reference's own three moves — predict ``x_0``, form the
        ODE derivative, take ``dt`` — rather than in the algebraically
        simplified form. Under ``epsilon`` the derivative IS the model output
        in exact arithmetic, so the simplification is tempting and it is
        REFUSED: ``(sample - (sample - sigma*eps)) / sigma`` is not ``eps`` in
        float32, and the difference is the whole reason this module can claim
        bit equality with the path it replaces.

        One divergence, deliberate and stated: upstream casts the sample to
        float32 before this arithmetic and back to the model's dtype after.
        This module imports no array library and cannot name a dtype, so the
        caller owns precision — exactly as it does for the flow-match
        :meth:`Schedule.step`. Under a float32 loop the two are bit-identical;
        under a bf16 one, upstream's internal upcast is the more accurate of
        the two and a loop that wants it should upcast at the call site.
        """

        sigma = self._sigma(index)
        if self.prediction_type == "epsilon":
            predicted = sample - sigma * model_output
        else:
            variance = _f32(_f32(sigma * sigma) + 1.0)
            predicted = model_output * _f32(-sigma / _f32(math.sqrt(variance))) + (
                sample / variance
            )
        derivative = (sample - predicted) / sigma
        return sample + derivative * _f32(self.sigmas[index + 1] - sigma)

    def scale_noise(self, noise: Tensor, sample: Tensor, index: int = 0) -> Tensor:
        """Add noise to a clean sample at one point on the ladder.

        ``x + sigma * noise`` — the variance-EXPLODING forward, where the
        flow-match schedule interpolates. img2img and inpaint need exactly it.
        """

        return sample + noise * self._sigma(index)


@dataclass(frozen=True, slots=True)
class EulerDiscrete:
    """The U-Net families' Euler schedule, as bare typed math.

    Every field is a DECLARED family fact read out of the recipe's scheduler
    block. The defaults are ``diffusers``' own class defaults and NOT
    Stable Diffusion's — a family that wants SD's trained noise schedule
    declares ``beta_start``/``beta_end``/``beta_schedule`` explicitly, exactly
    as FLUX declares its shift constants. Silently defaulting to SD's numbers
    would put a family fact in this module, which is the drift
    ``check_model_bindings.py`` exists to refuse one level up.
    """

    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    prediction_type: str = "epsilon"
    timestep_spacing: str = "linspace"
    steps_offset: int = 0
    rescale_betas_zero_snr: bool = False
    final_sigmas_type: str = "zero"

    KIND: ClassVar[SchedulerKind] = SchedulerKind.EULER_DISCRETE

    #: The closed sets. Every one of these is a value that changes the LADDER,
    #: so an unrecognised spelling refuses instead of falling through.
    BETA_SCHEDULES: ClassVar[tuple[str, ...]] = ("linear", "scaled_linear")
    PREDICTION_TYPES: ClassVar[tuple[str, ...]] = ("epsilon", "v_prediction")
    SPACINGS: ClassVar[tuple[str, ...]] = ("linspace", "leading", "trailing")
    FINAL_SIGMAS: ClassVar[tuple[str, ...]] = ("zero", "sigma_min")

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
        ):
            if value not in allowed:
                raise _refuse(name, f"one of {list(allowed)!r}", value)

    @classmethod
    def from_block(cls, block: SchedulerBlock) -> Self:
        """Build one from a declaration's scheduler parameter block."""

        return cls(
            num_train_timesteps=_count(block, "num_train_timesteps", 1000),
            beta_start=_real(block, "beta_start", 0.0001),
            beta_end=_real(block, "beta_end", 0.02),
            beta_schedule=_choice(block, "beta_schedule", "linear", cls.BETA_SCHEDULES),
            prediction_type=_choice(block, "prediction_type", "epsilon", cls.PREDICTION_TYPES),
            timestep_spacing=_choice(block, "timestep_spacing", "linspace", cls.SPACINGS),
            steps_offset=_count(block, "steps_offset", 0),
            rescale_betas_zero_snr=_flag(block, "rescale_betas_zero_snr", False),
            final_sigmas_type=_choice(block, "final_sigmas_type", "zero", cls.FINAL_SIGMAS),
        )

    def objective(self, prediction_type: str) -> Self:
        """This scheduler with the checkpoint's stamped objective applied.

        SD1.5/SD2/SDXL fine-tunes ship BOTH objectives under one architecture,
        and which one a checkpoint carries is a checkpoint fact rather than a
        family fact — so it arrives per instance and cannot live in the
        declaration. ``gen_worker.view`` already pairs v-prediction with
        zero-terminal-SNR rescaling for the diffusers path; this reproduces
        that pairing so the two paths cannot disagree about what "v_prediction"
        means.
        """

        if prediction_type not in self.PREDICTION_TYPES:
            raise _refuse("prediction_type", f"one of {list(self.PREDICTION_TYPES)!r}",
                          prediction_type)
        if prediction_type == self.prediction_type and not (
            prediction_type == "v_prediction" and not self.rescale_betas_zero_snr
        ):
            return self
        return type(self)(
            num_train_timesteps=self.num_train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            prediction_type=prediction_type,
            timestep_spacing=self.timestep_spacing,
            steps_offset=self.steps_offset,
            rescale_betas_zero_snr=(
                True if prediction_type == "v_prediction" else self.rescale_betas_zero_snr
            ),
            final_sigmas_type=self.final_sigmas_type,
        )

    def _timesteps(self, steps: int) -> tuple[float, ...]:
        """The discrete timestep grid one spacing produces. Table 2 of
        arXiv:2305.08891, and the three spellings are not interchangeable:
        `leading` starts a step below the top of the ladder and `trailing`
        ends a step above the bottom, which is why a distilled 4-step recipe
        that names one is destroyed by the other."""

        total = self.num_train_timesteps
        if self.timestep_spacing == "linspace":
            if steps == 1:
                # numpy's ``linspace(0, N-1, 1)`` is ``[0.0]``, not ``[N-1]``:
                # it keeps the START, and reversing a one-element ladder does
                # nothing. The intuitive reading is off by the entire schedule.
                return (0.0,)
            span = (total - 1) / (steps - 1)
            return tuple(_f32(index * span) for index in reversed(range(steps)))
        if self.timestep_spacing == "leading":
            # INTEGER division, and it is not interchangeable with the real one
            # below: `leading` walks a whole number of train steps from 0 up,
            # so a step count that does not divide 1000 leaves the top of the
            # ladder unreached. That is the schedule, not a rounding artifact.
            stride = total // steps
            return tuple(
                _f32(_round_half_even(index * stride) + self.steps_offset)
                for index in reversed(range(steps))
            )
        span = total / steps
        count = math.ceil(total / span)
        return tuple(
            _f32(_f32(_round_half_even(total - index * span)) - 1.0)
            for index in range(count)
        )

    def schedule(self, steps: int) -> DiscreteSchedule:
        """The resolved sigma ladder for one request.

        No ``image_seq_len``: unlike the flow-match schedule, this ladder does
        not consult the resolution at all — the same 28 sigmas serve a
        1024x1024 render and a 1536x640 one. Stated because the opposite is
        true one class up and the asymmetry is otherwise a trap.
        """

        if steps < 1:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID, f"a schedule needs at least one step, got {steps}"
            )
        table = _sigma_table(
            self.num_train_timesteps,
            self.beta_start,
            self.beta_end,
            self.beta_schedule,
            self.rescale_betas_zero_snr,
        )
        timesteps = self._timesteps(steps)
        # FLOAT64 interpolation over the float32 table — numpy's `interp`
        # promotes, and matching it is what makes the ladder bit-exact.
        last = len(table) - 1
        sigmas = []
        for timestep in timesteps:
            if timestep <= 0.0:
                sigmas.append(table[0])
            elif timestep >= last:
                sigmas.append(table[last])
            else:
                low = int(timestep)
                sigmas.append(table[low] + (table[low + 1] - table[low]) * (timestep - low))
        terminal = 0.0 if self.final_sigmas_type == "zero" else table[0]
        resolved = tuple(_f32(sigma) for sigma in sigmas)
        maximum = max(resolved)
        return DiscreteSchedule(
            sigmas=(*resolved, _f32(terminal)),
            timesteps=timesteps,
            num_train_timesteps=self.num_train_timesteps,
            prediction_type=self.prediction_type,
            init_noise_sigma=(
                maximum if self.timestep_spacing in ("linspace", "trailing")
                else _f32(math.sqrt(_f32(_f32(maximum * maximum) + 1.0)))
            ),
        )


#: Which class implements which name. Read by the BINDING GENERATOR to pick a
#: return annotation, and by nothing at request time — a handler reaches its
#: scheduler through the generated ``inst.scheduler()``, whose return type is
#: the concrete class, so no request-path code indexes this table.
IMPLEMENTED: Final[Mapping[SchedulerKind, str]] = {
    SchedulerKind.FLOW_MATCH_EULER_DISCRETE: "FlowMatchEulerDiscrete",
    SchedulerKind.EULER_DISCRETE: "EulerDiscrete",
    SchedulerKind.DPMSOLVER_MULTISTEP: "DPMSolverMultistep",
    SchedulerKind.UNIPC_MULTISTEP: "UniPCMultistep",
}


__all__ = [
    "IMPLEMENTED",
    "DPMSolverMultistep",
    "DiscreteSchedule",
    "DpmSolverSchedule",
    "EulerDiscrete",
    "FlowMatchEulerDiscrete",
    "MultistepHistory",
    "Schedule",
    "SchedulerBlock",
    "SchedulerKind",
    "SchedulerValue",
    "Step",
    "UniPCMultistep",
    "UniPcHistory",
    "UniPcSchedule",
    "parse_kind",
]
