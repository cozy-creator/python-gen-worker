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
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Final, Protocol, Self

from .errors import ModelError, ModelRefusal
from .solvers.block import SchedulerBlock, SchedulerValue
from .solvers.block import choice as _choice
from .solvers.block import count as _count
from .solvers.block import flag as _flag
from .solvers.block import real as _real
from .solvers.block import only as _only
from .solvers.block import refuse as _refuse
from .solvers.ladders import interpolate_table as _interpolate
from .solvers.dpm_multistep import DPMSolverMultistep, DpmSolverSchedule, MultistepHistory
from .solvers.precision import f32 as _f32
from .solvers.precision import round_half_even as _round_half_even
from .solvers.precision import alphas_cumprod as _alphas_cumprod
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
    #: Euler with an ancestral (stochastic) step — SDXL's DEFAULT sampler,
    #: reached as ``euler_a``. Implemented by :class:`EulerAncestralDiscrete`
    #: (pgw#1346 K10). It CONSUMES NOISE per step; see that class.
    EULER_ANCESTRAL_DISCRETE = "euler_ancestral_discrete"
    #: Deterministic DDIM (eta=0), reached as ``ddim`` / ``ddim_trailing``.
    #: Implemented by :class:`Ddim` (pgw#1346 K10). Alone among these it walks
    #: ALPHAS rather than sigmas, which is why it has its own schedule type.
    DDIM = "ddim"


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

    #: Every parameter the ``flow_match_euler_discrete`` KIND admits — which is
    #: wider than what THIS class reads, and deliberately so. The kind has two
    #: readers over one declared block: this class, and
    #: :class:`~gen_worker.model.flow_ladders.FlowMatchLadder`, which projects
    #: the same block plus ``shift_terminal`` / ``time_shift_type`` (pgw#1346
    #: B3a). The admissible key set belongs to the KIND, not to whichever reader
    #: a family happens to reach first — otherwise a legal declaration would
    #: refuse depending on which class read it (pgw#1346 K10).
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        "base_image_seq_len",
        "base_shift",
        "max_image_seq_len",
        "max_shift",
        "num_train_timesteps",
        "shift",
        "shift_terminal",
        "time_shift_type",
        "use_dynamic_shifting",
    )

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

        _only(block, cls.PARAMETERS)
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
class VarianceExploding:
    """One request's resolved sigma ladder, without the step that walks it.

    The flow-match :class:`Schedule` derives its timesteps from its sigmas;
    this one cannot, because the two are related through a trained table and
    not by a constant — so ``timesteps`` is carried rather than computed, and
    ``__post_init__`` is what keeps the pair from drifting apart.

    **The ladder and the step are separate facts, and this class is the
    evidence.** ``euler`` and ``euler_a`` resolve BIT-IDENTICAL sigmas from the
    same trained table and the same spacing, and then walk them differently:
    one deterministically, one by contracting to ``sigma_down`` and adding
    ``sigma_up`` of fresh noise. Everything they share lives here so the two
    cannot drift; ``step`` lives on the subclasses because its SIGNATURE
    differs — an ancestral step consumes noise and a deterministic one must not
    be able to accept it and ignore it.

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

    def predicted(self, index: int, model_output: Tensor, sample: Tensor) -> Tensor:
        """The denoised sample this step predicts — ``x_0``.

        Shared by both steps below because both reference implementations
        compute it identically, and because the ``v_prediction`` arm is the one
        piece of this arithmetic nobody rederives correctly from memory.
        """

        sigma = self._sigma(index)
        if self.prediction_type == "epsilon":
            return sample - sigma * model_output
        variance = _f32(_f32(sigma * sigma) + 1.0)
        return model_output * _f32(-sigma / _f32(math.sqrt(variance))) + (sample / variance)

    def scale_noise(self, noise: Tensor, sample: Tensor, index: int = 0) -> Tensor:
        """Add noise to a clean sample at one point on the ladder.

        ``x + sigma * noise`` — the variance-EXPLODING forward, where the
        flow-match schedule interpolates. img2img and inpaint need exactly it.
        """

        return sample + noise * self._sigma(index)


@dataclass(frozen=True, slots=True)
class DiscreteSchedule(VarianceExploding):
    """The DETERMINISTIC Euler walk down a variance-exploding ladder."""

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
        derivative = (sample - self.predicted(index, model_output, sample)) / sigma
        return sample + derivative * _f32(self.sigmas[index + 1] - sigma)


@dataclass(frozen=True, slots=True)
class AncestralSchedule(VarianceExploding):
    """The ANCESTRAL walk: contract to ``sigma_down``, then re-noise.

    An ancestral sampler does not step along the ODE to the next sigma. It
    steps to a SMALLER one (``sigma_down``) and adds back exactly the noise
    (``sigma_up``) that restores the variance the ladder expects — so the
    trajectory is stochastic and the sampler explores rather than descends.
    That is why ``euler_a`` is the default an SDXL request gets and why its
    output is not ``euler``'s output at the same seed.

    **The noise is a PARAMETER, never a source this object owns**, and the
    reason is the whole reproducibility argument this module is built on: a
    scheduler holding an RNG is a scheduler whose two concurrent requests share
    a stream, and this module imports no array library and could not seed one
    honestly anyway. The caller supplies the noise, and the catalog's serving
    half supplies it CPU-seeded and keyed by ``(request seed, step index)`` —
    see ``sdxl_serve.step_noise``, which states the keying — so two pods
    replaying one receipt draw the same tensor at step ``k`` regardless of what
    the loop around it did.
    """

    def ancestral(self, index: int) -> tuple[float, float]:
        """``(sigma_down, sigma_up)`` for one step.

        ``sigma_up`` is the variance handed back to the noise and ``sigma_down``
        is what is left for the deterministic move; ``sigma_up**2 +
        sigma_down**2 == sigma_next**2`` up to float32. Both are resolved in the
        reference's own float32 op order rather than folded, for the reason the
        module header gives at length.
        """

        near = self._sigma(index)
        far = self.sigmas[index + 1]
        near2 = _f32(near * near)
        far2 = _f32(far * far)
        up = _f32(math.sqrt(_f32(_f32(far2 * _f32(near2 - far2)) / near2)))
        down = _f32(math.sqrt(_f32(far2 - _f32(up * up))))
        return down, up

    def step(self, index: int, model_output: Tensor, sample: Tensor, noise: Tensor) -> Tensor:
        """One ancestral Euler step. ``noise`` is REQUIRED, never defaulted.

        Defaulting it to zeros would silently turn this into a worse
        :class:`DiscreteSchedule` — same signature, plausible image, wrong
        sampler — which is exactly the failure a required parameter costs
        nothing to prevent.
        """

        sigma = self._sigma(index)
        down, up = self.ancestral(index)
        derivative = (sample - self.predicted(index, model_output, sample)) / sigma
        return sample + derivative * _f32(down - sigma) + noise * up


@dataclass(frozen=True, slots=True)
class DdimSchedule:
    """One request's resolved DDIM trajectory. It walks ALPHAS, not sigmas.

    The odd one out on purpose, and the difference is not cosmetic:
    ``DDIMScheduler`` never forms a sigma ladder at all. It reads
    ``alphas_cumprod`` at the current and previous timestep and moves in the
    variance-PRESERVING parameterisation, which is why its ``init_noise_sigma``
    is 1.0 and its ``scale_model_input`` is the identity. Serving a DDIM
    trajectory through :class:`DiscreteSchedule`'s API would pre-scale the
    latents by ``1/sqrt(sigma**2+1)`` and start them at ~10x too much
    variance — no error, just a ruined image — so the two are different types.

    ``eta`` is fixed at 0: the deterministic sampler. Both endpoint names that
    reach DDIM (``ddim``, ``ddim_trailing``) are the deterministic one, and an
    ``eta > 0`` arm would be a noise-consuming step nothing declares.
    """

    #: The INTEGER train timesteps this trajectory visits, descending. Integer
    #: and not float: DDIM indexes ``alphas_cumprod`` with them directly, where
    #: the euler family interpolates a table at a fractional position.
    timesteps: tuple[int, ...]
    #: ``(alpha_prod_t, alpha_prod_t_prev)`` per step, resolved once.
    alphas: tuple[tuple[float, float], ...]
    num_train_timesteps: int
    prediction_type: str
    #: Always 1.0 — a variance-PRESERVING trajectory starts at unit variance.
    #: Carried rather than assumed so a loop can read it off any schedule.
    init_noise_sigma: float = 1.0

    def __post_init__(self) -> None:
        if not self.timesteps:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID, "a schedule needs at least one step"
            )
        if len(self.alphas) != len(self.timesteps):
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"{len(self.alphas)} alpha pairs do not walk "
                f"{len(self.timesteps)} timesteps",
            )

    def __len__(self) -> int:
        return len(self.timesteps)

    def _alphas(self, index: int) -> tuple[float, float]:
        if not 0 <= index < len(self):
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"step {index} is outside this schedule's {len(self)} steps",
            )
        return self.alphas[index]

    def scale_model_input(self, index: int, sample: Tensor) -> Tensor:
        """The IDENTITY, and it is here so a loop does not have to know that.

        ``DDIMScheduler.scale_model_input`` returns its argument: the
        variance-preserving parameterisation keeps the sample at unit scale, so
        there is nothing to divide out. Present rather than absent because the
        alternative is a caller branching on which scheduler it got, which is
        the branch that eventually forgets one.
        """

        self._alphas(index)
        return sample

    def predicted(self, index: int, model_output: Tensor, sample: Tensor) -> Tensor:
        """The denoised sample this step predicts — ``x_0``."""

        alpha, _ = self._alphas(index)
        beta = _f32(1.0 - alpha)
        if self.prediction_type == "epsilon":
            return (sample - _f32(math.sqrt(beta)) * model_output) / _f32(math.sqrt(alpha))
        return _f32(math.sqrt(alpha)) * sample - _f32(math.sqrt(beta)) * model_output

    def step(self, index: int, model_output: Tensor, sample: Tensor) -> Tensor:
        """One deterministic DDIM step — formula (12) of arXiv:2010.02502.

        ``x_{t-1} = sqrt(a_prev) * x_0_hat + sqrt(1 - a_prev) * eps_hat``, with
        ``eta = 0`` so the variance term vanishes. Written in the reference's
        own moves rather than simplified, for the same reason the Euler step is.
        """

        alpha, previous = self._alphas(index)
        original = self.predicted(index, model_output, sample)
        if self.prediction_type == "epsilon":
            residual = model_output
        else:
            beta = _f32(1.0 - alpha)
            residual = _f32(math.sqrt(alpha)) * model_output + _f32(math.sqrt(beta)) * sample
        direction = _f32(math.sqrt(_f32(1.0 - previous))) * residual
        return _f32(math.sqrt(previous)) * original + direction

    def scale_noise(self, noise: Tensor, sample: Tensor, index: int = 0) -> Tensor:
        """The variance-PRESERVING forward: ``sqrt(a) * x + sqrt(1-a) * noise``.

        Not ``x + sigma * noise``. Using the exploding form on an alpha
        trajectory produces a plausible, wrong img2img strength.
        """

        alpha, _ = self._alphas(index)
        return _f32(math.sqrt(alpha)) * sample + _f32(math.sqrt(_f32(1.0 - alpha))) * noise


@dataclass(frozen=True, slots=True)
class Trained:
    """The trained noise schedule the three U-Net-family kinds descend from.

    Every field is a DECLARED family fact read out of a recipe's scheduler
    block. The defaults are ``diffusers``' own class defaults and NOT
    Stable Diffusion's — a family that wants SD's trained noise schedule
    declares ``beta_start``/``beta_end``/``beta_schedule`` explicitly, exactly
    as FLUX declares its shift constants. Silently defaulting to SD's numbers
    would put a family fact in this module, which is the drift
    ``check_model_bindings.py`` exists to refuse one level up.

    Shared by :class:`EulerDiscrete`, :class:`EulerAncestralDiscrete` and
    :class:`Ddim` because they are three walks over ONE table, and a family
    declaring several of them (pgw#1346 K10 — the sampler is a tuned value)
    must not be able to state the same trained schedule three different ways.
    """

    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    prediction_type: str = "epsilon"
    timestep_spacing: str = "linspace"
    steps_offset: int = 0
    rescale_betas_zero_snr: bool = False

    #: The closed sets. Every one of these is a value that changes the LADDER,
    #: so an unrecognised spelling refuses instead of falling through.
    BETA_SCHEDULES: ClassVar[tuple[str, ...]] = ("linear", "scaled_linear")
    PREDICTION_TYPES: ClassVar[tuple[str, ...]] = ("epsilon", "v_prediction")
    SPACINGS: ClassVar[tuple[str, ...]] = ("linspace", "leading", "trailing")

    #: The parameters every kind below reads, before its own are added.
    TRAINED_PARAMETERS: ClassVar[tuple[str, ...]] = (
        "beta_end",
        "beta_schedule",
        "beta_start",
        "num_train_timesteps",
        "prediction_type",
        "rescale_betas_zero_snr",
        "steps_offset",
        "timestep_spacing",
    )

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
        ):
            if value not in allowed:
                raise _refuse(name, f"one of {list(allowed)!r}", value)

    @classmethod
    def _trained_kwargs(cls, block: SchedulerBlock) -> dict[str, SchedulerValue]:
        """The trained-schedule half of a block, parsed. Shared by every kind."""

        return {
            "num_train_timesteps": _count(block, "num_train_timesteps", 1000),
            "beta_start": _real(block, "beta_start", 0.0001),
            "beta_end": _real(block, "beta_end", 0.02),
            "beta_schedule": _choice(block, "beta_schedule", "linear", cls.BETA_SCHEDULES),
            "prediction_type": _choice(
                block, "prediction_type", "epsilon", cls.PREDICTION_TYPES
            ),
            "timestep_spacing": _choice(block, "timestep_spacing", "linspace", cls.SPACINGS),
            "steps_offset": _count(block, "steps_offset", 0),
            "rescale_betas_zero_snr": _flag(block, "rescale_betas_zero_snr", False),
        }

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
            raise _refuse(
                "prediction_type", f"one of {list(self.PREDICTION_TYPES)!r}", prediction_type
            )
        if prediction_type == self.prediction_type and not (
            prediction_type == "v_prediction" and not self.rescale_betas_zero_snr
        ):
            return self
        return replace(
            self,
            prediction_type=prediction_type,
            rescale_betas_zero_snr=(
                True if prediction_type == "v_prediction" else self.rescale_betas_zero_snr
            ),
        )

    def _grid(self, steps: int) -> tuple[float, ...]:
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
            _f32(_f32(_round_half_even(total - index * span)) - 1.0) for index in range(count)
        )

    def _ladder(self, steps: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """``(timesteps, sigmas)`` for the two variance-EXPLODING kinds.

        No ``image_seq_len``: unlike the flow-match schedule, this ladder does
        not consult the resolution at all — the same 28 sigmas serve a
        1024x1024 render and a 1536x640 one. Stated because the opposite is
        true for FLUX and the asymmetry is otherwise a trap.
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
        timesteps = self._grid(steps)
        return timesteps, _interpolate(table, timesteps)

    def _init_noise_sigma(self, sigmas: tuple[float, ...]) -> float:
        """``sigma_max`` under trailing/linspace and ``sqrt(sigma_max**2 + 1)``
        under leading — a fact of the SPACING, resolved once and carried."""

        maximum = max(sigmas)
        if self.timestep_spacing in ("linspace", "trailing"):
            return maximum
        return _f32(math.sqrt(_f32(_f32(maximum * maximum) + 1.0)))


@dataclass(frozen=True, slots=True)
class EulerDiscrete(Trained):
    """The U-Net families' DETERMINISTIC Euler schedule, as bare typed math.

    Reached as the ``euler`` and ``euler_trailing`` samplers. Everything about
    the trained noise schedule is :class:`Trained`'s; this class adds one
    parameter of its own, ``final_sigmas_type``, and the deterministic walk.
    """

    final_sigmas_type: str = "zero"

    KIND: ClassVar[SchedulerKind] = SchedulerKind.EULER_DISCRETE
    FINAL_SIGMAS: ClassVar[tuple[str, ...]] = ("zero", "sigma_min")
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        *Trained.TRAINED_PARAMETERS,
        "final_sigmas_type",
    )

    def __post_init__(self) -> None:
        Trained.__post_init__(self)
        if self.final_sigmas_type not in self.FINAL_SIGMAS:
            raise _refuse(
                "final_sigmas_type", f"one of {list(self.FINAL_SIGMAS)!r}", self.final_sigmas_type
            )

    @classmethod
    def from_block(cls, block: SchedulerBlock) -> Self:
        """Build one from a declaration's scheduler parameter block."""

        _only(block, cls.PARAMETERS)
        return cls(
            **cls._trained_kwargs(block),  # type: ignore[arg-type]
            final_sigmas_type=_choice(block, "final_sigmas_type", "zero", cls.FINAL_SIGMAS),
        )

    def schedule(self, steps: int) -> DiscreteSchedule:
        """The resolved sigma ladder for one request."""

        timesteps, sigmas = self._ladder(steps)
        # `sigma_min` keeps the smallest TRAINED sigma as the terminal rather
        # than landing on 0, which is a different final step and not a rounding
        # choice — a distilled recipe that names it is destroyed by `zero`.
        terminal = (
            0.0
            if self.final_sigmas_type == "zero"
            else _sigma_table(
                self.num_train_timesteps,
                self.beta_start,
                self.beta_end,
                self.beta_schedule,
                self.rescale_betas_zero_snr,
            )[0]
        )
        return DiscreteSchedule(
            sigmas=(*sigmas, _f32(terminal)),
            timesteps=timesteps,
            num_train_timesteps=self.num_train_timesteps,
            prediction_type=self.prediction_type,
            init_noise_sigma=self._init_noise_sigma(sigmas),
        )


@dataclass(frozen=True, slots=True)
class EulerAncestralDiscrete(Trained):
    """``euler_a`` — SDXL's DEFAULT sampler, and the one that consumes noise.

    Its LADDER is :class:`EulerDiscrete`'s, value for value: same trained
    table, same three spacings, same interpolation. Two things differ, and both
    are load-bearing:

    * there is **no** ``final_sigmas_type``. ``EulerAncestralDiscreteScheduler``
      always terminates at 0.0 and offers no choice, so declaring one here
      would be a parameter that changes nothing — refused by :func:`_only`
      rather than accepted and ignored;
    * the STEP contracts to ``sigma_down`` and adds ``sigma_up`` of fresh
      noise, which makes the trajectory stochastic. See
      :class:`AncestralSchedule` for where that noise comes from and how it is
      keyed, because "reproducible" is a claim about the noise and not about
      the ladder.

    This is the class pgw#1346 K10 exists for: SDXL's declared block was its
    TRAINED schedule under ``euler_discrete``, which is not the sampler most
    requests get — ``SdxlTuned.scheduler`` defaults to ``euler_a``.
    """

    KIND: ClassVar[SchedulerKind] = SchedulerKind.EULER_ANCESTRAL_DISCRETE
    PARAMETERS: ClassVar[tuple[str, ...]] = Trained.TRAINED_PARAMETERS

    @classmethod
    def from_block(cls, block: SchedulerBlock) -> Self:
        """Build one from a declaration's scheduler parameter block."""

        _only(block, cls.PARAMETERS)
        return cls(**cls._trained_kwargs(block))  # type: ignore[arg-type]

    def schedule(self, steps: int) -> AncestralSchedule:
        """The resolved sigma ladder for one request. Terminates at 0.0."""

        timesteps, sigmas = self._ladder(steps)
        return AncestralSchedule(
            sigmas=(*sigmas, 0.0),
            timesteps=timesteps,
            num_train_timesteps=self.num_train_timesteps,
            prediction_type=self.prediction_type,
            init_noise_sigma=self._init_noise_sigma(sigmas),
        )


@dataclass(frozen=True, slots=True)
class Ddim(Trained):
    """``ddim`` / ``ddim_trailing`` — deterministic, and NOT a sigma walk.

    Reached by three endpoint paths that pgw#1346 B2 enumerated: sd15's payload
    enum offers ``ddim``, sd15's ``generate_hyper`` pins ``ddim_trailing``
    unconditionally, and sdxl's enum offers ``ddim_trailing``. B2 recorded them
    and could not implement any of them, because the declaration held ONE
    scheduler; K10 is the change that lets a family declare this one BESIDE its
    euler entries.

    Two of its parameters are its own, and both are silent when wrong:

    * ``set_alpha_to_one`` decides whether the trajectory's LAST step targets
      ``alpha_bar = 1`` (a perfectly clean sample) or ``alphas_cumprod[0]``.
      Stable Diffusion's shipped ``scheduler_config.json`` says ``false`` and
      ``DDIMScheduler``'s class default says ``true``, so an omitted value
      resolves to the opposite of what every SD checkpoint was configured with;
    * ``clip_sample`` clamps the predicted ``x_0``. Every SD config sets it
      false. It is DECLARABLE and only ``false`` is implementable here — see
      :meth:`from_block` — because a true arm needs a tensor clamp with a
      declared range, and no endpoint on this fleet asks for one.
    """

    set_alpha_to_one: bool = True
    clip_sample: bool = False

    KIND: ClassVar[SchedulerKind] = SchedulerKind.DDIM
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        *Trained.TRAINED_PARAMETERS,
        "clip_sample",
        "set_alpha_to_one",
    )

    def __post_init__(self) -> None:
        Trained.__post_init__(self)
        if self.clip_sample:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                "clip_sample=True is declarable and not implemented: it needs a declared "
                "clip range and no endpoint on this fleet configures one. Declare it false "
                "or file the range, rather than getting an unclamped trajectory silently",
            )

    @classmethod
    def from_block(cls, block: SchedulerBlock) -> Self:
        """Build one from a declaration's scheduler parameter block."""

        _only(block, cls.PARAMETERS)
        return cls(
            **cls._trained_kwargs(block),  # type: ignore[arg-type]
            set_alpha_to_one=_flag(block, "set_alpha_to_one", True),
            clip_sample=_flag(block, "clip_sample", False),
        )

    def _grid(self, steps: int) -> tuple[float, ...]:
        """DDIM's INTEGER timestep grid, which is not euler's float one.

        ``leading`` and ``trailing`` agree with the euler family. ``linspace``
        does NOT: ``DDIMScheduler`` rounds it to integers and the euler
        schedulers keep the fractional position and interpolate their table at
        it. Sharing one grid would put a half-step error into every linspace
        DDIM request — visible as a slightly wrong image and nothing else.
        """

        total = self.num_train_timesteps
        if self.timestep_spacing == "linspace":
            if steps == 1:
                return (0.0,)
            span = (total - 1) / (steps - 1)
            return tuple(_round_half_even(index * span) for index in reversed(range(steps)))
        return Trained._grid(self, steps)

    def schedule(self, steps: int) -> DdimSchedule:
        """The resolved alpha trajectory for one request."""

        if steps < 1:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID, f"a schedule needs at least one step, got {steps}"
            )
        if steps > self.num_train_timesteps:
            raise ModelError(
                ModelRefusal.SCHEDULER_INVALID,
                f"DDIM walks the trained grid, so {steps} steps cannot exceed its "
                f"{self.num_train_timesteps} timesteps",
            )
        # `clamp_terminal` FALSE: `DDIMScheduler` does not overwrite the last
        # alpha under zero-terminal-SNR rescaling, where the euler schedulers
        # do. See `_alphas_cumprod`.
        table = _alphas_cumprod(
            self.num_train_timesteps,
            self.beta_start,
            self.beta_end,
            self.beta_schedule,
            self.rescale_betas_zero_snr,
            False,
        )
        final = 1.0 if self.set_alpha_to_one else table[0]
        stride = self.num_train_timesteps // steps
        timesteps = tuple(int(value) for value in self._grid(steps))
        alphas = tuple(
            (table[timestep], table[timestep - stride] if timestep - stride >= 0 else final)
            for timestep in timesteps
        )
        return DdimSchedule(
            timesteps=timesteps,
            alphas=alphas,
            num_train_timesteps=self.num_train_timesteps,
            prediction_type=self.prediction_type,
        )


#: Which class implements which name. Read by the BINDING GENERATOR to pick a
#: return annotation, and by nothing at request time — a handler reaches its
#: scheduler through the generated ``inst.scheduler()``, whose return type is
#: the concrete class (or the closed UNION of the ones the family declares), so
#: no request-path code indexes this table.
IMPLEMENTED: Final[Mapping[SchedulerKind, str]] = {
    SchedulerKind.FLOW_MATCH_EULER_DISCRETE: "FlowMatchEulerDiscrete",
    SchedulerKind.EULER_DISCRETE: "EulerDiscrete",
    SchedulerKind.DPMSOLVER_MULTISTEP: "DPMSolverMultistep",
    SchedulerKind.UNIPC_MULTISTEP: "UniPCMultistep",
    SchedulerKind.EULER_ANCESTRAL_DISCRETE: "EulerAncestralDiscrete",
    SchedulerKind.DDIM: "Ddim",
}


__all__ = [
    "AncestralSchedule",
    "DPMSolverMultistep",
    "Ddim",
    "DdimSchedule",
    "DiscreteSchedule",
    "DpmSolverSchedule",
    "EulerAncestralDiscrete",
    "EulerDiscrete",
    "FlowMatchEulerDiscrete",
    "IMPLEMENTED",
    "MultistepHistory",
    "Schedule",
    "SchedulerBlock",
    "SchedulerKind",
    "SchedulerValue",
    "Step",
    "Trained",
    "UniPCMultistep",
    "UniPcHistory",
    "UniPcSchedule",
    "VarianceExploding",
    "parse_kind",
]
