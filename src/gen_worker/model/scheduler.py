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
    #: The epsilon/v-prediction Euler schedule SDXL declares. DECLARED, not
    #: implemented here: pgw#1331 covers one family end to end and inventing
    #: SDXL's math with nothing measuring it is how a wrong schedule ships.
    EULER_DISCRETE = "euler_discrete"


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


#: What a scheduler block's values may be — ``recipe_v1``'s finite JSON scalars.
SchedulerValue = bool | int | float | str
SchedulerBlock = Mapping[str, SchedulerValue]

def _refuse(name: str, wanted: str, value: object) -> ModelError:
    return ModelError(
        ModelRefusal.SCHEDULER_INVALID,
        f"scheduler parameter {name!r} must be {wanted}, got {value!r}",
    )


def _flag(block: SchedulerBlock, name: str, default: bool) -> bool:
    """Read one declared boolean.

    Checked as ``bool`` and not as ``int`` even though ``bool`` IS an ``int`` in
    Python: a block that said ``use_dynamic_shifting: 1`` means something the
    author did not write, and accepting it is how a schedule silently changes.
    """

    value = block.get(name, default)
    if not isinstance(value, bool):
        raise _refuse(name, "a boolean", value)
    return value


def _count(block: SchedulerBlock, name: str, default: int) -> int:
    """Read one declared integer. ``bool`` is refused, for the reason above."""

    value = block.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _refuse(name, "an integer", value)
    return value


def _real(block: SchedulerBlock, name: str, default: float) -> float:
    """Read one declared real. An integer literal is a legal spelling of one."""

    value = block.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refuse(name, "a real number", value)
    return float(value)


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


#: Which class implements which name. Read by the BINDING GENERATOR to pick a
#: return annotation, and by nothing at request time — a handler reaches its
#: scheduler through the generated ``inst.scheduler()``, whose return type is
#: the concrete class, so no request-path code indexes this table.
IMPLEMENTED: Final[Mapping[SchedulerKind, str]] = {
    SchedulerKind.FLOW_MATCH_EULER_DISCRETE: "FlowMatchEulerDiscrete",
}


__all__ = [
    "IMPLEMENTED",
    "FlowMatchEulerDiscrete",
    "Schedule",
    "SchedulerBlock",
    "SchedulerKind",
    "SchedulerValue",
    "Step",
    "parse_kind",
]
