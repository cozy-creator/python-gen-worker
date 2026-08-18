"""The float32 discipline every declared solver reproduces, in ONE place.

pgw#1346 B2 established the rule and paid for it three times over: a scheduler
ladder that agrees with ``diffusers`` *algebraically* does not agree with it
*numerically*, because the reference is built at a precision — float32, stage by
stage, with a double accumulator inside ``cumprod`` — that the obvious float64
reading of the same formulae is 201 float32 ULP away from.

``euler_discrete``, ``dpmsolver_multistep`` and ``unipc_multistep`` all descend
from the SAME trained beta table, and a second copy of ``_sigma_table`` would be
a second declaration of the fleet's noise schedule — the drift
``check_model_bindings.py`` exists to refuse one level up, reappearing one level
down. So the primitives live here and every solver imports them.

**No array library, ever.** ``struct`` performs the one float32 round trip and
``math`` performs the rest. The adopt-only serve role (pgw#1328) holds this
module for free, and — the property that turned out to matter more — every value
below is IEEE double arithmetic with explicit narrowings, so it cannot vary with
which CPU kernel a pod's torch happened to dispatch. See
``tests/test_dit_solvers_pgw1346.py`` for the subprocess fence that proves it.
"""

from __future__ import annotations

import math
import struct
from functools import lru_cache
from typing import Final

#: One float32 round-trip.
_F32: Final = struct.Struct("<f")


def f32(value: float) -> float:
    """``value`` rounded to the nearest float32, as a Python float."""

    return float(_F32.unpack(_F32.pack(value))[0])


def linspace_f32(start: float, end: float, count: int) -> tuple[float, ...]:
    """``torch.linspace(start, end, count, dtype=torch.float32)``, exactly.

    torch's float32 CPU kernel computes a float32 ``step`` and walks OUTWARD
    FROM BOTH ENDS — ``start + step*i`` for the first half, ``end - step*(n-1-i)``
    for the second — so both endpoints land on their float32 selves. The
    straightforward ``start + (end-start)*i/(n-1)`` disagrees with it on 307 of
    1000 entries (B2, measured).

    This is the one place the module reproduces an IMPLEMENTATION rather than a
    contract, and torch's own two CPU kernels disagree with each other here by
    1 ULP on 145 of 1000 entries — which is why nothing downstream may assert
    bit-equality against a torch-derived reference.
    """

    first = f32(start)
    last = f32(end)
    if count == 1:
        return (first,)
    step = f32((last - first) / (count - 1))
    halfway = count // 2
    return tuple(
        f32(first + step * index) if index < halfway else f32(last - step * (count - 1 - index))
        for index in range(count)
    )


def linspace_f64(start: float, stop: float, count: int) -> tuple[float, ...]:
    """``numpy.linspace(start, stop, count)``, exactly — a DIFFERENT algorithm.

    numpy computes ``arange(count) * step`` and then ``+= start``, and finally
    OVERWRITES the last entry with ``stop`` verbatim. It does not walk in from
    both ends the way torch does, and the two therefore resolve different
    ladders from the same three arguments; the flow-sigma and Karras ladders
    below are numpy's, the beta table is torch's, and mixing them up is a silent
    schedule change rather than an error.

    ``numpy.linspace(a, b, 1)`` is ``[a]`` — it keeps the START. B2 found the
    same trap in the ``linspace`` timestep spacing at one step; it is restated
    here because a one-step distilled recipe is a real endpoint configuration.
    """

    if count < 1:
        raise ValueError(f"a linspace needs at least one point, got {count}")
    if count == 1:
        return (float(start),)
    step = (stop - start) / (count - 1)
    values = [index * step + start for index in range(count)]
    values[-1] = float(stop)
    return tuple(values)


def round_half_even(value: float) -> float:
    """numpy's ``round``: ties to even. Python's ``round`` agrees on floats."""

    return float(round(value))


def truncate_to_int(value: float) -> float:
    """``torch.from_numpy(x).to(torch.int64)`` — truncation TOWARD ZERO.

    Not rounding, and the difference is a whole timestep: the flow-sigma
    ladder's ``sigma * num_train_timesteps`` lands on 973.013 and the model is
    conditioned on **973**. Wan's own served grid (``[999, 973, 923, 800]`` at
    4 steps / shift 12) is the fixture that pins it.
    """

    return float(int(value))


@lru_cache(maxsize=32)
def sigma_table(
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
    beta_schedule: str,
    rescale_betas_zero_snr: bool,
) -> tuple[float, ...]:
    """The full ``num_train_timesteps``-long sigma table, ASCENDING.

    ``sqrt((1 - alphas_cumprod) / alphas_cumprod)`` — the variance-exploding
    ladder every diffusion-objective solver in this package interpolates. The
    three narrowings B2 measured are reproduced verbatim:

      (a) ``betas`` through torch's float32 ``linspace`` kernel;
      (b) ``alphas_cumprod`` accumulated in a DOUBLE running product and
          narrowed to float32 only on store, which is what torch's ``cumprod``
          does for float32 CPU tensors (a float32 accumulator is 15 ULP wrong);
      (c) ``1-ac`` and the quotient as SEPARATE float32 operations, because they
          are separate float32 tensor ops upstream — folding them into one
          double expression is a different, more accurate, WRONG number.

    One deliberate divergence, and it is the reason nothing here may be asserted
    in ULP: upstream spells the square root ``** 0.5``, which on a tensor is
    ``torch.pow`` and is NOT correctly rounded. This uses ``math.sqrt``, which
    is. Where the two differ, ours is the more accurate and the more
    deterministic of the pair, and no amount of care makes a correctly-rounded
    root bit-match a ``pow`` that is not.

    Cached: it depends on nothing per-request and every request of one family
    asks for the same one.
    """

    return tuple(
        f32(math.sqrt(f32(f32(1.0 - alpha) / alpha)))
        for alpha in alphas_cumprod(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            rescale_betas_zero_snr,
            True,
        )
    )


@lru_cache(maxsize=32)
def alphas_cumprod(
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
    beta_schedule: str,
    rescale_betas_zero_snr: bool,
    clamp_terminal: bool,
) -> tuple[float, ...]:
    """The trained ``alphas_cumprod`` table — the root every kind descends from.

    ``clamp_terminal`` is NOT a preference, and it is the one place the
    diffusion-objective kinds genuinely disagree about the same table
    (pgw#1346 K10). Under zero-terminal-SNR rescaling the euler schedulers and
    the multistep solvers overwrite the LAST entry with the smallest positive
    fp16 subnormal so the first SIGMA is finite; ``DDIMScheduler`` does not,
    because it walks alphas rather than sigmas and has no infinity to avoid.
    Reading one class's table into another is a silently different first step —
    the one furthest from the data manifold, so the most visible one — which is
    why the flag is a parameter here rather than a constant.
    """

    if beta_schedule == "scaled_linear":
        roots = linspace_f32(math.sqrt(beta_start), math.sqrt(beta_end), num_train_timesteps)
        betas = [f32(root * root) for root in roots]
    else:
        betas = list(linspace_f32(beta_start, beta_end, num_train_timesteps))

    if rescale_betas_zero_snr:
        betas = rescale_zero_terminal_snr(betas)

    cumulative = 1.0
    table: list[float] = []
    for beta in betas:
        cumulative *= f32(1.0 - beta)
        table.append(f32(cumulative))
    if rescale_betas_zero_snr and clamp_terminal:
        # Close to zero without being zero, so the first sigma is not inf.
        # Upstream's value verbatim: the smallest positive fp16 subnormal.
        table[-1] = 2.0**-24
    return tuple(table)


def rescale_zero_terminal_snr(betas: list[float]) -> list[float]:
    """Rescale betas so the terminal signal-to-noise ratio is zero.

    arXiv:2305.08891 Algorithm 1, and the transform ``gen_worker.view`` already
    turns on for every v-prediction checkpoint — so a v-pred fine-tune served
    through any solver here needs it to reach the same ladder the diffusers path
    reaches. Same float32 discipline as its caller.
    """

    cumulative = 1.0
    roots: list[float] = []
    for beta in betas:
        cumulative *= f32(1.0 - beta)
        roots.append(f32(math.sqrt(f32(cumulative))))
    first, last = roots[0], roots[-1]
    scale = f32(first / f32(first - last))
    shifted = [f32(f32(root - last) * scale) for root in roots]
    bars = [f32(root * root) for root in shifted]
    alphas = [bars[0]] + [f32(bars[index] / bars[index - 1]) for index in range(1, len(bars))]
    return [f32(1.0 - alpha) for alpha in alphas]


__all__ = [
    "alphas_cumprod",
    "f32",
    "linspace_f32",
    "linspace_f64",
    "rescale_zero_terminal_snr",
    "round_half_even",
    "sigma_table",
    "truncate_to_int",
]
