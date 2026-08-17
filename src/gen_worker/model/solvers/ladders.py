"""The sigma LADDERS a multistep solver may walk, as bare functions.

A solver is two separable things: the ladder it descends and the update rule it
applies between rungs. Splitting them is not tidiness — the SAME
``DPMSolverMultistep`` update runs on three different ladders depending on one
declared boolean, and ``dpmpp_2m`` vs ``dpmpp_2m_karras`` differ in NOTHING else.
Keeping the ladders as pure functions is what makes that statement checkable
instead of a claim about a branch buried in ``set_timesteps``.

Each function reproduces the precision of the reference stage by stage, per
pgw#1346 B2's rule. The stages differ per ladder and the differences are
load-bearing, so they are named at each function rather than summarized here.
"""

from __future__ import annotations

import math

from .precision import f32, linspace_f64, round_half_even, truncate_to_int


def interpolate_table(table: tuple[float, ...], timesteps: tuple[float, ...]) -> tuple[float, ...]:
    """``numpy.interp(timesteps, arange(len(table)), table)``.

    FLOAT64 interpolation over a float32 table — numpy's ``interp`` promotes,
    and matching that promotion is what keeps the resolved ladder on the
    reference's own value rather than a more-accurate neighbour of it.

    The multistep solvers reach this with INTEGER timesteps (their spacings emit
    ``int64``), so every interpolation lands exactly on a table entry; the
    fractional branch is here because ``euler_discrete``'s ``linspace`` spacing
    does not, and one interpolation shared beats two that may drift.
    """

    last = len(table) - 1
    resolved: list[float] = []
    for timestep in timesteps:
        if timestep <= 0.0:
            resolved.append(table[0])
        elif timestep >= last:
            resolved.append(table[last])
        else:
            low = int(timestep)
            resolved.append(table[low] + (table[low + 1] - table[low]) * (timestep - low))
    return tuple(resolved)


def karras_sigmas(sigma_min: float, sigma_max: float, steps: int, rho: float = 7.0) -> tuple[float, ...]:
    """The Karras ladder (arXiv:2206.00364 eq. 5), DESCENDING.

    ``rho = 7`` is the paper's value and is not a declared parameter anywhere in
    this fleet — ``diffusers`` hardcodes it too, so exposing it would invent an
    axis no endpoint can reach.

    Precision: the endpoints ``sigma_min``/``sigma_max`` arrive as float32 table
    entries (upstream takes them through ``.item()`` off a float32 tensor) and
    everything after is FLOAT64 — ``ramp`` is numpy's ``linspace``, not torch's,
    and the two are different algorithms. The result is narrowed to float32 once,
    at the end, by the caller that assembles the ladder.

    At one step ``ramp`` is ``[0.0]``, so the ladder is the single value
    ``sigma_max``: numpy's ``linspace(0, 1, 1)`` keeps the START. A one-step
    distilled recipe is a real configuration here, so this is measured rather
    than assumed.
    """

    if steps < 1:
        raise ValueError(f"a karras ladder needs at least one step, got {steps}")
    ramp = linspace_f64(0.0, 1.0, steps)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    return tuple((max_inv_rho + point * (min_inv_rho - max_inv_rho)) ** rho for point in ramp)


def exponential_sigmas(sigma_min: float, sigma_max: float, steps: int) -> tuple[float, ...]:
    """A geometric ladder: ``exp(linspace(log(sigma_max), log(sigma_min), n))``.

    **Not reachable from any endpoint in this fleet today**, and it is here for
    one reason: it is the sibling branch of :func:`karras_sigmas` inside the same
    ``set_timesteps``, and a solver that implements one branch of a two-branch
    switch has an untested edge rather than a missing feature. It is verified
    against ``diffusers`` exactly as the reachable ladders are, and
    ``tests/test_dit_solvers_pgw1346.py`` asserts that NO payload enum in the
    fleet selects it — so the day one does, the assertion is what tells us the
    fact changed, rather than a render.
    """

    if steps < 1:
        raise ValueError(f"an exponential ladder needs at least one step, got {steps}")
    return tuple(
        math.exp(point)
        for point in linspace_f64(math.log(sigma_max), math.log(sigma_min), steps)
    )


def flow_sigmas(steps: int, num_train_timesteps: int, shift: float) -> tuple[float, ...]:
    """The rectified-flow ladder a multistep solver walks, DESCENDING.

    This is NOT ``FlowMatchEulerDiscrete``'s ladder, and the difference is the
    thing that makes wan-2.2 and hidream reproducible rather than nearly so:
    ``FlowMatchEulerDiscrete`` spaces ``steps`` points from 1.0 down to
    ``1/steps``; the multistep solvers space ``steps + 1`` points from 1.0 down
    to ``1/num_train_timesteps`` and DROP THE LAST. Same shift map, different
    grid, and swapping them is a schedule change no assertion about "flow
    matching" would catch.

    The ``eps`` nudge at the top is upstream's and is required, not cosmetic: at
    ``sigma == 1`` the flow reading gives ``alpha_t = 1 - sigma = 0`` and
    ``log(alpha_t)`` is ``-inf``, so the first predictor step would produce NaN.
    Upstream subtracts ``1e-6``; reproducing the exact constant is what keeps our
    first timestep 999 rather than 1000.

    Precision: numpy ``linspace`` in float64, the shift map in float64, narrowed
    to float32 once by the caller.
    """

    if steps < 1:
        raise ValueError(f"a flow ladder needs at least one step, got {steps}")
    if num_train_timesteps < 1:
        raise ValueError(f"num_train_timesteps must be positive, got {num_train_timesteps}")
    raw = linspace_f64(1.0, 1.0 / num_train_timesteps, steps + 1)[:-1]
    shifted = [shift * point / (1.0 + (shift - 1.0) * point) for point in raw]
    if abs(shifted[0] - 1.0) < 1e-6:
        shifted[0] -= 1e-6
    return tuple(shifted)


def flow_timesteps(sigmas: tuple[float, ...], num_train_timesteps: int) -> tuple[float, ...]:
    """``(sigmas * num_train_timesteps)`` cast to int64 — TRUNCATED, not rounded.

    Kept beside :func:`flow_sigmas` because the pair is one fact: 0.973013 is
    timestep **973**, and rounding it to 973 by luck at four steps and to 974 at
    forty is the shape of a bug that only shows up on the long lane.
    """

    return tuple(truncate_to_int(sigma * num_train_timesteps) for sigma in sigmas)


def sigma_to_t(sigma: float, log_table: tuple[float, ...]) -> float:
    """Invert the trained table: which (fractional) train timestep is ``sigma``?

    Required by the Karras and exponential ladders and by nothing else: those
    two SYNTHESIZE sigmas that are not table entries, so the timestep the model
    is conditioned on has to be read back out of the table by interpolating in
    LOG-sigma. The reachable ladders that interpolate the table in the other
    direction already know their timesteps and never come here.

    Upstream's ``_sigma_to_t``, including its two clamps: the index is clipped so
    ``low_idx + 1`` is always in range, and the interpolation weight is clipped
    to ``[0, 1]`` so a sigma outside the table's span saturates at an endpoint
    instead of extrapolating off it.
    """

    log_sigma = math.log(max(sigma, 1e-10))
    # `cumsum(dists >= 0).argmax()` over an ASCENDING table is the index of the
    # last entry at or below `log_sigma` — and 0 when there is none, because
    # argmax over an all-zero cumsum returns the first index.
    below = sum(1 for entry in log_table if log_sigma - entry >= 0.0)
    low_idx = min(max(below - 1, 0), len(log_table) - 2)
    high_idx = low_idx + 1
    low, high = log_table[low_idx], log_table[high_idx]
    weight = (low - log_sigma) / (low - high)
    weight = min(max(weight, 0.0), 1.0)
    return (1.0 - weight) * low_idx + weight * high_idx


def log_table(table: tuple[float, ...]) -> tuple[float, ...]:
    """``numpy.log`` of the float32 sigma table, which is itself float32.

    Narrowed, because upstream takes the log of a float32 array and gets a
    float32 array back; the subsequent arithmetic in :func:`sigma_to_t` promotes
    to float64 exactly as numpy does when it meets a float64 scalar.
    """

    return tuple(f32(math.log(entry)) for entry in table)


def discrete_timesteps(
    spacing: str,
    steps: int,
    num_train_timesteps: int,
    steps_offset: int,
) -> tuple[float, ...]:
    """The multistep solvers' timestep grid — Table 2 of arXiv:2305.08891.

    **Deliberately not shared with** ``EulerDiscrete._timesteps``, and the reason
    is a real off-by-one rather than a preference: the multistep solvers space
    over ``num_inference_steps + 1`` points and drop the last, where Euler spaces
    over ``num_inference_steps``. At 28 steps under ``leading`` that is a stride
    of 34 here and 35 there — a different schedule, from the same three
    arguments. Folding the two into one helper is how the two families would
    quietly become one.

    All three branches are integer arithmetic on ``int64``, which is why the
    resolved grid is EXACT on every machine and is asserted as such.
    """

    if spacing == "linspace":
        # `linspace(...).round()[::-1][:-1]` — REVERSE first, THEN drop, which
        # keeps the TOP of the ladder and discards timestep 0. Dropping before
        # reversing keeps 0 and discards the top: the same three lines in the
        # other order serve a schedule that never reaches the noisy end.
        points = linspace_f64(0.0, num_train_timesteps - 1, steps + 1)
        return tuple(round_half_even(point) for point in reversed(points[1:]))
    if spacing == "leading":
        # INTEGER division, and the stride divides `steps + 1` rather than
        # `steps` — that is the whole difference from the Euler family's grid.
        stride = num_train_timesteps // (steps + 1)
        walk = [round_half_even(index * stride) for index in range(steps + 1)]
        return tuple(point + steps_offset for point in reversed(walk[1:]))
    if spacing == "trailing":
        span = num_train_timesteps / steps
        total = math.ceil(num_train_timesteps / span)
        return tuple(
            round_half_even(num_train_timesteps - index * span) - 1.0 for index in range(total)
        )
    raise ValueError(f"unknown timestep spacing {spacing!r}")


__all__ = [
    "discrete_timesteps",
    "exponential_sigmas",
    "flow_sigmas",
    "flow_timesteps",
    "interpolate_table",
    "karras_sigmas",
    "log_table",
    "sigma_to_t",
]
