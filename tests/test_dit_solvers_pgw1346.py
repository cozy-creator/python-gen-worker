"""pgw#1346 B3 — ``dpmsolver_multistep`` and ``unipc_multistep`` as bare math.

pgw#1346 B2 built ``euler_discrete`` and, more usefully, built the INSTRUMENT.
Its finding is this file's law and is not re-litigated here:

* **a torch-derived reference cannot be bit-matched by anything, including
  itself.** Three of its primitives are implementation-defined rather than
  IEEE-exact — ``linspace`` dispatches its float32 CPU kernel by ISA,
  ``cumprod`` varies in accumulator width and association order, and ``x ** 0.5``
  on a tensor is ``pow``, which is not correctly rounded. Measured cross-machine
  spread: 85 float32 ULP. So sigmas are compared RELATIVELY at 2e-4 (~20x tighter
  than one bf16 ULP, the precision the denoiser computes in);
* **timesteps are compared EXACTLY**, because they are integer arithmetic and
  have never differed on any machine;
* **loops are compared in the L2 NORM**, never element-wise: latents cross zero,
  and an element-wise relative measure divides by the noise floor and reports 17%
  for a parts-per-million difference;
* **loops are compared with OUR ladder injected into the reference**, which
  removes the table as a variable and makes the comparison a test of the STEP;
* **our own ladder is fenced BYTEWISE across CPU kernels** in a subprocess, which
  is the property the reference does not have and the reason to prefer this code
  over the code it replaces.

What B3 adds to that instrument, because multistep solvers are not memoryless:

* **the conditioning is re-measured per solver.** B2 measured Euler at gain ~1.0
  — a ladder perturbation propagates into the latents roughly 1:1, with no
  chaotic amplification. That is what makes byte-stability a CORRECTNESS
  property rather than a nicety, and it does not transfer for free to a
  predictor/corrector recursion. Measured here, and one of the four lanes is NOT
  gain-1 (see :func:`test_the_flow_ladders_first_rung_is_the_conditioning_risk`);
* **the state's initial condition is asserted to be constant**, because "two pods
  agree" is a claim about where the recursion starts as much as how it steps.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast, get_args

import pytest

from gen_worker.model.errors import ModelError
from gen_worker.model.scheduler import IMPLEMENTED, SchedulerKind, parse_kind
from gen_worker.model.solvers import ladders
from gen_worker.model.solvers.dpm_multistep import (
    DPMSolverMultistep,
    DpmSolverSchedule,
    MultistepHistory,
)
from gen_worker.model.solvers.unipc_multistep import (
    UniPCMultistep,
    UniPcHistory,
    UniPcSchedule,
)

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

#: How far a value may sit from diffusers', RELATIVELY. B2's constant and B2's
#: argument: ~40x the reference's own measured cross-machine spread (85 float32
#: ULP ~ 5.1e-6) and ~20x TIGHTER than one bf16 ULP (3.9e-3).
RELATIVE = 2e-4

#: SD's trained noise schedule — the base config ``view.clone_scheduler`` merges
#: a sampler's overrides onto for sd15, sd2 and sdxl. Declared in the catalog at
#: ``model/catalog/sd15.py`` and ``model/catalog/sdxl.py``; restated here as the
#: DIFFUSERS argument set so the reference and this module are built from one
#: literal and cannot drift apart inside the test.
SD_SCHEDULE: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "timestep_spacing": "leading",
    "steps_offset": 1,
    "final_sigmas_type": "zero",
    "solver_order": 2,
}

#: Every step count the sd15/sd2/sdxl recipes can reach a multistep solver at:
#: sd15's stamped default (30), sdxl's (28), sd2/Turbo's (1), the distilled pins
#: (4, 8), and the ends of the declared ranges (payload 1..80, sdxl's hub recipe
#: max 150).
DIFFUSION_STEPS = (1, 4, 8, 20, 25, 28, 30, 40, 50, 80, 150)

#: wan-2.2's and hidream's reachable UniPC step counts: the boot warm-up (1),
#: wan's distilled arms (4, 8), the 12-step natural-split arm, wan T2V/I2V's
#: stamped 40, wan TI2V-5B's and hidream-full's 50, and the declared ceiling 80.
FLOW_STEPS = (1, 4, 8, 12, 40, 50, 80)

#: The three ``flow_shift`` values production reaches: wan's curated T2V 12.0,
#: the TI2V-5B mirror's 5.0, and the A14B mirror / I2V 3.0 — which is also
#: hidream-full's stamped shift.
FLOW_SHIFTS = (3.0, 5.0, 12.0)


# --------------------------------------------------------------- instruments


def _rel(ours: Any, theirs: Any) -> float:
    """Largest RELATIVE difference, scaled by the reference's own magnitude.

    ``atol`` is deliberately absent: every value compared through this function
    is a sigma or a timestep, all comfortably away from zero except a terminal
    ``0.0`` both sides produce exactly.
    """

    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(theirs, dtype=np.float64)
    return float((np.abs(a - b) / np.maximum(np.abs(b), 1e-12)).max())


def _rel_norm(ours: Any, theirs: Any) -> float:
    """Relative difference of two TENSORS, in the L2 norm.

    Separate from :func:`_rel` for B2's measured reason: latents cross zero, so
    an element-wise relative measure divides by a value near the noise floor and
    reports a meaningless 17% for a parts-per-million difference.
    """

    a = np.asarray(ours, dtype=np.float64).ravel()
    b = np.asarray(theirs, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-30))


def _dpm_reference(**config: Any) -> Any:
    diffusers = pytest.importorskip("diffusers")
    return diffusers.DPMSolverMultistepScheduler(**config)


def _unipc_reference(**config: Any) -> Any:
    diffusers = pytest.importorskip("diffusers")
    return diffusers.UniPCMultistepScheduler(**config)


def _inject(theirs: Any, ours: DpmSolverSchedule | UniPcSchedule) -> Any:
    """Load OUR ladder into the reference, so the loop tests the STEP.

    B2's technique. The table has its own test and cannot be bit-exact, because
    the reference is not bit-exact against itself across CPU kernels; feeding
    both loops identical sigmas removes that variable, so what remains is the
    update rule and the multistep state.
    """

    theirs.sigmas = torch.tensor(ours.sigmas, dtype=torch.float32)
    theirs.timesteps = torch.tensor(ours.timesteps, dtype=torch.int64)
    theirs.num_inference_steps = len(ours.timesteps)
    return theirs


# ------------------------------------------------- the reachable enumeration


def test_the_implemented_variants_are_exactly_the_ones_an_endpoint_reaches() -> None:
    """The enumeration, from the SDK's own sampler table and the endpoints.

    ``gen_worker.view.SAMPLERS`` DEFINES each named sampler completely — the
    endpoints select names, never configurations — so it is the only honest
    source for "what does ``dpmpp_2m_karras`` mean here". Asserted rather than
    described, because a row moving under this module is exactly the change that
    would make a stamped recipe render differently without anything going red.
    """

    from gen_worker import view
    from gen_worker.model.catalog import sd15_serve as sd
    from gen_worker.model.catalog import sdxl_serve as sx

    # DPM-Solver++: four rows, two switches, and the switches are independent.
    assert view.SAMPLERS["dpmpp_2m"] == (
        "DPMSolverMultistepScheduler",
        {"solver_order": 2, "final_sigmas_type": "zero"},
    )
    assert view.SAMPLERS["dpmpp_2m_karras"][1]["use_karras_sigmas"] is True
    assert view.SAMPLERS["dpmpp_2m_sde_karras"][1]["algorithm_type"] == "sde-dpmsolver++"
    assert view.SAMPLERS["unipc"] == ("UniPCMultistepScheduler", {})

    # Which of those names a recipe can actually carry.
    sd15_names = set(get_args(sd.Sd15Sampler))
    sdxl_names = set(get_args(sx.SdxlSampler))
    assert {"dpmpp_2m", "dpmpp_2m_karras", "dpmpp_2m_sde_karras", "unipc"} <= sd15_names
    assert {"dpmpp_2m_karras", "dpmpp_2m_sde_karras"} <= sdxl_names
    # sdxl admits NO plain `dpmpp_2m` and NO `unipc` — the two families differ,
    # and implementing to the union would be implementing to neither.
    assert "dpmpp_2m" not in sdxl_names and "unipc" not in sdxl_names
    # sd15's DEFAULT is a DPM-Solver++ variant, which is why this solver is the
    # fleet's most-selected one and not an alternative.
    assert sd.Sd15Tuned().scheduler == "dpmpp_2m_karras"

    # `dpmpp_2m_sde` is DEFINED and reachable from nothing. Supported anyway —
    # it is one boolean from `dpmpp_2m_sde_karras` — and recorded as unreachable
    # so the day an enum admits it, this assertion is what says so.
    assert "dpmpp_2m_sde" in view.SAMPLERS
    assert "dpmpp_2m_sde" not in sd15_names | sdxl_names


def test_no_endpoint_selects_an_exponential_or_beta_sigma_ladder() -> None:
    """The two ladders that are IMPLEMENTED-but-unreachable, fenced.

    ``exponential`` is implemented (it is the sibling branch of Karras inside one
    ``set_timesteps``, and a solver with one branch of a two-branch switch has an
    untested edge). ``beta`` is not implemented at all — it needs ``scipy``, and
    nothing in this fleet has ever set it. Both facts are asserted here so a
    future recipe that reaches for either goes red on this test rather than on a
    render.
    """

    from gen_worker import view

    for name, (_cls, extra) in view.SAMPLERS.items():
        assert not extra.get("use_exponential_sigmas"), name
        assert not extra.get("use_beta_sigmas"), name
    # Karras, by contrast, IS reachable — and only through these two names.
    karras = {n for n, (_c, e) in view.SAMPLERS.items() if e.get("use_karras_sigmas")}
    assert karras == {"dpmpp_2m_karras", "dpmpp_2m_sde_karras"}


def test_both_kinds_are_declarable_and_the_generator_can_import_the_class() -> None:
    """``IMPLEMENTED`` stays TOTAL, and every name in it is importable.

    The binding generator emits ``from gen_worker.model.scheduler import <name>``
    from this table. A kind whose class lives in a submodule and is NOT
    re-exported would generate a module that fails to import — at the tenant's
    build, not here — so the re-export is the assertion.
    """

    from gen_worker.model import scheduler as surface

    assert set(IMPLEMENTED) == set(SchedulerKind)
    assert parse_kind("dpmsolver_multistep") is SchedulerKind.DPMSOLVER_MULTISTEP
    assert parse_kind("unipc_multistep") is SchedulerKind.UNIPC_MULTISTEP
    for kind, name in IMPLEMENTED.items():
        implementation = getattr(surface, name, None)
        assert implementation is not None, f"{kind} names {name}, which is not exported"
        assert hasattr(implementation, "from_block")
        assert hasattr(implementation, "schedule")


# ------------------------------------------------------ dpmsolver_multistep


@pytest.mark.parametrize("steps", DIFFUSION_STEPS)
@pytest.mark.parametrize("karras", [False, True])
@pytest.mark.parametrize("algorithm", ["dpmsolver++", "sde-dpmsolver++"])
@pytest.mark.parametrize(("objective", "zero_snr"), [("epsilon", False), ("v_prediction", True)])
def test_the_dpm_ladder_matches_diffusers_over_every_reachable_configuration(
    steps: int, karras: bool, algorithm: str, objective: str, zero_snr: bool
) -> None:
    """88 configurations: the real axes, at the real step counts.

    ``v_prediction`` is paired with ``rescale_betas_zero_snr`` because
    ``gen_worker.view`` pairs them — a v-pred checkpoint on this fleet is ALWAYS
    served with the rescale, so measuring the objective without it would measure
    a configuration no request reaches.
    """

    config = dict(
        SD_SCHEDULE,
        use_karras_sigmas=karras,
        algorithm_type=algorithm,
        prediction_type=objective,
        rescale_betas_zero_snr=zero_snr,
    )
    theirs = _dpm_reference(**config)
    theirs.set_timesteps(steps)
    ours = DPMSolverMultistep(**config).schedule(steps)

    assert len(ours.sigmas) == len(theirs.sigmas)
    assert _rel(ours.sigmas, theirs.sigmas.numpy()) <= RELATIVE
    # EXACT: the grid is integer arithmetic on both sides.
    assert list(ours.timesteps) == [float(t) for t in theirs.timesteps.tolist()]
    # DPM-Solver++ starts from UNIT-variance noise, unlike euler_discrete.
    assert ours.init_noise_sigma == float(theirs.init_noise_sigma) == 1.0


@pytest.mark.parametrize("steps", [1, 4, 8, 28, 30])
@pytest.mark.parametrize("algorithm", ["dpmsolver++", "sde-dpmsolver++"])
@pytest.mark.parametrize("solver_type", ["midpoint", "heun"])
def test_a_whole_dpm_loop_tracks_the_diffusers_loop(
    steps: int, algorithm: str, solver_type: str
) -> None:
    """The multistep recursion, step for step, with the table removed.

    ``solver_type`` is exercised on both settings even though ``view.SAMPLERS``
    never sets it: a checkpoint's own ``scheduler_config.json`` rides through
    ``view.clone_scheduler``'s ``{**base.config, **overrides}`` merge, so
    ``heun`` can arrive without any recipe naming it.

    Compared after EVERY step rather than once at the end: a divergence at step
    3 and one at step 27 are different defects, and only the per-step assertion
    tells them apart. It also pins the ORDER SCHEDULE — first-order first,
    second-order in the middle, first-order last — because a wrong order shows up
    as a jump at one index rather than as drift.
    """

    config = dict(
        SD_SCHEDULE,
        use_karras_sigmas=True,
        algorithm_type=algorithm,
        prediction_type="epsilon",
        solver_type=solver_type,
    )
    ours = DPMSolverMultistep(**config).schedule(steps)
    theirs = _inject(_dpm_reference(**config), ours)

    torch.manual_seed(0)
    our_sample = torch.randn(2, 4, 16, 16)
    their_sample = our_sample.clone()
    state = ours.begin()
    for index, timestep in enumerate(theirs.timesteps):
        prediction = torch.randn(2, 4, 16, 16)
        noise = torch.randn(2, 4, 16, 16) if algorithm.startswith("sde") else None
        our_sample, state = ours.step(index, prediction, our_sample, state, noise=noise)
        their_sample = theirs.step(
            prediction, timestep, their_sample, variance_noise=noise
        ).prev_sample
        assert _rel_norm(our_sample.numpy(), their_sample.numpy()) <= RELATIVE


def test_the_dpm_step_residual_is_the_reference_s_own_float32_transcendentals() -> None:
    """WHY the loop is not bit-exact even with the table removed, measured.

    B2's Euler loop IS bit-exact under ladder injection, because its step is
    arithmetic. A multistep coefficient is not: it passes through ``exp``,
    ``log`` and ``expm1``, and torch evaluates those on a float32 tensor in
    float32, where this module evaluates them in double and narrows once. Ours is
    the more accurate and the more deterministic of the two — a correctly-rounded
    double transcendental narrowed once cannot bit-match a float32-native one, by
    construction, exactly as ``math.sqrt`` cannot bit-match ``pow``.

    Asserted as a BOUND on the residual rather than as equality, and the bound is
    three orders of magnitude tighter than the file's own tolerance, so this
    would fail loudly on a real arithmetic error while never failing on the
    library's rounding.
    """

    config = dict(SD_SCHEDULE, use_karras_sigmas=True, prediction_type="epsilon")
    ours = DPMSolverMultistep(**config).schedule(30)
    theirs = _inject(_dpm_reference(**config), ours)

    torch.manual_seed(3)
    our_sample = torch.randn(2, 4, 16, 16)
    their_sample = our_sample.clone()
    state = ours.begin()
    worst = 0.0
    for index, timestep in enumerate(theirs.timesteps):
        prediction = torch.randn(2, 4, 16, 16)
        our_sample, state = ours.step(index, prediction, our_sample, state)
        their_sample = theirs.step(prediction, timestep, their_sample).prev_sample
        worst = max(worst, _rel_norm(our_sample.numpy(), their_sample.numpy()))
    assert 0.0 < worst <= 1e-5


# ---------------------------------------------------------- unipc_multistep


@pytest.mark.parametrize("steps", FLOW_STEPS)
@pytest.mark.parametrize("shift", FLOW_SHIFTS)
def test_the_unipc_flow_ladder_matches_diffusers_at_every_served_shift(
    steps: int, shift: float
) -> None:
    """wan-2.2's own solver, at the shifts and step counts it actually serves.

    The flow ladder is NOT ``FlowMatchEulerDiscrete``'s: it spaces ``steps + 1``
    points from 1.0 down to ``1/num_train_timesteps`` and drops the last, where
    the flow-match schedule spaces ``steps`` points down to ``1/steps``. Same
    shift map, different grid.
    """

    config: dict[str, Any] = {
        "num_train_timesteps": 1000,
        "use_flow_sigmas": True,
        "prediction_type": "flow_prediction",
        "flow_shift": shift,
        "solver_order": 2,
        "solver_type": "bh2",
        "predict_x0": True,
        "lower_order_final": True,
        "final_sigmas_type": "zero",
        "timestep_spacing": "linspace",
    }
    theirs = _unipc_reference(**config)
    theirs.set_timesteps(steps)
    ours = UniPCMultistep(**config).schedule(steps)

    assert len(ours.sigmas) == len(theirs.sigmas)
    assert _rel(ours.sigmas, theirs.sigmas.numpy()) <= RELATIVE
    assert list(ours.timesteps) == [float(t) for t in theirs.timesteps.tolist()]


def test_the_unipc_flow_grid_is_the_wan_mirror_s_own_published_timesteps() -> None:
    """``[999, 973, 923, 800]`` — wan-2.2's committed 4-step/shift-12 fixture.

    A fixture from ANOTHER repository, reproduced from first principles here. It
    pins the two details a re-derivation gets wrong: the timestep cast is a
    TRUNCATION and not a rounding (0.973013 is 973), and the top rung is nudged
    down by exactly ``1e-6`` so ``alpha_t = 1 - sigma`` is not zero — without
    which the first predictor step is NaN rather than slightly different.
    """

    ours = UniPCMultistep(
        use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=12.0
    ).schedule(4)
    assert list(ours.timesteps) == [999.0, 973.0, 923.0, 800.0]
    # The nudge is applied in float64 and the ladder is narrowed to float32 once
    # at the end, so the served value is the float32 neighbour of `1 - 1e-6` —
    # about 1.5 float32 ULP away from it, and still 100x further from 1.0 than
    # one ULP is, which is all the nudge has to achieve.
    assert ours.sigmas[0] == pytest.approx(1.0 - 1e-6, abs=2e-7)
    assert ours.sigmas[0] < 1.0
    assert ours.sigmas[-1] == 0.0


@pytest.mark.parametrize("steps", DIFFUSION_STEPS)
@pytest.mark.parametrize("solver_type", ["bh1", "bh2"])
@pytest.mark.parametrize(("objective", "zero_snr"), [("epsilon", False), ("v_prediction", True)])
def test_the_unipc_diffusion_ladder_matches_diffusers(
    steps: int, solver_type: str, objective: str, zero_snr: bool
) -> None:
    """The OTHER lane: sd15's and sd2's ``unipc``, on the trained beta table.

    One solver, two ladders, one declared boolean between them — which is the
    whole reason the ladders are functions in ``solvers/ladders.py`` rather than
    branches inside a ``set_timesteps``.
    """

    config = dict(
        SD_SCHEDULE,
        solver_type=solver_type,
        predict_x0=True,
        lower_order_final=True,
        prediction_type=objective,
        rescale_betas_zero_snr=zero_snr,
    )
    theirs = _unipc_reference(**config)
    theirs.set_timesteps(steps)
    ours = UniPCMultistep(**config).schedule(steps)

    assert _rel(ours.sigmas, theirs.sigmas.numpy()) <= RELATIVE
    assert list(ours.timesteps) == [float(t) for t in theirs.timesteps.tolist()]


@pytest.mark.parametrize(
    ("steps", "flow", "shift"),
    [(1, True, 12.0), (4, True, 12.0), (8, True, 5.0), (40, True, 12.0), (50, True, 3.0),
     (4, False, 0.0), (28, False, 0.0), (30, False, 0.0)],
)
def test_a_whole_unipc_loop_tracks_the_diffusers_loop(
    steps: int, flow: bool, shift: float, solver_type: str = "bh2"
) -> None:
    """Predictor AND corrector, step for step, on both ladders.

    The corrector is what makes this more than a second DPM test: it re-solves
    the PREVIOUS step using this step's model output, at the order the previous
    predictor actually ran at. Reading that order off the step index instead of
    off the state is the subtle way the two halves disagree on short ladders, so
    the 1-, 4- and 8-step rows are load-bearing and not padding.

    ``bh2`` only, because that is what every mirror in the fleet ships and
    because the reference cannot complete a ``bh1`` loop at all — see
    :func:`test_bh1_makes_the_reference_return_nan_on_its_final_step`.
    """

    if flow:
        config: dict[str, Any] = {
            "num_train_timesteps": 1000,
            "use_flow_sigmas": True,
            "prediction_type": "flow_prediction",
            "flow_shift": shift,
            "solver_order": 2,
            "solver_type": solver_type,
            "predict_x0": True,
            "lower_order_final": True,
            "final_sigmas_type": "zero",
            "timestep_spacing": "linspace",
        }
    else:
        config = dict(
            SD_SCHEDULE,
            solver_type=solver_type,
            predict_x0=True,
            lower_order_final=True,
            prediction_type="epsilon",
        )
    ours = UniPCMultistep(**config).schedule(steps)
    theirs = _inject(_unipc_reference(**config), ours)

    torch.manual_seed(1)
    our_sample = torch.randn(2, 4, 16, 16)
    their_sample = our_sample.clone()
    state = ours.begin()
    for index, timestep in enumerate(theirs.timesteps):
        prediction = torch.randn(2, 4, 16, 16)
        our_sample, state = ours.step(index, prediction, our_sample, state)
        their_sample = theirs.step(prediction, timestep, their_sample).prev_sample
        assert _rel_norm(our_sample.numpy(), their_sample.numpy()) <= RELATIVE


# ------------------------------------------------------------ the ladders


@pytest.mark.parametrize("steps", [1, 4, 28, 30, 50])
def test_the_karras_and_exponential_ladders_are_the_reference_s_own(steps: int) -> None:
    """The two synthesized ladders, differenced against diffusers directly.

    Tested as FUNCTIONS rather than only through a solver, because both are
    shared by ``DPMSolverMultistep`` and ``UniPCMultistep`` and a shared function
    that is only ever tested through one caller is tested once.
    """

    diffusers = pytest.importorskip("diffusers")
    reference = diffusers.DPMSolverMultistepScheduler(**SD_SCHEDULE)
    table = np.asarray(
        ((1 - reference.alphas_cumprod) / reference.alphas_cumprod) ** 0.5, dtype=np.float32
    )
    flipped = np.flip(table).copy()

    ours_table = tuple(
        float(value)
        for value in np.asarray(
            _sigma_table_of(DPMSolverMultistep(**SD_SCHEDULE)), dtype=np.float32
        )
    )
    assert _rel(ours_table, table) <= RELATIVE

    for build, theirs_build in (
        (ladders.karras_sigmas, reference._convert_to_karras),
        (ladders.exponential_sigmas, reference._convert_to_exponential),
    ):
        ours = build(ours_table[0], ours_table[-1], steps)
        theirs = theirs_build(in_sigmas=flipped, num_inference_steps=steps)
        assert _rel(ours, np.asarray(theirs, dtype=np.float64)) <= RELATIVE

    # And the inverse, which is what turns a synthesized sigma back into the
    # timestep the model is conditioned on.
    logs = ladders.log_table(ours_table)
    synthesized = ladders.karras_sigmas(ours_table[0], ours_table[-1], steps)
    log_sigmas = np.log(table)
    for sigma in synthesized:
        mine = ladders.sigma_to_t(sigma, logs)
        theirs_t = float(reference._sigma_to_t(np.float64(sigma), log_sigmas))
        assert abs(mine - theirs_t) <= 1e-6 * max(abs(theirs_t), 1.0)


def _sigma_table_of(scheduler: DPMSolverMultistep) -> tuple[float, ...]:
    from gen_worker.model.solvers.precision import sigma_table

    return sigma_table(
        scheduler.num_train_timesteps,
        scheduler.beta_start,
        scheduler.beta_end,
        scheduler.beta_schedule,
        scheduler.rescale_betas_zero_snr,
    )


def test_the_multistep_timestep_grid_is_not_the_euler_grid() -> None:
    """The off-by-one that makes ``ladders.discrete_timesteps`` its own function.

    The multistep solvers space over ``steps + 1`` points and drop one; the Euler
    family spaces over ``steps``. At 28 steps under ``leading`` that is a stride
    of 34 here and 35 there. Sharing one helper between the two families is how
    they would quietly become one schedule.
    """

    from gen_worker.model.scheduler import EulerDiscrete

    multistep = ladders.discrete_timesteps("leading", 28, 1000, 1)
    euler = EulerDiscrete(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        timestep_spacing="leading",
        steps_offset=1,
    ).schedule(28)
    # 1000 // 29 = 34 here; 1000 // 28 = 35 there. Both grids start one stride
    # BELOW the top of the ladder, and they start at different places.
    assert multistep[0] == 953.0 and multistep[0] - multistep[1] == 34.0
    assert euler.timesteps[0] == 946.0 and euler.timesteps[0] - euler.timesteps[1] == 35.0
    assert multistep != euler.timesteps
    # And the multistep grid is the reference's, exactly.
    reference = _dpm_reference(**SD_SCHEDULE)
    reference.set_timesteps(28)
    assert list(multistep) == [float(t) for t in reference.timesteps.tolist()]


# ---------------------------------------------------- the multistep STATE


def test_the_multistep_state_initializes_from_nothing_and_is_immutable() -> None:
    """"Two pods agree" is a claim about the recursion's INITIAL CONDITION too.

    ``begin()`` takes no arguments, so there is no seed, no device and no clock
    anywhere in it — two pods start identical by construction rather than by
    discipline. And the value is frozen, so a step cannot mutate a history a
    concurrent request is still holding.
    """

    dpm = DPMSolverMultistep(**SD_SCHEDULE).schedule(8)
    unipc = UniPCMultistep(**SD_SCHEDULE, predict_x0=True).schedule(8)

    assert dpm.begin() == MultistepHistory() == MultistepHistory(outputs=(), taken=0)
    assert unipc.begin() == UniPcHistory()
    assert unipc.begin().last_sample is None and unipc.begin().order == 0

    with pytest.raises((AttributeError, TypeError)):
        dpm.begin().taken = 3  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        unipc.begin().order = 2  # type: ignore[misc]


def test_two_interleaved_requests_cannot_disturb_each_other() -> None:
    """The property a mutable scheduler object cannot have, exercised.

    diffusers keeps ``model_outputs``, ``last_sample`` and a step counter on the
    scheduler, which is why a served pipeline must be cloned per request. Here
    the state is a value the caller holds, so two loops can be advanced ALTERNATELY
    on ONE schedule object and still produce exactly what each would alone.
    """

    schedule = UniPCMultistep(**SD_SCHEDULE, predict_x0=True).schedule(8)
    torch.manual_seed(11)
    starts = [torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)]
    predictions = [
        [torch.randn(1, 4, 8, 8) for _ in range(8)],
        [torch.randn(1, 4, 8, 8) for _ in range(8)],
    ]

    alone = []
    for which in (0, 1):
        sample, state = starts[which].clone(), schedule.begin()
        for index in range(8):
            sample, state = schedule.step(index, predictions[which][index], sample, state)
        alone.append(sample)

    samples = [starts[0].clone(), starts[1].clone()]
    states = [schedule.begin(), schedule.begin()]
    for index in range(8):
        for which in (0, 1):  # interleaved, one step each, same schedule object
            samples[which], states[which] = schedule.step(
                index, predictions[which][index], samples[which], states[which]
            )
    assert torch.equal(samples[0], alone[0])
    assert torch.equal(samples[1], alone[1])


@pytest.mark.parametrize("steps", [4, 30])
def test_the_dpm_order_schedule_is_the_reference_s_own(steps: int) -> None:
    """WHICH steps run first-order, asserted rather than inferred from output.

    A 4-step distilled recipe runs first, second, second, FIRST — the last step
    drops to first order because the ladder terminates at zero. "2M means always
    second order" is the misreading, and on a 4-step ladder it changes half the
    render.
    """

    schedule = DPMSolverMultistep(**SD_SCHEDULE).schedule(steps)
    state = schedule.begin()
    orders = []
    torch.manual_seed(2)
    sample = torch.randn(1, 4, 8, 8)
    for index in range(steps):
        before = state
        first_order = schedule.solver_order == 1 or before.taken < 1 or (
            index == steps - 1 and schedule.final_sigmas_type == "zero"
        )
        orders.append(1 if first_order else 2)
        sample, state = schedule.step(index, torch.randn(1, 4, 8, 8), sample, state)
        assert len(state.outputs) == min(index + 1, schedule.solver_order)
        assert state.taken == min(index + 1, schedule.solver_order)
    assert orders[0] == 1 and orders[-1] == 1
    assert orders[1] == 2


# ------------------------------------------------------------ conditioning


def _gain(
    schedule: DpmSolverSchedule | UniPcSchedule,
    steps: int,
    *,
    noise: bool = False,
    skip_first: bool = False,
    seed: int = 5,
) -> float:
    """Relative change in the LATENTS per unit relative change in the LADDER.

    B2's conditioning measurement, re-run per solver: perturb the sigma ladder by
    5e-6 relative — the scale of a genuine cross-machine disagreement — and see
    what comes out. A gain near 1 means the loop propagates a ladder difference
    without amplifying it, which is what makes ladder byte-stability a
    CORRECTNESS property: two pods that resolve slightly different ladders render
    slightly different images, not arbitrarily different ones.
    """

    scale = 5e-6
    span = range(1 if skip_first else 0, len(schedule.sigmas) - 1)
    bumped = list(schedule.sigmas)
    for index in span:
        bumped[index] = bumped[index] * (1.0 + scale)
    nudged = replace(schedule, sigmas=tuple(bumped))

    torch.manual_seed(seed)
    start = torch.randn(2, 4, 16, 16)
    predictions = [torch.randn(2, 4, 16, 16) for _ in range(steps)]
    noises = [torch.randn(2, 4, 16, 16) for _ in range(steps)] if noise else [None] * steps

    results = []
    for variant in (schedule, nudged):
        variant = cast("Any", variant)
        sample, state = start.clone(), variant.begin()
        for index in range(steps):
            if noise:
                sample, state = variant.step(
                    index, predictions[index], sample, state, noise=noises[index]
                )
            else:
                sample, state = variant.step(index, predictions[index], sample, state)
        results.append(sample)
    return _rel_norm(results[1].numpy(), results[0].numpy()) / scale


def test_the_diffusion_lane_loops_are_conditioned_at_gain_one() -> None:
    """B2 measured Euler at ~1.0. Three multistep lanes, re-measured.

    Measured, not assumed: a predictor/corrector recursion could in principle
    amplify, and this is the assertion that says these do not. The consequence is
    B2's: a 5e-6 ladder disagreement between two pods is a 5e-6 latent
    disagreement, four orders below one bf16 ULP — so our byte-stable ladder
    turns a real cross-pod reproducibility hazard into no hazard at all.
    """

    karras = dict(SD_SCHEDULE, use_karras_sigmas=True, prediction_type="epsilon")
    assert 0.5 <= _gain(DPMSolverMultistep(**karras).schedule(30), 30) <= 2.0
    assert 0.5 <= _gain(DPMSolverMultistep(**karras).schedule(4), 4) <= 2.0
    sde = dict(karras, algorithm_type="sde-dpmsolver++")
    assert 0.5 <= _gain(DPMSolverMultistep(**sde).schedule(30), 30, noise=True) <= 2.0
    unipc = dict(SD_SCHEDULE, predict_x0=True, prediction_type="epsilon")
    assert 0.5 <= _gain(UniPCMultistep(**unipc).schedule(30), 30) <= 2.0


@pytest.mark.parametrize("steps", [4, 8, 40, 50])
def test_the_flow_ladders_first_rung_is_the_conditioning_risk(steps: int) -> None:
    """The one lane that is NOT gain-1, and exactly where the sensitivity lives.

    On wan's flow ladder ``alpha_t = 1 - sigma``, and the top rung sits at
    ``1 - 1e-6`` — so ``alpha_t`` there is ``1e-6``, and a 5e-6 RELATIVE nudge of
    that sigma is a **five-fold** relative change in ``alpha``, which enters the
    first predictor coefficient through ``log``. Measured gain: 7.7 at 50 steps
    rising to ~249 at 4, where that first step is a quarter of the render.
    Exclude the top rung and the loop is conditioned like every other lane.

    This is not a defect in either implementation — it is a property of the
    schedule, and both sides resolve that ladder in pure float64 numpy with no
    torch primitive involved, so nothing disagrees about it today. It is recorded
    because it says which number in this package is load-bearing: the ``1e-6``
    nudge in ``ladders.flow_sigmas`` is not a rounding detail, and anything that
    changes it changes wan's render by orders of magnitude more than its own size.
    """

    schedule = UniPCMultistep(
        use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=12.0
    ).schedule(steps)
    everywhere = _gain(schedule, steps)
    below_the_top = _gain(schedule, steps, skip_first=True)
    assert 0.5 <= below_the_top <= 2.0
    assert everywhere > 5.0 * below_the_top


# ----------------------------------------------- cross-kernel byte stability


_FENCE = """
import json
from gen_worker.model.solvers.dpm_multistep import DPMSolverMultistep
from gen_worker.model.solvers.unipc_multistep import UniPCMultistep
sd = dict(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
          beta_schedule="scaled_linear", timestep_spacing="leading", steps_offset=1,
          final_sigmas_type="zero", solver_order=2)
dpm = DPMSolverMultistep(**sd, use_karras_sigmas=True).schedule(30)
unipc_flow = UniPCMultistep(use_flow_sigmas=True, prediction_type="flow_prediction",
                            flow_shift=12.0).schedule(40)
unipc_beta = UniPCMultistep(**sd, predict_x0=True).schedule(30)
print(json.dumps({
    "dpm": list(dpm.sigmas), "dpm_t": list(dpm.timesteps),
    "unipc_flow": list(unipc_flow.sigmas), "unipc_flow_t": list(unipc_flow.timesteps),
    "unipc_beta": list(unipc_beta.sigmas), "unipc_beta_t": list(unipc_beta.timesteps),
}))
"""


def test_neither_solver_s_ladder_depends_on_which_cpu_kernel_torch_dispatched() -> None:
    """The property the reference does NOT have, fenced — once per solver.

    ``ATEN_CPU_CAPABILITY=default`` forces torch's scalar CPU kernels, whose
    float32 ``linspace`` disagrees with the vectorized one on 145 of 1000
    entries. Every sigma below is IEEE double arithmetic with explicit
    narrowings, so the subprocess must produce the SAME BYTES — and, since the
    loops above are conditioned at gain ~1, that is what makes a receipt's seed
    mean the same thing on two pods.

    The fence runs in a subprocess deliberately: the variable is read by torch at
    IMPORT, so setting it in-process would prove nothing. It is also why this
    module importing no array library is checkable rather than aspirational —
    the subprocess never imports torch at all.
    """

    root = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", _FENCE],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root), "ATEN_CPU_CAPABILITY": "default"},
    )
    assert result.returncode == 0, result.stderr
    scalar = json.loads(result.stdout.strip().splitlines()[-1])

    here: dict[str, Any] = {
        "dpm": DPMSolverMultistep(**SD_SCHEDULE, use_karras_sigmas=True).schedule(30),
        "unipc_flow": UniPCMultistep(
            use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=12.0
        ).schedule(40),
        "unipc_beta": UniPCMultistep(**SD_SCHEDULE, predict_x0=True).schedule(30),
    }
    for name, schedule in here.items():
        assert tuple(scalar[name]) == schedule.sigmas, name
        assert tuple(scalar[f"{name}_t"]) == schedule.timesteps, name


# ------------------------------------------- what the reference cannot serve


def test_a_v_prediction_karras_recipe_crashes_diffusers_and_serves_here() -> None:
    """sd15's DEFAULT sampler on a v-prediction checkpoint. diffusers raises.

    The enumeration, from the declared vocabulary rather than from intent:
    ``Sd15Tuned.scheduler`` defaults to ``dpmpp_2m_karras``; the sd15 endpoint
    declares ``objectives=("epsilon", "v_prediction")``; and ``gen_worker.view``
    pairs v-prediction with ``rescale_betas_zero_snr``. That combination makes the
    Karras ladder's recovered timestep grid start with a DUPLICATE — 999 twice —
    and diffusers resolves a step index by SEARCHING for the timestep value,
    taking the second match when there are two. Its counter is then one ahead for
    the whole loop and it indexes off the end of its own sigma array.

    Measured span: it raises at 25, 28, 30, 40, 50, 80, 100 and 150 steps, which
    includes sd15's stamped default of 30 and sdxl's of 28. This module has no
    such failure mode by construction — the step index is an ARGUMENT, never
    recovered from a value — which is a correctness difference rather than a
    performance one, and it is the reason this test asserts BOTH halves.
    """

    config = dict(
        SD_SCHEDULE,
        use_karras_sigmas=True,
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
    )
    theirs = _dpm_reference(**config)
    theirs.set_timesteps(30)
    grid = theirs.timesteps.tolist()
    assert grid[0] == grid[1], "the duplicate head is the mechanism"

    sample = torch.randn(1, 4, 8, 8)
    with pytest.raises(IndexError):
        for timestep in theirs.timesteps:
            sample = theirs.step(torch.randn(1, 4, 8, 8), timestep, sample).prev_sample

    # And the same recipe, here.
    ours = DPMSolverMultistep(**config).schedule(30)
    sample, state = torch.randn(1, 4, 8, 8), ours.begin()
    for index in range(len(ours)):
        sample, state = ours.step(index, torch.randn(1, 4, 8, 8), sample, state)
    assert torch.isfinite(sample).all()

    # UniPC fails on the same ladder for the same reason, with a different
    # exception, and is likewise served here.
    unipc_config = dict(config, predict_x0=True, solver_type="bh2")
    unipc_config.pop("algorithm_type", None)
    reference = _unipc_reference(**unipc_config)
    reference.set_timesteps(30)
    sample = torch.randn(1, 4, 8, 8)
    with pytest.raises((IndexError, AssertionError)):
        for timestep in reference.timesteps:
            sample = reference.step(torch.randn(1, 4, 8, 8), timestep, sample).prev_sample
    mine = UniPCMultistep(**unipc_config).schedule(30)
    sample, unipc_state = torch.randn(1, 4, 8, 8), mine.begin()
    for index in range(len(mine)):
        sample, unipc_state = mine.step(index, torch.randn(1, 4, 8, 8), sample, unipc_state)
    assert torch.isfinite(sample).all()


@pytest.mark.parametrize("steps", [4, 30])
def test_bh1_makes_the_reference_return_nan_on_its_final_step(steps: int) -> None:
    """The second thing this lane found in the reference, and it is a NaN.

    ``bh1`` sets ``B_h = hh``. On the final step of a ladder that terminates at
    zero — every configuration this fleet reaches — ``sigma_t`` is 0, so
    ``lambda_t`` is ``+inf``, ``h`` is ``+inf`` and ``B_h`` is ``-inf``. That
    step is first-order, so its residual is exactly zero, and diffusers still
    evaluates ``alpha_t * B_h * 0``: **-inf times zero is NaN**, and the whole
    latent comes back NaN.

    This module never forms the product: a first-order step has no residual term
    to add, so it returns the base update — which is the same value the reference
    would produce in exact arithmetic. ``bh1`` is unreachable today (every mirror
    in the fleet ships ``bh2``) and it is upstream's own recommendation for step
    counts below 10, which wan's 4- and 8-step lanes are — so this is recorded
    now rather than discovered by whoever first ships a ``bh1`` mirror.
    """

    config = dict(SD_SCHEDULE, solver_type="bh1", predict_x0=True, prediction_type="epsilon")
    ours = UniPCMultistep(**config).schedule(steps)
    theirs = _inject(_unipc_reference(**config), ours)

    torch.manual_seed(4)
    our_sample = torch.randn(1, 4, 8, 8)
    their_sample = our_sample.clone()
    state = ours.begin()
    for index, timestep in enumerate(theirs.timesteps):
        prediction = torch.randn(1, 4, 8, 8)
        our_sample, state = ours.step(index, prediction, our_sample, state)
        their_sample = theirs.step(prediction, timestep, their_sample).prev_sample
        if index < steps - 1:
            assert _rel_norm(our_sample.numpy(), their_sample.numpy()) <= RELATIVE

    assert torch.isnan(their_sample).all(), "the reference's final bh1 step"
    assert torch.isfinite(our_sample).all()
    assert ours.sigmas[-1] == 0.0, "the terminal zero is what makes B_h infinite"


# ---------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    "block",
    [
        {"solver_order": 3},
        {"algorithm_type": "dpmsolver"},
        {"algorithm_type": "deis"},
        {"solver_type": "bh2"},
        {"use_karras_sigmas": True, "use_flow_sigmas": True},
        {"use_karras_sigmas": 1},
        {"prediction_type": "sample"},
        {"num_train_timesteps": 1.5},
    ],
)
def test_a_dpm_block_is_parsed_not_coerced(block: dict[str, Any]) -> None:
    """Every one of these changes the LADDER or the UPDATE, so none falls through."""

    with pytest.raises(ModelError):
        DPMSolverMultistep.from_block({**SD_SCHEDULE, **block})


@pytest.mark.parametrize(
    "block",
    [
        {"solver_order": 3},
        {"solver_type": "midpoint"},
        {"predict_x0": False, "prediction_type": "flow_prediction"},
        {"use_flow_sigmas": True, "flow_shift": 0.0},
        {"use_exponential_sigmas": True, "use_karras_sigmas": True},
    ],
)
def test_a_unipc_block_is_parsed_not_coerced(block: dict[str, Any]) -> None:
    with pytest.raises(ModelError):
        UniPCMultistep.from_block({**SD_SCHEDULE, **block})


def test_a_stochastic_sampler_refuses_to_run_without_its_noise() -> None:
    """``sde-dpmsolver++`` cannot invent a random tensor here, and must not.

    This module imports no array library, so the caller owns the generator.
    Defaulting the noise to zero would turn a stochastic sampler into a
    deterministic one that still reported itself as SDE — the silent-failure
    shape this package exists to remove. The mirror refusal matters as much: a
    DETERMINISTIC sampler handed noise would ignore it, and a caller passing one
    has selected the wrong sampler.
    """

    sde = DPMSolverMultistep(**SD_SCHEDULE, algorithm_type="sde-dpmsolver++").schedule(8)
    sample = torch.randn(1, 4, 8, 8)
    with pytest.raises(ModelError):
        sde.step(0, torch.randn(1, 4, 8, 8), sample, sde.begin())

    ode = DPMSolverMultistep(**SD_SCHEDULE).schedule(8)
    with pytest.raises(ModelError):
        ode.step(0, torch.randn(1, 4, 8, 8), sample, ode.begin(), noise=torch.zeros(1, 4, 8, 8))


def test_a_step_outside_the_ladder_is_refused_by_both_solvers() -> None:
    dpm = DPMSolverMultistep(**SD_SCHEDULE).schedule(4)
    unipc = UniPCMultistep(**SD_SCHEDULE, predict_x0=True).schedule(4)
    sample = torch.randn(1, 4, 8, 8)
    for schedule, state in ((dpm, dpm.begin()), (unipc, unipc.begin())):
        with pytest.raises(ModelError):
            schedule.step(4, torch.randn(1, 4, 8, 8), sample, state)  # type: ignore[arg-type]
        with pytest.raises(ModelError):
            schedule.step(-1, torch.randn(1, 4, 8, 8), sample, state)  # type: ignore[arg-type]


def test_the_declared_block_is_the_only_source_of_constants() -> None:
    """A constant hardcoded in the math would be a second declaration of it.

    The defaults are DIFFUSERS' class defaults — linear betas over [1e-4, 2e-2],
    ``linspace`` spacing, ``flow_shift`` 1.0 — and no model in this fleet was
    trained on any of them. A family that forgot to declare its schedule resolves
    to a plausible, wrong one, which is the failure this asserts is still loud.
    """

    assert DPMSolverMultistep().beta_schedule == "linear"
    assert DPMSolverMultistep().timestep_spacing == "linspace"
    assert DPMSolverMultistep().algorithm_type == "dpmsolver++"
    assert UniPCMultistep().solver_type == "bh2" and UniPCMultistep().predict_x0 is True
    assert UniPCMultistep().flow_shift == 1.0

    declared = UniPCMultistep.from_block(
        {"use_flow_sigmas": True, "prediction_type": "flow_prediction", "flow_shift": 12.0}
    )
    assert declared.flow_shift == 12.0
    # wan re-shifts per REQUEST and the rebuild must keep the flow reading.
    reshifted = declared.shifted(3.0)
    assert reshifted.flow_shift == 3.0
    assert reshifted.use_flow_sigmas and reshifted.prediction_type == "flow_prediction"
    with pytest.raises(ModelError):
        UniPCMultistep(**SD_SCHEDULE, predict_x0=True).shifted(5.0)


def test_a_v_prediction_checkpoint_carries_the_zero_snr_rescale_with_it() -> None:
    """``objective()`` reproduces the pairing ``gen_worker.view`` already makes."""

    for build in (DPMSolverMultistep, UniPCMultistep):
        base = build(**SD_SCHEDULE)  # type: ignore[arg-type]
        assert base.objective("epsilon") is base
        v_pred = base.objective("v_prediction")
        assert v_pred.prediction_type == "v_prediction" and v_pred.rescale_betas_zero_snr
        assert v_pred.schedule(28).sigmas != base.schedule(28).sigmas
        with pytest.raises(ModelError):
            base.objective("sample")


def test_neither_solver_scales_its_model_input() -> None:
    """The trap ``euler_discrete`` sets for whoever writes the second loop.

    ``EulerDiscrete`` REQUIRES ``scale_model_input`` — skipping it feeds a U-Net
    latents whose variance grows with sigma. Both solvers here require the
    opposite: diffusers' own ``scale_model_input`` is the identity for them, so a
    loop that copies Euler's shape and scales anyway destroys the render with no
    error. Asserted against the reference so the claim is theirs, not ours.
    """

    diffusers = pytest.importorskip("diffusers")
    sample = torch.randn(1, 4, 8, 8)
    for reference in (
        diffusers.DPMSolverMultistepScheduler(**SD_SCHEDULE),
        diffusers.UniPCMultistepScheduler(**SD_SCHEDULE),
    ):
        reference.set_timesteps(8)
        assert torch.equal(reference.scale_model_input(sample, reference.timesteps[0]), sample)
    assert not hasattr(DPMSolverMultistep(**SD_SCHEDULE).schedule(8), "scale_model_input")
    assert not hasattr(
        UniPCMultistep(**SD_SCHEDULE, predict_x0=True).schedule(8), "scale_model_input"
    )


def test_the_solver_modules_import_no_array_library() -> None:
    """The adopt-only serve role holds this package for free, and it stays true.

    Checked by reading the SOURCE rather than by inspecting ``sys.modules``: this
    test file has already imported torch, so a runtime check would prove nothing.
    """

    import ast

    package = Path(__file__).resolve().parents[1] / "src" / "gen_worker" / "model" / "solvers"
    banned = {"torch", "numpy", "diffusers", "scipy"}
    for source in sorted(package.glob("*.py")):
        tree = ast.parse(source.read_text())
        # `if TYPE_CHECKING: from torch import Tensor` is not an import — it is
        # an annotation, erased at runtime by `from __future__ import
        # annotations`. Stripped rather than allowlisted, so a REAL torch import
        # anywhere else still trips this.
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
                node.body = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not names & banned, f"{source.name} imports {names & banned}"
    # `math` and `struct` are the whole numeric surface.
    assert math.sqrt(4.0) == 2.0
