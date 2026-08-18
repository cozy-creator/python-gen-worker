"""pgw#1346 B2 — the U-Net families serve end to end, on bare typed math.

The Flux lane's four claims (pgw#1331), applied to the batch that owed the
harder half of them:

1. **The bare math is the same math.** ``euler_discrete``'s ladder descends
   from a trained noise schedule rather than from a closed form, so
   reproducing it means reproducing the PRECISION it was produced at — the
   float64 reading of the same algebra is 201 float32 ULP away, and one of
   these tests exists purely to keep that number from becoming folklore.
   Chasing it turned up the finding that reshaped this file: **the reference
   is not reproducible across machines.** Three torch primitives it depends on
   are implementation-defined rather than IEEE-exact — ``linspace`` dispatches
   its CPU kernel by ISA (145 of 1000 entries differ by 1 ULP), ``cumprod``
   varies in accumulator width and association order, and ``x ** 0.5`` is
   ``pow``, which is not correctly rounded where this module uses ``sqrt``,
   which is. Measured spread across machines: **85 float32 ULP**. So
   "bit-identical to diffusers" is not a claim ANY implementation can make,
   and three CI cycles were spent learning that a ULP bound is the wrong
   instrument rather than a number to widen.

   What is claimed, and fenced: the ladders agree RELATIVELY to ~20x tighter
   than one bf16 ULP; the timestep grid is matched EXACTLY (integer
   arithmetic, exact on every machine seen); the loop is bit-identical here
   once the table is removed as a variable; and — the claim that actually
   matters — **our ladder is byte-stable across CPU kernels where theirs is
   not**, which is a correctness property rather than a nicety, because the
   loop propagates a ladder difference roughly 1:1 into the latents.
2. **The whole composition is graph classes** — SDXL grew its two text towers,
   so no SD family leaves a tower riding an eager model at serve time.
3. **SD2 is not SD1.5 with different numbers.** B2's scoping proposed an
   instance; the shapes say a declaration. Tested, not asserted in a comment.
4. **The gaps are still shaped like refusals** — including the one this batch
   CLOSED, so the mechanism is fenced at the generator now that no catalog
   family exercises it.

The ddim decision (pgw#1346 owed it) is recorded at the bottom, as a test over
what the endpoints actually invoke rather than as prose.
"""

from __future__ import annotations

import json
import inspect
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_args

import pytest

from gen_worker.model.catalog import Sd2, Sd15, Sdxl
from gen_worker.model.catalog import sd15_serve as sd
from gen_worker.model.catalog import sdxl_serve as sx
from gen_worker.model.catalog.sd15 import SCHEDULER as SD_SCHEDULER
from gen_worker.model.catalog.sd15 import SD2, SD15
from gen_worker.model.catalog.sdxl import SCHEDULER as SDXL_SCHEDULER
from gen_worker.model.catalog.sdxl import TRAINED as SDXL_TRAINED
from gen_worker.model.catalog.sdxl import SDXL
from gen_worker.model.errors import ModelError, ModelRefusal
from gen_worker.model.scheduler import (
    IMPLEMENTED,
    AncestralSchedule,
    Ddim,
    DdimSchedule,
    DiscreteSchedule,
    EulerAncestralDiscrete,
    EulerDiscrete,
    SchedulerKind,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gen_worker.model.catalog._generated.sdxl import SdxlShape

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

#: Every step count the sdxl and sd15 endpoints can actually reach: their
#: stamped defaults (28 / 30 / 1), the distilled recipes their handlers pin
#: (Lightning 4 and 8, Hyper-SD 4, DMD2 4, SD-Turbo 1), and the ends of the
#: declared `steps` parameter range.
ENDPOINT_STEPS = (1, 4, 8, 20, 25, 28, 30, 40, 50, 100)

#: How far a value may sit from diffusers', RELATIVELY.
#:
#: Not a ULP count, and that is the lesson of three CI cycles: a bit-level
#: bound against this reference is unboundable, because every disagreement
#: traces to a torch primitive that is implementation-defined rather than
#: IEEE-exact. Three were identified, each measured:
#:
#:   * ``torch.linspace`` float32 CPU dispatches by ISA — its scalar and
#:     vectorized kernels differ on 145 of 1000 entries by 1 ULP;
#:   * ``torch.cumprod`` float32 CPU varies in accumulator width and
#:     association order, which is the dominant term — a 1000-term product
#:     drifts by tens of ULP;
#:   * ``x ** 0.5`` on a tensor is ``pow``, which is NOT correctly rounded,
#:     where this module uses ``math.sqrt``, which is.
#:
#: Observed spread across machines: **85 float32 ULP ≈ 5.1e-6 relative**. This
#: bound is 2e-4 — ~40x that, and still ~20x TIGHTER than one bf16 ULP
#: (3.9e-3), which is the precision the denoiser actually computes in. So the
#: two ladders are the same schedule by any measure that reaches a pixel.
#:
#: The bit-level claims live where they are actually true: on OUR OWN
#: cross-machine determinism, and on the timestep grid, which is integer
#: arithmetic and exact everywhere.
RELATIVE = 2e-4


def _ulp(ours: Any, theirs: Any) -> int:
    """Distance in float32 units in the last place. 0 is bit equality."""

    a = np.asarray(ours, dtype=np.float32)
    b = np.asarray(theirs, dtype=np.float32)
    return int(
        np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64)).max()
    )


def _rel(ours: Any, theirs: Any) -> float:
    """Largest RELATIVE difference, scaled by the reference's own magnitude.

    ``atol`` is deliberately absent: every value compared here is either a
    sigma (all comfortably above 1e-3) or terminates at an exact 0.0 that both
    sides produce, so a relative measure never divides by a noise floor.
    """

    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(theirs, dtype=np.float64)
    scale = np.maximum(np.abs(b), 1e-12)
    return float((np.abs(a - b) / scale).max())


def _rel_norm(ours: Any, theirs: Any) -> float:
    """Relative difference of two TENSORS, in the L2 norm.

    A separate function from :func:`_rel` on purpose, and the distinction is
    not pedantry: latents cross zero, so an element-wise relative measure
    divides by a value near the noise floor and reports a meaningless 17% for
    a difference that is genuinely parts-per-million. Measured — that number
    is what the first draft of the loop test would have produced. Sigmas never
    cross zero (the terminal 0.0 is exact on both sides), so they keep the
    element-wise measure, which is the stricter of the two there.
    """

    a = np.asarray(ours, dtype=np.float64).ravel()
    b = np.asarray(theirs, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-30))


def _reference(block: dict[str, Any], **overrides: Any) -> Any:
    diffusers = pytest.importorskip("diffusers")
    merged = {**block, **overrides}
    return diffusers.EulerDiscreteScheduler(
        num_train_timesteps=int(merged["num_train_timesteps"]),
        beta_start=float(merged["beta_start"]),
        beta_end=float(merged["beta_end"]),
        beta_schedule=str(merged["beta_schedule"]),
        timestep_spacing=str(merged["timestep_spacing"]),
        steps_offset=int(merged["steps_offset"]),
        prediction_type=str(merged["prediction_type"]),
        final_sigmas_type=str(merged["final_sigmas_type"]),
        rescale_betas_zero_snr=bool(merged.get("rescale_betas_zero_snr", False)),
        interpolation_type="linear",
    )


# ---------------------------------------------------------------- the bare math


@pytest.mark.parametrize("steps", ENDPOINT_STEPS)
@pytest.mark.parametrize(
    "spacing",
    [
        # `leading` is what every SD/SDXL checkpoint's own scheduler config
        # carries, so it is the `euler` sampler's spacing. `trailing` is the
        # `euler_trailing` sampler — SDXL-Lightning's published recipe, which
        # the sdxl endpoint's `_TURBO_RECIPES` pins for both its 4- and 8-step
        # arms. `linspace` is diffusers' class default and reachable only by a
        # checkpoint that declares nothing; included so the default is not the
        # one path nobody measured.
        "leading",
        "trailing",
        "linspace",
    ],
)
@pytest.mark.parametrize(
    ("objective", "zero_snr"), [("epsilon", False), ("v_prediction", True)]
)
def test_the_euler_discrete_ladder_matches_diffusers_to_the_reference_s_own_noise(
    steps: int, spacing: str, objective: str, zero_snr: bool
) -> None:
    """The bar was ONE float32 ULP (pgw#1331). Here is why that is the wrong
    instrument for THIS reference, and what replaces it.

    On the machine this was developed on the answer is **0** — exact bit
    equality on every value, under both of the CPU kernels available here. It
    is still not a portable claim, and three CI cycles established why:
    **every bit-level disagreement traces to a torch primitive that is
    implementation-defined rather than IEEE-exact** — ``linspace``'s ISA
    dispatch, ``cumprod``'s accumulation order, and ``pow`` where this module
    uses correctly-rounded ``sqrt``. Measured spread across machines: **85
    float32 ULP, ~5.1e-6 relative**. Bit-equality is simply not a property the
    reference possesses, so demanding it of an implementation is demanding the
    impossible — and each attempt to bound it in ULP was wrong by an order of
    magnitude, which is the argument for changing instrument rather than
    widening the number.

    So: the SIGMAS and ``init_noise_sigma`` are compared RELATIVELY, at ~20x
    tighter than one bf16 ULP. The TIMESTEP grid keeps its exact assertion —
    it is integer arithmetic and it has been exact on every machine, so a
    single ULP there would be a real defect rather than library noise.

    ``v_prediction`` is paired with ``rescale_betas_zero_snr`` because
    ``gen_worker.view`` pairs them: a v-pred checkpoint on this fleet is ALWAYS
    served with the zero-terminal-SNR rescale, so measuring the objective
    without it would measure a configuration no request reaches.
    """

    block = {
        **dict(SDXL_SCHEDULER),
        "timestep_spacing": spacing,
        "prediction_type": objective,
        "rescale_betas_zero_snr": zero_snr,
    }
    theirs = _reference(block)
    theirs.set_timesteps(steps)
    ours = EulerDiscrete.from_block(block).schedule(steps)

    assert len(ours.sigmas) == len(theirs.sigmas)
    assert _rel(ours.sigmas, theirs.sigmas.numpy()) <= RELATIVE
    # EXACT, and it stays exact: the grid is integer arithmetic that has never
    # differed on any machine this ran on.
    assert _ulp(ours.timesteps, theirs.timesteps.numpy()) == 0
    # Relative, because it is `max(sigmas)` and inherits the table's noise.
    assert _rel([ours.init_noise_sigma], [float(theirs.init_noise_sigma)]) <= RELATIVE


@pytest.mark.parametrize("spacing", ["leading", "trailing"])
@pytest.mark.parametrize(
    ("objective", "zero_snr"), [("epsilon", False), ("v_prediction", True)]
)
def test_a_whole_denoising_loop_tracks_the_diffusers_loop(
    spacing: str, objective: str, zero_snr: bool
) -> None:
    """28 steps of ``scale_model_input`` + ``step``, element for element.

    The ladder being right is half of it; the STEP is the other half, and it is
    the half where the algebraically obvious rewrite is wrong. Run at float32
    because that is where the two are comparable: upstream upcasts its sample
    to float32 internally and this module cannot name a dtype, so float32 is
    the precision at which "the same math" is a decidable claim.

    **Our ladder is loaded into the reference before the loop runs**, so this
    is a test of the STEP and not a second, weaker test of the table. With the
    table removed as a variable the two loops are bitwise identical on this
    machine — every element, all four configurations, under both local CPU
    kernels.

    It is still asserted RELATIVELY, and the reason is a fourth
    implementation-defined primitive that only this test could have exposed:
    ``scale_model_input`` divides by ``(sigma**2 + 1) ** 0.5``, and on a tensor
    that ``** 0.5`` is ``torch.pow``, which is **not correctly rounded** and
    varies by ISA. This module uses ``math.sqrt``, which is. So where the two
    differ, OURS is the more accurate and the more deterministic of the pair —
    and there is no amount of care that makes a correctly-rounded square root
    bit-match a ``pow`` that is not.
    """

    block = {
        **dict(SDXL_SCHEDULER),
        "timestep_spacing": spacing,
        "prediction_type": objective,
        "rescale_betas_zero_snr": zero_snr,
    }
    theirs = _reference(block)
    theirs.set_timesteps(28)
    schedule = EulerDiscrete.from_block(block).schedule(28)
    # Same ladder, both loops. `timesteps` is already exact under every kernel,
    # so the index each side resolves is unchanged by this.
    assert _ulp(schedule.timesteps, theirs.timesteps.numpy()) == 0
    theirs.sigmas = torch.tensor(schedule.sigmas, dtype=torch.float32)

    torch.manual_seed(0)
    ours_sample = torch.randn(2, 4, 32, 32) * float(schedule.init_noise_sigma)
    their_sample = ours_sample.clone()
    for index, timestep in enumerate(theirs.timesteps):
        prediction = torch.randn(2, 4, 32, 32)
        assert _rel_norm(
            schedule.scale_model_input(index, ours_sample).numpy(),
            theirs.scale_model_input(their_sample, timestep).numpy(),
        ) <= RELATIVE
        ours_sample = schedule.step(index, prediction, ours_sample)
        their_sample = theirs.step(prediction, timestep, their_sample).prev_sample
        # Compared after EVERY step rather than once at the end: a divergence
        # that appears at step 3 and one that appears at step 27 are different
        # defects, and only the per-step assertion tells them apart.
        assert _rel_norm(ours_sample.numpy(), their_sample.numpy()) <= RELATIVE


def test_our_ladder_does_not_depend_on_which_cpu_kernel_torch_dispatched() -> None:
    """The property the reference does NOT have, fenced.

    ``ATEN_CPU_CAPABILITY=default`` forces torch's scalar CPU kernels. Its
    float32 ``linspace`` disagrees with the vectorized one on 145 of 1000
    entries by 1 ULP, which is why diffusers' resolved ladder moves by up to
    6 ULP depending on the card — sorry, the CPU — a pod happened to rent.

    Ours cannot move: every operation is IEEE double arithmetic with one
    explicit narrowing, so the subprocess below must produce the SAME BYTES.
    This is the reproducibility claim that actually matters — a receipt's seed
    meaning the same thing on two pods — and it is a reason to prefer this
    module over the thing it replaces, not merely a tie.

    The reference is measured in the same subprocess rather than asserted to
    differ: a runner that already dispatches the scalar kernel would be
    comparing it against itself, and the point is that OURS cannot vary, not
    that theirs always does.
    """

    program = (
        "import sys, json;"
        "from gen_worker.model.scheduler import EulerDiscrete;"
        "from gen_worker.model.catalog.sdxl import SCHEDULER as B;"
        "from diffusers import EulerDiscreteScheduler as E;"
        "b=dict(B);"
        "o=EulerDiscrete.from_block(b).schedule(28);"
        "t=E(num_train_timesteps=1000,beta_start=0.00085,beta_end=0.012,"
        "beta_schedule='scaled_linear',timestep_spacing='leading',steps_offset=1,"
        "interpolation_type='linear',prediction_type='epsilon',final_sigmas_type='zero');"
        "t.set_timesteps(28);"
        "print(json.dumps({'ours': list(o.sigmas), 'theirs': t.sigmas.tolist()}))"
    )
    root = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root), "ATEN_CPU_CAPABILITY": "default"},
    )
    assert result.returncode == 0, result.stderr
    scalar = json.loads(result.stdout.strip().splitlines()[-1])

    here = EulerDiscrete.from_block(dict(SDXL_SCHEDULER)).schedule(28)
    # THE claim: byte-identical across kernels.
    assert tuple(scalar["ours"]) == here.sigmas
    # And still agrees with the reference, whichever kernel it dispatched.
    assert _rel(here.sigmas, scalar["theirs"]) <= RELATIVE


def test_the_float64_reading_of_the_same_algebra_is_measurably_wrong() -> None:
    """Why the implementation narrows to float32 stage by stage.

    This is the finding pgw#1331 asked every scheduler lane to state, made
    executable so it cannot rot into a comment nobody re-derives: computing the
    identical algebra honestly in float64 is HUNDREDS of float32 ULP away from
    what the fleet serves today, because ``alphas_cumprod`` is a thousand-term
    cumulative product and float32 is the precision it is DEFINED at.
    """

    theirs = _reference(dict(SDXL_SCHEDULER))
    theirs.set_timesteps(28)

    total = int(SDXL_SCHEDULER["num_train_timesteps"])
    start = math.sqrt(float(SDXL_SCHEDULER["beta_start"]))
    end = math.sqrt(float(SDXL_SCHEDULER["beta_end"]))
    running = 1.0
    table = []
    for index in range(total):
        root = start + (end - start) * index / (total - 1)
        running *= 1.0 - root * root
        table.append(math.sqrt((1.0 - running) / running))

    reference_table = np.asarray(
        ((1 - theirs.alphas_cumprod) / theirs.alphas_cumprod) ** 0.5, dtype=np.float32
    )
    # 201 ULP on the trained table itself…
    assert _ulp(table, reference_table) > 200

    naive = [table[int(t)] + (table[int(t) + 1] - table[int(t)]) * (t - int(t))
             for t in theirs.timesteps.tolist()]
    # …and 25 on the 28 sigmas a request actually walks. Interpolation averages
    # some of the drift away, which is exactly why "it looks fine" is not a
    # measurement: still 25x the bar, and monotonically worse with step count.
    assert _ulp(naive + [0.0], theirs.sigmas.numpy()) > 15

    # …and the shipped implementation is STRICTLY CLOSER on the same machine,
    # which is the claim the narrowing exists to make. Asserted as a
    # comparison rather than a threshold on purpose: both sides move together
    # when the reference's own kernels change, so a fixed number here would be
    # the fourth wrong constant this file learned not to write.
    ours = EulerDiscrete.from_block(dict(SDXL_SCHEDULER)).schedule(28).sigmas
    assert _rel(ours, theirs.sigmas.numpy()) < _rel(naive + [0.0], theirs.sigmas.numpy())
    assert _rel(ours, theirs.sigmas.numpy()) <= RELATIVE


def test_the_declared_block_is_the_scheduler_s_only_source_of_constants() -> None:
    """A constant hardcoded in the math would be a second declaration of it.

    Sharper here than it is for Flux: :class:`EulerDiscrete` defaults to
    DIFFUSERS' class defaults — linear betas over [1e-4, 2e-2] with linspace
    spacing — which no Stable Diffusion was ever trained on. A family that
    forgot to declare its schedule would resolve to a plausible, wrong one, so
    the defaults are deliberately NOT Stable Diffusion's.
    """

    bare = EulerDiscrete()
    assert (bare.beta_schedule, bare.timestep_spacing, bare.steps_offset) == (
        "linear",
        "linspace",
        0,
    )
    for spec, block in ((SDXL, SDXL_SCHEDULER), (SD15, SD_SCHEDULER), (SD2, SD_SCHEDULER)):
        assert spec.schedulers
        declared = EulerDiscrete.from_block(dict(block))
        assert declared.beta_schedule == "scaled_linear"
        assert declared.beta_start == 0.00085 and declared.beta_end == 0.012
        assert (declared.timestep_spacing, declared.steps_offset) == ("leading", 1)
        # Every DECLARED sampler restates the same trained noise schedule; what
        # differs between them is the kind and the spacing, never the betas.
        # pgw#1346 K10's whole risk is a family declaring one trained table
        # three slightly different ways, so this is the assertion that refuses
        # it.
        for entry in spec.schedulers.values():
            resolved = dict(entry.parameters)
            for shared in ("beta_start", "beta_end", "beta_schedule", "num_train_timesteps"):
                assert resolved[shared] == block[shared]


@pytest.mark.parametrize(
    "block",
    [
        {"timestep_spacing": "backwards"},
        {"beta_schedule": "cosine"},
        {"prediction_type": "flow_prediction"},
        {"final_sigmas_type": "sigma_max"},
        {"num_train_timesteps": 1.5},
        {"steps_offset": True},
    ],
)
def test_a_scheduler_block_is_parsed_not_coerced(block: dict[str, Any]) -> None:
    """Every one of these changes the LADDER, so none of them falls through."""

    with pytest.raises(ModelError):
        EulerDiscrete.from_block({**dict(SDXL_SCHEDULER), **block})


def test_a_v_prediction_checkpoint_carries_the_zero_snr_rescale_with_it() -> None:
    """``objective()`` reproduces the pairing ``gen_worker.view`` already makes.

    The two paths must not be able to disagree about what "v_prediction" means:
    the diffusers path sets ``rescale_betas_zero_snr=True`` whenever the
    stamped objective is v-pred, so this one does too, and it is the SCHEDULER
    that carries it rather than every caller remembering.
    """

    base = EulerDiscrete.from_block(dict(SDXL_SCHEDULER))
    assert base.prediction_type == "epsilon" and not base.rescale_betas_zero_snr
    v_pred = base.objective("v_prediction")
    assert v_pred.prediction_type == "v_prediction" and v_pred.rescale_betas_zero_snr
    assert base.objective("epsilon") is base
    assert v_pred.schedule(28).sigmas != base.schedule(28).sigmas
    with pytest.raises(ModelError):
        base.objective("sample")


def test_a_schedule_terminating_anywhere_but_zero_is_refused() -> None:
    with pytest.raises(ModelError):
        DiscreteSchedule(
            sigmas=(14.6, 0.1),
            timesteps=(999.0,),
            num_train_timesteps=1000,
            prediction_type="epsilon",
            init_noise_sigma=14.6,
        )
    with pytest.raises(ModelError):
        DiscreteSchedule(
            sigmas=(14.6, 1.0, 0.0),
            timesteps=(999.0,),
            num_train_timesteps=1000,
            prediction_type="epsilon",
            init_noise_sigma=14.6,
        )


# ------------------------------------------------------- the whole composition


def test_every_declarable_scheduler_has_math_and_the_gap_is_still_fenced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``IMPLEMENTED`` is TOTAL, and the absent-method mechanism still works.

    pgw#1331 proved the mechanism through SDXL, whose scheduler had no math.
    Every name carries math now, so nothing in the catalog exercises the gap
    and the mechanism has to be fenced at the GENERATOR instead — otherwise the
    next declared-but-unimplemented scheduler discovers it by shipping.

    K10 makes the fence sharper than it was, because the miss is now PER
    SAMPLER: dropping ONE kind must remove only the arms that named it and
    leave the rest of the accessor intact, and dropping every kind a family
    declares must remove the method entirely.
    """

    from gen_worker.model import codegen

    assert set(IMPLEMENTED) == set(SchedulerKind)
    assert {name: kind.value for name, kind in Sdxl.SCHEDULERS.items()} == {
        "ddim_trailing": "ddim",
        "dpmpp_2m_karras": "dpmsolver_multistep",
        "dpmpp_2m_sde_karras": "dpmsolver_multistep",
        "euler_a": "euler_ancestral_discrete",
        "euler_trailing": "euler_discrete",
    }
    assert {name: kind.value for name, kind in Sd15.SCHEDULERS.items()} == {
        "ddim": "ddim",
        "ddim_trailing": "ddim",
        "dpmpp_2m": "dpmsolver_multistep",
        "dpmpp_2m_karras": "dpmsolver_multistep",
        "dpmpp_2m_sde_karras": "dpmsolver_multistep",
        "euler": "euler_discrete",
        "euler_a": "euler_ancestral_discrete",
        "unipc": "unipc_multistep",
    }
    assert isinstance(Sdxl.fake().scheduler(), EulerAncestralDiscrete)

    def render() -> str:
        return codegen.render_module(
            Sdxl.EXPORT,
            spec_module="gen_worker.model.catalog.sdxl",
            spec_attr="SDXL",
        )

    rendered = render()
    assert "def scheduler(self" in rendered
    assert (
        "SdxlDeclaredSampler = Literal['ddim_trailing', 'dpmpp_2m_karras', "
        "'dpmpp_2m_sde_karras', 'euler_a', 'euler_trailing']"
    ) in rendered

    # Drop ONE kind: the accessor survives, minus that sampler.
    monkeypatch.delitem(IMPLEMENTED, SchedulerKind.DDIM)
    partial = render()
    assert "def scheduler(self" in partial
    assert (
        "SdxlDeclaredSampler = Literal['dpmpp_2m_karras', 'dpmpp_2m_sde_karras', "
        "'euler_a', 'euler_trailing']"
    ) in partial
    # And the DECLARED-but-unimplemented sampler is still in the class facts,
    # so the declaration is not quietly rewritten to match the implementation.
    assert "'ddim_trailing': SchedulerKind.DDIM," in partial
    assert "Ddim.from_block" not in partial

    # Drop them all: no method at all, so a handler that wants one gets an
    # AttributeError from its own type checker (pgw#1331's mechanism).
    monkeypatch.delitem(IMPLEMENTED, SchedulerKind.EULER_DISCRETE)
    monkeypatch.delitem(IMPLEMENTED, SchedulerKind.EULER_ANCESTRAL_DISCRETE)
    monkeypatch.delitem(IMPLEMENTED, SchedulerKind.DPMSOLVER_MULTISTEP)
    without = render()
    assert "def scheduler(self" not in without


def test_sdxl_declares_every_stage_of_the_pipeline() -> None:
    """Two text towers, the U-Net and the VAE decoder — nothing left eager."""

    assert tuple(runner.name for runner in SDXL.runners) == (
        "clip_g",
        "clip_l",
        "decoder",
        "denoiser",
    )
    assert SDXL.loop is not None
    assert tuple(stage.runner for stage in SDXL.loop.stages) == (
        "clip_l",
        "clip_g",
        "denoiser",
        "decoder",
    )
    # bigG returns BOTH of the things SDXL reads from it, from ONE pass.
    assert len(Sdxl.EXPORT.runner("clip_g").variants[0].outputs) == 2
    assert len(Sdxl.EXPORT.runner("clip_l").variants[0].outputs) == 1


def test_sd2_is_a_declaration_and_not_an_instance_of_sd15() -> None:
    """B2's scoping proposed an ``sd2`` INSTANCE. The shapes refuse it.

    An instance is weights + tuned values + a ref label; these two do not share
    a traced class or even a weight SHAPE. Every number below is a different
    cross-attention projection in the U-Net, so one compiled graph cannot serve
    both and one exhaustive ``Literal`` must not name both.
    """

    from gen_worker.model.catalog.sd15 import SD2_UNET, SD15_UNET

    assert SD15.name == "sd15" and SD2.name == "sd2"
    assert SD15_UNET["cross_attention_dim"] != SD2_UNET["cross_attention_dim"]
    assert SD15_UNET["attention_head_dim"] != SD2_UNET["attention_head_dim"]
    assert SD15_UNET["use_linear_projection"] != SD2_UNET["use_linear_projection"]
    # The runner SET is identical, which is exactly why the set is not the test.
    assert tuple(r.name for r in SD15.runners) == tuple(r.name for r in SD2.runners)
    assert Sd15.EXPORT_DIGEST != Sd2.EXPORT_DIGEST


def test_the_shape_axis_is_the_endpoints_own_bucket_grid() -> None:
    """Nine SDXL shapes and seven SD1.5 ones, non-square, packed onto one axis.

    Two axes would be a CROSS PRODUCT — 81 and 49 traced classes for 9 and 7
    real ones — so the pair is packed. The packing has to round-trip and the
    latent order has to be (rows, cols): 1344x768 and 768x1344 are different
    conv graphs, and transposing them picks a wrong class rather than making a
    wrong image.
    """

    assert len(sx.SHAPE_BUCKETS) == 9 and len(sd.SD15_SHAPE_BUCKETS) == 7
    for width, height in sx.SHAPES:
        code = sx.pack_shape(width, height)
        assert sx.unpack_shape(code) == (width, height)
        assert sx.latent_shape(code) == (height // 8, width // 8)
    assert sx.latent_shape(sx.pack_shape(1344, 768)) == (96, 168)
    assert sx.latent_shape(sx.pack_shape(768, 1344)) == (168, 96)
    # The bucket set the family declares IS the set the generated Literal has.
    assert set(SDXL.axis_values["shape"]) == set(sx.SHAPE_BUCKETS)
    # `sd15_serve`'s loop is annotated with `Sd15Shape` for BOTH families, so
    # the two grids being identical is a load-bearing fact and not a
    # coincidence: if one moves, this fails before the annotation lies.
    assert sd.SD15_SHAPE_BUCKETS == sd.SD2_SHAPE_BUCKETS
    assert set(SD15.axis_values["shape"]) == set(SD2.axis_values["shape"])


def test_the_tuned_schemas_carry_every_field_the_endpoints_stamp() -> None:
    """``inst.tuned`` replaces ``ctx.defaults`` BY VALUE or it loses recipes.

    A field this schema lacks is a stamped catalog value that stops reaching
    the handler silently, and a sampler name absent from the Literal makes an
    already-stamped recipe undecodable. Both were true of the first draft of
    ``SdxlTuned``: ``quality_preamble`` was missing and three of the endpoint's
    six sampler names were.
    """

    sdxl_fields = set(Sdxl.Tuned.__struct_fields__)
    assert {"scheduler", "steps", "guidance", "quality_preamble", "negative",
            "max_guidance"} <= sdxl_fields
    assert set(get_args(sx.SdxlSampler)) == {
        "euler_a", "dpmpp_2m_karras", "dpmpp_2m_sde_karras",
        "lcm", "euler_trailing", "ddim_trailing",
    }
    assert {"scheduler", "num_inference_steps", "guidance", "negative"} <= set(
        Sd15.Tuned.__struct_fields__
    )
    # SD2's schema is its OWN (th#1139), and the DEFAULTS are the difference
    # that matters: a Turbo recipe validated against SD1.5's vocabulary would
    # accept a 30-step CFG-7 stamping that destroys a one-step checkpoint.
    assert Sd15.Tuned().num_inference_steps == 30 and Sd15.Tuned().guidance == 7.0
    assert Sd2.Tuned().num_inference_steps == 1 and Sd2.Tuned().guidance == 0.0
    assert Sdxl.Tuned().steps == 28 and Sdxl.Tuned().scheduler == "euler_a"


# -------------------------------------------------------- real inference shape


def test_a_fake_backed_sdxl_generates_an_image_through_the_typed_callables() -> None:
    """The whole loop, hubless and cardless, through the real code path.

    Not a mock of the SDK — it IS the SDK with the one part that needs a card
    replaced. So this exercises both text towers, the CFG batching, the
    micro-conditioning block, the bit-exact schedule and four typed callables.
    """

    instance = Sdxl.fake()
    # `pack_shape` returns a plain int; the binding wants the closed Literal.
    # An endpoint reaches it through a `BucketMap` instead of a cast — this is
    # the one place a test has to spell the value itself.
    shape = cast("SdxlShape", sx.pack_shape(1024, 1024))
    seen: list[tuple[int, int]] = []
    image = sx.generate(
        instance,
        shape=shape,
        positive=sx.token_ids([1, 2, 3], device="cpu"),
        negative=sx.token_ids([], device="cpu"),
        steps=3,
        guidance=6.0,
        seed=7,
        on_step=lambda index, total: seen.append((index, total)),
    )
    assert seen == [(0, 3), (1, 3), (2, 3)]
    assert tuple(image.shape) == (1, 3, 1024, 1024)
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
    again = sx.generate(
        instance,
        shape=shape,
        positive=sx.token_ids([1, 2, 3], device="cpu"),
        negative=sx.token_ids([], device="cpu"),
        steps=3,
        guidance=6.0,
        seed=7,
    )
    assert torch.equal(image, again)


@pytest.mark.parametrize(
    ("model", "width", "height"),
    [(Sd15, 512, 512), (Sd15, 768, 512), (Sd2, 512, 512)],
)
def test_a_fake_backed_sd_generates_an_image_through_the_typed_callables(
    model: Any, width: int, height: int
) -> None:
    """Both U-Net declarations, including the non-square bucket.

    768x512 is here on purpose: it is the shape the packed axis exists for, and
    a transposed latent would produce a (1, 3, 512, 768) image that reads as
    "close enough" in a shape assertion written loosely.
    """

    # SD1.5's stamped DEFAULT sampler is `dpmpp_2m_karras`, which pgw#1346 K10
    # deliberately does NOT declare (a multistep solver, owed to B3/B4), so a
    # default-tuned instance refuses rather than renders. Stamping `euler` here
    # is the by-value equivalent of what a catalog slot does, and the refusal
    # itself is asserted in its own test below.
    instance = model.fake(tuned=model.Tuned(scheduler="euler"))
    shape = cast("sd.AnyShape", sd.pack_shape(width, height))
    image = sd.generate(
        instance,
        shape=shape,
        positive=sd.token_ids([1, 2, 3], device="cpu"),
        negative=sd.token_ids([], device="cpu"),
        steps=2,
        guidance=7.0,
        seed=3,
    )
    assert tuple(image.shape) == (1, 3, height, width)
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0


def test_the_initial_latents_start_at_the_ladders_own_scale() -> None:
    """A variance-EXPLODING schedule does not start at unit variance.

    ``leading`` spacing makes ``init_noise_sigma`` ``sqrt(sigma_max**2 + 1)``.
    Starting at 1.0 instead is the classic silent way to render a washed-out
    image — no error, just a wrong picture — so the scale is asserted.

    ~10.8 and not the ~14.6 of the full trained table, because ``leading``
    starts a whole step BELOW the top of the ladder (its highest timestep at 28
    steps is 946, not 999). That gap is exactly the difference the three
    spacings exist to express, and it is why a distilled recipe naming one is
    destroyed by another.
    """

    schedule = EulerDiscrete.from_block(dict(SDXL_SCHEDULER)).schedule(28)
    assert schedule.timesteps[0] == 946.0
    assert 10.0 < schedule.init_noise_sigma < 11.0
    latents = sx.initial_latents(
        shape=sx.pack_shape(1024, 1024),
        seed=1,
        device="cpu",
        dtype=torch.float32,
        sigma=schedule.init_noise_sigma,
    )
    assert tuple(latents.shape) == (1, 4, 128, 128)
    assert float(latents.std()) > 10.0
    other = sx.initial_latents(
        shape=sx.pack_shape(1024, 1024),
        seed=2,
        device="cpu",
        dtype=torch.float32,
        sigma=schedule.init_noise_sigma,
    )
    assert not torch.equal(latents, other)


def test_the_ie740_serving_floors_migrated_by_value() -> None:
    """The PARSED NUMBERS, because "by value" is a claim about numbers.

    Both are production incidents rather than margins. sdxl's ``vram12g`` is
    the value ie#704's stopgap block itself calls "the honest SERVING floor"
    and names as its revert target, after one overloaded scalar
    over-constrained serving to fix mint placement; its ``sm89+`` is the
    DECODABILITY floor for the rowwise handle, not the fastest one. sd15's
    ``vram6g`` is what both of its ``@endpoint``s declare.

    Also asserted: the floors do NOT ride the export digest. A serving floor is
    a fact about the MACHINE and a compiled graph's identity is graph x sm x toolchain,
    so restating a floor must not re-key every artifact — which is why
    ``check_model_bindings.py`` stays byte-clean across this change.
    """

    sdxl = SDXL.layout_requirements
    assert sdxl["cozy.fp8-rowwise@1"].minimum.min_sm == 89
    assert sdxl["plain.bf16@1"].minimum.min_vram_gb == 12.0
    assert dict(SDXL.layouts or {}) == {"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")}

    for spec in (SD15, SD2):
        assert spec.layout_requirements["plain.bf16@1"].minimum.min_vram_gb == 6.0
        assert dict(spec.layouts or {}) == {"*": ("plain.bf16@1",)}

    # The floor is declaration-side only; the committed export does not carry it.
    assert "layout_requirements" not in Sdxl.EXPORT.dumps()


# ----------------------------------------------- what the compiled gauntlet needs


def test_a_mint_can_be_narrowed_to_one_shape_so_a_gauntlet_row_is_affordable() -> None:
    """``--bucket shape=…``, and why it had to exist before the sm_86 row runs.

    B2 grew SDXL from 2 buckets x 2 runners (4 classes) to 9 buckets x 2
    shape-bearing runners + 2 towers (**20 classes**). #1348's row 14 mints
    sdxl on an rtx-a4500 and adopts on an rtx-a4000; at 20 classes it pays 5x
    what it used to for 19 classes it never adopts. Narrowing to the one shape
    the row's workload renders makes it 3 — denoiser, decoder, and the towers
    that declare no shape axis at all and must NOT be dropped by a shape
    filter, which is the half of this that is easy to get wrong.
    """

    from gen_worker.model.mint import variants_of

    everything = variants_of(SDXL)
    assert len(everything) == 20

    one_shape = variants_of(SDXL, buckets={"shape": [sx.pack_shape(1024, 1024)]})
    assert {runner.name for runner, _, _ in one_shape} == {
        "clip_g", "clip_l", "decoder", "denoiser"
    }
    assert len(one_shape) == 4

    just_the_unet = variants_of(
        SDXL, only=("denoiser",), buckets={"shape": [sx.pack_shape(1024, 1024)]}
    )
    assert len(just_the_unet) == 1
    assert just_the_unet[0][1] == {"shape": sx.pack_shape(1024, 1024)}

    # An axis or a value the family does not declare is REFUSED, not silently
    # empty: a typo'd `--bucket` that minted zero classes would report a green
    # row that compiled nothing, which is the rig's own most expensive lesson.
    with pytest.raises(ModelError):
        variants_of(SDXL, buckets={"resolution": [1024]})
    with pytest.raises(ModelError):
        variants_of(SDXL, buckets={"shape": [5120512]})


# ---------------------------------------------- pgw#1346 K10: the scheduler SET
#
# B2 closed with a blocker rather than a deferral: the sampler is a TUNED value
# and `GraphModelSpec.scheduler` was single-valued, so a second implemented
# kind would have been a class no catalog declaration could attach to. The
# consequence B2 measured is the reason this is not a nicety — SDXL's DEFAULT
# sampler is `euler_a`, so the family's one declared block was its TRAINED
# schedule and NOT the one most requests asked for.
#
# The instrument is B2's, unchanged and for its reasons: relative agreement at
# 2e-4 against the reference (never ULP — the reference disagrees with itself
# by up to 85 ULP across machines), exact timesteps, L2 for anything that
# crosses zero, and our own ladder fenced byte-identical across CPU kernels,
# which is the property the reference does not have.


def _ancestral_reference(block: dict[str, Any], **overrides: Any) -> Any:
    diffusers = pytest.importorskip("diffusers")
    merged = {**block, **overrides}
    return diffusers.EulerAncestralDiscreteScheduler(
        num_train_timesteps=int(merged["num_train_timesteps"]),
        beta_start=float(merged["beta_start"]),
        beta_end=float(merged["beta_end"]),
        beta_schedule=str(merged["beta_schedule"]),
        timestep_spacing=str(merged["timestep_spacing"]),
        steps_offset=int(merged["steps_offset"]),
        prediction_type=str(merged["prediction_type"]),
        rescale_betas_zero_snr=bool(merged.get("rescale_betas_zero_snr", False)),
    )


def _ddim_reference(block: dict[str, Any], **overrides: Any) -> Any:
    diffusers = pytest.importorskip("diffusers")
    merged = {**block, **overrides}
    return diffusers.DDIMScheduler(
        num_train_timesteps=int(merged["num_train_timesteps"]),
        beta_start=float(merged["beta_start"]),
        beta_end=float(merged["beta_end"]),
        beta_schedule=str(merged["beta_schedule"]),
        timestep_spacing=str(merged["timestep_spacing"]),
        steps_offset=int(merged["steps_offset"]),
        prediction_type=str(merged["prediction_type"]),
        rescale_betas_zero_snr=bool(merged.get("rescale_betas_zero_snr", False)),
        set_alpha_to_one=bool(merged.get("set_alpha_to_one", True)),
        clip_sample=False,
    )


def _trained(**overrides: Any) -> dict[str, Any]:
    """The SDXL trained schedule, minus the parameters that are one KIND's."""

    return {**dict(SDXL_TRAINED), **overrides}


# ------------------------------------------------------------ the declaration


def test_the_declaration_carries_a_scheduler_SET_keyed_by_the_tuned_sampler() -> None:
    """K10's whole shape, in the two vocabularies it keeps apart.

    The KEY is a sampler — a value ``inst.tuned.scheduler`` can hold. The
    ``name`` is a scheduler KIND. They are not the same vocabulary and the
    mapping between them is many-to-one in both directions: ``euler`` and
    ``euler_trailing`` are ONE kind under two spacings, and ``ddim`` and
    ``euler_a`` are two kinds under one trained table.
    """

    assert {name: entry.name for name, entry in SDXL.schedulers.items()} == {
        "ddim_trailing": "ddim",
        "dpmpp_2m_karras": "dpmsolver_multistep",
        "dpmpp_2m_sde_karras": "dpmsolver_multistep",
        "euler_a": "euler_ancestral_discrete",
        "euler_trailing": "euler_discrete",
    }
    for spec in (SD15, SD2):
        assert {name: entry.name for name, entry in spec.schedulers.items()} == {
            "ddim": "ddim",
            "ddim_trailing": "ddim",
            "dpmpp_2m": "dpmsolver_multistep",
            "dpmpp_2m_karras": "dpmsolver_multistep",
            "dpmpp_2m_sde_karras": "dpmsolver_multistep",
            "euler": "euler_discrete",
            "euler_a": "euler_ancestral_discrete",
            "unipc": "unipc_multistep",
        }
    # Sorted, so the export is canonical and the digest is stable under a
    # re-ordered declaration.
    for spec in (SDXL, SD15, SD2):
        assert list(spec.schedulers) == sorted(spec.schedulers)
        assert [row.sampler for row in export_model_of(spec).schedulers] == sorted(
            spec.schedulers
        )


def export_model_of(spec: Any) -> Any:
    """The committed export for a declaration, by family name."""

    return {"sdxl": Sdxl, "sd15": Sd15, "sd2": Sd2}[spec.name].EXPORT


def test_every_declared_sampler_is_a_name_the_tuned_schema_admits() -> None:
    """The set is keyed by ``tuned.scheduler``, so it must be KEYABLE by it.

    A declared sampler the tuned Literal cannot hold would be a scheduler no
    checkpoint could ever select — the mirror image of the gap K10 closed, and
    just as silent. Asserted as a SUBSET and not equality on purpose: the
    reverse inclusion is the staged gap, and it has its own test below.
    """

    assert set(SDXL.schedulers) <= set(get_args(sx.SdxlSampler))
    assert set(SD15.schedulers) <= set(get_args(sd.Sd15Sampler))
    assert set(SD2.schedulers) <= set(get_args(sd.Sd15Sampler))


def test_a_family_with_one_sampler_declares_a_set_of_one_and_nothing_changes() -> None:
    """B2's held migration needs NOTHING from this change.

    The single-scheduler families migrate mechanically to a set of one, and the
    generated accessor keeps its exact previous shape: no argument, one
    concrete return type. That is what makes the K10 declaration change safe to
    land ahead of the endpoint migration rather than beside it.
    """

    from gen_worker.model.catalog import Flux1Dev

    assert list(Flux1Dev.SCHEDULERS) == ["flow_match_euler"]
    scheduler = Flux1Dev.fake().scheduler()
    assert type(scheduler).__name__ == "FlowMatchEulerDiscrete"
    # No `name=` parameter at all on the single-sampler accessor.
    assert "name" not in inspect.signature(Flux1Dev.scheduler).parameters


def test_a_stamped_sampler_the_sdk_cannot_serve_is_REFUSED_and_never_substituted() -> None:
    """The staged gap, made loud.

    ``Sd15Tuned``'s own DEFAULT is ``dpmpp_2m_karras``, which is a
    ``DPMSolverMultistep`` — a MULTISTEP solver that reads the previous step's
    model output and therefore does not fit the request-scoped, historyless
    schedule this SDK serves. It is owed to B3/B4, which own that solver for
    the DiT fleet anyway.

    What must NOT happen is the thing a single-valued field made inevitable:
    serving the family's other schedule under the requested sampler's name. The
    image would be plausible and wrong, and nothing would say so. So the miss
    is a refusal that names BOTH sides.
    """

    # `lcm` is the ONE name still owed: `LCMScheduler` has no module in
    # `model/solvers/`. Everything else the two endpoints admit is declarable
    # — the multistep pair arrived with B3-math, which landed ahead of this
    # lane and is additive to the set mechanism.
    with pytest.raises(ModelError) as exc:
        Sd15.fake(tuned=Sd15.Tuned(scheduler="lcm")).scheduler()
    assert exc.value.reason is ModelRefusal.SCHEDULER_UNDECLARED
    assert "lcm" in str(exc.value)
    assert "'euler_a'" in str(exc.value)

    # sd15's own DEFAULT is `dpmpp_2m_karras`, and it is SERVABLE. That is the
    # live half of K10 for this family: before the set, the declaration named
    # `euler_discrete` while every default-tuned request asked for a solver
    # nothing could attach.
    assert Sd15.Tuned().scheduler == "dpmpp_2m_karras"
    assert type(Sd15.fake().scheduler()).__name__ == "DPMSolverMultistep"

    # Every name that is owed, enumerated here so the ledger is executable
    # rather than prose. When B3/B4 land a solver, this set shrinks and this
    # test is what says so.
    owed_sdxl = set(get_args(sx.SdxlSampler)) - set(SDXL.schedulers)
    owed_sd15 = set(get_args(sd.Sd15Sampler)) - set(SD15.schedulers)
    assert owed_sdxl == {"lcm"}
    assert owed_sd15 == {"lcm"}

    # SDXL's and SD2's DEFAULTS are covered, which is the live half of K10:
    # `euler_a` is what most SDXL requests actually ask for.
    assert Sdxl.Tuned().scheduler == "euler_a" and "euler_a" in SDXL.schedulers
    assert Sd2.Tuned().scheduler == "euler_a" and "euler_a" in SD2.schedulers


def test_a_declared_block_may_not_carry_a_parameter_its_kind_never_reads() -> None:
    """The silent failure a scheduler SET introduces, fenced.

    With one scheduler a block was copied once. With several, the copy is per
    sampler and the kinds do not read the same parameters:
    ``EulerAncestralDiscrete`` has no ``final_sigmas_type`` and ``Ddim`` has no
    ``final_sigmas_type`` either, while ``Ddim`` has two nobody else has. A
    stale key would change NOTHING and say nothing, so every kind refuses one.
    """

    with pytest.raises(ModelError):
        EulerAncestralDiscrete.from_block(_trained(final_sigmas_type="zero"))
    with pytest.raises(ModelError):
        Ddim.from_block(_trained(final_sigmas_type="zero"))
    with pytest.raises(ModelError):
        EulerDiscrete.from_block(_trained(set_alpha_to_one=True))
    # …and the ones each kind DOES read still resolve.
    assert EulerAncestralDiscrete.from_block(_trained()).timestep_spacing == "leading"
    assert Ddim.from_block(_trained(set_alpha_to_one=False)).set_alpha_to_one is False

    # `clip_sample=True` is declarable and NOT implementable: it needs a clip
    # range no endpoint declares. Refused rather than ignored.
    with pytest.raises(ModelError):
        Ddim.from_block(_trained(clip_sample=True))


def test_the_sampler_vocabulary_is_shared_so_two_families_cannot_disagree() -> None:
    """``euler_a`` means one thing on this fleet, not one thing per family.

    sdxl and sd15 are two declarations over ONE endpoint-visible sampler
    vocabulary. If each spelled the mapping for itself they could drift into
    scheduling different things under one name, which is a defect no test in
    either family would catch.
    """

    from gen_worker.model.catalog.sd_samplers import SD_SAMPLERS, sd_schedulers  # noqa: F401

    for sampler in set(SDXL.schedulers) & set(SD15.schedulers):
        assert SDXL.schedulers[sampler].name == SD15.schedulers[sampler].name
    assert set(SDXL.schedulers) | set(SD15.schedulers) <= set(SD_SAMPLERS)
    # A trained block carrying a kind-specific parameter is refused where the
    # SET is built, not at the first request that reaches that kind.
    with pytest.raises(ModelError):
        sd_schedulers({**dict(SDXL_TRAINED), "final_sigmas_type": "zero"}, ("euler_a",))
    with pytest.raises(ModelError):
        sd_schedulers(dict(SDXL_TRAINED), ("lcm",))

    # And the vocabulary is not this module's invention: `gen_worker.view`
    # already DEFINES each sampler name completely for the `Slot`-served
    # endpoints — the diffusers class plus its config overrides. That table is
    # the fleet's single source for "what does `dpmpp_2m_karras` mean", so the
    # declared blocks are asserted against it rather than restated from memory.
    from gen_worker.view import SAMPLERS

    classes = {
        "ddim": "DDIMScheduler",
        "dpmsolver_multistep": "DPMSolverMultistepScheduler",
        "euler_ancestral_discrete": "EulerAncestralDiscreteScheduler",
        "euler_discrete": "EulerDiscreteScheduler",
        "unipc_multistep": "UniPCMultistepScheduler",
    }
    for sampler, (kind, overrides) in SD_SAMPLERS.items():
        reference, config = SAMPLERS[sampler]
        assert classes[kind] == reference, sampler
        for field, value in config.items():
            assert overrides[field] == value, (sampler, field)


# ------------------------------------------------------- euler_ancestral maths


@pytest.mark.parametrize("steps", ENDPOINT_STEPS)
@pytest.mark.parametrize("spacing", ["leading", "trailing", "linspace"])
@pytest.mark.parametrize(
    ("objective", "zero_snr"), [("epsilon", False), ("v_prediction", True)]
)
def test_the_euler_ancestral_ladder_matches_diffusers(
    steps: int, spacing: str, objective: str, zero_snr: bool
) -> None:
    """``euler_a``'s LADDER is ``euler``'s, and its step is not.

    Both halves are worth asserting. The ladder, because the two classes share
    a trained table and a spacing and would be a real defect if they diverged;
    and separately (below) the step, because that is where they differ.

    Instrument per B2's carry-forward: relative at 2e-4, timesteps EXACT.
    """

    block = _trained(
        timestep_spacing=spacing, prediction_type=objective, rescale_betas_zero_snr=zero_snr
    )
    theirs = _ancestral_reference(block)
    theirs.set_timesteps(steps)
    ours = EulerAncestralDiscrete.from_block(block).schedule(steps)

    assert len(ours.sigmas) == len(theirs.sigmas)
    assert _rel(ours.sigmas, theirs.sigmas.numpy()) <= RELATIVE
    assert _ulp(ours.timesteps, theirs.timesteps.numpy()) == 0
    assert _rel([ours.init_noise_sigma], [float(theirs.init_noise_sigma)]) <= RELATIVE

    # The ladder IS euler's, value for value, under the same block. This is
    # what lets `euler` and `euler_a` share one `_sigma_table`.
    euler = EulerDiscrete.from_block({**block, "final_sigmas_type": "zero"}).schedule(steps)
    assert ours.sigmas == euler.sigmas and ours.timesteps == euler.timesteps
    assert ours.init_noise_sigma == euler.init_noise_sigma


@pytest.mark.parametrize("spacing", ["leading", "trailing"])
@pytest.mark.parametrize(
    ("objective", "zero_snr"), [("epsilon", False), ("v_prediction", True)]
)
def test_a_whole_ancestral_loop_tracks_the_diffusers_loop(
    spacing: str, objective: str, zero_snr: bool
) -> None:
    """28 stochastic steps, ours against theirs, with the table removed.

    Our ladder is loaded into the reference first, exactly as B2's euler loop
    test does, so this measures the STEP. The noise is the other variable and
    it is removed the same way: the reference's generator is re-seeded to this
    step's own key before each call, so both sides draw the SAME tensor and
    what remains is the ancestral arithmetic — ``sigma_up``, ``sigma_down``,
    and the contracted ``dt``.

    Measured on this machine: BIT-EXACT, all four configurations. Asserted
    relatively anyway, for B2's reason — ``pow`` is not correctly rounded and
    varies by ISA, so a bitwise assertion against this reference is not a
    portable claim about anything.
    """

    block = _trained(
        timestep_spacing=spacing, prediction_type=objective, rescale_betas_zero_snr=zero_snr
    )
    theirs = _ancestral_reference(block)
    theirs.set_timesteps(28)
    schedule = EulerAncestralDiscrete.from_block(block).schedule(28)
    assert _ulp(schedule.timesteps, theirs.timesteps.numpy()) == 0
    theirs.sigmas = torch.tensor(schedule.sigmas, dtype=torch.float32)

    torch.manual_seed(0)
    ours_sample = torch.randn(2, 4, 32, 32) * float(schedule.init_noise_sigma)
    their_sample = ours_sample.clone()
    generator = torch.Generator(device="cpu")
    for index, timestep in enumerate(theirs.timesteps):
        prediction = torch.randn(2, 4, 32, 32)
        assert _rel_norm(
            schedule.scale_model_input(index, ours_sample).numpy(),
            theirs.scale_model_input(their_sample, timestep).numpy(),
        ) <= RELATIVE
        seed = sx.step_seed(11, index)
        noise = torch.randn(
            prediction.shape,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            dtype=torch.float32,
        )
        ours_sample = schedule.step(index, prediction, ours_sample, noise)
        generator.manual_seed(seed)
        their_sample = theirs.step(
            prediction, timestep, their_sample, generator=generator
        ).prev_sample
        assert _rel_norm(ours_sample.numpy(), their_sample.numpy()) <= RELATIVE


def test_the_ancestral_split_conserves_the_ladders_variance() -> None:
    """``sigma_up**2 + sigma_down**2 == sigma_next**2``, which is WHY it works.

    An ancestral step is not a worse Euler step; it is an Euler step to a
    SMALLER sigma with the difference handed back as noise. If the split did
    not conserve the variance the trajectory would drift off the ladder the
    denoiser was trained on, and the failure would look like a slightly
    over- or under-denoised image rather than an error.
    """

    schedule = EulerAncestralDiscrete.from_block(_trained()).schedule(28)
    for index in range(len(schedule)):
        down, up = schedule.ancestral(index)
        target = schedule.sigmas[index + 1] ** 2
        assert abs(down**2 + up**2 - target) <= max(target, 1e-12) * RELATIVE
        # It steps DOWN past the next sigma, which is the whole point.
        assert down <= schedule.sigmas[index + 1] + 1e-9
    # The last step lands exactly on the clean sample: nothing is re-noised
    # after the final one, or the image would come back with noise on it.
    assert schedule.ancestral(len(schedule) - 1) == (0.0, 0.0)


def test_an_ancestral_step_cannot_be_taken_without_noise() -> None:
    """A defaulted noise argument would be a silently different sampler.

    ``euler_a`` with zero noise is ``euler`` with a contracted step — a
    plausible image, a wrong sampler, and no error anywhere. So the parameter
    is required, and the two schedule types are different types rather than one
    type with an optional argument.
    """

    ancestral = EulerAncestralDiscrete.from_block(_trained()).schedule(4)
    deterministic = EulerDiscrete.from_block(dict(SDXL_SCHEDULER)).schedule(4)
    sample = torch.randn(1, 4, 8, 8)
    prediction = torch.randn(1, 4, 8, 8)
    with pytest.raises(TypeError):
        ancestral.step(0, prediction, sample)  # type: ignore[call-arg]
    # And with noise it is genuinely a different trajectory from euler's.
    noise = torch.randn(1, 4, 8, 8)
    assert not torch.equal(
        ancestral.step(0, prediction, sample, noise),
        deterministic.step(0, prediction, sample),
    )


# -------------------------------------------------------------- the noise story


def test_the_per_step_noise_is_keyed_so_two_pods_resolve_the_same_tensor() -> None:
    """The reproducibility claim ``euler_a`` needs, and it is NOT the ladder's.

    ``initial_latents`` already argues that a request's noise must mean the
    same thing on two pods, and CPU-seeds for it. An ancestral sampler draws
    again at EVERY step, so that argument has to cover the whole stream or it
    covers almost none of it.

    The stream is KEYED by ``(seed, index)`` rather than run from one advancing
    generator. Both are reproducible across pods; only the keyed one is
    reproducible across LOOP SHAPES, so a preview pass, a resumed loop or a
    reordered call site cannot re-roll the tail. The mix is splitmix64's
    finalizer because ``seed + index`` would make step ``k`` of seed ``s``
    identical to step ``k-1`` of seed ``s+1`` — adjacent receipts sharing
    noise, which is exactly the correlation a seed is supposed to prevent.
    """

    shape = sx.pack_shape(1024, 1024)
    draw = lambda seed, index: sx.step_noise(  # noqa: E731
        shape=shape, seed=seed, index=index, device="cpu", dtype=torch.float32
    )
    # Same key, same bytes — this is the whole claim, and it holds for a fresh
    # call rather than for a position in a stream.
    assert torch.equal(draw(7, 3), draw(7, 3))
    assert tuple(draw(7, 3).shape) == (1, 4, 128, 128)
    # Different step, different noise. Different seed, different noise.
    assert not torch.equal(draw(7, 3), draw(7, 4))
    assert not torch.equal(draw(7, 3), draw(8, 3))
    # …and NOT merely different: the shift-by-one correlation `seed + index`
    # would introduce is absent.
    assert not torch.equal(draw(7, 4), draw(8, 3))
    assert sx.step_seed(7, 4) != sx.step_seed(8, 3)
    # The two families key identically — one story, not one per family.
    assert sx.step_seed(7, 3) == sd.step_seed(7, 3)


def test_the_whole_ancestral_loop_is_reproducible_from_one_seed() -> None:
    """End to end, through the real serving path, on a fake backing.

    The claim a receipt makes is about the LATENTS: this catalog's fake decoder
    is deliberately input-insensitive, so asserting on the image would assert
    nothing. Two runs at one seed must agree bit for bit; two seeds must not.
    """

    instance = Sdxl.fake()
    assert isinstance(sx.schedule_for(instance, steps=3), AncestralSchedule)
    shape = cast("SdxlShape", sx.pack_shape(1024, 1024))

    def run(seed: int) -> Any:
        schedule = sx.schedule_for(instance, steps=3)
        latents = sx.initial_latents(
            shape=int(shape), seed=seed, device="cpu", dtype=torch.float32,
            sigma=schedule.init_noise_sigma,
        )
        for _, latents in sx.denoise(
            instance,
            shape=shape,
            latents=latents,
            prompt_embeds=torch.zeros(2, sx.TEXT_TOKENS, sx.CROSS_ATTENTION_WIDTH),
            pooled=torch.zeros(2, sx.CLIP_G_WIDTH),
            schedule=schedule,
            guidance=6.0,
            seed=seed,
        ):
            pass
        return latents

    assert torch.equal(run(5), run(5))
    assert not torch.equal(run(5), run(6))


# ---------------------------------------------------------------- the ddim maths


@pytest.mark.parametrize("steps", ENDPOINT_STEPS)
@pytest.mark.parametrize("spacing", ["leading", "trailing", "linspace"])
@pytest.mark.parametrize("set_alpha_to_one", [True, False])
@pytest.mark.parametrize(
    ("objective", "zero_snr"), [("epsilon", False), ("v_prediction", True)]
)
def test_the_ddim_trajectory_matches_diffusers(
    steps: int, spacing: str, set_alpha_to_one: bool, objective: str, zero_snr: bool
) -> None:
    """DDIM walks ALPHAS, so this compares alphas — and the timesteps are ints.

    ``set_alpha_to_one`` is an axis rather than a fixed value because the class
    default (True) is the OPPOSITE of what every Stable Diffusion
    ``scheduler_config.json`` ships (False), and it changes only the LAST
    step — the one that decides whether the trajectory lands on a clean sample.
    A test that fixed it would measure the arm the fleet does not serve.

    ``linspace`` is here for a specific reason: ``DDIMScheduler`` ROUNDS its
    linspace grid to integers where the euler family keeps the fractional
    position and interpolates a table at it. Sharing one grid function between
    them would put a half-step error into every linspace DDIM request.
    """

    block = _trained(
        timestep_spacing=spacing,
        prediction_type=objective,
        rescale_betas_zero_snr=zero_snr,
        set_alpha_to_one=set_alpha_to_one,
        clip_sample=False,
    )
    theirs = _ddim_reference(block)
    theirs.set_timesteps(steps)
    ours = Ddim.from_block(block).schedule(steps)

    # EXACT: DDIM's grid is integer arithmetic on both sides.
    assert ours.timesteps == tuple(int(value) for value in theirs.timesteps.tolist())
    assert ours.init_noise_sigma == 1.0 == float(theirs.init_noise_sigma)

    stride = 1000 // steps
    reference = []
    for timestep in ours.timesteps:
        previous = timestep - stride
        reference.append(
            (
                float(theirs.alphas_cumprod[timestep]),
                float(theirs.alphas_cumprod[previous])
                if previous >= 0
                else float(theirs.final_alpha_cumprod),
            )
        )
    assert _rel(
        [value for pair in ours.alphas for value in pair],
        [value for pair in reference for value in pair],
    ) <= RELATIVE


def test_ddim_does_not_clamp_the_terminal_alpha_where_the_euler_family_does() -> None:
    """The one place three kinds genuinely disagree about ONE trained table.

    Under zero-terminal-SNR rescaling ``EulerDiscreteScheduler`` and
    ``EulerAncestralDiscreteScheduler`` overwrite ``alphas_cumprod[-1]`` with
    the smallest positive fp16 subnormal so the first sigma is finite;
    ``DDIMScheduler`` does not, because it never forms a sigma and has no
    infinity to avoid. Sharing one cached table between them without the flag
    would put a wrong FIRST step into every v-prediction DDIM request — the
    step furthest from the data manifold, so the most visible one.
    """

    from gen_worker.model.solvers.precision import alphas_cumprod as _alphas_cumprod

    clamped = _alphas_cumprod(1000, 0.00085, 0.012, "scaled_linear", True, True)
    plain = _alphas_cumprod(1000, 0.00085, 0.012, "scaled_linear", True, False)
    assert clamped[-1] == 2.0**-24
    assert plain[-1] != clamped[-1]
    assert clamped[:-1] == plain[:-1]
    # Without the rescale the flag changes nothing, so it never fires by
    # accident on the epsilon rows every SD checkpoint actually serves.
    assert _alphas_cumprod(1000, 0.00085, 0.012, "scaled_linear", False, True) == (
        _alphas_cumprod(1000, 0.00085, 0.012, "scaled_linear", False, False)
    )

    # And the reference agrees with our unclamped table where DDIM reads it.
    theirs = _ddim_reference(_trained(prediction_type="v_prediction",
                                      rescale_betas_zero_snr=True))
    theirs.set_timesteps(28)
    assert _rel(plain, theirs.alphas_cumprod.numpy()) <= RELATIVE


@pytest.mark.parametrize("spacing", ["leading", "trailing"])
@pytest.mark.parametrize(
    ("objective", "zero_snr"), [("epsilon", False), ("v_prediction", True)]
)
def test_a_whole_ddim_loop_tracks_the_diffusers_loop(
    spacing: str, objective: str, zero_snr: bool
) -> None:
    """28 deterministic steps, with OUR table injected so this is the step.

    Same discipline as B2's euler loop and the ancestral one above: the table
    is removed as a variable by loading ours into the reference, and what is
    left is the arithmetic. ``scale_model_input`` is asserted too, because
    DDIM's is the IDENTITY where the euler family divides by
    ``sqrt(sigma**2+1)`` — feeding a DDIM trajectory through the euler
    pre-scale is a wrong image and never an error.

    Measured on this machine: BIT-EXACT across all four configurations.
    """

    from gen_worker.model.solvers.precision import alphas_cumprod as _alphas_cumprod

    block = _trained(
        timestep_spacing=spacing,
        prediction_type=objective,
        rescale_betas_zero_snr=zero_snr,
        set_alpha_to_one=False,
        clip_sample=False,
    )
    theirs = _ddim_reference(block)
    theirs.set_timesteps(28)
    table = _alphas_cumprod(1000, 0.00085, 0.012, "scaled_linear", zero_snr, False)
    theirs.alphas_cumprod = torch.tensor(table, dtype=torch.float32)
    theirs.final_alpha_cumprod = torch.tensor(table[0], dtype=torch.float32)
    schedule = Ddim.from_block(block).schedule(28)

    torch.manual_seed(0)
    ours_sample = torch.randn(2, 4, 32, 32)
    their_sample = ours_sample.clone()
    for index, timestep in enumerate(theirs.timesteps):
        prediction = torch.randn(2, 4, 32, 32)
        # THE IDENTITY, asserted rather than skipped — and asserted about OUR
        # function ALONE, which is what makes it exact. The two samples have
        # already diverged by this point in the loop (parts per million, but
        # not zero), so `torch.equal` ACROSS the two loops is the same
        # unportable claim B2 spent three CI cycles retracting: it held on the
        # development machine and failed on the runner. Both halves are here
        # because they say different things — ours pre-scales by nothing, and
        # theirs does the same to its own sample.
        assert torch.equal(schedule.scale_model_input(index, ours_sample), ours_sample)
        assert torch.equal(
            theirs.scale_model_input(their_sample, timestep), their_sample
        )
        assert _rel_norm(
            schedule.scale_model_input(index, ours_sample).numpy(),
            theirs.scale_model_input(their_sample, timestep).numpy(),
        ) <= RELATIVE
        ours_sample = schedule.step(index, prediction, ours_sample)
        their_sample = theirs.step(prediction, int(timestep), their_sample).prev_sample
        assert _rel_norm(ours_sample.numpy(), their_sample.numpy()) <= RELATIVE


def test_a_ddim_trajectory_is_not_a_sigma_ladder_and_says_so_in_its_type() -> None:
    """Three schedules, three types, and the differences are load-bearing.

    A common supertype would have to hide exactly what separates them: DDIM
    has no sigmas and starts at unit variance, the ancestral step consumes
    noise, and the euler step does not. Every one of those is a wrong IMAGE
    rather than an error when it is got wrong, so the union is the honest
    return type and ``isinstance`` is the honest dispatch.
    """

    instance = Sdxl.fake()
    assert isinstance(instance.scheduler("ddim_trailing").schedule(4), DdimSchedule)
    assert isinstance(instance.scheduler("euler_a").schedule(4), AncestralSchedule)
    assert isinstance(instance.scheduler("euler_trailing").schedule(4), DiscreteSchedule)
    # An `AncestralSchedule` is NOT a `DiscreteSchedule`, so a loop that only
    # handles the deterministic step cannot silently accept it.
    assert not isinstance(
        instance.scheduler("euler_a").schedule(4), DiscreteSchedule
    )

    ddim = instance.scheduler("ddim_trailing").schedule(4)
    assert ddim.init_noise_sigma == 1.0
    assert not hasattr(ddim, "sigmas")
    # DDIM cannot walk more steps than the trained grid has timesteps.
    with pytest.raises(ModelError):
        instance.scheduler("ddim_trailing").schedule(1001)


# ------------------------------------------------------- cross-kernel stability


@pytest.mark.parametrize("sampler", ["ddim_trailing", "euler_a"])
def test_the_new_ladders_do_not_depend_on_which_cpu_kernel_torch_dispatched(
    sampler: str,
) -> None:
    """B2's byte-stability fence, extended to the two new kinds.

    This is the property the reference does NOT have and the reason this module
    is an improvement rather than a transcription: ``torch``'s float32 CPU
    ``linspace`` dispatches by ISA, so diffusers resolves a different ladder
    depending on which CPU a pod rented. Ours cannot — every operation is IEEE
    double arithmetic with one explicit narrowing — so the subprocess below,
    forced onto the scalar kernels, must produce the SAME BYTES.

    It matters more for an ancestral sampler than for a deterministic one:
    ``sigma_up`` and ``sigma_down`` are derived from the ladder and then
    MULTIPLIED INTO NOISE, so a ladder that moved by a ULP would move the
    stochastic component of every step, not just the trajectory.
    """

    program = (
        "import json;"
        "from gen_worker.model.catalog import Sdxl;"
        f"s=Sdxl.fake().scheduler({sampler!r}).schedule(28);"
        "print(json.dumps("
        "{'t': [float(x) for x in s.timesteps],"
        " 'v': [float(x) for pair in s.alphas for x in pair]"
        " if hasattr(s, 'alphas') else list(s.sigmas)}"
        "))"
    )
    root = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root), "ATEN_CPU_CAPABILITY": "default"},
    )
    assert result.returncode == 0, result.stderr
    scalar = json.loads(result.stdout.strip().splitlines()[-1])

    here = Sdxl.fake().scheduler(cast("Any", sampler)).schedule(28)
    values = (
        [value for pair in here.alphas for value in pair]
        if isinstance(here, DdimSchedule)
        else list(here.sigmas)
    )
    assert scalar["t"] == [float(value) for value in here.timesteps]
    assert scalar["v"] == values


@pytest.mark.parametrize("sampler", sorted(SDXL.schedulers))
def test_every_declared_sdxl_sampler_renders_through_the_real_serving_path(
    sampler: str,
) -> None:
    """The claim the SET makes, exercised once per member.

    A declared scheduler that no loop can actually walk is the mirror image of
    the gap K10 closed, and just as quiet — the declaration would look complete
    and the first real request would fail. So every member is served end to
    end, hubless and cardless, through the same ``generate`` an endpoint calls.

    Five schedule types over five samplers, and the loop dispatches on the TYPE
    rather than on the name: the multistep solvers carry history between steps
    and are NOT pre-scaled (their ``init_noise_sigma`` is 1.0), the ancestral
    one consumes keyed noise, and ``sde-dpmsolver++`` consumes it too. Getting
    any of those wrong is a plausible wrong image, never an error.
    """

    instance = Sdxl.fake(tuned=Sdxl.Tuned(scheduler=cast("Any", sampler)))
    shape = cast("SdxlShape", sx.pack_shape(1024, 1024))
    seen: list[int] = []
    image = sx.generate(
        instance,
        shape=shape,
        positive=sx.token_ids([1, 2, 3], device="cpu"),
        negative=sx.token_ids([], device="cpu"),
        steps=4,
        guidance=6.0,
        seed=7,
        on_step=lambda index, total: seen.append(index),
    )
    assert seen == [0, 1, 2, 3]
    assert tuple(image.shape) == (1, 3, 1024, 1024)
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0


@pytest.mark.parametrize("sampler", sorted(SD15.schedulers))
def test_every_declared_sd15_sampler_renders_through_the_real_serving_path(
    sampler: str,
) -> None:
    """The same claim for the family whose DEFAULT is a multistep solver."""

    instance = Sd15.fake(tuned=Sd15.Tuned(scheduler=cast("Any", sampler)))
    image = sd.generate(
        instance,
        shape=cast("sd.AnyShape", sd.pack_shape(512, 512)),
        positive=sd.token_ids([1, 2, 3], device="cpu"),
        negative=sd.token_ids([], device="cpu"),
        steps=4,
        guidance=7.0,
        seed=3,
    )
    assert tuple(image.shape) == (1, 3, 512, 512)
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
