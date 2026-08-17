"""pgw#1346 B3a — the flow-match ladders the three DiT families actually walk.

The B2 lane established the instrument this file uses, and it did so the hard
way (two wrong ULP bounds in a row): **do not bound a torch-derived reference
in ULP at all.** Diffusers' ladder is not reproducible against itself across
CPU kernels, so bit-equality is not a property it possesses. What is asserted
here instead is:

* the SIGMAS agree with ``diffusers.FlowMatchEulerDiscreteScheduler``
  RELATIVELY, at 2e-4 — ~20x tighter than one bf16 ULP (3.9e-3), the precision
  the denoiser actually computes in;
* OUR OWN ladder is BYTE-IDENTICAL across torch's CPU kernels, which is the
  property the reference does not have and the reason to prefer this module;
* the loop is compared in the L2 NORM, never element-wise, because latents
  cross zero.

The flow-match ladder is much simpler than B2's euler_discrete one — no trained
beta table, no thousand-term cumulative product — so the agreement here is far
tighter than the bound. The bound is still relative, deliberately: the moment a
future runner dispatches a different kernel, an exact assertion would be
red for a reason that has nothing to do with this fleet.

**The finding this file exists to record**: the W2 plan scoped B3 expecting
"explicit-sigma ladders (new)" for the few-step and DMD lanes. Measured against
all three pipelines, none is owed — every one hands ``set_timesteps`` the same
``linspace(1.0, 1/steps, steps)`` the SDK already synthesizes, and a distilled
lane differs by its step COUNT and its shift. What IS owed, and is what
:mod:`gen_worker.model.flow_ladders` adds, is Qwen-Image's ``shift_terminal``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gen_worker.model.catalog.ernie import SCHEDULER as ERNIE_SCHEDULER
from gen_worker.model.catalog.qwen_image import SCHEDULER as QWEN_SCHEDULER
from gen_worker.model.catalog.z_image import SCHEDULER as Z_SCHEDULER
from gen_worker.model.errors import ModelError
from gen_worker.model.flow_ladders import FlowMatchLadder
from gen_worker.model.scheduler import FlowMatchEulerDiscrete

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

#: B2's bound, carried forward unchanged. See this module's header.
RELATIVE = 2e-4

#: Every step count the three endpoints can reach: 1 (boot warm-up), 4, the two
#: distilled recipes (8 for the qwen Lightning and z-image PAI lanes and for
#: ernie Turbo, 9 for z-image's official DMD card), the three base defaults
#: (28 z-image/ernie, 30 qwen), and the declared range ends.
STEPS = (1, 4, 8, 9, 28, 30, 50, 80)


def _rel(ours: Any, theirs: Any) -> float:
    """Largest RELATIVE difference, scaled by the reference's own magnitude."""

    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(theirs, dtype=np.float64)
    return float((np.abs(a - b) / np.maximum(np.abs(b), 1e-12)).max())


def _raw_sigmas(steps: int) -> list[float]:
    """The raw ladder all three pipelines hand ``set_timesteps``.

    qwen spells it ``np.linspace(1.0, 1/steps, steps)``, z-image spells it
    ``torch.linspace(1.0, 1/steps, steps)`` inside
    ``get_default_z_image_sigmas``, and ERNIE spells it
    ``torch.linspace(1.0, 0.0, steps + 1)[:-1]``. All three are the same points.
    """

    return list(np.linspace(1.0, 1 / steps, steps))


def _reference(block: Any, **overrides: Any) -> Any:
    merged = {**dict(block), **overrides}
    return diffusers.FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=int(merged["num_train_timesteps"]),
        shift=float(merged["shift"]),
        use_dynamic_shifting=bool(merged["use_dynamic_shifting"]),
        base_shift=float(merged.get("base_shift", 0.5)),
        max_shift=float(merged.get("max_shift", 1.15)),
        base_image_seq_len=int(merged.get("base_image_seq_len", 256)),
        max_image_seq_len=int(merged.get("max_image_seq_len", 4096)),
        shift_terminal=merged.get("shift_terminal", None),
        time_shift_type=str(merged.get("time_shift_type", "exponential")),
    )


# ------------------------------------------------------- the explicit-sigma answer


@pytest.mark.parametrize("steps", STEPS)
def test_ernies_own_raw_ladder_is_the_one_the_sdk_already_synthesizes(steps: int) -> None:
    """``linspace(1, 0, steps + 1)[:-1]`` IS ``linspace(1, 1/steps, steps)``.

    ERNIE is the family whose pipeline spells its ladder most differently from
    the other two, and it is the same ladder. This is the measurement that
    retires the batch plan's "explicit-sigma ladders (new)" line for B3a: what a
    distilled lane needs is a step count, not a table.
    """

    ernie_style = torch.linspace(1.0, 0.0, steps + 1)[:-1].tolist()
    assert _rel(ernie_style, _raw_sigmas(steps)) <= RELATIVE


# ------------------------------------------------------------ the static ladders


@pytest.mark.parametrize("steps", STEPS)
@pytest.mark.parametrize(
    ("block", "shift", "what"),
    [
        (Z_SCHEDULER, None, "z-image base"),
        # The official DMD Turbo checkpoint's OWN published shift. It arrives
        # as a tuned value rather than a second declaration, so it is exercised
        # here through the same override the serve half applies.
        (Z_SCHEDULER, 3.0, "z-image turbo (DMD)"),
        (ERNIE_SCHEDULER, None, "ernie, both checkpoints"),
    ],
)
def test_the_static_shift_ladders_match_diffusers(
    steps: int, block: Any, shift: float | None, what: str
) -> None:
    """Two families, three published shifts, one ladder implementation.

    Neither family sets ``use_dynamic_shifting``, so the resolution is not an
    axis of these ladders at all — which is why neither declaration carries a
    packed-token bucket for the SCHEDULE's sake, only for the graph's.
    """

    scheduler = FlowMatchEulerDiscrete.from_block(dict(block))
    if shift is not None:
        scheduler = type(scheduler)(
            num_train_timesteps=scheduler.num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=scheduler.use_dynamic_shifting,
            base_shift=scheduler.base_shift,
            max_shift=scheduler.max_shift,
            base_image_seq_len=scheduler.base_image_seq_len,
            max_image_seq_len=scheduler.max_image_seq_len,
        )
    ours = scheduler.schedule(steps)

    theirs = _reference(block, **({} if shift is None else {"shift": shift}))
    theirs.set_timesteps(sigmas=_raw_sigmas(steps))

    assert len(ours.sigmas) == len(theirs.sigmas)
    assert _rel(ours.sigmas, theirs.sigmas.numpy()) <= RELATIVE, what
    assert _rel(ours.timesteps, theirs.timesteps.numpy()) <= RELATIVE, what


def test_z_images_two_checkpoints_walk_measurably_different_ladders() -> None:
    """Why ``shift`` had to become a tuned field rather than stay implicit.

    The base checkpoint publishes 6.0 and the official Turbo 3.0. Serving the
    DMD lane on the base ladder is not a rounding difference — it moves the
    ladder by tens of percent on a nine-step walk, which is most of what a
    distilled recipe IS.
    """

    base = FlowMatchEulerDiscrete.from_block(dict(Z_SCHEDULER))
    turbo = type(base)(
        num_train_timesteps=base.num_train_timesteps,
        shift=3.0,
        use_dynamic_shifting=False,
    )
    ladder_base = base.schedule(9).sigmas
    ladder_turbo = turbo.schedule(9).sigmas
    assert _rel(ladder_turbo, ladder_base) > 0.2
    # ...and both still terminate exactly where a schedule must.
    assert ladder_base[-1] == ladder_turbo[-1] == 0.0


# ------------------------------------------------------- the stretched ladder


#: The packed token counts Qwen-Image's fourteen presets produce, at the two
#: extremes and the anchor: 720x1280 is the smallest, 1328x1328 the square
#: default, 1664x928 the widest of the 1.7 MP tier.
QWEN_SEQ_LENS = (45 * 80, 83 * 83, 58 * 104)


@pytest.mark.parametrize("steps", [count for count in STEPS if count > 1])
@pytest.mark.parametrize("image_seq_len", QWEN_SEQ_LENS)
def test_the_qwen_ladder_matches_diffusers_including_the_terminal_stretch(
    steps: int, image_seq_len: int
) -> None:
    """The whole reason :mod:`gen_worker.model.flow_ladders` exists.

    Dynamic shifting AND a terminal stretch, differenced against the reference
    across every step count the endpoint reaches and across the extremes of its
    preset grid — the ladder is resolution-dependent here, unlike the other two
    families, so the sequence length is a real axis of this comparison.
    """

    ours = FlowMatchLadder.from_block(dict(QWEN_SCHEDULER)).schedule(
        steps, image_seq_len=image_seq_len
    )
    theirs = _reference(QWEN_SCHEDULER)
    block = dict(QWEN_SCHEDULER)
    mu = FlowMatchEulerDiscrete.from_block(block).mu(image_seq_len)
    theirs.set_timesteps(sigmas=_raw_sigmas(steps), mu=mu)

    assert len(ours.sigmas) == len(theirs.sigmas)
    assert _rel(ours.sigmas, theirs.sigmas.numpy()) <= RELATIVE
    assert _rel(ours.timesteps, theirs.timesteps.numpy()) <= RELATIVE


@pytest.mark.parametrize("steps", (8, 30))
def test_the_sdk_scheduler_alone_would_serve_an_unstretched_ladder(steps: int) -> None:
    """The trap this module removes, measured rather than asserted in prose.

    ``instance.scheduler()`` resolves a ``FlowMatchEulerDiscrete`` from the SAME
    declared block, and that class does not read ``shift_terminal`` — so it
    would walk a ladder whose last evaluated sigma is the shifted ``1/steps``
    instead of 0.02. The declaration would say the ladder was stretched and
    every request would walk an unstretched one.
    """

    block = dict(QWEN_SCHEDULER)
    seq_len = 83 * 83
    stretched = FlowMatchLadder.from_block(block).schedule(steps, image_seq_len=seq_len)
    unstretched = FlowMatchEulerDiscrete.from_block(block).schedule(
        steps, image_seq_len=seq_len
    )

    # The declared terminal, reached exactly (in double arithmetic, so this is
    # an equality and not a tolerance).
    assert stretched.sigmas[-2] == pytest.approx(0.02, abs=1e-12)
    assert unstretched.sigmas[-2] != pytest.approx(0.02, abs=1e-3)
    # Every sigma moves, not just the last — the ladder is rescaled in
    # `1 - sigma` space — and the walk still starts at 1.0.
    assert stretched.sigmas[0] == pytest.approx(1.0, abs=1e-12)
    assert stretched.sigmas[1] != unstretched.sigmas[1]
    # The terminal 0.0 the Schedule appends is the point the last step LANDS
    # on and is never stretched.
    assert stretched.sigmas[-1] == 0.0


def test_a_one_step_stretched_ladder_is_refused_here_and_is_NaN_upstream() -> None:
    """The transform's one degenerate input, and what the reference does with it.

    A one-step ladder is the single sigma 1.0, so there is no span to rescale.
    Diffusers divides 0 by 0 and returns NaN — measured below, not assumed —
    which renders nothing and explains nothing. This module refuses with the
    reason instead.

    **Migration note for the qwen-image endpoint (pgw#1346 W2):** the one place
    the fleet reaches a one-step walk is BOOT WARM-UP (`_warm_steps` clamps
    every regime to 1 on `ctx.boot_warmup`), where the pass exists to trace a
    graph and its output is discarded. A migrated handler must resolve its warm
    ladder at 2 steps or skip the ladder entirely; it must not hand this a 1.
    """

    with pytest.raises(ModelError, match="at least two steps"):
        FlowMatchLadder.from_block(dict(QWEN_SCHEDULER)).schedule(
            1, image_seq_len=6889
        )

    theirs = _reference(QWEN_SCHEDULER)
    mu = FlowMatchEulerDiscrete.from_block(dict(QWEN_SCHEDULER)).mu(6889)
    theirs.set_timesteps(sigmas=_raw_sigmas(1), mu=mu)
    assert bool(torch.isnan(theirs.sigmas).any())


def test_a_stretch_of_a_ladder_that_already_terminates_there_is_the_identity() -> None:
    """The transform's fixed point, which is what makes it safe to always apply."""

    ladder = FlowMatchLadder.from_block({**dict(QWEN_SCHEDULER), "shift_terminal": 0.02})
    sigmas = ladder.schedule(12, image_seq_len=4096).sigmas[:-1]
    assert ladder.stretch(sigmas) == pytest.approx(sigmas, abs=1e-12)


def test_our_stretched_ladder_does_not_depend_on_which_cpu_kernel_torch_dispatched() -> None:
    """The property the reference does NOT have, fenced — B2's instrument, reused.

    ``ATEN_CPU_CAPABILITY=default`` forces torch's scalar CPU kernels. Ours is
    IEEE double arithmetic throughout, so the subprocess must produce the SAME
    BYTES. That is the reproducibility claim that matters: a receipt's seed and
    step count meaning the same ladder on two pods.
    """

    program = (
        "import json;"
        "from gen_worker.model.flow_ladders import FlowMatchLadder;"
        "from gen_worker.model.catalog.qwen_image import SCHEDULER as B;"
        "print(json.dumps(list("
        "FlowMatchLadder.from_block(dict(B)).schedule(30, image_seq_len=6889).sigmas)))"
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

    here = FlowMatchLadder.from_block(dict(QWEN_SCHEDULER)).schedule(
        30, image_seq_len=6889
    )
    assert tuple(scalar) == here.sigmas


# ------------------------------------------------------------------- refusals


def test_linear_time_shifting_is_refused_rather_than_silently_exponential() -> None:
    """No checkpoint this fleet serves publishes it, so it has no measurement."""

    with pytest.raises(ModelError, match="linear time shifting"):
        FlowMatchLadder.from_block(
            {**dict(QWEN_SCHEDULER), "time_shift_type": "linear"}
        )


@pytest.mark.parametrize("value", [1.0, 1.5, -0.1, True, "0.02"])
def test_an_unusable_shift_terminal_is_refused_at_the_declaration(value: object) -> None:
    """A stretch to 1.0 divides by zero and a bool is not a number somebody meant."""

    with pytest.raises(ModelError):
        FlowMatchLadder.from_block(
            {**dict(QWEN_SCHEDULER), "shift_terminal": value}  # type: ignore[dict-item]
        )


def test_an_absent_shift_terminal_is_not_zero() -> None:
    """ERNIE publishes ``shift_terminal: null``; z-image publishes no key at all.

    Both must resolve to "no stretch", and the declaration carries neither key —
    a scheduler block holds finite JSON scalars, so ``null`` cannot be declared
    and 0.0 would be a DIFFERENT ladder rather than a faithful transcription.
    """

    for block in (ERNIE_SCHEDULER, Z_SCHEDULER):
        ladder = FlowMatchLadder.from_block(dict(block))
        assert ladder.shift_terminal is None
        assert ladder.schedule(8).sigmas == FlowMatchEulerDiscrete.from_block(
            dict(block)
        ).schedule(8).sigmas
