"""The Stable Diffusion lineage's SAMPLER vocabulary, in one place.

pgw#1346 K10: ``GraphModelSpec.schedulers`` is a SET keyed by the sampler name
a checkpoint is stamped with, because the sampler is a checkpoint fact and not
a family constant. This module answers the one question that set-keying
introduces and that no single family owns:

    what does the name ``euler_a`` MEAN?

It is not family truth — ``sdxl`` and ``sd15`` are two declarations offering
overlapping subsets of ONE endpoint-visible vocabulary, and if each spelled
``euler_a`` for itself the two could drift into meaning different schedules
under one name. It is also not scheduler-math truth:
:mod:`gen_worker.model.scheduler` deliberately knows no sampler names at all,
only KINDS.

So the mapping lives here, once, as a pair per name: the scheduler KIND, and
the parameters that name overrides on the family's own trained schedule. The
resolved blocks are what ride the export digest, so the DECLARATION still
carries every value verbatim — this module only keeps two families from
disagreeing about which values those are.

**The names come from ``gen_worker.view.SAMPLERS``, which already DEFINES each
one completely** — the diffusers class plus the config overrides — for the
``Slot``-served endpoints. That table is the single source for "what does
``dpmpp_2m_karras`` mean here", and the test suite asserts this module agrees
with it rather than restating it from memory.

**One name is still owed: ``lcm``.** ``LCMScheduler`` has no implementation in
``model/solvers/`` yet. A checkpoint stamped with it is REFUSED by name
(``SCHEDULER_UNDECLARED``), never served under a neighbouring schedule — which
is the whole reason the set is exhaustive rather than defaulted. Every other
name the two endpoints admit is declarable: the multistep pair arrived with
pgw#1346 B3 (``dpmsolver_multistep``, ``unipc_multistep``), which landed ahead
of this lane and is additive to it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from .. import scheduler
from ..errors import ModelError, ModelRefusal
from ..scheduler import IMPLEMENTED, SchedulerBlock, SchedulerValue, Trained, parse_kind
from ..spec import Scheduler

#: What each endpoint-visible sampler NAME means: the scheduler kind, and what
#: that name overrides on the family's trained schedule.
#:
#: ``steps_offset`` is set to 0 by both trailing entries. It changes nothing —
#: neither reference implementation reads it outside the ``leading`` branch —
#: and it is stated anyway, because a declaration that carries a checkpoint's
#: ``steps_offset=1`` beside ``timestep_spacing="trailing"`` reads as though
#: the offset applies.
SD_SAMPLERS: Final[Mapping[str, tuple[str, Mapping[str, SchedulerValue]]]] = {
    #: The trained schedule walked deterministically. Every SD/SDXL
    #: checkpoint's own `scheduler_config.json` spacing.
    "euler": ("euler_discrete", {"final_sigmas_type": "zero"}),
    #: SDXL-Lightning's published recipe, which the sdxl endpoint pins for both
    #: its 4- and 8-step turbo arms. `trailing` ends a step above the bottom of
    #: the ladder; a distilled 4-step recipe is destroyed by `leading`.
    "euler_trailing": (
        "euler_discrete",
        {"timestep_spacing": "trailing", "steps_offset": 0, "final_sigmas_type": "zero"},
    ),
    #: SDXL's DEFAULT sampler, and sd2/SD-Turbo's. Same ladder as `euler`,
    #: stochastic step — see `EulerAncestralDiscrete`. No `final_sigmas_type`:
    #: the ancestral scheduler has none and declaring one is refused.
    "euler_a": ("euler_ancestral_discrete", {}),
    #: sd15's payload enum offers this one directly.
    "ddim": ("ddim", {"set_alpha_to_one": False, "clip_sample": False}),
    #: sd15's `generate_hyper` pins this UNCONDITIONALLY (a handler reaches
    #: DDIM with no payload involved) and sdxl's enum offers it.
    "ddim_trailing": (
        "ddim",
        {
            "timestep_spacing": "trailing",
            "steps_offset": 0,
            "set_alpha_to_one": False,
            "clip_sample": False,
        },
    ),
    #: DPM-Solver++ (2M) on the trained beta ladder — sd15's plain `dpmpp_2m`.
    "dpmpp_2m": (
        "dpmsolver_multistep",
        {"solver_order": 2, "final_sigmas_type": "zero"},
    ),
    #: The fleet's most-selected sampler and `Sd15Tuned`'s own DEFAULT: the same
    #: solver on a KARRAS ladder, which is one declared boolean and nothing else.
    "dpmpp_2m_karras": (
        "dpmsolver_multistep",
        {"solver_order": 2, "use_karras_sigmas": True, "final_sigmas_type": "zero"},
    ),
    #: STAMPED on two live sdxl catalog entries. The stochastic algorithm on the
    #: same Karras ladder.
    "dpmpp_2m_sde_karras": (
        "dpmsolver_multistep",
        {
            "solver_order": 2,
            "algorithm_type": "sde-dpmsolver++",
            "use_karras_sigmas": True,
            "final_sigmas_type": "zero",
        },
    ),
    #: sd15's payload enum offers it. `view.SAMPLERS` overrides NOTHING, so this
    #: is `UniPCMultistepScheduler`'s own defaults on the family's trained table.
    "unipc": ("unipc_multistep", {}),
}


def sd_schedulers(
    trained: SchedulerBlock, samplers: tuple[str, ...]
) -> dict[str, Scheduler]:
    """One family's declared scheduler set, from its trained noise schedule.

    ``trained`` carries ONLY the trained-schedule parameters — the betas, the
    objective, the spacing and offset the checkpoint's own config states. A
    kind-specific parameter here would be a family declaring, say,
    ``final_sigmas_type`` for a scheduler that has none, so it is refused
    rather than dropped.

    Every resolved block is CONSTRUCTED before it is returned. A block that
    would refuse at serve time refuses at declaration import instead, on the
    author's machine, with no pod behind it.
    """

    unknown = sorted(set(trained) - set(Trained.TRAINED_PARAMETERS))
    if unknown:
        raise ModelError(
            ModelRefusal.SCHEDULER_INVALID,
            f"the trained schedule may not carry {unknown[0]!r}: it is a parameter of ONE "
            f"scheduler kind, so it belongs in this sampler's overrides. It reads "
            f"{list(Trained.TRAINED_PARAMETERS)!r}",
        )
    resolved: dict[str, Scheduler] = {}
    for sampler in samplers:
        if sampler not in SD_SAMPLERS:
            raise ModelError(
                ModelRefusal.SCHEDULER_UNDECLARED,
                f"sampler {sampler!r} has no meaning in the SD vocabulary; it reads "
                f"{sorted(SD_SAMPLERS)!r}",
            )
        kind, overrides = SD_SAMPLERS[sampler]
        block: dict[str, SchedulerValue] = {**trained, **overrides}
        # Construct it now: `from_block` is where an unread parameter, an
        # unrecognised spacing or an unimplementable arm is refused. Resolved
        # through `IMPLEMENTED`, the ONE table that pairs a kind with its
        # class, rather than through a second mapping that can disagree.
        implementation: Any = getattr(scheduler, IMPLEMENTED[parse_kind(kind)])
        implementation.from_block(block)
        resolved[sampler] = Scheduler(kind, block)
    return resolved


__all__ = ["SD_SAMPLERS", "sd_schedulers"]
