"""The declared solvers, one module each, and the pieces they share.

``model/scheduler.py`` is the SURFACE — the closed :class:`SchedulerKind` set,
the parser, and the ``IMPLEMENTED`` table the binding generator reads. This
package is the MATH behind it, split so a solver is a file rather than a region
of one:

* :mod:`.precision` — the float32 discipline every ladder is reproduced at;
* :mod:`.block` — reading one declared scheduler block, refusing not coercing;
* :mod:`.ladders` — the sigma ladders (trained-table, Karras, exponential, flow)
  and the timestep grids, as pure functions;
* :mod:`.dpm_multistep` — DPM-Solver++ (2M), the fleet's most-selected sampler;
* :mod:`.unipc_multistep` — UniPC, the video fleet's trained solver.

Nothing here imports ``torch``, ``numpy`` or ``diffusers``. Samples and model
outputs are tensor OPERANDS, never named types, so an adopt-only serve role
(pgw#1328) holds the whole package for free — and, the property that turned out
to matter more, every scalar is IEEE double arithmetic with explicit narrowings
and therefore cannot vary with the CPU kernel a pod's torch dispatched.
"""

from __future__ import annotations

from .block import SchedulerBlock, SchedulerValue
from .dpm_multistep import DPMSolverMultistep, DpmSolverSchedule, MultistepHistory
from .unipc_multistep import UniPCMultistep, UniPcHistory, UniPcSchedule

__all__ = [
    "DPMSolverMultistep",
    "DpmSolverSchedule",
    "MultistepHistory",
    "SchedulerBlock",
    "SchedulerValue",
    "UniPCMultistep",
    "UniPcHistory",
    "UniPcSchedule",
]
