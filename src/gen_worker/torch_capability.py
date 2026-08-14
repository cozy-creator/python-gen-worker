"""Torch is a CAPABILITY, not a precondition for booting.

A worker with no torch is a first-class shape, not a degenerate GPU worker
(`accelerator='none'`, never `'cpu'`). A bare torch import anywhere on the boot
path —

    entrypoint.py -> env_seal.establish()
                  -> settings_authority.impose_torch()
                  -> host_isa.impose()
                  -> guard_closure.establish_posture()

— kills a torchless image at ``phase=env_seal`` before it can advertise a
single function, and no env knob skips it.

The seal is about REPRODUCIBILITY of the execution environment. On a worker with
no torch there is no inductor/dynamo config to impose, so the honest behaviour is
to record that fact: ``torch: "absent"`` is a real, keyable environment fact, and
a seal that states it is STRONGER than one that cannot be computed. Adding a
~2-3 GB torch to images whose entire value is being cheap CPU pods would encode
an SDK defect as a fleet requirement instead.

Deliberately NOT memoized. The answer after the first call is a ``sys.modules``
lookup, and a cached verdict is exactly the shape that leaks between tests
(an import-blocking test would poison every later test in the session). Only the
log line is once-per-process.
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)

#: The declared seal value for a worker that has no torch.
ABSENT = "absent"

_logged = False


def torch_or_none() -> Optional[ModuleType]:
    """The live ``torch`` module, or ``None`` on a torchless worker.

    Only ``ImportError`` counts as absent. A torch that is installed but broken
    raises, loudly, as it should.
    """
    global _logged
    try:
        import torch
    except ImportError as exc:
        if not _logged:
            _logged = True
            logger.info(
                "torch is ABSENT (%s) — sealing torch=%r; the ISA clamp and the "
                "guard posture have nothing to impose on this worker (pgw#788)",
                exc, ABSENT)
        return None
    return torch


def present() -> bool:
    """Whether this worker has torch at all."""
    return torch_or_none() is not None


__all__ = ["ABSENT", "present", "torch_or_none"]
