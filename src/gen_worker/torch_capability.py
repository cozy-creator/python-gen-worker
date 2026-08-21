"""Torch is a CAPABILITY, not a precondition for booting."""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)

ABSENT = "absent"

_logged = False


def torch_or_none() -> Optional[ModuleType]:
    """The live ``torch`` module, or ``None`` on a torchless worker."""
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
