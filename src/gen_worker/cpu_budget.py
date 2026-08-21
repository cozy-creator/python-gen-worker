"""Per-group host-CPU budget."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict

from . import hostfacts
from .procsplit import host_siblings

logger = logging.getLogger(__name__)


def cpu_allowance() -> float:
    """The narrowest true bound on this process's CPU, in fractional cores."""
    return hostfacts.cpu_allowance().cores


def per_group_threads(allowance: float, groups: int) -> int:
    """Intra-op threads one execution group may use."""
    return max(1, int(math.floor(allowance / max(1, int(groups)))))


def impose_intra_op_threads(groups: int) -> Dict[str, Any]:
    """Size torch's intra-op (and, when still settable, inter-op) pools for ONE group's share of this process."""

    siblings = host_siblings()
    effective = max(1, int(groups)) * siblings
    facts: Dict[str, Any] = {"execution_groups": int(groups), "host_siblings": siblings,
                             "concurrency": effective}
    try:
        import torch
    except Exception:  # noqa: BLE001  (torch-free contexts: tools, tests)
        facts["skipped"] = "torch unavailable"
        return facts
    allowance = cpu_allowance()
    budget = per_group_threads(allowance, effective)
    default = int(torch.get_num_threads())
    imposed = min(default, budget)
    facts.update(allowance=round(allowance, 2), budget=budget,
                 torch_default=default, imposed=imposed)
    if imposed == default:
        return facts
    try:
        torch.set_num_threads(imposed)
        facts["intra_op"] = int(torch.get_num_threads())
    except Exception as exc:  # noqa: BLE001
        facts["intra_op_error"] = str(exc)[:200]
    try:
        torch.set_num_interop_threads(imposed)
        facts["inter_op"] = int(torch.get_num_interop_threads())
    except Exception as exc:  # noqa: BLE001
        facts["inter_op_error"] = str(exc)[:120]
    logger.info(
        "pgw#782 cpu budget: %d local group(s) x %d sibling child(ren) = %d "
        "sharing a %.2f-core allowance -> intra-op threads %d (torch default "
        "was %d)",
        facts["execution_groups"], siblings, effective, allowance, imposed, default)
    return facts


__all__ = [
    "cpu_allowance",
    "impose_intra_op_threads",
    "per_group_threads",
]
