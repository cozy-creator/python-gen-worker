from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from .lane_ladder import (
    VERDICT_ABSENT,
    VERDICT_DERIVABLE,
    VERDICT_INCOMPATIBLE,
    VERDICT_SATISFIES,
    VERDICTS,
    CardFacts,
)

logger = logging.getLogger(__name__)


def host_card_facts() -> CardFacts:
    """The card this process is on, as a value."""
    try:
        import torch
    except ImportError:
        return CardFacts(sm=0, name="no-torch")
    if not torch.cuda.is_available():
        return CardFacts(sm=0, name="no-cuda")
    try:
        major, minor = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        free_total = torch.cuda.mem_get_info()
        vram_gb = float(free_total[1]) / (1024.0 ** 3)
    except Exception:  # noqa: BLE001 — a card census must not fail a boot
        logger.warning("lane ladder: the card census raised; treating this "
                       "host as sm0", exc_info=True)
        return CardFacts(sm=0, name="census-failed")
    return CardFacts(sm=int(major) * 10 + int(minor), vram_gb=vram_gb,
                     name=str(name))


class HostKernelGates:
    """The host's own veto, read from the modules that already own it."""

    def w8a8_mode(self) -> str:
        try:
            from ..models.w8a8 import w8a8_gemm_mode
        except ImportError:
            return ""
        try:
            return str(w8a8_gemm_mode() or "")
        except Exception:  # noqa: BLE001 — an unqualified gate is a rejection
            logger.warning("lane ladder: w8a8_gemm_mode() raised; the fp8 rung "
                           "is unqualified on this host", exc_info=True)
            return ""

    def w4a4_mode(self) -> str:
        try:
            from ..models.w4a4 import w4a4_gemm_mode
        except ImportError:
            return ""
        try:
            return str(w4a4_gemm_mode() or "")
        except Exception:  # noqa: BLE001
            logger.warning("lane ladder: w4a4_gemm_mode() raised; the nvfp4 "
                           "rung is unqualified on this host", exc_info=True)
            return ""


class BindingVerdicts:
    """What the DEPLOY staged, per lane contract."""

    def __init__(
        self,
        *,
        trees: Mapping[str, Any] | None = None,
        verdicts: Mapping[str, str] | None = None,
        sizes: Mapping[str, int] | None = None,
    ) -> None:
        self._trees = dict(trees or {})
        self._verdicts = {
            str(k): str(v) for k, v in (verdicts or {}).items()
            if str(v) in VERDICTS
        }
        self._sizes = {str(k): int(v) for k, v in (sizes or {}).items()}

    @classmethod
    def of(cls, binding: Any) -> "BindingVerdicts":
        """Read the lane facts off a `DeployBinding`."""
        return cls(
            trees=getattr(binding, "lane_trees", None),
            verdicts=getattr(binding, "lane_verdicts", None),
            sizes=getattr(binding, "lane_bytes", None),
        )

    def for_single_lane(self, contract_id: str, tree: Any) -> "BindingVerdicts":
        """The one-lane deployment: the staged tree satisfies the one contract."""
        if not contract_id or contract_id in self._trees:
            return self
        merged = dict(self._trees)
        merged[str(contract_id)] = tree
        return BindingVerdicts(trees=merged, verdicts=self._verdicts,
                               sizes=self._sizes)

    def tree_for(self, contract_id: str) -> Optional[Any]:
        return self._trees.get(str(contract_id))

    def verdict(self, contract_id: str) -> str:
        key = str(contract_id or "")
        if not key:
            return VERDICT_ABSENT
        stated = self._verdicts.get(key)
        if stated == VERDICT_INCOMPATIBLE:
            return VERDICT_INCOMPATIBLE
        if key in self._trees:
            return VERDICT_SATISFIES
        if stated == VERDICT_DERIVABLE:
            return VERDICT_DERIVABLE
        if stated == VERDICT_SATISFIES:
            return VERDICT_DERIVABLE
        return VERDICT_ABSENT

    def transfer_bytes(self, contract_id: str) -> int:
        return int(self._sizes.get(str(contract_id), 0))


__all__ = ["BindingVerdicts", "HostKernelGates", "host_card_facts"]
