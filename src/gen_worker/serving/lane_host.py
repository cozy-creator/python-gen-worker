"""Production wiring for the lane ladder's ports (pgw#1606).

`lane_ladder` is deliberately pure — every collaborator is a port, so the whole
selection walk is provable on a CPU box against fabricated cards. This module
is where the real card, the real kernel gates and the real staged bytes are
plugged in, and it is the ONLY place that touches torch or the deploy binding.
"""

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
    """The card this process is on, as a value.

    `sm` is `major * 10 + minor`, the integer form the rest of the fleet
    already uses (`w8a8.py:355-357`, `hostfacts.py:283`, `placement.py:126`).
    A host with no CUDA answers `sm=0`, which floors every quantized rung out
    by the ordinary `min_sm` comparison rather than by a special case.
    """
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
    """The host's own veto, read from the modules that already own it.

    Both reads are `lru_cache(1)`'d in their own modules, so the 4096-cubed
    profitability benchmark behind them is paid ONCE per process no matter how
    many models this endpoint declares. That is the whole reason the ladder
    consumes this rather than growing a second opinion — the second opinion
    would cost a second benchmark and could disagree with the loader that
    actually runs.
    """

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
    """What the DEPLOY staged, per lane contract.

    **Why this is a read and not a computation.** The tri-state verdict
    (`Satisfies | DerivableVia | Incompatible`, tensorfs#123) is Go-only today
    — `tensorfs/verdict.go:177`, with no pyo3 binding — and its one caller is
    the hub's bind gate (`th internal/bindgate/bindgate.go:317`), which runs it
    at BIND time, before this pod existed. So the authoritative answer already
    exists and travels in the binding. Recomputing it here would be a fifth
    copy of the pattern matcher (tensorfs#129 already counts three) and could
    disagree with the gate that admitted the deployment.

    This is not the hub round-trip the coordinator ruled out: nothing here
    calls the hub. It reads a fact the binding already carries.

    **SETTLED 2026-08-20, and the earlier plan here was WRONG.** This docstring
    used to promise that a pyo3 `Verdict` binding would land and a lane the
    binding said nothing about would then be "computed locally from the two
    documents". The pgw#1599 lane declined to vendor `Contract.verdict` into
    pgw, and that decision is correct: the verdict is an ADMIT decision, not a
    lookup. Computing it here would make the worker a second author of
    admissibility, free to admit a binding the gate would have refused — the
    failure that is invisible precisely because both sides believe they agree.

    So there is no local-computation arm and there will not be one. A contract
    the binding says nothing about stays `absent`, which the ladder degrades
    into a priced conversion ask. That is the safe direction: pgw can ask for
    bytes it was not given, and cannot admit bytes it was not told about.
    """

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
        """Read the lane facts off a `DeployBinding`.

        A single-lane binding (every deployment in the fleet today) carries no
        lane map at all; its one staged tree IS the answer for its one declared
        contract, and `for_single_lane` below supplies that. This constructor
        is the multi-lane path and degrades to "nothing staged" rather than
        guessing, because guessing here serves the wrong numerics silently.
        """
        return cls(
            trees=getattr(binding, "lane_trees", None),
            verdicts=getattr(binding, "lane_verdicts", None),
            sizes=getattr(binding, "lane_bytes", None),
        )

    def for_single_lane(self, contract_id: str, tree: Any) -> "BindingVerdicts":
        """The one-lane deployment: the staged tree satisfies the one contract.

        Kept explicit rather than folded into `of()` so that a MULTI-lane
        binding that forgot its map cannot silently inherit "everything is
        staged" from the single-lane convenience.
        """
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
            # The gate says the bytes fit and yet nothing is staged for this
            # pod. That is a staging hole, not a satisfied lane — report it as
            # derivable so the ladder asks for the tree instead of trying to
            # load one that is not there.
            return VERDICT_DERIVABLE
        return VERDICT_ABSENT

    def transfer_bytes(self, contract_id: str) -> int:
        return int(self._sizes.get(str(contract_id), 0))


__all__ = ["BindingVerdicts", "HostKernelGates", "host_card_facts"]
