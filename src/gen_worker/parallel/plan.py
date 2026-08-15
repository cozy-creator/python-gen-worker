"""GroupPlan: rank 0 decides, every rank obeys — the whole doctrine, in one file.

Every collective is a rendezvous. gen-worker is full of per-rank adaptive
decisions that read *this card's* measured free VRAM: the fit ladder, degraded
mode, residency LRU, compile arm/disarm, ``gate_functions``. Any
control-flow difference between ranks either

- **hangs** — a different number of collectives, or
- **silently corrupts** — the same collectives over different weights. This is
  the nastier one (the LoRA attach case fails quietly, not loudly).

**Ruling: rank 0 decides, broadcasts the decision, and every
rank obeys it unconditionally — including obeying a decision that is wrong for
its own card. A rank that cannot honour the broadcast fails the WHOLE GROUP
loudly; it never adapts locally.**

That last clause is why this module exists as a type rather than as a comment:
the only place a follower is allowed to disagree is by raising
:class:`RankDivergence`, which is fatal for the group.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BootPlan:
    """Everything a follower needs to build the SAME pipeline.

    Lives beside :class:`GroupPlan` because they are the same act — rank 0
    decides, every rank obeys — and they ride the arm command together.
    """

    modules: Tuple[str, ...] = ()
    function_name: str = ""
    slot: str = ""
    # slot -> the pod-shared CAS path. One copy of the bytes, N mappings.
    path: str = ""
    cache_dir: str = ""
    degree: int = 1
    dtype: str = ""
    storage_dtype: str = ""


class RankDivergence(RuntimeError):
    """A rank could not honour the group's plan.

    NEVER caught and turned into a local fallback — that is precisely the
    silent-corruption path. It fails the group.
    """

    def __init__(self, rank: int, field_name: str, detail: str) -> None:
        self.rank = int(rank)
        self.field_name = field_name
        super().__init__(
            f"rank_divergence rank={rank} field={field_name}: {detail}"
        )


@dataclass(frozen=True)
class GroupPlan:
    """Every adaptive decision that must be identical across a group's ranks.

    Across GROUPS these may legitimately differ — group 0 may be degraded
    while group 1 is not, because they are independent placements on
    independent cards (EXECUTION-TOPOLOGY-DESIGN §5.3). Within a group at
    D > 1 they must be identical or the collective hangs.
    """

    # Which precision lane the group serves (bf16 / w8a8 / fp8-storage / ...).
    precision_execution_lane: str = ""
    # The w8a8 GEMM mode. Under context parallelism this MUST be rowwise:
    # a per-tensor ACTIVATION scale is computed from the local shard, so two
    # ranks holding different sequence shards derive different scales and
    # quantize the same logical tensor differently. Rowwise scales are
    # per-token, hence shard-invariant.
    gemm_mode: str = ""
    # The degraded-mode plan, or "" for the undegraded ladder rung.
    degraded_plan: str = ""
    # Compile arm/disarm must be collective: a group where rank 0 armed and
    # rank 1 fell back to eager runs two different graphs.
    compile_armed: bool = False
    compiled_graph_key: str = ""
    # The exact resident LoRA set, ordered. Attach/detach is a COLLECTIVE
    # decision (SEQPAR-DESIGN §5.3): sharded attention operates on
    # activations, LoRA on weights, so a rank that attaches while another
    # does not produces divergent weights and silently wrong output.
    loras: Tuple[Tuple[str, float], ...] = ()
    # The sharding degree this plan was decided for.
    sp_degree: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    def refuse_unless_cp_safe(self) -> None:
        """Refuse a plan that cannot be correct under sharding, BEFORE the
        group renders anything. Typed, never a downgrade."""
        if self.sp_degree <= 1:
            return
        if self.gemm_mode and self.gemm_mode != "rowwise":
            raise RankDivergence(
                0, "gemm_mode",
                f"w8a8 gemm_mode={self.gemm_mode!r} is not shard-invariant: a "
                "per-tensor activation scale is derived from the LOCAL "
                "sequence shard, so each rank quantizes the same logical "
                "tensor differently and the group produces silently wrong "
                "output. Sequence parallelism requires rowwise "
                "(per-token) activation scales.",
            )

    def assert_agrees(self, other: "GroupPlan", *, rank: int) -> None:
        """A follower's locally-derived view against rank 0's delivered plan.

        Every follower derives its own group facts from its own materialized
        pipeline and holds them against the plan the arm command carried — a
        disagreement (mismatched card, different toolchain) fails the group
        loudly. There is deliberately no ``broadcast_plan`` collective: the plan
        rides the command channel with the BootPlan, so plan delivery can
        neither hang nor desynchronize a process group.
        """
        mine, theirs = asdict(self), asdict(other)
        for key in sorted(mine):
            if mine[key] != theirs[key]:
                raise RankDivergence(
                    rank, key,
                    f"rank-0 broadcast {theirs[key]!r}, this rank derived "
                    f"{mine[key]!r} — the group fails; a rank NEVER adapts "
                    "locally",
                )
