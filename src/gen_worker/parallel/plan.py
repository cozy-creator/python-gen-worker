"""GroupPlan: rank 0 decides, every rank obeys — the whole doctrine, in one file.

Every collective is a rendezvous. gen-worker is full of per-rank adaptive
decisions that read *this card's* measured free VRAM: the fit ladder, degraded
mode (ie#468), residency LRU, compile arm/disarm, ``gate_functions``. Any
control-flow difference between ranks either

- **hangs** — a different number of collectives, or
- **silently corrupts** — the same collectives over different weights. This is
  the nastier one (the LoRA attach case fails quietly, not loudly).

**Ruling (pgw#748 §5.4): rank 0 decides, broadcasts the decision, and every
rank obeys it unconditionally — including obeying a decision that is wrong for
its own card. A rank that cannot honour the broadcast fails the WHOLE GROUP
loudly; it never adapts locally.**

That last clause is why this module exists as a type rather than as a comment:
the only place a follower is allowed to disagree is by raising
:class:`RankDivergence`, which is fatal for the group.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


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
    precision_lane: str = ""
    # The w8a8 GEMM mode. Under context parallelism this MUST be rowwise:
    # a per-tensor ACTIVATION scale is computed from the local shard, so two
    # ranks holding different sequence shards derive different scales and
    # quantize the same logical tensor differently. Rowwise scales are
    # per-token, hence shard-invariant.
    gemm_mode: str = ""
    # The ie#468 degraded-mode plan, or "" for the undegraded ladder rung.
    degraded_plan: str = ""
    # Compile arm/disarm must be collective: a group where rank 0 armed and
    # rank 1 fell back to eager runs two different graphs.
    compile_armed: bool = False
    compile_cell_key: str = ""
    # The exact resident LoRA set, ordered. Attach/detach is a COLLECTIVE
    # decision (SEQPAR-DESIGN §5.3): sharded attention operates on
    # activations, LoRA on weights, so a rank that attaches while another
    # does not produces divergent weights and silently wrong output.
    loras: Tuple[Tuple[str, float], ...] = ()
    # The sharding degree this plan was decided for.
    sp_degree: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    def encode(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True).encode()

    @classmethod
    def decode(cls, raw: bytes) -> "GroupPlan":
        obj = json.loads(raw.decode())
        obj["loras"] = tuple(tuple(x) for x in obj.get("loras") or ())
        return cls(**obj)

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
        """A follower's local view against the broadcast plan."""
        mine, theirs = asdict(self), asdict(other)
        for key in sorted(mine):
            if mine[key] != theirs[key]:
                raise RankDivergence(
                    rank, key,
                    f"rank-0 broadcast {theirs[key]!r}, this rank derived "
                    f"{mine[key]!r} — the group fails; a rank NEVER adapts "
                    "locally",
                )


def broadcast_plan(
    plan: Optional[GroupPlan],
    *,
    rank: int,
    group: Any = None,
) -> GroupPlan:
    """Rank 0's plan, delivered to every rank.

    Followers pass ``None``; they receive rank 0's. Implemented over a byte
    tensor rather than ``broadcast_object_list`` so the payload is explicit
    and the same code path works on gloo (the CPU test rig) and NCCL.
    """
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        # Degree 1: the plan is whatever rank 0 decided, unbroadcast.
        if plan is None:
            raise RankDivergence(rank, "-", "no plan and no process group")
        plan.refuse_unless_cp_safe()
        return plan

    device = None
    if dist.get_backend(group) == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())

    if rank == 0:
        assert plan is not None, "rank 0 must decide the plan"
        plan.refuse_unless_cp_safe()
        payload = plan.encode()
        size = torch.tensor([len(payload)], dtype=torch.int64, device=device)
    else:
        size = torch.tensor([0], dtype=torch.int64, device=device)
    dist.broadcast(size, src=0, group=group)

    n = int(size.item())
    if rank == 0:
        buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
        if device is not None:
            buf = buf.to(device)
    else:
        buf = torch.empty(n, dtype=torch.uint8, device=device)
    dist.broadcast(buf, src=0, group=group)

    decided = GroupPlan.decode(bytes(buf.cpu().numpy().tobytes()))
    if rank != 0 and plan is not None:
        # A follower that formed its own opinion says so, loudly, once — the
        # disagreement is evidence about the fleet (a mismatched card, a
        # different toolchain), not a reason to serve two graphs.
        try:
            plan.assert_agrees(decided, rank=rank)
        except RankDivergence as exc:
            logger.error("%s", exc)
            raise
    return decided
