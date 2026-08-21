"""GroupPlan: rank 0 decides, every rank obeys — the whole doctrine, in one file."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BootPlan:
    """Everything a follower needs to build the SAME pipeline."""

    modules: Tuple[str, ...] = ()
    function_name: str = ""
    slot: str = ""
    path: str = ""
    cache_dir: str = ""
    degree: int = 1
    dtype: str = ""
    storage_dtype: str = ""


class RankDivergence(RuntimeError):
    """A rank could not honour the group's plan."""

    def __init__(self, rank: int, field_name: str, detail: str) -> None:
        self.rank = int(rank)
        self.field_name = field_name
        super().__init__(
            f"rank_divergence rank={rank} field={field_name}: {detail}"
        )


@dataclass(frozen=True)
class GroupPlan:
    """Every adaptive decision that must be identical across a group's ranks."""

    precision_execution_lane: str = ""
    gemm_mode: str = ""
    degraded_plan: str = ""
    compile_armed: bool = False
    compiled_graph_key: str = ""
    loras: Tuple[Tuple[str, float], ...] = ()
    sp_degree: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    def refuse_unless_cp_safe(self) -> None:
        """Refuse a plan that cannot be correct under sharding, BEFORE the group renders anything."""
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
        """A follower's locally-derived view against rank 0's delivered plan."""
        mine, theirs = asdict(self), asdict(other)
        for key in sorted(mine):
            if mine[key] != theirs[key]:
                raise RankDivergence(
                    rank, key,
                    f"rank-0 broadcast {theirs[key]!r}, this rank derived "
                    f"{mine[key]!r} — the group fails; a rank NEVER adapts "
                    "locally",
                )
