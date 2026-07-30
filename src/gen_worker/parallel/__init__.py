"""Sequence-parallel execution runtime (pgw#748 phase 1).

One execution GROUP of degree D is executed by D ranks: rank 0 is the worker
process itself (it owns the pipeline object, the request payload, VAE decode,
mp4 export, ctx and the output path), and D−1 spawned siblings do nothing but
the identical SPMD forward.

Every group owns a NON-default torch.distributed process group (pgw#773) —
the worker process is rank 0 of every group, so groups can never share the
default world — and rank-0 commands ride mp queues, never collectives
(pgw#774), so an idle or wedged group can always be torn down without one.

Wired from the delivered ``ExecutionTopology`` (``parallel="sequence"``), never
from an endpoint declaration and never from a request field: the degree is
derived from placement, and placement is the worker's.
"""

from .cp import (
    ContextParallelUnavailable,
    CpComms,
    UngatedShardedForward,
    gated_call,
    in_gated_call,
    install_context_parallel,
    refuse_unless_divisible,
    refuse_unless_shard_invariant_quant,
)
from .group import (
    FollowerChannel,
    RankGroup,
    RankGroupError,
    RankSpec,
    init_rank,
)
from .plan import GroupPlan, RankDivergence

__all__ = [
    "ContextParallelUnavailable",
    "CpComms",
    "FollowerChannel",
    "GroupPlan",
    "RankDivergence",
    "RankGroup",
    "RankGroupError",
    "RankSpec",
    "UngatedShardedForward",
    "gated_call",
    "in_gated_call",
    "init_rank",
    "install_context_parallel",
    "refuse_unless_divisible",
    "refuse_unless_shard_invariant_quant",
]
