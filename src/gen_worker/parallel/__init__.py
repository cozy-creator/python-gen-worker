"""Sequence-parallel execution runtime (pgw#748 phase 1).

One execution GROUP of degree D is executed by D ranks in one NCCL world: rank
0 is the worker process itself (it owns the pipeline object, the request
payload, VAE decode, mp4 export, ctx and the output path), and D−1 spawned
siblings do nothing but the identical SPMD forward.

Wired from the delivered ``ExecutionTopology`` (``parallel="sequence"``), never
from an endpoint declaration and never from a request field: the degree is
derived from placement, and placement is the worker's.
"""

from .cp import (
    ContextParallelUnavailable,
    install_context_parallel,
    refuse_unless_divisible,
    refuse_unless_shard_invariant_quant,
)
from .group import RankGroup, RankGroupError, RankSpec, init_rank
from .plan import GroupPlan, RankDivergence, broadcast_plan

__all__ = [
    "ContextParallelUnavailable",
    "GroupPlan",
    "RankDivergence",
    "RankGroup",
    "RankGroupError",
    "RankSpec",
    "broadcast_plan",
    "init_rank",
    "install_context_parallel",
    "refuse_unless_divisible",
    "refuse_unless_shard_invariant_quant",
]
