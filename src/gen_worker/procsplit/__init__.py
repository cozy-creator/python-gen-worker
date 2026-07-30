"""pgw#763 layer 1: control/compute process split.

Parent = control plane (gRPC stream, identity, JWT, job accounting) — never
imports torch. Child = compute plane (executor, CUDA context, models, endpoint
handler code). On child death the parent reports typed FATALs for in-flight
jobs, keeps the pod's connection identity alive, and respawns the child.

Off by default: ``GEN_WORKER_PROCESS_SPLIT=1`` enables the split.
"""

from __future__ import annotations

import os

ENV_SPLIT = "GEN_WORKER_PROCESS_SPLIT"
ENV_CHILD = "GEN_WORKER_COMPUTE_CHILD"
ENV_SOCKET = "GEN_WORKER_CHILD_SOCKET"
ENV_CHILD_CMD = "GEN_WORKER_CHILD_CMD"
ENV_WATCHDOG_PING_S = "GEN_WORKER_CHILD_WATCHDOG_PING_S"
# pgw#771: the LOOP-INDEPENDENT liveness channel. The frame ping above rides
# the child's event loop, which a long inductor compile starves — so on its own
# it turns a live compile into a SIGKILL labelled watchdog_hang (th#1299 one
# layer down). A thread writes this fd instead, carrying the same kernel-
# accounted evidence activity.watchdog already trusts.
ENV_LIVENESS_FD = "GEN_WORKER_CHILD_LIVENESS_FD"


def split_enabled() -> bool:
    return os.environ.get(ENV_SPLIT, "").strip().lower() in ("1", "true", "yes")


def is_compute_child() -> bool:
    return bool(os.environ.get(ENV_CHILD, "").strip())


__all__ = [
    "ENV_CHILD",
    "ENV_CHILD_CMD",
    "ENV_LIVENESS_FD",
    "ENV_SOCKET",
    "ENV_SPLIT",
    "ENV_WATCHDOG_PING_S",
    "is_compute_child",
    "split_enabled",
]
