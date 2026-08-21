"""Control/compute process split. Parent = control plane (gRPC stream, identity, JWT, job accounting) — never imports torch; child = compute plane (executor, CUDA, models, endpoint handler code). On child death the parent reports typed FATALs, keeps the pod's connection identity, and respawns. The split is UNCONDITIONAL — there is no single-process mode. Every name in this module is PLATFORM-RESERVED hub-side and injected at pod launch; a tenant can neither attach them as org envs nor bake them into a release — GEN_WORKER_CHILD_CMD is the argv the control parent execs, which tenant-settable would be control-plane RCE."""

from __future__ import annotations

import os

ENV_CHILD = "GEN_WORKER_COMPUTE_CHILD"
ENV_SOCKET = "GEN_WORKER_CHILD_SOCKET"
ENV_CHILD_CMD = "GEN_WORKER_CHILD_CMD"
ENV_WATCHDOG_PING_S = "GEN_WORKER_CHILD_WATCHDOG_PING_S"
ENV_LIVENESS_FD = "GEN_WORKER_CHILD_LIVENESS_FD"
ENV_SESSION_ID = "GEN_WORKER_SESSION_ID"

ENV_TOPOLOGY = "WORKER_EXECUTION_TOPOLOGY"
ENV_GROUP_ORDINAL = "GEN_WORKER_GROUP_ORDINAL"
ENV_HOST_SIBLINGS = "GEN_WORKER_HOST_SIBLINGS"


def host_siblings() -> int:
    """Compute children sharing this container's CPU/RAM quota (>= 1)."""
    try:
        return max(1, int(os.environ.get(ENV_HOST_SIBLINGS, "") or 1))
    except ValueError:
        return 1


def group_ordinal() -> int:
    try:
        return max(0, int(os.environ.get(ENV_GROUP_ORDINAL, "") or 0))
    except ValueError:
        return 0


def is_compute_child() -> bool:
    return bool(os.environ.get(ENV_CHILD, "").strip())


EXIT_JOB_RECYCLE = 75


__all__ = [
    "ENV_CHILD",
    "ENV_CHILD_CMD",
    "ENV_GROUP_ORDINAL",
    "ENV_HOST_SIBLINGS",
    "ENV_LIVENESS_FD",
    "ENV_SESSION_ID",
    "ENV_SOCKET",
    "ENV_TOPOLOGY",
    "ENV_WATCHDOG_PING_S",
    "EXIT_JOB_RECYCLE",
    "group_ordinal",
    "host_siblings",
    "is_compute_child",
]
