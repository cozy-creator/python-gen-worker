"""Control/compute process split.

Parent = control plane (gRPC stream, identity, JWT, job accounting) — never
imports torch. Child = compute plane (executor, CUDA context, models, endpoint
handler code). On child death the parent reports typed FATALs for in-flight
jobs, keeps the pod's connection identity alive, and respawns the child.

The split is unconditional: every worker process runs as a control parent that
execs one compute child per execution group. There is no single-process mode.

Delta 0: every name in this module is PLATFORM-RESERVED hub-side (tensorhub
``internal/api/endpoint_env_reserved.go`` + the orchestrator twin) and injected
at pod-launch. A tenant can neither attach them as org envs nor bake them into a
release: the boundary exists to contain tenant code, so it must not be that
code's option — and ``GEN_WORKER_CHILD_CMD`` is the argv the control parent
execs, which tenant-settable would make control-plane RCE.
"""

from __future__ import annotations

import os

ENV_CHILD = "GEN_WORKER_COMPUTE_CHILD"
ENV_SOCKET = "GEN_WORKER_CHILD_SOCKET"
ENV_CHILD_CMD = "GEN_WORKER_CHILD_CMD"
ENV_WATCHDOG_PING_S = "GEN_WORKER_CHILD_WATCHDOG_PING_S"
# The LOOP-INDEPENDENT liveness channel. The frame ping above rides the child's
# event loop, which a long inductor compile starves — so on its own it turns a
# live compile into a SIGKILL labelled watchdog_hang. A thread writes this fd
# instead, carrying the same kernel-accounted evidence activity.watchdog trusts.
ENV_LIVENESS_FD = "GEN_WORKER_CHILD_LIVENESS_FD"
# The PARENT mints the worker session id once and passes it to every child.
# Child-minted it changes on every respawn, and the hub rejects cross-session
# shadow state, so a respawned child's shadow state is silently invalidated. The
# parent owns the session, so it survives child respawns. At G>1 all children
# share it (the hub sees one worker, one session).
ENV_SESSION_ID = "GEN_WORKER_SESSION_ID"

# One child per EXECUTION GROUP. The hub's delivered packing is the
# only source of G — there is no new knob to mis-set.
ENV_TOPOLOGY = "WORKER_EXECUTION_TOPOLOGY"
# Which group this child owns. It labels logs/dials and selects one-writer
# per-group filesystem carriers (inductor cache and post-mortem markers); it
# never changes serving policy, because the child is rewritten into a
# SINGLE-group worker over its own cards (see procsplit.group).
ENV_GROUP_ORDINAL = "GEN_WORKER_GROUP_ORDINAL"
# How many compute children share this pod's cgroup. The one fact a child
# needs about its siblings: the CPU and host-RAM budgets
# are pod-wide quotas that must be divided by it, and the child's own rewritten
# topology says G == 1.
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
    "group_ordinal",
    "host_siblings",
    "is_compute_child",
]
