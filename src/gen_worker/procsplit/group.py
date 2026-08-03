"""pgw#783: the execution GROUP as an OS process — the child plan.

pgw#782 measured it: four execution groups in ONE interpreter serve 0.94x of
serial (21% on every card); four PROCESSES with one group each serve **4.00x**
at 91-93% util, same pod, same weights. So the child of pgw#763's split is not
one process — it is one process PER EXECUTION GROUP.

This module is the pure half of that: given the delivered topology, what
children exist and what env does each one get. No spawning, no I/O, no torch.

**The design decision that makes N children cheap: every child is a
SINGLE-GROUP worker over its own cards.** Child ``g`` is handed
``CUDA_VISIBLE_DEVICES`` naming only its group's physical devices and a
topology rewritten to ``D x D`` (locally ``G == 1``), so inside the child:

* ``current_device_group()`` is 0 and ``group_cuda_device()`` is plain
  ``"cuda"`` — the placement path every single-group pod has always used;
* ``Executor._gpu_slots`` is 1, so pgw#779's "the slot semaphore is a count,
  not a per-group permit" dissolves: the count IS the permit when the process
  is the group;
* pgw#748's multi-group hazards (thread device pinning, the group ordinal in
  ``instance_key``, the refusal to serve async handlers on a wide worker) have
  nothing left to be about;
* pgw#777's process-global compile plane stops having G concurrent users, and
  pgw#784's mint delegate works per child by construction.

**At G == 1 the env delta is EMPTY** — no ``CUDA_VISIBLE_DEVICES``, no topology
rewrite, no per-group anything — so a one-child worker is byte-identical to
what pgw#763 stage 1 shipped and measured. That is asserted, not asserted-ish:
see ``tests/test_group_processes_pgw783.py``.

A ``D > 1`` group stays ONE process. Its devices are one logical accelerator —
the model's own arrangement (``parallel="internal"``) or a platform collective
(``sequence``/``cfg``) — and both need a single Python driver holding the ranks.
Splitting one would need a real distributed launcher, and nothing measured says
a D>1 group is GIL-bound the way G independent D=1 groups are: inside it there
is ONE launch loop feeding a collective, not D competing ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

from ..topology import ExecutionTopology
from . import ENV_GROUP_ORDINAL, ENV_HOST_SIBLINGS, ENV_TOPOLOGY

__all__ = ["ChildGroup", "GroupPlan"]


@dataclass(frozen=True)
class ChildGroup:
    """One execution group, and the OS process that will own it."""

    ordinal: int
    devices: Tuple[int, ...]
    socket_path: str
    env: Mapping[str, str] = field(default_factory=dict)

    @property
    def degree(self) -> int:
        return len(self.devices)

    @property
    def label(self) -> str:
        return f"g{self.ordinal}[{','.join(str(d) for d in self.devices)}]"


@dataclass(frozen=True)
class GroupPlan:
    """The full set of children a topology asks for."""

    topology: ExecutionTopology
    children: Tuple[ChildGroup, ...]

    @property
    def execution_groups(self) -> int:
        return len(self.children)

    def child(self, ordinal: int) -> ChildGroup:
        return self.children[int(ordinal)]

    # ---- routing ---------------------------------------------------------

    def route(self, gpu_index: Optional[int]) -> int:
        """Which child serves a dispatch naming ``gpu_index``.

        ``ResolvedCompute.gpu_index`` names the group's RANK-0 PHYSICAL device
        (0, D, 2D, ...), so this is exactly ``Executor._dispatch_group``'s
        derivation moved one process earlier — the parent ROUTES, it never
        schedules; the hub already picked the slot.

        Single-group pods keep the historical behaviour exactly (there is only
        one group to mean). At G>1 a non-rank-0 or missing index is the same
        typed refusal the executor now raises (pgw#779, security-deltas base):
        flooring a hub/worker packing disagreement onto group 0 is the silent
        bug that piles every mis-dispatch onto the busiest card.
        """
        if self.topology.execution_groups <= 1:
            return 0
        if gpu_index is None:
            raise ValueError(
                f"dispatch carries no resolved compute on a {self.topology} "
                "pod: the execution group cannot be derived"
            )
        return self.topology.group_ordinal_exact(int(gpu_index))

    def local_gpu_index(self, ordinal: int) -> int:
        """The ``gpu_index`` the CHILD must see.

        Under ``CUDA_VISIBLE_DEVICES`` the child's world starts at 0, so a
        physical rank-0 of ``g*D`` becomes local 0. One field, rewritten as the
        RunJob is routed; at G == 1 it is the identity (0 -> 0).
        """
        del ordinal
        return 0

    # ---- construction ----------------------------------------------------

    @classmethod
    def for_topology(
        cls,
        topology: ExecutionTopology,
        *,
        socket_path: str,
    ) -> "GroupPlan":
        groups = int(topology.execution_groups)
        children = []
        for ordinal in range(groups):
            devices = tuple(topology.group(ordinal).devices)
            children.append(ChildGroup(
                ordinal=ordinal,
                devices=devices,
                socket_path=_socket_for(socket_path, ordinal, groups),
                env=_child_env(topology, ordinal, devices),
            ))
        return cls(topology=topology, children=tuple(children))


def _socket_for(base: str, ordinal: int, groups: int) -> str:
    """Stage 1's exact path at G == 1; one socket per child beyond that."""
    if groups <= 1:
        return base
    if base.endswith(".sock"):
        return f"{base[:-5]}-g{ordinal}.sock"
    return f"{base}-g{ordinal}"


def _child_env(
    topology: ExecutionTopology, ordinal: int, devices: Tuple[int, ...],
) -> Dict[str, str]:
    """The env DELTA applied on top of the parent's own environment.

    Empty at G == 1 — deliberately, and asserted by the identity test. The
    per-group compile/inductor cache dirs and the host-RAM divisor named in the
    design note are 783-C's wiring (they need consumers in ``cpu_budget`` /
    ``probe_host_ram``); what this stage fixes is the CONTRACT, so the child
    already carries the sibling count it will be budgeted against.
    """
    if topology.execution_groups <= 1:
        return {}
    local = ExecutionTopology(
        gpu_count=len(devices),
        gpus_per_execution_group=topology.gpus_per_execution_group,
        parallel=topology.parallel,
    )
    return {
        # The child's local cuda:0..D-1 ARE its group. It cannot see, let alone
        # allocate on, a sibling's card — the strongest VRAM isolation
        # available, and it costs nothing.
        "CUDA_VISIBLE_DEVICES": ",".join(str(d) for d in devices),
        # Locally G == 1: the child is the most-tested shape in the worker.
        ENV_TOPOLOGY: json.dumps(local.as_dict(), separators=(",", ":")),
        # Labelling only (logs, dials, per-group dirs). No behaviour reads it.
        ENV_GROUP_ORDINAL: str(ordinal),
        # The ONE thing a child must know about its siblings: how many
        # processes share this pod's cgroup. Without it pgw#782's cpu_budget
        # divisor (the delivered group count, now rewritten to 1) would let
        # every child take the whole CPU quota and reinstate exactly the
        # 192-threads-on-32-cores misconfiguration pgw#782 removed; and
        # pgw#752's host-RAM probe would tell each child the whole pod's RAM
        # is its own, making pgw#763's move guard G-times too permissive on
        # precisely the pods most likely to OOM.
        ENV_HOST_SIBLINGS: str(int(topology.execution_groups)),
    }
