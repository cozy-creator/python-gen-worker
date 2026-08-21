"""The execution GROUP as an OS process — the child plan."""

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

    def route(self, gpu_index: Optional[int]) -> int:
        """Which child serves a dispatch naming ``gpu_index``."""
        if self.topology.execution_groups <= 1:
            return 0
        if gpu_index is None:
            raise ValueError(
                f"dispatch carries no resolved compute on a {self.topology} "
                "pod: the execution group cannot be derived"
            )
        return self.topology.group_ordinal_exact(int(gpu_index))

    def local_gpu_index(self, ordinal: int) -> int:
        """The ``gpu_index`` the CHILD must see."""
        del ordinal
        return 0

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
    if groups <= 1:
        return base
    if base.endswith(".sock"):
        return f"{base[:-5]}-g{ordinal}.sock"
    return f"{base}-g{ordinal}"


def _child_env(
    topology: ExecutionTopology, ordinal: int, devices: Tuple[int, ...],
) -> Dict[str, str]:
    if topology.execution_groups <= 1:
        return {}
    local = ExecutionTopology(
        gpu_count=len(devices),
        gpus_per_execution_group=topology.gpus_per_execution_group,
        parallel=topology.parallel,
    )
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(str(d) for d in devices),
        ENV_TOPOLOGY: json.dumps(local.as_dict(), separators=(",", ":")),
        ENV_GROUP_ORDINAL: str(ordinal),
        ENV_HOST_SIBLINGS: str(int(topology.execution_groups)),
    }
