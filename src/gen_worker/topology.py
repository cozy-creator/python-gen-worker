"""Execution topology: the hub's staffing decision, delivered as ONE env var — WORKER_EXECUTION_TOPOLOGY={"gpu_count","gpus_per_execution_group","execution_groups","parallel"} (tensorhub internal/orchestrator/topology). WORKER_ is a reserved endpoint-env prefix hub-side, so this is a trusted wire fact, never an operator knob. Derivation, identical on both sides: D = gpus_per_execution_group; G = gpu_count / D; group g owns devices [g*D, (g+1)*D); ResolvedCompute.gpu_index names the group's RANK-0 device. An absent ENV VAR is legal (a CPU pod keeps the single slot); an absent field inside a present object — and any unknown key — is a TYPED refusal: the field set is CLOSED, and ignoring what you do not understand is how a hub that believes it bought degree 2 gets served degree 1 in silence. An "execution group" (the serving unit) is NOT a torch ProcessGroup (the communication handle); each execution group owns exactly one non-default ProcessGroup."""

from __future__ import annotations

import contextvars
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from .models.residency import REPLICATED, SHARDED, DeviceGroup
from .host_canary import is_fabric_wedge, sp_admits
from .hostfacts import cuda_ready

logger = logging.getLogger(__name__)

_current_group: contextvars.ContextVar[int] = contextvars.ContextVar(
    "gen_worker_device_group", default=0
)


def current_device_group() -> int:
    return _current_group.get()


_installed: Optional["ExecutionTopology"] = None


def install_topology(topo: Optional["ExecutionTopology"]) -> None:
    """Publish the delivered packing for the ordinal -> device translation."""
    global _installed
    _installed = topo


def group_devices(ordinal: Optional[int] = None) -> Tuple[int, ...]:
    """The CUDA device indices a group owns, from the DELIVERED topology."""
    g = current_device_group() if ordinal is None else int(ordinal)
    topo = _installed
    if topo is None or g < 0 or g >= topo.execution_groups:
        return (max(0, g),)
    return topo.group(g).devices


def group_rank0_device(ordinal: Optional[int] = None) -> int:
    """The group's rank-0 card — the device its weights and its rank-0 work belong on, and the index the hub dispatches as ``gpu_index``."""
    return group_devices(ordinal)[0]


def pin_cuda_device_for_group() -> None:
    """Set THIS THREAD's current CUDA device to the current group's rank-0 card."""
    device = group_rank0_device()
    if device <= 0:
        return
    try:
        import torch

        if cuda_ready() and device < torch.cuda.device_count():
            torch.cuda.set_device(device)
    except Exception:  # noqa: BLE001 - placement is best-effort here; the
        logger.warning("could not pin thread to device %d", device, exc_info=True)


class device_group_scope:
    """Serve this block as ``ordinal``'s group."""

    def __init__(self, ordinal: int) -> None:
        self._ordinal = int(ordinal)
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> int:
        self._token = _current_group.set(self._ordinal)
        return self._ordinal

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _current_group.reset(self._token)
            self._token = None

ENV_VAR = "WORKER_EXECUTION_TOPOLOGY"

KEY_GPU_COUNT = "gpu_count"
KEY_GPUS_PER_GROUP = "gpus_per_execution_group"
KEY_EXECUTION_GROUPS = "execution_groups"
KEY_PARALLEL = "parallel"

_KNOWN_KEYS = frozenset({
    KEY_GPU_COUNT, KEY_GPUS_PER_GROUP, KEY_EXECUTION_GROUPS, KEY_PARALLEL,
})

MAX_GPU_COUNT = 1024

_INT64_MIN = -(2 ** 63)
_INT64_MAX = 2 ** 63 - 1

PARALLEL_NONE = ""
PARALLEL_INTERNAL = "internal"
PARALLEL_SEQUENCE = "sequence"
PARALLEL_CFG = "cfg"

_PARALLEL_VALUES = (PARALLEL_NONE, PARALLEL_INTERNAL, PARALLEL_SEQUENCE, PARALLEL_CFG)

_PLATFORM_PARALLEL = (PARALLEL_SEQUENCE, PARALLEL_CFG)


class TopologyError(ValueError):
    """A present-but-illegal topology."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class ExecutionTopology:
    """One pod's delivered packing."""

    gpu_count: int = 1
    gpus_per_execution_group: int = 1
    parallel: str = PARALLEL_NONE

    def __post_init__(self) -> None:
        if self.gpu_count < 1:
            raise TopologyError(
                "topology_gpu_count_invalid", f"gpu_count={self.gpu_count}"
            )
        if self.gpu_count > MAX_GPU_COUNT:
            raise TopologyError(
                "topology_gpu_count_invalid",
                f"gpu_count={self.gpu_count} exceeds the maximum {MAX_GPU_COUNT}",
            )
        if self.gpus_per_execution_group < 1:
            raise TopologyError(
                "topology_degree_invalid", f"gpus_per_execution_group={self.gpus_per_execution_group}"
            )
        if self.gpus_per_execution_group > self.gpu_count:
            raise TopologyError(
                "topology_degree_invalid",
                f"gpus_per_execution_group={self.gpus_per_execution_group} exceeds "
                f"gpu_count={self.gpu_count}",
            )
        if self.gpu_count % self.gpus_per_execution_group != 0:
            raise TopologyError(
                "topology_degree_not_divisor",
                f"gpus_per_execution_group={self.gpus_per_execution_group} does not divide "
                f"gpu_count={self.gpu_count}",
            )
        if self.parallel not in _PARALLEL_VALUES:
            raise TopologyError(
                "topology_parallel_unknown", f"parallel={self.parallel!r}"
            )
        if self.gpus_per_execution_group > 1 and self.parallel == PARALLEL_NONE:
            raise TopologyError(
                "topology_parallel_required",
                f"gpus_per_execution_group={self.gpus_per_execution_group} needs a parallel mechanism",
            )
        if self.gpus_per_execution_group == 1 and self.parallel != PARALLEL_NONE:
            raise TopologyError(
                "topology_parallel_without_degree",
                f"parallel={self.parallel!r} at gpus_per_execution_group=1",
            )

    @property
    def execution_groups(self) -> int:
        """G — execution groups, which IS the worker's slot count."""
        return self.gpu_count // self.gpus_per_execution_group

    @property
    def slots(self) -> int:
        return self.execution_groups

    @property
    def degree(self) -> int:
        return self.gpus_per_execution_group

    @property
    def platform_parallel(self) -> bool:
        """True when the PLATFORM installs the sharding (and therefore when a fabric miss may demote this pod)."""
        return self.parallel in _PLATFORM_PARALLEL

    @property
    def placement_mode(self) -> str:
        """How a materialization occupies a group."""
        return SHARDED if self.parallel == PARALLEL_INTERNAL else REPLICATED

    def group_ordinal(self, gpu_index: int) -> int:
        """Which group a dispatched ``ResolvedCompute.gpu_index`` names."""
        idx = int(gpu_index)
        if idx < 0 or idx >= self.gpu_count or idx % self.gpus_per_execution_group != 0:
            ordinal = max(0, min(self.execution_groups - 1, idx // self.gpus_per_execution_group))
            logger.warning(
                "topology: dispatched gpu_index=%d is not a rank-0 device of "
                "%s; serving it as group %d", idx, self, ordinal,
            )
            return ordinal
        return idx // self.gpus_per_execution_group

    def group_ordinal_exact(self, gpu_index: int) -> int:
        """``group_ordinal`` without the floor: a typed refusal instead."""
        idx = int(gpu_index)
        if idx < 0 or idx >= self.gpu_count or idx % self.gpus_per_execution_group != 0:
            raise TopologyError(
                "topology_dispatch_gpu_index_invalid",
                f"gpu_index={idx} is not a rank-0 device of {self} "
                f"(expected one of "
                f"{[g * self.gpus_per_execution_group for g in range(self.execution_groups)]})",
            )
        return idx // self.gpus_per_execution_group

    def device_group(self, gpu_index: int) -> DeviceGroup:
        """The DeviceGroup a job dispatched to ``gpu_index`` executes on."""
        return self.group(self.group_ordinal(gpu_index))

    def group(self, ordinal: int) -> DeviceGroup:
        g = int(ordinal)
        if g < 0 or g >= self.execution_groups:
            raise TopologyError(
                "topology_group_out_of_range",
                f"group {g} of {self.execution_groups} in {self}",
            )
        base = g * self.gpus_per_execution_group
        return DeviceGroup(
            devices=tuple(range(base, base + self.gpus_per_execution_group)),
            placement_mode=self.placement_mode,
        )

    def all_groups(self) -> Tuple[DeviceGroup, ...]:
        return tuple(self.group(g) for g in range(self.execution_groups))

    def demoted(self) -> "ExecutionTopology":
        """The hub's ``topology_demoted_fabric_not_nvlink`` re-pack, computed locally: G = gpu_count, D = 1."""
        return ExecutionTopology(gpu_count=self.gpu_count, gpus_per_execution_group=1)

    def as_dict(self) -> dict:
        d: dict[str, Any] = {
            KEY_GPU_COUNT: self.gpu_count,
            KEY_GPUS_PER_GROUP: self.gpus_per_execution_group,
            KEY_EXECUTION_GROUPS: self.execution_groups,
        }
        if self.parallel:
            d[KEY_PARALLEL] = self.parallel
        return d

    def __str__(self) -> str:  # pragma: no cover - logging sugar
        return (
            f"{self.execution_groups}x{self.gpus_per_execution_group}"
            f"{'/' + self.parallel if self.parallel else ''}"
        )

    @classmethod
    def single(cls) -> "ExecutionTopology":
        """The historical, and still the default, shape: one slot, one device."""
        return cls()

    @classmethod
    def decode(cls, raw: str) -> "ExecutionTopology":
        try:
            obj = json.loads(raw)
        except Exception as exc:
            raise TopologyError(
                "topology_decode_failed", f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise TopologyError(
                "topology_decode_failed", f"expected an object, got {type(obj).__name__}"
            )

        unknown = sorted(k for k in obj if k not in _KNOWN_KEYS)
        if unknown:
            raise TopologyError(
                "topology_unknown_field",
                f"unrecognised topology field(s) {unknown} in {raw!r}; this "
                f"worker knows {sorted(_KNOWN_KEYS)}. Refusing rather than "
                "reading the fields it names as absent — absent means ONE "
                "slot, so ignoring them would silently serve less than the "
                "hub bought",
            )

        def _opt(key: str) -> Optional[int]:
            if key not in obj or obj[key] is None:
                return None
            value = obj[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TopologyError(
                    "topology_decode_failed", f"{key} is not an integer: {value!r}"
                )
            if not _INT64_MIN <= value <= _INT64_MAX:
                raise TopologyError(
                    "topology_decode_failed", f"{key} is not an integer in range: {value!r}"
                )
            return value

        parallel_raw = obj.get(KEY_PARALLEL)
        if parallel_raw is None:
            parallel = ""
        elif not isinstance(parallel_raw, str):
            raise TopologyError(
                "topology_decode_failed", f"parallel is not a string: {parallel_raw!r}"
            )
        else:
            parallel = parallel_raw
        gpu_count = _opt(KEY_GPU_COUNT)
        if gpu_count is None:
            raise TopologyError(
                "topology_gpu_count_invalid",
                f"{KEY_GPU_COUNT} is absent; a present topology must declare it",
            )
        degree = _opt(KEY_GPUS_PER_GROUP)
        if degree is None:
            raise TopologyError(
                "topology_degree_invalid",
                f"{KEY_GPUS_PER_GROUP} is absent; a present topology must "
                "declare it",
            )
        topo = cls(
            gpu_count=gpu_count,
            gpus_per_execution_group=degree,
            parallel=parallel.strip(),
        )
        declared = _opt(KEY_EXECUTION_GROUPS)
        if declared is not None and declared != topo.execution_groups:
            raise TopologyError(
                "topology_execution_groups_disagree",
                f"declared execution_groups={declared} but gpu_count/"
                f"gpus_per_execution_group={topo.execution_groups}",
            )
        return topo

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None
    ) -> "ExecutionTopology":
        """Decode the delivered topology."""
        source = os.environ if env is None else env
        raw = (source.get(ENV_VAR) or "").strip()
        if not raw:
            return cls.single()
        topo = cls.decode(raw)
        logger.info(
            "EXECUTION_TOPOLOGY gpu_count=%d gpus_per_execution_group=%d groups=%d parallel=%s",
            topo.gpu_count, topo.gpus_per_execution_group, topo.execution_groups, topo.parallel or "-",
        )
        return topo


def delivered_topology(
    env: Optional[Mapping[str, str]] = None,
    *,
    interconnect: Optional[str] = None,
    peer_gbps: Optional[float] = None,
    peer_access: Optional[bool] = None,
) -> ExecutionTopology:
    """The topology this worker will actually execute."""

    topo = ExecutionTopology.from_env(env)
    if interconnect is None and topo.gpu_count > 1:
        from .host_canary import get_host_canary

        canary = get_host_canary()
        interconnect = canary.interconnect
        if peer_gbps is None:
            peer_gbps = canary.peer_gbps
        if peer_access is None:
            peer_access = canary.peer_access
    if topo.gpu_count > 1 and is_fabric_wedge(bool(peer_access), float(peer_gbps or 0.0)):
        raise TopologyError(
            "topology_fabric_wedged_peer_access_zero_bandwidth",
            f"{topo}: peer access reported with 0.0 GB/s measured — every "
            "collective on this host blocks forever (fleet survey, machine "
            "class reproduced twice). Refusing at boot so the hub re-packs "
            "rather than stranding requests",
        )
    if not topo.platform_parallel:
        if env is None:
            install_topology(topo)
        return topo
    if not sp_admits(interconnect or "", float(peer_gbps or 0.0)):
        demoted = topo.demoted()
        logger.warning(
            "topology_demoted_fabric_not_nvlink: measured interconnect=%r "
            "peer_gbps=%.2f, %s -> %s (the sharded tier is sold on a fabric "
            "this pod did not prove)",
            interconnect or "", float(peer_gbps or 0.0), topo, demoted,
        )
        if env is None:
            install_topology(demoted)
        return demoted
    refuse_unless_groups_can_coexist(topo)
    if env is None:
        install_topology(topo)
    return topo


def refuse_unless_groups_can_coexist(topo: ExecutionTopology) -> None:
    """Refuse a multi-device group the worker cannot actually execute, by name."""
    if topo.degree <= 1 or topo.parallel == PARALLEL_SEQUENCE:
        return
    raise TopologyError(
        "topology_group_parallel_unsupported",
        f"{topo}: parallel={topo.parallel!r} has no degree-{topo.degree} "
        "runtime in this worker — only 'sequence' shards a group's work. "
        "Refusing at boot so the hub re-packs, rather than holding "
        f"{topo.degree} cards per slot and serving one card's worth",
    )
