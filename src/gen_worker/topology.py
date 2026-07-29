"""Execution topology: the hub's staffing decision, delivered as one env var.

The hub decides how many GPUs a pod gets, how they are packed into execution
groups, and which parallelism (if any) the platform installs inside a group.
The worker never invents any of it. The whole contract is
``WORKER_EXECUTION_TOPOLOGY`` (th#1285, tensorhub ``internal/orchestrator/
topology/topology.go``)::

    WORKER_EXECUTION_TOPOLOGY={"gpu_count":4,"group_degree":2,"groups":2,
                               "parallel":"sequence"}

``WORKER_`` is a reserved endpoint-env prefix hub-side, so this is a trusted
wire fact and NOT an operator knob: nothing in gen-worker may set it to change
behaviour, and nothing may default it to anything but "one slot".

Derivation, identical on both sides, from ``(gpu_count, group_degree)`` alone::

    D = group_degree                    # devices per execution group
    G = gpu_count / group_degree        # execution groups == serving slots
    group g owns devices [g*D, (g+1)*D)

``ResolvedCompute.gpu_index`` is unchanged on the job wire and now names the
**rank-0 device of the group** (0, D, 2D, ...), so at D == 1 it is byte-identical
to what has always shipped.

Absent is a legal state, never an error: every CPU pod and every pod created
before the field existed has no topology and keeps the historical single slot.
A value that is *present but malformed* is a typed refusal — never a silent
fallback — because it can only mean a producer that is not the hub.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
from dataclasses import dataclass
from typing import Iterator, Mapping, Optional, Tuple

from .models.residency import REPLICATED, SHARDED, DeviceGroup

logger = logging.getLogger(__name__)

# Which execution group the current task/thread is serving. Stamped once per
# job from the dispatched rank-0 device and inherited by every coroutine and
# ``asyncio.to_thread`` hop the job makes (contextvars propagate into both),
# which is what lets per-group bookkeeping ride the call graph the job already
# threads instead of a new parameter on 200 functions. Default 0 = the only
# group a single-slot worker has ever had.
_current_group: contextvars.ContextVar[int] = contextvars.ContextVar(
    "gen_worker_device_group", default=0
)


def current_device_group() -> int:
    return _current_group.get()


def set_device_group(ordinal: int) -> contextvars.Token:
    return _current_group.set(int(ordinal))


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

# ``parallel`` — who shards the work across a group's devices.
PARALLEL_NONE = ""            # group is one device
PARALLEL_INTERNAL = "internal"  # the MODEL spans the devices by its own arrangement
PARALLEL_SEQUENCE = "sequence"  # the PLATFORM installs Ulysses sequence parallelism
PARALLEL_CFG = "cfg"            # the PLATFORM splits the CFG batch

_PARALLEL_VALUES = (PARALLEL_NONE, PARALLEL_INTERNAL, PARALLEL_SEQUENCE, PARALLEL_CFG)

# Only these make a bandwidth promise, so only these are fabric-gated and only
# these may be demoted. Demoting an ``internal`` group would take a model's own
# cards away and break a release that has always worked.
_PLATFORM_PARALLEL = (PARALLEL_SEQUENCE, PARALLEL_CFG)


class TopologyError(ValueError):
    """A present-but-illegal topology. ``code`` is one of the hub's stable
    refusal strings, so both sides name the same fault."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class ExecutionTopology:
    """One pod's delivered packing. Immutable; derived, never negotiated."""

    gpu_count: int = 1
    group_degree: int = 1
    parallel: str = PARALLEL_NONE

    def __post_init__(self) -> None:
        if self.gpu_count < 1:
            raise TopologyError(
                "topology_gpu_count_invalid", f"gpu_count={self.gpu_count}"
            )
        if self.group_degree < 1:
            raise TopologyError(
                "topology_degree_invalid", f"group_degree={self.group_degree}"
            )
        if self.gpu_count % self.group_degree != 0:
            raise TopologyError(
                "topology_degree_not_divisor",
                f"group_degree={self.group_degree} does not divide "
                f"gpu_count={self.gpu_count}",
            )
        if self.parallel not in _PARALLEL_VALUES:
            raise TopologyError(
                "topology_parallel_unknown", f"parallel={self.parallel!r}"
            )
        if self.group_degree > 1 and self.parallel == PARALLEL_NONE:
            raise TopologyError(
                "topology_parallel_required",
                f"group_degree={self.group_degree} needs a parallel mechanism",
            )
        if self.group_degree == 1 and self.parallel != PARALLEL_NONE:
            raise TopologyError(
                "topology_parallel_without_degree",
                f"parallel={self.parallel!r} at group_degree=1",
            )

    # ---- derived facts -----------------------------------------------------

    @property
    def groups(self) -> int:
        """G — execution groups, which IS the worker's slot count."""
        return self.gpu_count // self.group_degree

    @property
    def slots(self) -> int:
        return self.groups

    @property
    def degree(self) -> int:
        return self.group_degree

    @property
    def platform_parallel(self) -> bool:
        """True when the PLATFORM installs the sharding (and therefore when a
        fabric miss may demote this pod)."""
        return self.parallel in _PLATFORM_PARALLEL

    @property
    def placement_mode(self) -> str:
        """How a materialization occupies a group. Sequence/CFG parallelism
        replicate the weights and shard activations; an ``internal`` group is
        the model's own arrangement, which is what a device map shards."""
        return SHARDED if self.parallel == PARALLEL_INTERNAL else REPLICATED

    def group_ordinal(self, gpu_index: int) -> int:
        """Which group a dispatched ``ResolvedCompute.gpu_index`` names."""
        idx = int(gpu_index)
        if idx < 0 or idx >= self.gpu_count or idx % self.group_degree != 0:
            # The hub always dispatches a group's rank-0 device. Anything else
            # is a hub/worker disagreement about the packing; floor it to a
            # real group rather than index off the end, and say so.
            ordinal = max(0, min(self.groups - 1, idx // self.group_degree))
            logger.warning(
                "topology: dispatched gpu_index=%d is not a rank-0 device of "
                "%s; serving it as group %d", idx, self, ordinal,
            )
            return ordinal
        return idx // self.group_degree

    def device_group(self, gpu_index: int) -> DeviceGroup:
        """The DeviceGroup a job dispatched to ``gpu_index`` executes on."""
        return self.group(self.group_ordinal(gpu_index))

    def group(self, ordinal: int) -> DeviceGroup:
        g = int(ordinal)
        if g < 0 or g >= self.groups:
            raise TopologyError(
                "topology_group_out_of_range",
                f"group {g} of {self.groups} in {self}",
            )
        base = g * self.group_degree
        return DeviceGroup(
            devices=tuple(range(base, base + self.group_degree)),
            placement_mode=self.placement_mode,
        )

    def all_groups(self) -> Tuple[DeviceGroup, ...]:
        return tuple(self.group(g) for g in range(self.groups))

    def demoted(self) -> "ExecutionTopology":
        """The hub's ``topology_demoted_fabric_not_nvlink`` re-pack, computed
        locally: G = gpu_count, D = 1. Both sides read the same measured
        interconnect, so they agree by construction (§2a's Phase-3 seam does
        not need a HelloAck field for a mechanism the worker also measures)."""
        return ExecutionTopology(gpu_count=self.gpu_count, group_degree=1)

    def as_dict(self) -> dict:
        d = {
            "gpu_count": self.gpu_count,
            "group_degree": self.group_degree,
            "groups": self.groups,
        }
        if self.parallel:
            d["parallel"] = self.parallel
        return d

    def __str__(self) -> str:  # pragma: no cover - logging sugar
        return (
            f"{self.groups}x{self.group_degree}"
            f"{'/' + self.parallel if self.parallel else ''}"
        )

    # ---- decoding ----------------------------------------------------------

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

        def _int(key: str, default: int) -> int:
            if key not in obj or obj[key] is None:
                return default
            value = obj[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TopologyError(
                    "topology_decode_failed", f"{key} is not a number: {value!r}"
                )
            if int(value) != value:
                raise TopologyError(
                    "topology_decode_failed", f"{key} is not an integer: {value!r}"
                )
            return int(value)

        parallel = obj.get("parallel") or ""
        if not isinstance(parallel, str):
            raise TopologyError(
                "topology_decode_failed", f"parallel is not a string: {parallel!r}"
            )
        topo = cls(
            gpu_count=_int("gpu_count", 1),
            group_degree=_int("group_degree", 1),
            parallel=parallel.strip(),
        )
        # ``groups`` is derived and the producer always recomputes it, so a
        # disagreement means the value did not come from the hub. Refuse it
        # rather than pick a winner — a wrong slot count is a wrong fleet
        # capacity, and it is silent.
        declared_groups = _int("groups", topo.groups)
        if declared_groups != topo.groups:
            raise TopologyError(
                "topology_groups_disagree",
                f"declared groups={declared_groups} but "
                f"gpu_count/group_degree={topo.groups}",
            )
        return topo

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None
    ) -> "ExecutionTopology":
        """Decode the delivered topology. Absent/blank ⇒ one slot (legal)."""
        source = os.environ if env is None else env
        raw = (source.get(ENV_VAR) or "").strip()
        if not raw:
            return cls.single()
        topo = cls.decode(raw)
        logger.info(
            "EXECUTION_TOPOLOGY gpu_count=%d group_degree=%d groups=%d parallel=%s",
            topo.gpu_count, topo.group_degree, topo.groups, topo.parallel or "-",
        )
        return topo


def delivered_topology(
    env: Optional[Mapping[str, str]] = None,
    *,
    interconnect: Optional[str] = None,
) -> ExecutionTopology:
    """The topology this worker will actually execute.

    Applies the same fabric gate the hub applies at Hello: a group the
    *platform* shards is sold on a proven interconnect, so if this pod's own
    boot canary does not report ``nvlink`` the group is demoted to ``G×1``
    rather than served at a promise the hardware cannot keep. ``internal``
    groups are never demoted — the devices are the model's, not the platform's.

    ``interconnect`` defaults to the shipped boot canary's measurement.
    """
    topo = ExecutionTopology.from_env(env)
    if not topo.platform_parallel:
        return topo
    if interconnect is None:
        from .host_canary import get_host_canary

        interconnect = get_host_canary().interconnect
    from .host_canary import INTERCONNECT_NVLINK

    if interconnect != INTERCONNECT_NVLINK:
        demoted = topo.demoted()
        logger.warning(
            "topology_demoted_fabric_not_nvlink: measured interconnect=%r, "
            "%s -> %s (the sharded tier is sold on a fabric this pod does "
            "not have)", interconnect or "", topo, demoted,
        )
        return demoted
    return topo
