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
from typing import Any, Mapping, Optional, Tuple

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


# The delivered packing, published process-wide so the ambient group ordinal
# can be TRANSLATED into the cards that group actually owns. An ordinal is not
# a device index — at degree D group g owns ``[g*D, (g+1)*D)`` — and the
# contextvar above carries only the ordinal, so the translation needs the
# topology. One pod has exactly one delivered topology, which is why this is a
# module fact and not a parameter on every placement call. Absent (harness
# attach, `cli serve`, every test that builds none) ⇒ the identity mapping,
# which is what a one-device-per-group worker has always done.
_installed: Optional["ExecutionTopology"] = None


def install_topology(topo: Optional["ExecutionTopology"]) -> None:
    """Publish the delivered packing for the ordinal -> device translation."""
    global _installed
    _installed = topo


def installed_topology() -> Optional["ExecutionTopology"]:
    return _installed


def group_devices(ordinal: Optional[int] = None) -> Tuple[int, ...]:
    """The CUDA device indices a group owns, from the DELIVERED topology.

    ``(ordinal,)`` when no topology describes that group: at degree 1 the
    ordinal IS the device, which is every pod that has ever shipped.
    """
    g = current_device_group() if ordinal is None else int(ordinal)
    topo = _installed
    if topo is None or g < 0 or g >= topo.groups:
        return (max(0, g),)
    return topo.group(g).devices


def group_rank0_device(ordinal: Optional[int] = None) -> int:
    """The group's rank-0 card — the device its weights and its rank-0 work
    belong on, and the index the hub dispatches as ``gpu_index``."""
    return group_devices(ordinal)[0]


def pin_cuda_device_for_group() -> None:
    """Set THIS THREAD's current CUDA device to the current group's rank-0 card.

    `torch.cuda.set_device` is thread-local and the load path never called it:
    handler threads did (from ``ResolvedCompute.gpu_index``), but the SETUP and
    materialization threads did not, so every group's weights were placed by a
    plain ``.to("cuda")`` onto whatever card that pool thread happened to point
    at — card 0. Measured live on a 4xL40S pod: `current_device` was correctly
    cuda:0..3 per group while the WEIGHTS sat on cuda:0,0,0,3 (pgw#748 DP
    width-4 acceptance). Cheap and idempotent; a no-op without CUDA and for
    the group whose rank-0 card is 0.

    The card comes from the delivered topology, never from the ordinal: on a
    ``2x2`` pod group 1 owns cards 2-3, so pinning to device 1 would place
    group 1's weights on group 0's follower card (pgw#773 residual).
    """
    device = group_rank0_device()
    if device <= 0:
        return
    try:
        import torch

        if torch.cuda.is_available() and device < torch.cuda.device_count():
            torch.cuda.set_device(device)
    except Exception:  # noqa: BLE001 - placement is best-effort here; the
        # residency promote still targets `cuda:N` explicitly.
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

    def group_ordinal_exact(self, gpu_index: int) -> int:
        """``group_ordinal`` without the floor: a typed refusal instead.

        The floor exists so a single-group pod cannot index off the end. On a
        wide pod flooring is the silent bug (pgw#779): every dispatch the hub
        got wrong lands on group 0, which is also the group that is always
        busiest.
        """
        idx = int(gpu_index)
        if idx < 0 or idx >= self.gpu_count or idx % self.group_degree != 0:
            raise TopologyError(
                "topology_dispatch_gpu_index_invalid",
                f"gpu_index={idx} is not a rank-0 device of {self} "
                f"(expected one of "
                f"{[g * self.group_degree for g in range(self.groups)]})",
            )
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
        locally: G = gpu_count, D = 1. Both sides read the same measurement —
        but "agree by construction" holds only while both apply the same
        PREDICATE over it. pgw#818 is what happened when the hub grew a
        bandwidth floor and this side kept class-only: in the disagreement
        band a 2x2 pod refused half of every dispatch forever. The predicate
        is now shared (``host_canary.sp_admits`` == hub ``topology.SPAdmits``,
        one constant each side, same number); there is deliberately still no
        HelloAck demote field — two independent gates over one measurement is
        the design, and the constants must move together."""
        return ExecutionTopology(gpu_count=self.gpu_count, group_degree=1)

    def as_dict(self) -> dict:
        d: dict[str, Any] = {
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
    peer_gbps: Optional[float] = None,
    peer_access: Optional[bool] = None,
) -> ExecutionTopology:
    """The topology this worker will actually execute.

    Applies the same fabric gate the hub applies at Hello — the SAME two-term
    predicate (pgw#818): a group the *platform* shards is sold on a proven
    interconnect, so unless this pod's own boot canary measures ``nvlink``
    AND ``peer_gbps >= SP_MIN_PEER_GBPS`` the group is demoted to ``G×1``
    rather than served at a promise the hardware cannot keep. Class alone is
    not enough — a degraded host prints ``nvlink`` at 30 GB/s, and in that
    band a hub that demoted while the worker did not left half of every 2x2
    pod's dispatches in a permanent retry loop. ``internal`` groups are never
    bandwidth-demoted — the devices are the model's, not the platform's.

    One refusal outranks everything on ANY multi-GPU topology: a WEDGED
    fabric (peer access reported, bandwidth exactly zero — the collective
    hangs with no error) raises typed at boot, so the hub re-packs instead of
    racing its own quarantine drain against this worker reaching serving.

    ``interconnect``/``peer_gbps``/``peer_access`` default to the shipped
    boot canary's measurement.

    Reading the REAL environment (``env is None``, i.e. every production and
    boot call) also PUBLISHES the result: the placement helpers translate a
    group ordinal into the cards that group owns, and only the delivered —
    possibly fabric-demoted — packing can do that. An explicit ``env`` is a
    caller asking a question about some other pod, so it publishes nothing.
    """
    from .host_canary import is_fabric_wedge, sp_admits

    topo = ExecutionTopology.from_env(env)
    # An explicit ``interconnect`` is a caller asking about SOME pod — the
    # unsupplied axes default fail-closed (0.0 / False) rather than reading
    # this process's canary into another pod's question. Production calls
    # pass nothing and read the pod's own measurement.
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
    """Refuse a multi-device group the worker cannot actually execute, by name.

    ``G>1 ∧ D>1`` is SERVED now (pgw#773 residual, lifted). Both original
    reasons are closed: each group owns a non-default process group over its
    own store (so two groups cannot corrupt each other's collectives), and
    placement no longer reads the group ordinal as a device index — every card
    is derived from this topology, so group 1 of a `2x2` pod places on its own
    cards 2-3. Live-accepted on 4xH100-80 SXM NVLink: degree 2, degree 4 and
    two concurrent degree-2 groups, each bit-identical to degree 1.

    What is still refused is a degree>1 group whose sharding NOTHING in this
    worker installs — at boot that is ``cfg`` (the other platform-sharded
    mechanism the wire can carry). It would hold D cards for one slot and then
    serve it unsharded: a fraction of the tier that was sold, silently. A boot
    refusal is the only thing that makes the hub re-pack.
    """
    if topo.degree <= 1 or topo.parallel == PARALLEL_SEQUENCE:
        return
    raise TopologyError(
        "topology_group_parallel_unsupported",
        f"{topo}: parallel={topo.parallel!r} has no degree-{topo.degree} "
        "runtime in this worker — only 'sequence' shards a group's work. "
        "Refusing at boot so the hub re-packs, rather than holding "
        f"{topo.degree} cards per slot and serving one card's worth",
    )
