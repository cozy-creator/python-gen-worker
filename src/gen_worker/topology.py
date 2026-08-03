"""Execution topology: the hub's staffing decision, delivered as one env var.

The hub decides how many GPUs a pod gets, how they are packed into execution
groups, and which parallelism (if any) the platform installs inside a group.
The worker never invents any of it. The whole contract is
``WORKER_EXECUTION_TOPOLOGY`` (th#1285, tensorhub ``internal/orchestrator/
topology/topology.go``)::

    WORKER_EXECUTION_TOPOLOGY={"gpu_count":4,"gpus_per_execution_group":2,
                               "execution_groups":2,"parallel":"sequence"}

``WORKER_`` is a reserved endpoint-env prefix hub-side, so this is a trusted
wire fact and NOT an operator knob: nothing in gen-worker may set it to change
behaviour, and nothing may default it to anything but "one slot".

Derivation, identical on both sides, from ``(gpu_count,
gpus_per_execution_group)`` alone::

    D = gpus_per_execution_group        # devices per execution group
    G = gpu_count / D                   # execution groups == serving slots
    group g owns devices [g*D, (g+1)*D)

``gpu_count = execution_groups × gpus_per_execution_group`` is not bookkeeping;
it **encodes a partition invariant**. An execution group exclusively owns its
devices — no card is in two groups, and no card is idle. Should GPU sharing
between groups ever be introduced, that arithmetic stops holding and every
consumer that derives G from it refuses loudly (``topology_degree_not_divisor``,
``topology_execution_groups_disagree``) instead of quietly serving a packing
nobody chose. Keep it that way: derive, never store a fourth independent fact.

``ResolvedCompute.gpu_index`` is unchanged on the job wire and names the
**rank-0 device of the group** (0, D, 2D, ...), so at D == 1 it is byte-identical
to what has always shipped.

An absent ENV VAR is a legal state, never an error: every CPU pod and every pod
created before the field existed has no topology and keeps the historical
single slot. An absent FIELD inside a *present* object is the opposite — a
typed refusal (th#1385), because both producers always write ``gpu_count`` and
the degree, so an object missing one did not come from a producer this contract
knows, and defaulting it reads as ONE SLOT. Everything *present but not fully
recognised* is likewise a typed refusal — never a silent fallback. That covers
unknown keys (``topology_unknown_field``): the field set is CLOSED, so growing
the contract is itself a two-release transition, exactly like the th#1375
rename below. The alternative — ignore what you do not understand — is how a
hub that believes it bought degree 2 gets served degree 1 in silence.

An "execution group" is NOT a PyTorch ``ProcessGroup``
---------------------------------------------------
They sound identical and appear within a few lines of each other in the process
-split code, and they are different things. An **execution group** is the
*serving unit*: the set of cards that cooperate on one request, and the thing
whose count is this worker's slot count. A ``torch.distributed.ProcessGroup``
is the *communication handle* those ranks collect over. Each execution group
owns one (non-default, over its own store), so the relationship is
one-ProcessGroup-per-execution-group — not a synonym.

Translating from another stack
------------------------------
The same two numbers exist everywhere; only the names differ. Ours are spelled
out because a bare ``groups`` collides with process groups, parameter groups
and co-location groups, all of which live in these files:

===========================  ==================================================
ours                         theirs
===========================  ==================================================
``gpus_per_execution_group``  ``tensor_parallel_size`` (vLLM, SGLang);
                              ``context_parallel_size`` / ``cp_size``
                              (Megatron); ``sequence_parallel_size``
                              (DeepSpeed-Ulysses)
``execution_groups``          ``data_parallel_size``; ``num_replicas``
                              (Ray Serve); instance count (Triton)
===========================  ==================================================

Note what those names do that ours does not: each of them fuses the *width* of
a group with the *technique* used to fill it, so a stack that wants Ulysses
instead of tensor parallelism needs a different field. We keep ``parallel`` as
a separate field precisely so the width is named independently of the mechanism
— ``gpus_per_execution_group=2`` means two cards serve one request whether the
sharding is ``sequence``, ``cfg``, or the model's own ``internal`` arrangement.
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
    if topo is None or g < 0 or g >= topo.execution_groups:
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

# The wire keys. The set is CLOSED — anything else is ``topology_unknown_field``
# (see the module docstring), which is what makes "present but unrecognised" a
# refusal instead of a silent fall back to one slot.
KEY_GPU_COUNT = "gpu_count"
KEY_GPUS_PER_GROUP = "gpus_per_execution_group"
KEY_EXECUTION_GROUPS = "execution_groups"
KEY_PARALLEL = "parallel"

# th#1375's pre-rename spelling. Accepted here, and emitted alongside the new
# spelling by ``as_dict``, for exactly one release: the hub and the workers
# deploy independently and prod pins gen-worker 0.79.0, so no flip day exists on
# which every reader speaks the new names. Carrying both makes deploy ORDER
# irrelevant rather than merely survivable. th#1376 deletes them, after which
# they fall through to ``topology_unknown_field`` by construction.
LEGACY_KEY_GPUS_PER_GROUP = "group_degree"
LEGACY_KEY_EXECUTION_GROUPS = "groups"

_KNOWN_KEYS = frozenset({
    KEY_GPU_COUNT, KEY_GPUS_PER_GROUP, KEY_EXECUTION_GROUPS, KEY_PARALLEL,
    LEGACY_KEY_GPUS_PER_GROUP, LEGACY_KEY_EXECUTION_GROUPS,
})

# th#1385/pgw#870: a CEILING, not just a floor, and the same number on both
# sides (tensorhub ``topology.MaxGPUCount``). Without one the decoder accepts
# any wire integer and ``all_groups()`` / ``group()`` over it is unbounded work
# on a number no producer could have meant. The largest pod the fleet rents is
# 8 cards.
MAX_GPU_COUNT = 1024

# The wire integers are int64 on the hub side, so a value outside that range is
# not a number this contract can carry — it is a typed decode refusal, matching
# what `encoding/json` does with it.
_INT64_MIN = -(2 ** 63)
_INT64_MAX = 2 ** 63 - 1

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

    # ---- derived facts -----------------------------------------------------

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
        if idx < 0 or idx >= self.gpu_count or idx % self.gpus_per_execution_group != 0:
            # The hub always dispatches a group's rank-0 device. Anything else
            # is a hub/worker disagreement about the packing; floor it to a
            # real group rather than index off the end, and say so.
            ordinal = max(0, min(self.execution_groups - 1, idx // self.gpus_per_execution_group))
            logger.warning(
                "topology: dispatched gpu_index=%d is not a rank-0 device of "
                "%s; serving it as group %d", idx, self, ordinal,
            )
            return ordinal
        return idx // self.gpus_per_execution_group

    def group_ordinal_exact(self, gpu_index: int) -> int:
        """``group_ordinal`` without the floor: a typed refusal instead.

        The floor exists so a single-group pod cannot index off the end. On a
        wide pod flooring is the silent bug (pgw#779): every dispatch the hub
        got wrong lands on group 0, which is also the group that is always
        busiest.
        """
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
        return ExecutionTopology(gpu_count=self.gpu_count, gpus_per_execution_group=1)

    def as_dict(self) -> dict:
        d: dict[str, Any] = {
            KEY_GPU_COUNT: self.gpu_count,
            KEY_GPUS_PER_GROUP: self.gpus_per_execution_group,
            KEY_EXECUTION_GROUPS: self.execution_groups,
            # th#1376 removes these two. They are here so this worker's own
            # produced topology (the parent stamps one per split child, and
            # ``procsplit`` children may run a different build during a rolling
            # image swap) is readable by a pre-th#1375 reader.
            LEGACY_KEY_GPUS_PER_GROUP: self.gpus_per_execution_group,
            LEGACY_KEY_EXECUTION_GROUPS: self.execution_groups,
        }
        if self.parallel:
            d[KEY_PARALLEL] = self.parallel
        return d

    def __str__(self) -> str:  # pragma: no cover - logging sugar
        return (
            f"{self.execution_groups}x{self.gpus_per_execution_group}"
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

        # THE guard. A key this worker does not recognise means the value did
        # not come from a hub this worker understands — most likely one half of
        # a rename that only landed on one side. Ignoring it would read the
        # field it names as ABSENT, and absent is legal and means one slot, so
        # the pod would serve degree 1 while the hub billed the degree it
        # bought. Refuse by name instead, listing what was and was not known,
        # so the fault reads as a version skew rather than a config bug.
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
            """One optional wire integer. Absent and ``null`` are both "not
            written"; anything that is not a JSON integer is a TYPED decode
            refusal.

            pgw#870: this used to accept any ``(int, float)`` and then compare
            ``int(value) != value``, which let an integral float through
            (``2.0`` -> 2, where the hub refuses) and let ``NaN``/``Infinity``
            — which ``json.loads`` accepts as non-standard literals — reach
            ``int()`` and escape as an UNTYPED ``ValueError``/``OverflowError``
            past every caller that catches ``TopologyError``. A float is not an
            integer; say so instead of coercing.
            """
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

        def _aliased(key: str, legacy_key: str) -> Optional[int]:
            """One field under either spelling (th#1375/#1376).

            Both present must AGREE. During the transition every hub emits
            both, so a disagreement is not a stale producer to be tolerated —
            it is a producer whose two spellings describe two different
            packings, and picking either one is picking silently.
            """
            new_value = _opt(key)
            legacy_value = _opt(legacy_key)
            if legacy_value is None:
                return new_value
            if new_value is not None and new_value != legacy_value:
                raise TopologyError(
                    "topology_alias_disagree",
                    f"{key}={new_value} but {legacy_key}={legacy_value} — one "
                    "producer wrote both spellings of the same field with "
                    "different values; refusing rather than choosing",
                )
            if new_value is None:
                logger.warning(
                    "topology: %r is the pre-th#1375 spelling of %r and is "
                    "accepted for this release only (th#1376 removes it)",
                    legacy_key, key,
                )
                return legacy_value
            return new_value

        # pgw#870: TYPE-CHECK BEFORE DEFAULTING. This was
        # ``obj.get(KEY_PARALLEL) or ""``, the `or 1` launder in a second
        # field: every falsy non-string (``false``, ``0``, ``[]``, ``{}``)
        # became PARALLEL_NONE before the isinstance guard could see it, so
        # the guard was unreachable for exactly the values it was written for.
        parallel_raw = obj.get(KEY_PARALLEL)
        if parallel_raw is None:
            parallel = ""
        elif not isinstance(parallel_raw, str):
            raise TopologyError(
                "topology_decode_failed", f"parallel is not a string: {parallel_raw!r}"
            )
        else:
            parallel = parallel_raw
        # th#1385: an ABSENT field is a refusal, not a default. The hub and
        # this worker both always emit gpu_count and the degree (under one
        # spelling or the other), so an object missing either did not come
        # from a producer this contract knows — and a default reads it as
        # ONE SLOT, which is the exact silence th#1375 exists to prevent.
        # (An absent ENV VAR is still legal and still means one slot; that is
        # `from_env`, not this.)
        gpu_count = _opt(KEY_GPU_COUNT)
        if gpu_count is None:
            raise TopologyError(
                "topology_gpu_count_invalid",
                f"{KEY_GPU_COUNT} is absent; a present topology must declare it",
            )
        degree = _aliased(KEY_GPUS_PER_GROUP, LEGACY_KEY_GPUS_PER_GROUP)
        if degree is None:
            raise TopologyError(
                "topology_degree_invalid",
                f"{KEY_GPUS_PER_GROUP} is absent under both spellings; a present "
                "topology must declare it",
            )
        topo = cls(
            gpu_count=gpu_count,
            gpus_per_execution_group=degree,
            parallel=parallel.strip(),
        )
        # ``execution_groups`` is derived and the producer always recomputes
        # it, so a disagreement means the value did not come from the hub.
        # Refuse it rather than pick a winner — a wrong slot count is a wrong
        # fleet capacity, and it is silent.
        declared = _aliased(KEY_EXECUTION_GROUPS, LEGACY_KEY_EXECUTION_GROUPS)
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
        """Decode the delivered topology. Absent/blank ⇒ one slot (legal)."""
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
