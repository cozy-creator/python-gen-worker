"""Residency v2 program suite (pgw#641 umbrella; design of record:
WORKER-RESIDENCY-DESIGN.md). One designed home for the program's rows —
new stages add rows here, not new per-issue files (pgw#645).

Covered trains, all REAL planner/admission/eviction logic; fakes only at the
torch/CUDA boundary per th#1105:

A. pgw#648 — DeviceGroup accounting. VRAM is never summed across groups: a
   3x24GB pod is three 24GB pools, not one 72GB pool (the live admission
   bug that admitted a 30GB model fitting on no single card).
B. pgw#641 Stage 2 — admission leases. A job's refs are victim-protected
   from admission (including refs with no entry yet — the executing() pin
   no-op'd on those), and not-yet-loaded bytes are RESERVED so concurrent
   admissions cannot double-book free VRAM. ``fits`` is the cheap honest
   "can this worker serve this now?" query.
C. pgw#647 — concurrency contract. Handlers on ONE live instance are
   single-flight by default (mutable graph buffers); ``reentrant=True`` is
   the explicit opt-in, classes only.
D. pgw#652 — activation-aware admission. Concurrency costs transient VRAM
   (latents + attention workspace), not only weights. The claim is LEARNED
   from measured peaks and SUMS across concurrent leases; nothing is
   declared by an endpoint.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, List, Tuple

import msgspec
import pytest

from gen_worker.api import Resources, endpoint
from gen_worker.executor import Executor
from gen_worker.models import residency as residency_mod
from gen_worker.models.residency import DeviceGroup, Residency, Tier
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

_GiB = 1024 ** 3


# ---------------------------------------------------------------------------
# A. DeviceGroup: per-group accounting, never a cross-card sum (pgw#648).
# ---------------------------------------------------------------------------


class _FakeCuda:
    """torch.cuda boundary fake: a pod with N cards of ``free`` bytes each."""

    def __init__(self, count: int, free: int, total: int) -> None:
        self._count, self._free, self._total = count, free, total

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return self._count

    def mem_get_info(self, device: int = 0) -> Tuple[int, int]:
        if not 0 <= int(device) < self._count:
            raise RuntimeError(f"invalid device ordinal {device}")
        return self._free, self._total


@pytest.fixture()
def three_by_24gb(monkeypatch: pytest.MonkeyPatch) -> _FakeCuda:
    torch = pytest.importorskip("torch")
    fake = _FakeCuda(count=3, free=24 * _GiB, total=24 * _GiB)
    monkeypatch.setattr(torch.cuda, "is_available", fake.is_available)
    monkeypatch.setattr(torch.cuda, "device_count", fake.device_count)
    monkeypatch.setattr(torch.cuda, "mem_get_info", fake.mem_get_info)
    return fake


def test_device_group_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        DeviceGroup(devices=())
    with pytest.raises(ValueError, match="unique"):
        DeviceGroup(devices=(0, 0))
    with pytest.raises(ValueError, match=">= 0"):
        DeviceGroup(devices=(-1,))
    assert DeviceGroup().primary == 0
    assert DeviceGroup(devices=(2, 1)).primary == 2


def test_free_vram_is_one_pool_never_the_pod_sum(three_by_24gb: _FakeCuda) -> None:
    # The live pgw#648 bug: 3x24GB reported 72GB free and admitted a 30GB
    # model that fits nowhere. The default group is ONE card's pool.
    assert DeviceGroup().free_vram_bytes() == 24 * _GiB
    assert residency_mod._default_free_vram_bytes() == 24 * _GiB
    # A registry probes its OWN group only.
    assert Residency().free_vram_bytes() == 24 * _GiB
    assert Residency(device_group=DeviceGroup(devices=(2,))).free_vram_bytes() == 24 * _GiB


def test_multi_device_group_sums_within_the_group_only(three_by_24gb: _FakeCuda) -> None:
    # A group SPANNING devices (future TP mesh) sums its own members — that
    # is one placement unit by definition — and nothing else.
    assert DeviceGroup(devices=(1, 2)).free_vram_bytes() == 48 * _GiB
    # Devices the host does not have contribute 0, not an exception.
    assert DeviceGroup(devices=(0, 7)).free_vram_bytes() == 24 * _GiB


# ---------------------------------------------------------------------------
# B. Admission leases (pgw#641 Stage 2).
# ---------------------------------------------------------------------------


class _FakeModule:
    def __init__(self) -> None:
        self.device = "cuda"

    def to(self, device: str) -> "_FakeModule":
        self.device = device
        return self


def _budget_residency(gib: int) -> Residency:
    def move(obj: Any, device: str) -> None:
        to = getattr(obj, "to", None)
        if callable(to):
            to(device)

    return Residency(vram_budget_bytes=gib * _GiB, move_fn=move)


def test_lease_protects_refs_with_no_entry_yet() -> None:
    """The structural gw#409 gap: executing() no-op'd on refs without
    entries, so an entry created mid-job was demotable until the inner pin.
    A lease protects the REF from admission on, whenever its entry appears."""
    res = _budget_residency(24)
    lease = res.admit({"cold-pick": 5 * _GiB})
    assert res.in_use("cold-pick")  # leased, even with no entry yet
    res.track_vram("cold-pick", _FakeModule(), vram_bytes=5 * _GiB)
    assert res.demote("cold-pick") is False
    assert res.evict("cold-pick") is False
    assert res.release_to_disk("cold-pick") is False
    assert "cold-pick" not in res.lru_vram_victims()
    lease.release()
    assert not res.in_use("cold-pick")
    assert res.demote("cold-pick") is True


def test_reservation_blocks_double_booking() -> None:
    """Two concurrent admissions must not book the same free bytes: while a
    lease holds a 10 GiB claim, the honest answer to a 12 GiB question is
    NO even though the physical probe still reports 14 GiB free."""
    res = _budget_residency(24)
    res.track_vram("resident-a", _FakeModule(), vram_bytes=10 * _GiB)
    assert res.free_vram_bytes() == 14 * _GiB

    lease = res.admit({"incoming-x": 10 * _GiB})
    # incoming-x's claim is outstanding: 14 free - 10 reserved + 10
    # reclaimable (resident-a is unleased and demotable) = 14 available.
    assert res.fits({"query-y": 12 * _GiB}) is True   # 12 + 2 margin == 14
    assert res.fits({"query-y": 13 * _GiB}) is False  # 13 + 2 > 14
    lease.release()
    # Claim gone: 14 free + 10 reclaimable covers it.
    assert res.fits({"query-y": 13 * _GiB}) is True

    # A ref already VRAM-resident asks for nothing.
    assert res.fits({"resident-a": 10 * _GiB}) is True


def test_make_room_excludes_own_refs_but_counts_others() -> None:
    res = _budget_residency(24)
    res.track_vram("resident-a", _FakeModule(), vram_bytes=10 * _GiB)
    lease = res.admit({"incoming-x": 12 * _GiB})
    try:
        # The admitted job's OWN make_room: its reservation is the demand
        # being satisfied — 14 GiB free covers 12 + 2 without any demotion.
        assert res.make_room(12 * _GiB, for_refs=("incoming-x",)) is True
        assert res.tier("resident-a") is Tier.VRAM

        # A COMPETING make_room sees only 14 - 12 = 2 GiB headroom and must
        # demote the idle resident to reach 4 + 2.
        assert res.make_room(4 * _GiB) is True
        assert res.tier("resident-a") is Tier.RAM
    finally:
        lease.release()


def test_track_vram_consumes_the_reservation() -> None:
    res = _budget_residency(24)
    lease = res.admit({"incoming-x": 10 * _GiB})
    try:
        assert res.fits({"query-y": 13 * _GiB}) is False  # claim outstanding
        # The load lands: bytes are now IN the measured pool, claim consumed
        # — no double count (24 - 10 tracked = 14 free; leased entry is not
        # reclaimable; 13 + 2 > 14 stays an honest NO, 12 + 2 fits).
        res.track_vram("incoming-x", _FakeModule(), vram_bytes=10 * _GiB)
        assert res.fits({"query-y": 12 * _GiB}) is True
        assert res.fits({"query-y": 13 * _GiB}) is False
    finally:
        lease.release()


def test_two_leases_on_one_ref_claim_max_not_sum() -> None:
    res = _budget_residency(24)
    l1 = res.admit({"shared-pick": 10 * _GiB})
    l2 = res.admit({"shared-pick": 10 * _GiB})
    # One future load, one claim: 24 - 10 = 14 available -> 12 + 2 fits.
    assert res.fits({"query-y": 12 * _GiB}) is True
    assert res.fits({"query-y": 13 * _GiB}) is False
    l1.release()
    assert res.fits({"query-y": 13 * _GiB}) is False  # l2 still claims 10
    l2.release()
    assert res.fits({"query-y": 13 * _GiB}) is True
    assert res.leased_refs() == []


# ---------------------------------------------------------------------------
# C. Single-flight per instance (pgw#647) — real Executor, real dispatch.
# ---------------------------------------------------------------------------


class _SfIn(msgspec.Struct):
    text: str = "x"


class _SfOut(msgspec.Struct):
    y: str


def _run_two_jobs(ep_cls: type, timeout: float = 30.0) -> List[pb.JobResult]:
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    specs = extract_specs(ep_cls)
    ex = Executor(specs, _send)

    def _results() -> List[pb.JobResult]:
        return [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]

    async def run() -> List[pb.JobResult]:
        for rid in ("r-1", "r-2"):
            await ex.handle_run_job(pb.RunJob(
                request_id=rid, attempt=1, function_name=specs[0].name,
                input_payload=msgspec.msgpack.encode(_SfIn())))
        deadline = asyncio.get_running_loop().time() + timeout
        while len(_results()) < 2:
            if asyncio.get_running_loop().time() > deadline:
                pytest.fail(f"jobs did not finish: {sent}")
            await asyncio.sleep(0.01)
        return _results()

    return asyncio.run(run())


def test_single_flight_serializes_handlers_on_one_instance() -> None:
    """Two concurrent dispatches to the SAME instance must never overlap in
    the handler by default: the instance's graph holds mutable buffers
    (resident LoRA branch), so overlap is corruption, not concurrency. With
    the per-record run gate, overlap is impossible regardless of timing."""
    state = {"active": 0, "max_active": 0}
    guard = threading.Lock()

    @endpoint(resources=Resources())
    class SingleFlightEp:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _SfIn) -> _SfOut:
            with guard:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            try:
                threading.Event().wait(0.15)  # widen any would-be overlap
            finally:
                with guard:
                    state["active"] -= 1
            return _SfOut(y="ok")

    results = _run_two_jobs(SingleFlightEp)
    assert [r.status for r in results] == [pb.JOB_STATUS_OK] * 2
    assert state["max_active"] == 1, "handlers overlapped on one instance"


def test_reentrant_opt_in_allows_true_overlap() -> None:
    """``reentrant=True`` classes opt out of the gate: each handler waits
    for the OTHER to enter before returning — completing at all PROVES both
    were in flight simultaneously (under single-flight this would time out)."""
    entered = [threading.Event(), threading.Event()]
    order = {"n": 0}
    guard = threading.Lock()

    @endpoint(resources=Resources(), reentrant=True)
    class ReentrantEp:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _SfIn) -> _SfOut:
            with guard:
                me = order["n"]
                order["n"] += 1
            entered[me].set()
            other = entered[1 - me]
            assert other.wait(20.0), "no overlap: sibling never entered"
            return _SfOut(y="ok")

    results = _run_two_jobs(ReentrantEp)
    assert [r.status for r in results] == [pb.JOB_STATUS_OK] * 2


def test_reentrant_is_a_class_only_declaration() -> None:
    with pytest.raises(ValueError, match="class endpoints only"):
        @endpoint(reentrant=True)
        def loose_fn(ctx, payload: _SfIn) -> _SfOut:  # pragma: no cover
            return _SfOut(y="no")

    @endpoint(resources=Resources(), reentrant=True)
    class Marked:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _SfIn) -> _SfOut:  # pragma: no cover
            return _SfOut(y="ok")

    spec = extract_specs(Marked)[0]
    assert spec.reentrant is True

    @endpoint(resources=Resources())
    class Unmarked:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _SfIn) -> _SfOut:  # pragma: no cover
            return _SfOut(y="ok")

    assert extract_specs(Unmarked)[0].reentrant is False


# ---------------------------------------------------------------------------
# D. Activation-aware admission (pgw#652).
# ---------------------------------------------------------------------------


def test_activation_is_learned_only_from_measurement() -> None:
    """Nothing is reserved on the strength of a declaration: an unmeasured
    function claims 0. The hint then jumps UP instantly (under-reserving is
    what OOMs) and decays gradually, so one outlier does not tax residency
    depth forever."""
    res = _budget_residency(24)
    assert res.activation_hint("sdxl.generate") == 0

    assert res.record_activation("sdxl.generate", 3 * _GiB) == 3 * _GiB
    # A bigger observation is adopted immediately, in full.
    assert res.record_activation("sdxl.generate", 9 * _GiB) == 9 * _GiB
    # Smaller ones bleed the high-water down instead of dropping to it.
    after_one = res.record_activation("sdxl.generate", 1 * _GiB)
    assert after_one == 9 * _GiB - (9 * _GiB) // 8  # one 12.5% bleed, not a drop
    for _ in range(60):
        res.record_activation("sdxl.generate", 1 * _GiB)
    assert res.activation_hint("sdxl.generate") == 1 * _GiB

    # An unnamed key is not a silent global bucket.
    assert res.record_activation("", 5 * _GiB) == 0
    assert res.activation_hint("") == 0


def test_activation_claims_sum_across_concurrent_leases() -> None:
    """Weight claims MAX across leases (one future load serves both jobs);
    activation claims SUM (each concurrent request allocates its own). The
    live pgw#652 hazard: interleaving OOMs the moment it starts working."""
    res = _budget_residency(24)
    res.track_vram("weights", _FakeModule(), vram_bytes=12 * _GiB)
    assert res.free_vram_bytes() == 12 * _GiB

    # Two concurrent requests on the same resident weights: no weight claim
    # at all (already booked), but 3 GiB of activation each.
    l1 = res.admit({"weights": 12 * _GiB}, activation_bytes=3 * _GiB)
    l2 = res.admit({"weights": 12 * _GiB}, activation_bytes=3 * _GiB)
    try:
        # 12 free - 6 claimed = 6 available (the resident weights are leased,
        # hence not reclaimable). 4 + 2 margin fits; 5 + 2 does not.
        assert res.fits({"other": 4 * _GiB}) is True
        assert res.fits({"other": 5 * _GiB}) is False
        # A THIRD request must count its own activation alongside its weights.
        assert res.fits({"other": 4 * _GiB}, activation_bytes=1 * _GiB) is False
    finally:
        l1.release()
        l2.release()
    assert res.fits({"other": 5 * _GiB}) is True


def test_make_room_protects_other_requests_activation_not_its_own() -> None:
    """A load must not eat the workspace a running request is about to use —
    but it must not double-count its OWN activation either (already allocated,
    hence already visible in the measured free bytes)."""
    res = _budget_residency(24)
    res.track_vram("idle-victim", _FakeModule(), vram_bytes=6 * _GiB)
    running = res.admit({"hot": 0}, activation_bytes=4 * _GiB)
    loader = res.admit({"incoming": 0}, activation_bytes=4 * _GiB)
    try:
        # 18 GiB free; the OTHER request's 4 GiB workspace is off limits, the
        # loader's own is not => 14 GiB usable, so 11 + 2 needs no demotion...
        assert res.make_room(11 * _GiB, for_refs=("incoming",)) is True
        assert res.tier("idle-victim") is Tier.VRAM
        # ...but 13 + 2 does, and the unleased idle entry is the victim.
        assert res.make_room(13 * _GiB, for_refs=("incoming",)) is True
        assert res.tier("idle-victim") is Tier.RAM
    finally:
        running.release()
        loader.release()


def test_negative_activation_delta_is_not_evidence() -> None:
    # Weights evicted mid-job can put the peak BELOW the baseline. That is a
    # measurement artifact, not a 0-byte workspace: clamp, never learn junk.
    res = _budget_residency(24)
    assert res.record_activation("sdxl.generate", -5 * _GiB) == 0
    assert res.activation_hint("sdxl.generate") == 0


class _FakeVram:
    """torch.cuda boundary fake: a card whose allocator reports a baseline at
    handler start and a higher peak once the handler has run."""

    def __init__(self, baseline: int, peak: int) -> None:
        self.baseline, self.peak = baseline, peak

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def current_device(self) -> int:
        return 0

    def mem_get_info(self, device: int = 0) -> Tuple[int, int]:
        return 24 * _GiB - self.baseline, 24 * _GiB

    def memory_allocated(self, device: Any = None) -> int:
        return self.baseline

    def max_memory_allocated(self, device: Any = None) -> int:
        return self.peak

    def reset_peak_memory_stats(self, device: Any = None) -> None:
        return None


def test_executor_learns_activation_from_the_measured_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The REAL dispatch path teaches the store: run a job, and the function's
    activation hint becomes (measured peak - what was already allocated when
    it took the GPU). No test re-implements that subtraction — the executor
    does it and the assertion reads the store."""
    torch = pytest.importorskip("torch")
    fake = _FakeVram(baseline=6 * _GiB, peak=9 * _GiB)
    for name in ("is_available", "device_count", "current_device", "mem_get_info",
                 "memory_allocated", "max_memory_allocated", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, getattr(fake, name))

    @endpoint(resources=Resources(gpu=True))
    class ActivationEp:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _SfIn) -> _SfOut:
            return _SfOut(y="ok")

    specs = extract_specs(ActivationEp)
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor(specs, _send)
    res = ex.store.residency
    assert res.activation_hint(specs[0].name) == 0

    async def run() -> None:
        await ex.handle_run_job(pb.RunJob(
            request_id="act-1", attempt=1, function_name=specs[0].name,
            input_payload=msgspec.msgpack.encode(_SfIn())))
        deadline = asyncio.get_running_loop().time() + 30.0
        while not any(m.WhichOneof("msg") == "job_result" for m in sent):
            if asyncio.get_running_loop().time() > deadline:
                pytest.fail(f"job did not finish: {sent}")
            await asyncio.sleep(0.01)

    asyncio.run(run())
    result = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"][0]
    assert result.status == pb.JOB_STATUS_OK
    # 9 GiB peak - 6 GiB already allocated = a 3 GiB workspace, learned.
    assert res.activation_hint(specs[0].name) == 3 * _GiB
