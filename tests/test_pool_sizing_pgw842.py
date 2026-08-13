"""pgw#842: the entry pool's width must be EXPLAINABLE and MONOTONE.

Two real L4 mints of the same 72-entry sdxl regional cell, back to back, are
the specimen this file is written against:

    attempt ten  (0.86.0, 16 vcpu / 62 GB): K=5, compile_s 1314.94, wall 347.94
    attempt eleven (0.89.0, 21 vcpu / 83 GB): K=3, compile_s 1327.23, wall 554.78

Identical compile work (+0.9 %), 59 % more wall, on a BIGGER host — the whole
regression is the width. And nothing hub-side recorded why: `entry_workers`
was the only pool number that ever reached a hub row, so the binding
constraint on those two pods is unrecoverable and the pods are gone.

So the tests here are of two kinds:

* **the inputs cannot lie** — the two readings that are not monotone in the
  box (a cgroup RAM headroom that shrinks as the pod does I/O; a free-VRAM
  sample taken while the tenant holds an activation set) are driven through
  the REAL production functions against synthetic hosts;
* **the decision cannot be silent** — a real pool run's real width and real
  ledger have to come out the other end of the real parent-side relay as a
  hub event that names the binding constraint.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import activity as activity_mod
from gen_worker import aot_compile_pool as pool
from gen_worker import aot_mint, mint_delegate, mint_process
from gen_worker.pb import worker_scheduler_pb2 as pb

_GIB = 1024 ** 3


# ---------------------------------------------------------------------------
# The RAM reading: a bound that moved with the pod's history, not its size
# ---------------------------------------------------------------------------


def _cgroup_v2(
    tmp_path: Path, *, limit: int, current: int, inactive_file: int = 0,
) -> Path:
    root = tmp_path / "cgroup"
    root.mkdir(parents=True, exist_ok=True)
    (root / "memory.max").write_text(f"{limit}\n")
    (root / "memory.current").write_text(f"{current}\n")
    (root / "memory.stat").write_text(
        f"anon {max(0, current - inactive_file)}\n"
        f"file {inactive_file}\n"
        f"inactive_file {inactive_file}\n"
        f"slab_reclaimable 0\n")
    return root


def _meminfo(tmp_path: Path, available_kb: int) -> Path:
    path = tmp_path / "meminfo"
    path.write_text(
        "MemTotal:       131072000 kB\n"
        f"MemAvailable:   {available_kb} kB\n")
    return path


def test_page_cache_is_not_charged_against_the_pool(tmp_path: Path) -> None:
    """`memory.current` counts page cache, and a mint generates GBs of it —
    weights, the toolchain the seal hashes, every staged program.

    Sizing the pool on ``memory.max - memory.current`` therefore narrows it in
    proportion to how much I/O the pod has ALREADY done: a bound that tracks
    history instead of the box. Reclaimable file pages come back the moment
    anything needs them, so the working set is what the pool must respect.
    """
    limit = 62 * _GIB
    # 50 GiB charged, 40 GiB of it reclaimable page cache: the working set is
    # 10 GiB and the pool may have the other 52.
    root = _cgroup_v2(
        tmp_path, limit=limit, current=50 * _GIB, inactive_file=40 * _GIB)
    facts = pool.memory_facts(
        meminfo=_meminfo(tmp_path, 100 * 1024 * 1024), cgroup_root=root)

    assert facts.basis == "cgroup"
    assert facts.cgroup_reclaimable_bytes == 40 * _GIB
    assert facts.available_bytes == 52 * _GIB, (
        f"{facts} — the naive `max - current` answer is 12 GiB, which would "
        f"size this pool at K=2 on a 62 GB pod for no reason but its own "
        f"read history")


def test_the_ram_bound_is_monotone_in_the_pods_ram(tmp_path: Path) -> None:
    """Same workload, bigger cgroup limit, never a narrower answer."""
    seen: List[Tuple[int, int]] = []
    for gb in (32, 62, 83, 128):
        root = _cgroup_v2(
            tmp_path / f"h{gb}", limit=gb * _GIB,
            current=int(0.8 * gb) * _GIB, inactive_file=int(0.6 * gb) * _GIB)
        avail = pool.memory_facts(
            meminfo=_meminfo(tmp_path, 200 * 1024 * 1024),
            cgroup_root=root).available_bytes
        seen.append((gb, avail))
    assert seen == sorted(seen), seen


# ---------------------------------------------------------------------------
# The VRAM reading — DELETED WITH ITS BOUND
# ---------------------------------------------------------------------------
#
# pgw#842's third bound divided free VRAM by a per-entry device ask, and this
# section covered the READING behind it: a single `mem_get_info` taken beside
# a live tenant forward reads that forward's activation set as gone, so the
# figure was sampled over a short window and every sample kept. Correct, and
# now moot — §4.33 deleted the bound, so there is no free-VRAM figure to
# sample and `device_facts` / `DeviceFacts` / `CardCensus` are gone with it.
#
# What pgw#842 is actually ABOUT survives in full below: a K nobody can
# explain is a K nobody can fix, and a bigger host may never yield a narrower
# pool. Both properties now hold over the two bounds that remain.


# ---------------------------------------------------------------------------
# The width itself: monotone in the box, on the two real hosts' shapes
# ---------------------------------------------------------------------------


_HOSTS = {
    # The two pods this issue was filed off, plus the extremes around them.
    "attempt-ten": (16, 62),
    "attempt-eleven": (21, 83),
    "narrow": (8, 32),
    "fat": (48, 128),
}


def test_a_bigger_host_never_yields_a_narrower_pool(
    tmp_path: Path,
) -> None:
    """The property the two mints violated, stated as a law.

    Both hosts' RAM goes through the REAL cgroup reading (same workload, same
    page cache, different limit) rather than being handed in as a number, so
    a regression in `memory_facts` fails here too.
    """
    widths: List[Tuple[Tuple[int, int], int]] = []
    for name, (vcpus, ram_gb) in sorted(_HOSTS.items(), key=lambda kv: kv[1]):
        root = _cgroup_v2(
            tmp_path / name, limit=ram_gb * _GIB,
            # Identical work on both: 12 GiB resident, 9 GiB of page cache.
            current=21 * _GIB, inactive_file=9 * _GIB)
        avail = pool.memory_facts(
            meminfo=_meminfo(tmp_path, 200 * 1024 * 1024),
            cgroup_root=root).available_bytes
        width = pool.entry_workers(
            72, vcpus=vcpus, available_bytes=avail,
            device_lock=True)
        widths.append(((vcpus, ram_gb), width.workers))
    assert [k for _, k in widths] == sorted(k for _, k in widths), widths
    # And specifically: the pair that started this.
    ten = dict(widths)[_HOSTS["attempt-ten"]]
    eleven = dict(widths)[_HOSTS["attempt-eleven"]]
    assert eleven >= ten, (
        f"21 vcpu / 83 GB gave K={eleven} where 16 vcpu / 62 GB gave K={ten} "
        f"— that is pgw#842 verbatim")


def test_the_width_names_its_binding_constraint_and_its_readings() -> None:
    """A K nobody can explain is a K nobody can fix."""
    # 13 GiB available - 4 GiB tenant reserve = 9 GiB / 3 GiB per entry -> 3.
    width = pool.entry_workers(
        72, vcpus=21, available_bytes=13 * _GIB, device_lock=True)
    facts = width.facts()
    assert width.binding == "host-memory", width.reason
    assert facts["underwidth"] == 5, facts
    # Every reading that fed a bound, and what KIND of reading it was.
    for key in ("binding", "underwidth", "ceiling", "cpu_basis", "mem_basis",
                "per_entry_rss_basis",
                "cgroup_reclaimable_bytes", "host_available_bytes",
                "os_cpu_count", "affinity_cpus", "quota_cores"):
        assert key in facts, f"{key} missing from {sorted(facts)}"
    # pgw#877's three DEVICE provenances died with the axis. The
    # one per-entry footprint left still says whether it was measured or
    # defaulted, because a default must never read like a measurement.
    common: Dict[str, Any] = dict(
        entries=72, vcpus=21, available_bytes=60 * _GIB, device_lock=True)
    assert pool.entry_workers(**common).per_entry_rss_basis == "default"
    assert pool.entry_workers(
        peak_rss_bytes=2 * _GIB, **common).per_entry_rss_basis == "measured"


def test_the_advertised_cores_are_not_the_ones_the_pool_believes() -> None:
    """`cpu_facts` records all three readings, so an advertised-vs-enforced
    gap (RunPod's `host_vcpus` vs the cgroup quota) is visible in the record
    instead of being the difference between two pods nobody kept."""
    facts = pool.cpu_facts()
    assert facts.vcpus >= 1 and facts.basis in {
        "quota", "affinity", "cpu_count"}
    assert facts.vcpus == min(
        [facts.os_cpu_count, facts.affinity_cpus]
        + ([max(1, int(facts.quota_cores + 0.5))]
           if facts.quota_cores > 0 else []))


# ---------------------------------------------------------------------------
# The decision reaches the hub — real pool, real ledger, real relay
# ---------------------------------------------------------------------------


torch = pytest.importorskip("torch")

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

_HIDDEN = 96


def _program(seed: int) -> Any:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(_HIDDEN, _HIDDEN)

        def forward(self, x: Any) -> Any:
            return torch.tanh(self.a(x)) * (1.0 + seed)

    return torch.export.export(Tiny(), (torch.randn(4, _HIDDEN),))


def _relayed(table: Dict[str, Any]) -> List[pb.ActivityUpdate]:
    """The REAL parent-side relay: the child's report in, hub messages out."""
    sent: List[pb.ActivityUpdate] = []
    loop = asyncio.new_event_loop()

    async def _send(msg: pb.WorkerMessage) -> None:
        if msg.WhichOneof("msg") == "activity_update":
            sent.append(msg.activity_update)

    outcome = mint_process.MintOutcome(
        status=mint_process.MINTED, elapsed_s=12.5,
        report=mint_process.MintReport(
            status=mint_process.MINTED, elapsed_s=12.5, mint_phases=table))
    try:
        activity_mod.bind_sink(_send, loop)
        mint_delegate._emit_aot_phases(
            outcome, family="sdxl", execution_lane="w8a8-lora64")
        loop.run_until_complete(asyncio.sleep(0.05))
    finally:
        activity_mod.bind_sink(None, None)
        loop.close()
    return sent


def test_a_real_pools_width_and_ledger_land_hub_side(tmp_path: Path) -> None:
    """End to end over the path that was dark.

    The pool's own pgw#830 ledger event is emitted from the mint CHILD, which
    holds no orchestrator session — so it has never reached a hub row, and
    neither had the width block, which `_mint_phase_table` has always built
    and `emit_phase_events` never emitted. Both now ride the parent's relay.
    """
    entries = [(f"unet/adapter=true/dim={i}", _program(i)) for i in range(2)]
    width = pool.entry_workers(
        len(entries), vcpus=16, available_bytes=64 * _GIB,
        device_lock=True, limit=2)
    assert width.workers == 2, width.reason
    box = pool.EntryCompilePool(
        tmp_path / "pool", width=width,
        inductor_configs={"compile_threads": 2},
        cache_dir=str(tmp_path / "cache"))
    out = box.compile(entries)
    assert set(out) == {name for name, _ in entries}

    ledger = {
        **box.ledger.facts(),
        "peak_concurrency": box.peak_concurrency,
        "peak_child_rss_bytes": box.peak_rss_bytes,
    }
    table = aot_mint._mint_phase_table([], {"total_s": 12.5}, None, width,
                                       ledger)
    updates = _relayed(table)

    pool_events = [
        u for u in updates
        if u.kind == aot_mint.MINT_PHASES_KIND
        and u.phase == aot_mint.POOL_PHASE]
    assert len(pool_events) == 1, (
        f"the width decision did not reach the hub: "
        f"{[(u.kind, u.phase) for u in updates]}")
    detail = pool_events[0].detail
    assert f"entry_workers={width.workers}" in detail
    assert f"binding={width.binding}" in detail
    for key in ("cpu_workers", "mem_workers",
                "pool_efficiency", "pool_idle_s", "peak_concurrency",
                "cpu_basis", "mem_basis"):
        assert key in detail, f"{key} missing from the pool event: {detail}"
    assert pool_events[0].duration_ms > 0, (
        "the pool's wall clock is the span this event measures")
    # Observed, not intended: the pool really did overlap.
    assert box.peak_concurrency == 2, box.peak_concurrency


def test_a_narrow_pool_says_so_in_the_first_line() -> None:
    """The standing rule: no silent decisions. A pool held below what the
    cell could use names the shortfall and its cause up front — attempt
    eleven's 59 % was invisible precisely because nothing did."""
    width = pool.entry_workers(
        72, vcpus=21, available_bytes=13 * _GIB, device_lock=True)
    assert width.workers == 3 and width.underwidth == 5, width.reason
    table = aot_mint._mint_phase_table([], {"total_s": 554.78}, None, width,
                                       {"pool_wall_s": 453.0})
    updates = _relayed(table)
    detail = next(
        u.detail for u in updates if u.phase == aot_mint.POOL_PHASE)
    assert "underwidth=5" in detail and "held by host-memory" in detail, detail
