"""pgw#1607: the checkpoint juggle's card-side legs.

Run inside a coordinator-arbitrated GPU window (the box's RTX 4070), under
the micro-rig carve-out and nothing more: the checkpoints are GENERATED on
this box (random-init toy trees, < 500 MB total, nothing downloaded), the
device budget is enforced through the arena, the compile phase is STUBBED —
the zero-re-arm proof here is pointer/forward-identity, and the
real-compiled arm runs pod-side (phase 4).

    nice -n 19 .venv/bin/python benchmarks/checkpoint_juggle_pgw1607.py --arms ABCDEF

* **A — distinct outputs.** Same input, same seed: serving ck0 -> y0, switch
  ck1 -> y1, switch back ck0 -> y0'. y0 != y1 (the switch actually switched)
  and y0 == y0' BITWISE (a round trip restores the checkpoint exactly).
* **B — zero re-arms.** Every managed parameter's data_ptr and every
  module's forward identity are snapshotted before the first switch and
  asserted UNCHANGED after every switch; `juggler.rearms == 0`.
* **C — integrity.** After each switch, every backed region is read back
  D2H and its digest compared against the ingest digest banked for the
  serving checkpoint. Content-level, this lane's half of the va#12 split.
* **D — the franken fence (RED leg).** A refill is KILLED mid-switch by
  fault injection at the copy seam. The region must go INVALID, serving
  must REFUSE under both identities, and an unpatched re-switch must
  recover to green digests.
* **E — teardown.** release + `committed == mapped == 0`.
* **F — the numbers.** Warm switch wall vs the transfer bound (a pinned
  H2D floor measured on THIS card in the same window), serving-switch ~ 0,
  disk-cold switch, and a 200-request zipf juggle vs the single-checkpoint
  baseline.

Discipline: fleet line asserted first, load gate at 24, `uptime` recorded,
one heavy thing at a time, verdict JSON written before exit.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

MIB = 1 << 20
GIB = 1 << 30

#: Micro-rig bound: generated weights, total under 500 MB.
CHECKPOINTS = 4
SEED = 1607

SCRATCH = Path(
    os.environ.get(
        "PGW1607_SCRATCH",
        os.path.expanduser("~/.cache/cozy/pgw1607"),
    )
)


def loud(msg: str) -> None:
    print(f"[pgw1607] {msg}", flush=True)


def load_gate() -> None:
    load1 = os.getloadavg()[0]
    if load1 > 24:
        raise SystemExit(f"load gate: 1-min load {load1:.1f} > 24; refusing to start")
    loud(f"load gate ok (1-min {load1:.1f}); uptime: "
         f"{subprocess.run(['uptime'], capture_output=True, text=True).stdout.strip()}")


# ---------------------------------------------------------------------------
# The toy lane: a real tree with stream-sized leaves, generated checkpoints
# ---------------------------------------------------------------------------


def build_template(torch: Any, nn: Any) -> Any:
    class ToyLane(nn.Module):
        """~118 MB fp32: three stream-sized leaves + a core, real forward."""

        def __init__(self) -> None:
            super().__init__()
            self.blk0 = nn.Linear(2048, 2048, bias=False)  # 16 MiB
            self.blk1 = nn.Linear(2048, 2048, bias=False)  # 16 MiB
            self.blk2 = nn.Linear(2048, 4096, bias=False)  # 32 MiB
            self.blk3 = nn.Linear(4096, 2048, bias=False)  # 32 MiB
            self.mid = nn.Linear(1024, 1024, bias=False)  # 4 MiB
            self.head = nn.Linear(2048, 64, bias=False)  # 0.5 MiB -> core
            self.norm = nn.LayerNorm(2048)

        def forward(self, x: Any) -> Any:
            h = torch.tanh(self.blk0(x))
            h = torch.tanh(self.blk1(h))
            h = torch.tanh(self.blk3(torch.tanh(self.blk2(h))))
            return self.head(self.norm(h))

    return ToyLane()


def generate_checkpoints(torch: Any, template: Any, directory: Path) -> List[Path]:
    """CHECKPOINTS distinct random inits of the SAME architecture, on disk.

    Real safetensors files via the test fixture's writer (one header
    implementation on each side of the seam).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from test_checkpoint_juggle import write_safetensors  # noqa: E402

    out = []
    for i in range(CHECKPOINTS):
        g = torch.Generator().manual_seed(SEED + i)
        state = {}
        for key, tensor in template.state_dict().items():
            state[key] = torch.randn(tensor.shape, generator=g, dtype=tensor.dtype)
        ck_dir = directory / f"ck{i}"
        ck_dir.mkdir(parents=True, exist_ok=True)
        write_safetensors(ck_dir / "weights.safetensors", state)
        out.append(ck_dir)
    total = sum(
        f.stat().st_size for d in out for f in d.glob("*.safetensors")
    )
    assert total < 500 * MIB, f"micro-rig bound violated: {total} bytes generated"
    loud(f"generated {CHECKPOINTS} checkpoints, {total / MIB:.1f} MiB total (< 500 MiB bound)")
    return out


# ---------------------------------------------------------------------------
# Rig assembly
# ---------------------------------------------------------------------------


def make_rig(torch: Any, nn: Any, ck_dirs: List[Path], *, budget_bytes: int):
    from gen_worker.models.arena_residency import ArenaResidency
    from gen_worker.models.checkpoint_juggle import CheckpointJuggler, read_manifest

    template = build_template(torch, nn)
    manifests = {f"ck{i}": read_manifest(d) for i, d in enumerate(ck_dirs)}
    # Serve ck0: load its bytes into the live tree, move to the card.
    state = {}
    m0 = manifests["ck0"]
    for key, src in m0.items():
        t = torch.empty(src.shape, dtype=template.state_dict()[key].dtype)
        with open(src.path, "rb") as fh:
            fh.seek(src.offset)
            fh.readinto(memoryview(t.view(torch.uint8).view(-1).numpy()))
        state[key] = t
    template.load_state_dict(state)
    template.to("cuda").eval()

    triples = {k: (s.path, s.offset, s.length) for k, s in m0.items()}
    residency = ArenaResidency(
        [("root", template)],
        device="cuda",
        budget_bytes=budget_bytes,
        triples=triples,
        min_stream_bytes=1 * MIB,
    )
    residency.engage()
    juggler = CheckpointJuggler(residency, "ck0", m0)
    for i in range(1, CHECKPOINTS):
        juggler.admit(f"ck{i}", manifests[f"ck{i}"])
    return template, residency, juggler


def forward_once(torch: Any, juggler: Any, template: Any, x: Any) -> Any:
    juggler.assert_servable()
    with torch.no_grad():
        return template(x).clone()


def backed_region_digest(torch: Any, residency: Any, region: Any) -> str:
    """D2H readback digest of a backed region's WEIGHT bytes (slot-bounded)."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    for slot in region.slots:
        view = residency._view(slot)
        host = view.detach().to("cpu")
        h.update(host.view(torch.uint8).view(-1).numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------


def leg_a_distinct_outputs(torch: Any, juggler: Any, template: Any) -> Dict[str, Any]:
    x = torch.zeros(8, 2048, device="cuda")
    x[:, :4] = 1.0
    y0 = forward_once(torch, juggler, template, x)
    juggler.switch_to("ck1")
    y1 = forward_once(torch, juggler, template, x)
    juggler.switch_to("ck0")
    y0_again = forward_once(torch, juggler, template, x)
    distinct = not torch.equal(y0, y1)
    restored = torch.equal(y0, y0_again)
    assert distinct, "outputs identical across a switch — the swap did not swap"
    assert restored, "round trip did not restore ck0 bitwise"
    loud("leg A GREEN: outputs are checkpoint-distinct and the round trip is bitwise")
    return {"distinct": distinct, "restored_bitwise": restored}


def leg_b_zero_rearms(torch: Any, juggler: Any, template: Any) -> Dict[str, Any]:
    residency = juggler.residency
    before_ptrs = {}
    before_fwd = {}
    for region in residency.layout.regions:
        if residency.is_resident(region.name):
            for slot in region.slots:
                before_ptrs[(slot.leaf, slot.attr)] = int(
                    residency._view(slot).data_ptr()
                )
    for name, module in template.named_modules():
        before_fwd[name] = id(module.forward.__func__ if hasattr(module.forward, "__func__") else module.forward)
    x = torch.randn(4, 2048, generator=torch.Generator().manual_seed(7)).cuda()
    for target in ("ck1", "ck2", "ck3", "ck0"):
        juggler.switch_to(target)
        forward_once(torch, juggler, template, x)
    moved = []
    for region in residency.layout.regions:
        if residency.is_resident(region.name):
            for slot in region.slots:
                ptr = int(residency._view(slot).data_ptr())
                if before_ptrs.get((slot.leaf, slot.attr)) != ptr:
                    moved.append(f"{slot.leaf}.{slot.attr}")
    rebound = [
        name for name, module in template.named_modules()
        if before_fwd.get(name) != id(module.forward.__func__ if hasattr(module.forward, "__func__") else module.forward)
    ]
    assert not moved, f"data_ptr moved across switches: {moved[:5]}"
    assert not rebound, f"forward identity changed: {rebound[:5]}"
    assert juggler.rearms == 0, f"rearms counted: {juggler.rearms}"
    loud(f"leg B GREEN: {len(before_ptrs)} pointers fixed, 0 re-arms across 4 switches")
    return {"pointers_checked": len(before_ptrs), "rearms": juggler.rearms}


def leg_c_integrity(torch: Any, juggler: Any) -> Dict[str, Any]:
    residency = juggler.residency
    checked = 0
    for target in ("ck2", "ck3"):
        juggler.switch_to(target)
        image = juggler.catalog.warm(target)
        assert image is not None, "integrity leg needs the warm tier"
        for region in residency.layout.regions:
            if not residency.is_resident(region.name):
                continue
            got = backed_region_digest(torch, residency, region)
            want = image.region_digests[region.name]
            assert got == want, (
                f"region {region.name!r}: VRAM bytes digest {got} != ingest {want}"
            )
            checked += 1
    juggler.switch_to("ck0")
    loud(f"leg C GREEN: {checked} backed-region digests match ingest across 2 switches")
    return {"regions_checked": checked}


def leg_d_franken_fence(torch: Any, juggler: Any, template: Any) -> Dict[str, Any]:
    """Kill a refill mid-switch; the fence must refuse; recovery must serve."""
    from gen_worker.models.checkpoint_juggle import RegionInvalid

    real = juggler._refill_backed
    calls = {"n": 0}

    def dying_refill(region: Any, image: Any, manifest: Any, stream: Any) -> int:
        calls["n"] += 1
        if calls["n"] == 2:
            # First region lands (state: valid@target); the second dies with
            # SOME of its bytes possibly moved — the franken shape.
            raise RuntimeError("injected: H2D died mid-transfer")
        return real(region, image, manifest, stream)

    juggler._refill_backed = dying_refill  # fault injection at the copy seam
    died = False
    try:
        juggler.switch_to("ck1")
    except RuntimeError as exc:
        died = "injected" in str(exc)
    finally:
        juggler._refill_backed = real
    assert died, "the injection did not fire; the leg proved nothing"

    refused = {"as_old": False, "as_new": False}
    try:
        juggler.assert_servable()
    except RegionInvalid:
        refused["as_old"] = True
    juggler.serving_id = "ck1"  # even claiming the new identity must refuse
    try:
        juggler.assert_servable()
    except RegionInvalid:
        refused["as_new"] = True
    juggler.serving_id = "ck0"
    assert refused["as_old"] and refused["as_new"], (
        f"franken state served: {refused}"
    )

    # Recovery: an unpatched re-switch is idempotent and lands green.
    juggler.switch_to("ck1")
    juggler.assert_servable()
    image = juggler.catalog.warm("ck1")
    residency = juggler.residency
    for region in residency.layout.regions:
        if residency.is_resident(region.name):
            assert backed_region_digest(torch, residency, region) == \
                image.region_digests[region.name]
    juggler.switch_to("ck0")
    loud("leg D GREEN (red arm fired): mid-refill kill -> INVALID -> refused "
         "under BOTH identities -> idempotent re-switch recovered to green digests")
    return {"refused": refused, "recovered": True}


def leg_e_teardown(torch: Any, residency: Any) -> Dict[str, Any]:
    stats_before = residency.stats()
    residency.release()
    stats = dict(residency.arena.stats())
    committed, mapped = int(stats["committed_bytes"]), int(stats["mapped_bytes"])
    assert committed == 0 and mapped == 0, (
        f"teardown discipline violated: committed={committed} mapped={mapped}"
    )
    loud(f"leg E GREEN: committed==mapped==0 at teardown "
         f"(peak backed regions {stats_before.get('backed_regions')})")
    return {"committed": committed, "mapped": mapped}


def measure_h2d_floor(torch: Any, nbytes: int) -> float:
    """Pinned H2D GB/s on THIS card, measured in the same window."""
    src = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    for _ in range(2):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    del src, dst
    torch.cuda.empty_cache()
    return nbytes / best / 1e9


def leg_f_numbers(torch: Any, juggler: Any, template: Any) -> Dict[str, Any]:
    residency = juggler.residency
    lane_bytes = sum(
        r.span for r in residency.layout.regions if residency.is_resident(r.name)
    )
    floor_gbps = measure_h2d_floor(torch, max(64 * MIB, lane_bytes))
    x = torch.zeros(4, 2048, device="cuda")

    # Warm switches, round-robin, median of ~11.
    walls: List[float] = []
    seq = [f"ck{i % CHECKPOINTS}" for i in range(12)]
    for target in seq:
        if target == juggler.serving_id:
            continue
        report = juggler.switch_to(target)
        walls.append(report.wall_s)
    warm_median = statistics.median(walls)
    warm_gbps = (lane_bytes / warm_median / 1e9) if warm_median else 0.0

    # Serving no-op.
    noop = juggler.switch_to(juggler.serving_id)

    # Cold, both shapes. (1) evicted image, switch rebuilds it: disk -> image
    # -> VRAM, the D5 cold path. (2) hysteresis-cold: image refused, the
    # switch streams disk-direct into the arena.
    cold_target = "ck1" if juggler.serving_id != "ck1" else "ck2"
    juggler.catalog.evict(cold_target)
    t0 = time.perf_counter()
    rebuild_report = juggler.switch_to(cold_target)
    cold_rebuild_wall = time.perf_counter() - t0
    assert rebuild_report.tier == "host-warm", rebuild_report.tier

    direct_target = "ck2" if cold_target != "ck2" else "ck3"
    juggler.catalog.evict(direct_target)
    juggler.catalog._evicted_epoch[direct_target] = juggler.catalog.pressure_epoch
    t0 = time.perf_counter()
    direct_report = juggler.switch_to(direct_target)
    cold_direct_wall = time.perf_counter() - t0
    assert direct_report.tier == "disk-cold", direct_report.tier
    del juggler.catalog._evicted_epoch[direct_target]

    # Zipf-ish juggle vs single-checkpoint baseline, 200 requests each.
    import random

    rng = random.Random(SEED)
    ids = [f"ck{i}" for i in range(CHECKPOINTS)]
    weights = [1.0 / (i + 1) ** 1.1 for i in range(CHECKPOINTS)]
    t0 = time.perf_counter()
    for _ in range(200):
        forward_once(torch, juggler, template, x)
    torch.cuda.synchronize()
    baseline = time.perf_counter() - t0
    t0 = time.perf_counter()
    switches_before = juggler.switches
    for _ in range(200):
        target = rng.choices(ids, weights)[0]
        if target != juggler.serving_id:
            juggler.switch_to(target)
        forward_once(torch, juggler, template, x)
    torch.cuda.synchronize()
    juggled = time.perf_counter() - t0
    zipf_switches = juggler.switches - switches_before

    out = {
        "lane_backed_bytes": lane_bytes,
        "h2d_floor_gbps": round(floor_gbps, 2),
        "warm_switch_median_s": round(warm_median, 4),
        "warm_switch_gbps": round(warm_gbps, 2),
        "warm_vs_floor": round(warm_gbps / floor_gbps, 3) if floor_gbps else None,
        "serving_noop_s": round(noop.wall_s, 6),
        "cold_rebuild_switch_s": round(cold_rebuild_wall, 4),
        "cold_disk_direct_switch_s": round(cold_direct_wall, 4),
        "zipf_200req_s": round(juggled, 3),
        "zipf_switches": zipf_switches,
        "single_ck_200req_s": round(baseline, 3),
        "zipf_overhead": round(juggled / baseline, 3) if baseline else None,
    }
    loud(f"leg F: {json.dumps(out)}")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="ABCDEF")
    parser.add_argument("--budget-mib", type=int, default=192,
                        help="arena VRAM budget; the whole rig stays under the "
                             "micro-rig 4 GiB device bound")
    parser.add_argument("--out", default=str(SCRATCH / "verdict.json"))
    args = parser.parse_args()

    from gen_worker.rigcheck import assert_fleet_line

    assert_fleet_line("pgw#1607 checkpoint-juggle rig")
    load_gate()
    guard = os.environ.get("GEN_WORKER_HOST_MOVE_GUARD", "1")
    if guard == "0":
        raise SystemExit("GEN_WORKER_HOST_MOVE_GUARD=0: refusing to run")

    import torch
    import torch.nn as nn

    assert torch.cuda.is_available(), "no CUDA device; this is the card-side rig"
    budget = args.budget_mib * MIB
    assert budget <= 4 * GIB, "micro-rig device bound is 4 GiB"

    SCRATCH.mkdir(parents=True, exist_ok=True)
    ck_dirs = generate_checkpoints(torch, build_template(torch, nn), SCRATCH / "cks")
    template, residency, juggler = make_rig(torch, nn, ck_dirs, budget_bytes=budget)
    loud(f"rig up: layout {residency.layout.virtual_bytes / MIB:.1f} MiB virtual, "
         f"budget {args.budget_mib} MiB, plan resident="
         f"{len(residency.plan.all_resident)} streamed={len(residency.plan.streamed)}")

    verdict: Dict[str, Any] = {
        "issue": "pgw#1607", "when": time.strftime("%F %T"),
        "budget_mib": args.budget_mib, "arms": args.arms, "legs": {},
    }
    failed = False
    try:
        if "A" in args.arms:
            verdict["legs"]["A"] = leg_a_distinct_outputs(torch, juggler, template)
        if "B" in args.arms:
            verdict["legs"]["B"] = leg_b_zero_rearms(torch, juggler, template)
        if "C" in args.arms:
            verdict["legs"]["C"] = leg_c_integrity(torch, juggler)
        if "D" in args.arms:
            verdict["legs"]["D"] = leg_d_franken_fence(torch, juggler, template)
        if "F" in args.arms:
            verdict["legs"]["F"] = leg_f_numbers(torch, juggler, template)
    except Exception as exc:  # noqa: BLE001
        failed = True
        verdict["failure"] = f"{type(exc).__name__}: {exc}"
        loud(f"RED: {verdict['failure']}")
    finally:
        if "E" in args.arms:
            try:
                verdict["legs"]["E"] = leg_e_teardown(torch, residency)
            except Exception as exc:  # noqa: BLE001
                failed = True
                verdict["legs"]["E"] = {"failure": f"{type(exc).__name__}: {exc}"}
                loud(f"RED teardown: {exc}")
        juggler.catalog.close()
        gc.collect()
        torch.cuda.empty_cache()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(verdict, indent=2))
        loud(f"verdict -> {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
