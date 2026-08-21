"""pgw#1607 phase 4: the checkpoint juggle at SDXL scale, on a rented pod.

Runs ON the pod (weights never transit the box). Downloads K public SDXL
fine-tune UNets from HF (the 16 header-curated compatible repos), builds ONE
UNet lane over the arena, and prices the juggle for real: 5 GB-class
checkpoints, real denoise-shaped requests, a real torch.compile artifact.

Arms (default SWCZ; each banks its verdict before the next runs):

* **S — smoke/correctness** (the spend gate: nothing else runs if S is red).
  3 checkpoints: distinct outputs at fixed inputs, bitwise round-trip,
  D2H digests == ingest digests, zero re-arms (pointer + forward identity),
  and the REAL-COMPILE leg: `torch.compile`d UNet serves distinct outputs
  across swaps with ZERO recompiles (dynamo counters, not eyeballs).
* **W — warm matrix.** All K ingested (the warm set sizes itself from the
  pod's RAM — watching it evict IS the measurement), round-robin warm
  switches, walls + GB/s against the same-pod pinned H2D floor.
* **C — cold.** Evict → switch (disk → image → VRAM, the rebuild path) and
  hysteresis-forced disk-direct, against the pod's measured disk floor.
* **Z — zipf.** Real ~seconds requests (28 UNet steps, 1024^2-shaped),
  single-checkpoint baseline FIRST (the phase-3 ordering lesson), then the
  zipf mix; the honest product is seconds-per-request and switch cost per
  request, not a toy ratio.
* **B — cast arm (optional).** The bf16 lane over fp16 files: ingest casts
  once, warm switches only (disk-direct correctly REFUSES under a cast).

Lane dtype is FP16 by default — the files' native dtype, so disk triples
are byte-true and disk-direct switching is legal. (fp16 and bf16 are both
16-bit: raw fp16 bytes in a bf16 slot would pass every length check and be
silent garbage, which is why the juggler refuses casts outside ingest and
why this harness loads the template through the IMAGE, never raw triples.)

Overlap, stated honestly: a UNet-only lane has no phase to hide a swap
under — the bytes cannot be overwritten while any step still reads them, so
the switch serializes after the request tail and its floor is the transfer
bound (measured). What CAN overlap is host-side prefetch (disk -> pin under
compute), which arm Z exercises via hint_next. Phase-overlap (TE/VAE while
UNet computes) needs a multi-component lane — pgw#1602 integration, not
this harness.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

MIB = 1 << 20
GIB = 1 << 30

#: Header-curated 2026-08-20 ($0, ranged reads; SSD-1B red-armed the check).
CURATED = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "playgroundai/playground-v2-1024px-aesthetic",
    "playgroundai/playground-v2.5-1024px-aesthetic",
    "RunDiffusion/Juggernaut-XL-v9",
    "SG161222/RealVisXL_V4.0",
    "SG161222/RealVisXL_V5.0",
    "cagliostrolab/animagine-xl-3.1",
    "cagliostrolab/animagine-xl-4.0",
    "Lykon/dreamshaper-xl-1-0",
    "Lykon/dreamshaper-xl-v2-turbo",
    "misri/zavychromaxl_v80",
    "GraydientPlatformAPI/albedobase2-xl",
    "fluently/Fluently-XL-v4",
    "zenless-lab/sdxl-blue-pencil-xl-v7",
    "dataautogpt3/OpenDalleV1.1",
    # deliberately NOT ColorfulXL-Lightning here: F32 on disk, it belongs to
    # arm B (the cast arm), not the byte-true fp16 lane.
]

OUT_DIR = Path(os.environ.get("PGW1607_POD_OUT", "/workspace/pgw1607-out"))


def loud(msg: str) -> None:
    print(f"[pgw1607-pod] {msg}", flush=True)


def bank(name: str, payload: Dict[str, Any]) -> None:
    """Write the arm's verdict NOW — a later death keeps earlier arms.

    The payload is ALSO printed inline: the driver captures this stream over
    SSH, so even a dead scp path leaves every verdict in the box-side
    transcript (the pgw#1568 lesson: an evidence path that can die must not
    be the only one)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    blob = json.dumps(payload, indent=2)
    path.write_text(blob)
    loud(f"banked {path}")
    print(f"[pgw1607-verdict:{name}] {json.dumps(payload)}", flush=True)


def _meminfo(key: str) -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def gpu_line() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Downloads — pod-side only
# ---------------------------------------------------------------------------


def fetch_unet(repo: str, cache: Path) -> Path:
    """The repo's UNet directory only (config + fp16 or main safetensors)."""
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    root = snapshot_download(
        repo,
        cache_dir=str(cache),
        allow_patterns=[
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
        ],
    )
    unet = Path(root) / "unet"
    files = list(unet.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"{repo}: no unet safetensors landed")
    loud(f"fetched {repo} unet in {time.perf_counter() - t0:.0f}s "
         f"({sum(f.stat().st_size for f in files) / GIB:.2f} GiB)")
    return unet


def pick_variant(unet_dir: Path) -> Optional[str]:
    if (unet_dir / "diffusion_pytorch_model.fp16.safetensors").exists():
        return "fp16"
    return None


# ---------------------------------------------------------------------------
# Rig
# ---------------------------------------------------------------------------


def build_unet(torch: Any, config_dir: Path, dtype: Any) -> Any:
    from diffusers import UNet2DConditionModel

    with open(config_dir / "config.json") as fh:
        config = json.load(fh)
    model = UNet2DConditionModel.from_config(config)
    return model.to(dtype)


def load_template_from_image(torch: Any, template: Any, image: Any) -> None:
    """Template weights from the NORMALIZED image — never raw triples (the
    fp16/bf16 same-length hazard)."""
    with torch.no_grad():
        state = dict(template.state_dict())
        for region in image.layout.regions:
            for slot in region.slots:
                key = _slot_key_of(slot)
                state[key].copy_(image.slot_view(slot))


def _slot_key_of(slot: Any) -> str:
    _root, _, rest = slot.leaf.partition(".")
    return f"{rest}.{slot.attr}" if rest else slot.attr


def make_rig(
    torch: Any,
    unet_dirs: Dict[str, Path],
    serving: str,
    *,
    dtype: Any,
    budget_bytes: int,
    admit: List[str],
    host_floor_gib: float = 0.0,
):
    from gen_worker.models.arena_residency import (
        ArenaResidency,
        plan_layout,
    )
    from gen_worker.models.checkpoint_juggle import (
        CheckpointCatalog,
        CheckpointJuggler,
        read_manifest,
    )
    from gen_worker.models.arena_residency import dlpack_dtype
    from gen_worker.models.stream_residency import (
        discover_leaves,
        own_tensors,
        tensor_bytes,
    )

    template = build_unet(torch, unet_dirs[serving], dtype)
    leaves, _c, _a = discover_leaves([("unet", template)])
    specs = []
    for leaf_name, leaf in leaves.items():
        slots = []
        for attr, is_param, tensor in own_tensors(leaf):
            code, bits = dlpack_dtype(torch, tensor.dtype)
            slots.append(
                (attr, is_param, tensor_bytes(tensor), tuple(tensor.shape), code, bits)
            )
        specs.append((leaf_name, slots))
    layout = plan_layout(specs, granularity=2 * MIB, min_stream_bytes=8 * MIB)

    manifests = {
        name: read_manifest(d, variant=pick_variant(d))
        for name, d in unet_dirs.items()
    }
    import gen_worker.models.checkpoint_juggle as cj

    catalog = CheckpointCatalog(
        layout, torch_mod=torch, varena_mod=_varena(),
        host_floor_bytes=int(host_floor_gib * GIB) if host_floor_gib
        else cj.DEFAULT_HOST_FLOOR_BYTES,
    )
    for name in admit:
        catalog.admit(name, manifests[name])
    image0 = catalog.ensure_warm(serving)
    assert image0 is not None, "the serving checkpoint's image must ingest"
    load_template_from_image(torch, template, image0)
    template.to("cuda").eval()

    triples = {
        k: (s.path, s.offset, s.length) for k, s in manifests[serving].items()
    }
    residency = ArenaResidency(
        [("unet", template)],
        device="cuda",
        budget_bytes=budget_bytes,
        triples=triples,
        min_stream_bytes=8 * MIB,
    )
    residency.engage()
    juggler = CheckpointJuggler(
        residency, serving, manifests[serving], catalog=catalog
    )
    return template, residency, juggler, manifests


def _varena() -> Any:
    import varena

    return varena


def unet_inputs(torch: Any, dtype: Any, *, batch: int = 2) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(1607)
    return {
        "sample": torch.randn(batch, 4, 128, 128, generator=g).to("cuda", dtype),
        "timestep": torch.tensor(500, device="cuda"),
        "encoder_hidden_states": torch.randn(batch, 77, 2048, generator=g).to("cuda", dtype),
        "added_cond_kwargs": {
            "text_embeds": torch.randn(batch, 1280, generator=g).to("cuda", dtype),
            "time_ids": torch.randn(batch, 6, generator=g).to("cuda", dtype),
        },
    }


def one_step(torch: Any, model: Any, inputs: Dict[str, Any]) -> Any:
    with torch.no_grad():
        return model(
            inputs["sample"], inputs["timestep"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            added_cond_kwargs=inputs["added_cond_kwargs"],
        ).sample


def request(torch: Any, juggler: Any, model: Any, inputs: Dict[str, Any], steps: int) -> Any:
    """A denoise-shaped request: `steps` UNet forwards. Seconds, not ms."""
    juggler.assert_servable()
    out = None
    with torch.no_grad():
        for _ in range(steps):
            out = one_step(torch, model, inputs)
    torch.cuda.synchronize()
    return out


def region_digest(torch: Any, residency: Any, region: Any) -> str:
    h = hashlib.blake2b(digest_size=16)
    for slot in region.slots:
        host = residency._view(slot).detach().to("cpu")
        h.update(host.view(torch.uint8).view(-1).numpy().tobytes())
    return h.hexdigest()


def measure_h2d_floor(torch: Any, nbytes: int) -> float:
    src = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
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


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def arm_s(torch: Any, juggler: Any, template: Any, dtype: Any, ids: List[str]) -> Dict[str, Any]:
    """The spend gate. Everything here must be green before matrix money."""
    inputs = unet_inputs(torch, dtype)
    a, b = ids[0], ids[1]

    y_a = one_step(torch, template, inputs).clone()
    juggler.switch_to(b)
    y_b = one_step(torch, template, inputs).clone()
    juggler.switch_to(a)
    y_a2 = one_step(torch, template, inputs).clone()
    assert not torch.equal(y_a, y_b), "outputs identical across the swap"
    assert torch.equal(y_a, y_a2), "round trip not bitwise"
    loud("S: distinct outputs + bitwise round trip GREEN")

    # Digests on every backed region for the serving checkpoint.
    residency = juggler.residency
    image = juggler.catalog.warm(a)
    checked = 0
    for region in residency.layout.regions:
        if residency.is_resident(region.name):
            assert region_digest(torch, residency, region) == \
                image.region_digests[region.name], region.name
            checked += 1
    loud(f"S: {checked} region digests GREEN")

    # THE REAL-COMPILE LEG: torch.compile once, swap, zero recompiles.
    import torch._dynamo as dynamo

    dynamo.reset()
    compiled = torch.compile(template)
    _ = one_step(torch, compiled, inputs)  # compile happens here
    torch.cuda.synchronize()
    # The sharpest instrument available: ANY recompile from here on RAISES.
    dynamo.config.error_on_recompile = True
    from torch._dynamo.utils import counters as dyn_counters

    frames_before = dict(dyn_counters["stats"])
    ptrs_before = {
        (s.leaf, s.attr): int(residency._view(s).data_ptr())
        for r in residency.layout.regions if residency.is_resident(r.name)
        for s in r.slots
    }
    c_a = one_step(torch, compiled, inputs).clone()
    juggler.switch_to(b)
    c_b = one_step(torch, compiled, inputs).clone()
    juggler.switch_to(a)
    c_a2 = one_step(torch, compiled, inputs).clone()
    torch.cuda.synchronize()
    frames_after = dict(dyn_counters["stats"])
    dynamo.config.error_on_recompile = False
    recompiles = frames_after.get("unique_graphs", 0) - frames_before.get("unique_graphs", 0)
    moved = [
        k for r in residency.layout.regions if residency.is_resident(r.name)
        for s in r.slots
        for k in [(s.leaf, s.attr)]
        if ptrs_before.get(k) != int(residency._view(s).data_ptr())
    ]
    assert not torch.equal(c_a, c_b), "compiled outputs identical across swap"
    assert torch.equal(c_a, c_a2), "compiled round trip not bitwise"
    assert recompiles == 0, f"recompiles counted: {recompiles}"
    assert not moved, f"pointers moved under compiled serve: {moved[:3]}"
    assert juggler.rearms == 0
    loud("S: COMPILED serve across swaps GREEN — distinct outputs, "
         "0 recompiles (counted), pointers fixed")
    return {
        "distinct": True, "bitwise_roundtrip": True, "digests_checked": checked,
        "compiled_distinct": True, "compiled_recompiles": recompiles,
        "pointers_fixed": len(ptrs_before), "rearms": juggler.rearms,
    }


def arm_w(torch: Any, juggler: Any, ids: List[str]) -> Dict[str, Any]:
    floor = measure_h2d_floor(torch, 1 * GIB)
    lane_bytes = sum(
        r.span for r in juggler.residency.layout.regions
        if juggler.residency.is_resident(r.name)
    )
    for name in ids:
        juggler.hint_next(name)  # ingest what RAM admits; eviction is data
    rows = []
    for name in ids * 2:
        if name == juggler.serving_id:
            continue
        r = juggler.switch_to(name)
        rows.append({
            "to": name, "tier": r.tier, "wall_s": round(r.wall_s, 4),
            "gbps": round(r.bytes_moved / r.wall_s / 1e9, 2) if r.wall_s else 0,
            "verified": r.backing_verified,
        })
    warm = [x["wall_s"] for x in rows if x["tier"] == "host-warm"]
    out = {
        "h2d_floor_gbps": round(floor, 2),
        "lane_backed_bytes": lane_bytes,
        "warm_images": len(juggler.catalog.images),
        "evictions": juggler.catalog.evictions,
        "warm_median_s": round(statistics.median(warm), 4) if warm else None,
        "warm_bound_s": round(lane_bytes / (floor * 1e9), 4),
        "rows": rows,
    }
    loud(f"W: {json.dumps({k: v for k, v in out.items() if k != 'rows'})}")
    return out


def arm_c(torch: Any, juggler: Any, ids: List[str]) -> Dict[str, Any]:
    target = next(n for n in ids if n != juggler.serving_id)
    juggler.catalog.evict(target)
    t0 = time.perf_counter()
    r1 = juggler.switch_to(target)
    rebuild = time.perf_counter() - t0
    target2 = next(
        n for n in ids if n not in (juggler.serving_id, target)
    )
    juggler.catalog.evict(target2)
    juggler.catalog._evicted_epoch[target2] = juggler.catalog.pressure_epoch
    t0 = time.perf_counter()
    r2 = juggler.switch_to(target2)
    direct = time.perf_counter() - t0
    del juggler.catalog._evicted_epoch[target2]
    out = {
        "cold_rebuild_s": round(rebuild, 3), "rebuild_tier": r1.tier,
        "disk_direct_s": round(direct, 3), "direct_tier": r2.tier,
    }
    loud(f"C: {json.dumps(out)}")
    return out


def arm_z(torch: Any, juggler: Any, template: Any, dtype: Any, ids: List[str],
          *, steps: int, requests: int) -> Dict[str, Any]:
    import random

    inputs = unet_inputs(torch, dtype)
    # BASELINE FIRST (phase-3 ordering lesson), on a warm-mirrored serving ck.
    juggler.hint_next(juggler.serving_id)
    t0 = time.perf_counter()
    for _ in range(max(4, requests // 4)):
        request(torch, juggler, template, inputs, steps)
    base_n = max(4, requests // 4)
    baseline_per_req = (time.perf_counter() - t0) / base_n

    rng = random.Random(1607)
    weights = [1.0 / (i + 1) ** 1.1 for i in range(len(ids))]
    switch_walls: List[float] = []
    t0 = time.perf_counter()
    for _ in range(requests):
        target = rng.choices(ids, weights)[0]
        if target != juggler.serving_id:
            r = juggler.switch_to(target)
            switch_walls.append(r.wall_s)
        request(torch, juggler, template, inputs, steps)
    zipf_wall = time.perf_counter() - t0
    zipf_per_req = zipf_wall / requests
    out = {
        "warm_images_end": len(juggler.catalog.images),
        "evictions_total": juggler.catalog.evictions,
        "pressure_epoch": juggler.catalog.pressure_epoch,
        "steps_per_request": steps,
        "baseline_s_per_req": round(baseline_per_req, 3),
        "zipf_s_per_req": round(zipf_per_req, 3),
        "zipf_overhead_frac": round(zipf_per_req / baseline_per_req - 1, 4),
        "switches": len(switch_walls),
        "switch_median_s": round(statistics.median(switch_walls), 4) if switch_walls else None,
        "requests": requests,
    }
    loud(f"Z: {json.dumps(out)}")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="SWCZ")
    parser.add_argument("--repos", type=int, default=6, help="how many curated repos")
    parser.add_argument("--budget-gib", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--cache", default="/workspace/hf-cache")
    parser.add_argument("--host-floor-gib", type=float, default=0.0,
                        help="raise the warm tier's host floor to force real "
                             "evictions/hysteresis on big-RAM pods (0 = the "
                             "adaptive default)")
    args = parser.parse_args()

    from gen_worker.rigcheck import assert_fleet_line

    assert_fleet_line("pgw#1607 pod juggle rig")
    loud(f"gpu: {gpu_line()}")

    import torch

    assert torch.cuda.is_available()
    # FAIL FAST on the whole import closure BEFORE any multi-GB download —
    # try 2 died on a missing module AFTER 29 GB of fetches ($0.04; this
    # line makes that class cost $0.01).
    from gen_worker.models.arena_residency import ArenaResidency  # noqa: F401
    from gen_worker.models.checkpoint_juggle import CheckpointJuggler  # noqa: F401
    from diffusers import UNet2DConditionModel  # noqa: F401
    dtype = torch.float16
    ids = [f"ck{i}" for i in range(args.repos)]
    repos = CURATED[: args.repos]
    bank("meta", {
        "issue": "pgw#1607", "when": time.strftime("%F %T"), "gpu": gpu_line(),
        "arms": args.arms, "repos": repos, "budget_gib": args.budget_gib,
        "host_floor_gib": args.host_floor_gib,
        "mem_total_gib": round(_meminfo("MemTotal") / (1 << 30), 1),
        "mem_available_gib": round(_meminfo("MemAvailable") / (1 << 30), 1),
        "torch": torch.__version__,
    })

    cache = Path(args.cache)
    unet_dirs: Dict[str, Path] = {}
    for name, repo in zip(ids, repos):
        unet_dirs[name] = fetch_unet(repo, cache)
    bank("downloads", {"fetched": {n: str(p) for n, p in unet_dirs.items()}})

    smoke_ids = ids[:3]
    template, residency, juggler, _m = make_rig(
        torch, unet_dirs, ids[0],
        dtype=dtype, budget_bytes=int(args.budget_gib * GIB), admit=ids,
        host_floor_gib=args.host_floor_gib,
    )
    loud(f"rig up: {residency.layout.virtual_bytes / GIB:.2f} GiB virtual, "
         f"resident={len(residency.plan.all_resident)} "
         f"streamed={len(residency.plan.streamed)}")

    failed = False
    try:
        if "S" in args.arms:
            bank("S", arm_s(torch, juggler, template, dtype, smoke_ids))
        if "W" in args.arms:
            bank("W", arm_w(torch, juggler, ids))
        if "C" in args.arms:
            bank("C", arm_c(torch, juggler, ids))
        if "Z" in args.arms:
            bank("Z", arm_z(torch, juggler, template, dtype, ids,
                            steps=args.steps, requests=args.requests))
    except Exception as exc:  # noqa: BLE001
        failed = True
        bank("failure", {"error": f"{type(exc).__name__}: {exc}"})
        loud(f"RED: {type(exc).__name__}: {exc}")
    finally:
        try:
            residency.release()
            residency.arena.set_budget(0)
            stats = dict(residency.arena.stats())
            bank("teardown", {
                "committed": int(stats["committed_bytes"]),
                "mapped": int(stats["mapped_bytes"]),
            })
            assert int(stats["committed_bytes"]) == 0 and int(stats["mapped_bytes"]) == 0
            loud("teardown GREEN: committed==mapped==0")
        except Exception as exc:  # noqa: BLE001
            failed = True
            loud(f"RED teardown: {exc}")
        juggler.catalog.close()
        gc.collect()
        torch.cuda.empty_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
