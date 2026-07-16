"""gw#551 pod validation (H100 SECURE).

Reproduces the te#79 crash shapes at real scale, proves the lane-gate fix
with alternating traffic, and measures swap performance (pageable-before vs
pinned-after; RAM->VRAM and disk->VRAM).

Usage on the pod:
    ARM=before python3 gw551_pod_validate.py   # 0.29.0 (pre-fix) semantics
    ARM=after  python3 gw551_pod_validate.py   # branch (lane gate + pinned swap)

Both arms drive the REAL Residency machinery; the lanes are diffusers-shaped
pipelines whose exclusive transformers overcommit the card exactly like the
merged qwen endpoint (2 x ~38 GiB + ~15 GiB shared encoder on 80 GiB).
"""

from __future__ import annotations

import json
import logging
import os
import time

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("gw551")

ARM = os.environ.get("ARM", "after")
GiB = 1024 ** 3
LANE_GB = float(os.environ.get("LANE_GB", "38"))
SHARED_GB = float(os.environ.get("SHARED_GB", "15"))
DTYPE = torch.bfloat16
WIDTH = 8192


def big_module(gb: float) -> nn.Module:
    """~gb GiB of bf16 Linear weights, uninitialized (values irrelevant)."""
    per_layer = WIDTH * WIDTH * 2
    layers = max(1, int(gb * GiB / per_layer))
    m = nn.Sequential(*[
        nn.Linear(WIDTH, WIDTH, bias=False, dtype=DTYPE, device="meta")
        for _ in range(layers)
    ])
    return m.to_empty(device="cpu")


class LanePipe:
    """diffusers-shaped pipeline: latents are created on the device it
    BELIEVES it is on (first transformer param); the shared encoder's cuda
    output feeds the transformer — the two te#79 crash shapes when demoted."""

    def __init__(self, transformer: nn.Module, encoder: nn.Module, name: str) -> None:
        self.transformer = transformer
        self.encoder = encoder
        self.name = name

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    def __call__(self, seed: int | None = None) -> str:
        dev = self.device
        if seed is not None:
            gen = torch.Generator(device="cuda")  # ctx.generator on a cuda host
            gen.manual_seed(seed)
            latents = torch.randn(1, WIDTH, device=dev, dtype=DTYPE, generator=gen)
        else:
            latents = torch.randn(1, WIDTH, device=dev, dtype=DTYPE)
        emb = self.encoder[0](latents.to(next(self.encoder.parameters()).device))
        out = self.transformer[0](emb)  # addmm: cuda emb x lane weights
        torch.cuda.synchronize()
        return str(out.device)


def build(res) -> tuple["LanePipe", "LanePipe"]:
    from gen_worker.models.memory import estimate_cuda_resident_gb
    from gen_worker.models.residency import LoadedComponentKey

    t0 = time.monotonic()
    shared = big_module(SHARED_GB)
    lanes = {"t2i": big_module(LANE_GB), "edit": big_module(LANE_GB)}
    log.info("built modules in %.1fs", time.monotonic() - t0)

    shared.cuda()  # shared encoder: resident + refcount-held (gw#479 shape)
    key = LoadedComponentKey.for_component(
        content_digest="gw551-shared-te", component="text_encoder")
    res.acquire_shared(key, lambda: shared,
                       vram_bytes=int(estimate_cuda_resident_gb(shared) * GiB))

    pipes = []
    for name, tr in lanes.items():
        ref = f"tensorhub/gw551-{name}"
        excl = nn.ModuleDict({"transformer": tr})
        need = int(LANE_GB * GiB)
        res.make_room(need)
        if res.free_vram_bytes() >= need + 2 * GiB:
            t0 = time.monotonic()
            tr.cuda()
            torch.cuda.synchronize()
            log.info("lane %s loaded to VRAM in %.2fs", name, time.monotonic() - t0)
            res.track_vram(ref, excl,
                           vram_bytes=int(estimate_cuda_resident_gb(tr) * GiB))
        else:
            log.info("lane %s stays host-resident (RAM tier)", name)
            res.track_ram(ref, excl)
        pipes.append(LanePipe(tr, shared, name))
    return pipes[0], pipes[1]


def main() -> None:
    from gen_worker.models.residency import Residency, Tier

    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info()
    log.info("ARM=%s card=%s total=%.1fGiB free=%.1fGiB gen_worker=%s",
             ARM, torch.cuda.get_device_name(0), total / GiB, free / GiB,
             __import__("gen_worker").__file__)
    assert 2 * LANE_GB + SHARED_GB > total / GiB, "not overcommitted; resize"

    res = Residency()
    t2i, edit = build(res)
    report: dict = {"arm": ARM, "card": torch.cuda.get_device_name(0),
                    "lane_gb": LANE_GB, "shared_gb": SHARED_GB}

    refs = {"t2i": "tensorhub/gw551-t2i", "edit": "tensorhub/gw551-edit"}
    if ARM == "after":
        from gen_worker.api.errors import RetryableError
        from gen_worker.models.lane_gate import LaneGate, arm_lane_gate

        for p in (t2i, edit):
            assert arm_lane_gate(p, LaneGate(
                ref=refs[p.name], residency=res, label=refs[p.name],
                retry_exc=RetryableError))

    # ---- Part 1: the te#79 crash shapes (requests on the demoted lane). ----
    demoted = t2i if res.tier(refs["t2i"]) is Tier.RAM else edit
    log.info("demoted lane after setup: %s", demoted.name)
    part1 = {}
    for label, seed in (("seeded_generator", 17), ("unseeded_addmm", None)):
        try:
            t0 = time.monotonic()
            dev = demoted(seed=seed)
            part1[label] = {"ok": True, "device": dev,
                            "wall_s": round(time.monotonic() - t0, 2)}
            log.info("demoted-lane call (%s) SERVED on %s", label, dev)
        except Exception as exc:  # noqa: BLE001
            part1[label] = {"ok": False,
                            "error": f"{type(exc).__name__}: {exc}"}
            log.error("demoted-lane call (%s) CRASHED: %s: %s",
                      label, type(exc).__name__, exc)
            if ARM == "after":
                raise
    report["demoted_lane_calls"] = part1

    # ---- Part 2 (after only): alternating traffic, measured swaps. ---------
    if ARM == "after":
        timings = []
        for i in range(3):
            for pipe in (t2i, edit):
                t0 = time.monotonic()
                dev = pipe(seed=i)
                wall = time.monotonic() - t0
                assert dev.startswith("cuda"), dev
                assert res.tier(refs[pipe.name]) is Tier.VRAM
                timings.append({"round": i, "lane": pipe.name,
                                "wall_s": round(wall, 3)})
                log.info("round %d lane %s: %.2fs", i, pipe.name, wall)
        report["alternating"] = timings
        report["transition_stats"] = res.transition_stats()

    # ---- Part 3: raw swap timing on one ~LANE_GB transformer. --------------
    tr = t2i.transformer
    if ARM == "after":
        from gen_worker.models.pinned_swap import swap_module

        if res.tier(refs["t2i"]) is Tier.RAM:
            assert res.promote(refs["t2i"])
        torch.cuda.synchronize()
        t0 = time.perf_counter(); swap_module(tr, "cpu"); d1 = time.perf_counter() - t0
        torch.cuda.empty_cache()
        t0 = time.perf_counter(); swap_module(tr, "cuda"); p1 = time.perf_counter() - t0
        t0 = time.perf_counter(); swap_module(tr, "cpu"); d2 = time.perf_counter() - t0
        torch.cuda.empty_cache()
        t0 = time.perf_counter(); swap_module(tr, "cuda"); p2 = time.perf_counter() - t0
        report["swap_timing"] = {
            "demote_first_s": round(d1, 3), "promote_pinned_s": round(p1, 3),
            "demote_pointer_swap_s": round(d2, 3),
            "promote_pinned_2_s": round(p2, 3),
            "promote_gib_per_s": round(LANE_GB / min(p1, p2), 1)}
        swap_module(tr, "cpu")
        swap_module(edit.transformer, "cpu")
    else:
        if next(tr.parameters()).device.type != "cuda":
            res.make_room(int(LANE_GB * GiB))
            tr.cuda(); torch.cuda.synchronize()
        t0 = time.perf_counter(); tr.cpu(); d1 = time.perf_counter() - t0
        torch.cuda.empty_cache()
        t0 = time.perf_counter(); tr.cuda(); torch.cuda.synchronize(); p1 = time.perf_counter() - t0
        report["swap_timing"] = {
            "demote_pageable_s": round(d1, 3),
            "promote_pageable_s": round(p1, 3),
            "promote_gib_per_s": round(LANE_GB / p1, 1)}
        tr.cpu()
        edit.transformer.cpu()
    torch.cuda.empty_cache()

    # ---- Part 4: disk -> VRAM cold path (safetensors, load direct to GPU). -
    if os.environ.get("SKIP_DISK") != "1":
        from safetensors.torch import load_file, save_file

        path = "/tmp/gw551_lane.safetensors"
        sd = dict(tr.state_dict())
        t0 = time.perf_counter(); save_file(sd, path); d_save = time.perf_counter() - t0
        del sd
        os.system("sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null")
        t0 = time.perf_counter()
        sd = load_file(path, device="cuda")
        torch.cuda.synchronize()
        d_cold = time.perf_counter() - t0
        report["disk_to_vram"] = {"save_s": round(d_save, 1),
                                  "load_to_cuda_s": round(d_cold, 1),
                                  "gib_per_s": round(LANE_GB / d_cold, 1)}
        del sd
        os.remove(path)

    print("GW551_REPORT " + json.dumps(report))


if __name__ == "__main__":
    main()
