#!/usr/bin/env python3
"""svdq runtime bench, EDIT family: ONE arm per process.

The image+text-conditioned twin of ``bench_arm.py``. Same shapes, same
timing/reporting contract, same executed-lane recording — so the qwen-image
table and the qwen-image-edit table are read side by side.

Arms: ``bf16`` (unquantized reference transformer) and ``fp8`` (the PUBLISHED
``#fp8-w8a8`` flavor through the production w8a8 loader). There is no nvfp4
arm: no 4-bit qwen-image-edit artifact exists, and this instrument never
fabricates one.

Inputs are the REAL gate inputs (see edit_prompts.json provenance), verified
by sha256 at load: a benchmark whose conditioning image is synthetic noise is
not measuring the edit workload.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

from bench_arm import StepTimer, load_bf16, load_fp8, save_webp  # noqa: F401


def load_befores(spec: dict, before_dir: Path, log) -> dict:
    """BEFORE image bytes, sha256-verified against the recorded gate inputs."""
    out = {}
    from PIL import Image
    for r in spec["edit"]:
        p = before_dir / r["before"]
        raw = p.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        assert got == r["before_sha256"], (
            f"{r['id']}: BEFORE {p} sha256 {got} != recorded "
            f"{r['before_sha256']} — the conditioning input is not the one "
            f"the gate ran on")
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        assert list(img.size) == r["before_size"], (
            f"{r['id']}: BEFORE size {img.size} != {r['before_size']}")
        out[r["id"]] = img
        log(f"BEFORE {r['id']} {p.name} {img.size} sha256={got[:16]} ok")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["bf16", "fp8"])
    ap.add_argument("--fp8-tree", default="")
    ap.add_argument("--reference-tree", required=True)
    ap.add_argument("--before-dir", default="/root/before")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--repeat-id", default="m01")
    ap.add_argument("--rows", default="")
    ap.add_argument("--norm-rows", type=int, default=3)
    ap.add_argument("--norm-shape", default="1024x1024x30x4.0")
    args = ap.parse_args()

    def log(m):
        print(f"[edit:{args.arm}] {m}", flush=True)

    import torch
    assert torch.cuda.is_available()
    dev = torch.cuda.get_device_name(0)
    sm = "".join(str(x) for x in torch.cuda.get_device_capability())
    _free, total = torch.cuda.mem_get_info()
    log(f"device={dev} sm_{sm} torch={torch.__version__} "
        f"vram={total / 1e9:.1f}GB")

    spec = json.loads(Path(args.prompts).read_text())
    if args.rows:
        keep = [r.strip() for r in args.rows.split(",") if r.strip()]
        spec["edit"] = [r for r in spec["edit"] if r["id"] in keep]
        assert spec["edit"], f"no rows matched {keep}"
    befores = load_befores(spec, Path(args.before_dir), log)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_load0 = time.perf_counter()
    if args.arm == "bf16":
        den, rtinfo = load_bf16(Path(args.reference_tree), log)
    else:
        den, rtinfo = load_fp8(Path(args.fp8_tree), log)
    if args.compile:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 32
        den.forward = torch.compile(den.forward, dynamic=None)
        rtinfo["compiled"] = "torch.compile(dynamic=None)"
        log("denoiser forward compiled (production posture)")

    import diffusers
    import transformers
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        args.reference_tree, torch_dtype=torch.bfloat16, transformer=den)
    assert type(pipe).__name__ == "QwenImageEditPlusPipeline", (
        f"expected the edit pipeline, got {type(pipe).__name__}")
    offload = total < 60e9
    if offload:
        pipe.enable_model_cpu_offload()
        log("model_cpu_offload enabled (<60GB card); denoise stays on cuda")
    else:
        pipe = pipe.to("cuda")
    load_s = time.perf_counter() - t_load0
    log(f"pipeline {type(pipe).__name__} ready in {load_s:.1f}s "
        f"diffusers={diffusers.__version__} "
        f"transformers={transformers.__version__}")

    steps, cfg = int(spec["steps"]), float(spec["true_cfg_scale"])
    neg, (w, h) = spec["negative_prompt"], spec["resolution"]

    def render_at(row, n_steps, rw, rh, rcfg):
        timer = StepTimer()
        g = torch.Generator(device="cuda").manual_seed(row["seed"])
        t0 = time.perf_counter()
        img = pipe(image=[befores[row["id"]]], prompt=row["instruction"],
                   num_inference_steps=n_steps, width=rw, height=rh,
                   generator=g, true_cfg_scale=rcfg, negative_prompt=neg,
                   callback_on_step_end=timer).images[0]
        wall = time.perf_counter() - t0
        deltas = [timer.ts[i] - timer.ts[i - 1]
                  for i in range(1, len(timer.ts))]
        return img, wall, deltas

    log("warmup render (4 steps, not counted)")
    wimg, _ww, _wd = render_at(spec["edit"][0], 4, w, h, cfg)
    save_webp(wimg, out_dir / "warmup.webp")
    torch.cuda.reset_peak_memory_stats()

    rows = []
    for r in spec["edit"]:
        img, wall, deltas = render_at(r, steps, w, h, cfg)
        save_webp(img, out_dir / f"{r['id']}.webp")
        steady = deltas[1:] if len(deltas) > 2 else deltas
        rows.append({"id": r["id"], "seed": r["seed"], "e2e_s": wall,
                     "steps": steps, "step_deltas": deltas,
                     "step_mean_s": sum(steady) / len(steady)})
        log(f"{r['id']} e2e={wall:.2f}s "
            f"step_mean={rows[-1]['step_mean_s'] * 1000:.0f}ms")

    rep_row = next((r for r in spec["edit"] if r["id"] == args.repeat_id),
                   spec["edit"][0])
    repeats = []
    for i in range(args.repeats):
        _img, wall, deltas = render_at(rep_row, steps, w, h, cfg)
        steady = deltas[1:] if len(deltas) > 2 else deltas
        repeats.append({"e2e_s": wall,
                        "step_mean_s": sum(steady) / len(steady)})
        log(f"repeat{i} {rep_row['id']} e2e={wall:.2f}s")

    norm = None
    if args.norm_rows > 0 and args.norm_shape:
        nw, nh, nsteps, ncfg = args.norm_shape.split("x")
        nw, nh, nsteps, ncfg = int(nw), int(nh), int(nsteps), float(ncfg)
        log(f"normalized block {nw}x{nh} steps={nsteps} true_cfg={ncfg} "
            f"({'2 fwd' if ncfg > 1.0 else '1 fwd'}/step)")
        nrows = []
        wimg2, _w2, _d2 = render_at(spec["edit"][0], 4, nw, nh, ncfg)
        save_webp(wimg2, out_dir / "norm_warmup.webp")
        for r in spec["edit"][:args.norm_rows]:
            img, wall, deltas = render_at(r, nsteps, nw, nh, ncfg)
            save_webp(img, out_dir / f"norm_{r['id']}.webp")
            steady = deltas[1:] if len(deltas) > 2 else deltas
            nrows.append({"id": r["id"], "e2e_s": wall,
                          "step_mean_s": sum(steady) / len(steady)})
            log(f"norm {r['id']} e2e={wall:.2f}s "
                f"step_mean={nrows[-1]['step_mean_s'] * 1000:.0f}ms")
        ne = [r["e2e_s"] for r in nrows]
        ns = [r["step_mean_s"] for r in nrows]
        norm = {"shape": [nw, nh], "steps": nsteps, "true_cfg_scale": ncfg,
                "forwards_per_step": 2 if ncfg > 1.0 else 1, "rows": nrows,
                "e2e_mean_s": sum(ne) / len(ne), "e2e_min_s": min(ne),
                "step_mean_s": sum(ns) / len(ns),
                "images_per_hour": 3600.0 / (sum(ne) / len(ne))}
        log(f"NORM e2e_mean={norm['e2e_mean_s']:.2f}s "
            f"step={norm['step_mean_s'] * 1000:.0f}ms")

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_res = torch.cuda.max_memory_reserved()
    e2e = [r["e2e_s"] for r in rows[1:]]
    stepms = [r["step_mean_s"] for r in rows[1:]]
    report = {
        "arm": args.arm, "family": "edit", "runtime": rtinfo, "device": dev,
        "sm": sm, "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "offload": "model_cpu_offload" if offload else "none",
        "load_s": load_s,
        "recipe": {"steps": steps, "true_cfg_scale": cfg,
                   "resolution": [w, h], "negative_prompt": neg,
                   "set": spec["set"],
                   "inputs": [{"id": r["id"], "before": r["before"],
                               "before_sha256": r["before_sha256"],
                               "instruction": r["instruction"]}
                              for r in spec["edit"]],
                   "note": "per step = 2 transformer forwards (true CFG); "
                           "condition image is resized to a 1024^2 area by "
                           "the pipeline, so the sequence carries the edit "
                           "tokens on top of the output latents"},
        "images": rows, "repeats": repeats, "normalized": norm,
        "summary": {
            "e2e_mean_s": sum(e2e) / len(e2e),
            "e2e_min_s": min(e2e), "e2e_max_s": max(e2e),
            "step_mean_s": sum(stepms) / len(stepms),
            "images_per_hour": 3600.0 / (sum(e2e) / len(e2e)),
            "peak_vram_alloc_gb": peak_alloc / 2**30,
            "peak_vram_reserved_gb": peak_res / 2**30,
        },
    }
    (out_dir / f"bench_{args.arm}.json").write_text(json.dumps(report, indent=1))
    s = report["summary"]
    nz = (f" | norm {norm['shape'][0]}^2/{norm['steps']}: "
          f"e2e={norm['e2e_mean_s']:.2f}s "
          f"step={norm['step_mean_s'] * 1000:.0f}ms" if norm else "")
    log(f"DONE e2e_mean={s['e2e_mean_s']:.2f}s "
        f"step_mean={s['step_mean_s'] * 1000:.0f}ms "
        f"imgs/hr={s['images_per_hour']:.1f} "
        f"peak_vram={s['peak_vram_alloc_gb']:.1f}GB load={load_s:.1f}s{nz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
