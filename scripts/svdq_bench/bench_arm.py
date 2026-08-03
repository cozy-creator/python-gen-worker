#!/usr/bin/env python3
"""svdq runtime bench: ONE arm per process (pgw#865 instrument).

Arms: ``ours`` (gen-worker svdq lane — baseline or fused, chosen by the
per-process lane decision), ``bf16`` (the unquantized reference transformer on
the same card), ``nunchaku`` (their wheel, sm_120a only).

Renders cozy-hard-eval-v1 (20 prompts, 28 steps, true_cfg 4.5, 1328x1328,
fixed seeds, negative " ") from the SAME nunchaku fp4_r128 checkpoint bytes,
with per-step timing (cuda-sync at each step end), e2e wall per image, and
peak VRAM. Repeats one row 3x for variance. Saves product-grade webps
(q95 method4) + a JSON report. Run on-pod only.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path


class StepTimer:
    def __init__(self) -> None:
        self.ts: list[float] = []

    def __call__(self, pipe, i, t, kw):
        import torch
        torch.cuda.synchronize()
        self.ts.append(time.perf_counter())
        return {}


def save_webp(img, dest: Path) -> None:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="WEBP", quality=95, lossless=False, method=4)
    dest.write_bytes(buf.getvalue())


def load_ours(ckpt: Path, log):
    import torch
    from gen_worker.models.svdq import detect_svdq_artifact
    from gen_worker.models.svdq_native import (
        load_svdq_native_denoiser, svdq_native_available, svdq_native_reason)

    art = detect_svdq_artifact(ckpt)
    assert art is not None, f"no svdq artifact detected at {ckpt}"
    from gen_worker.models import native_kernels as nk
    lane = nk.svdq_linear_lane()
    mod_lane = nk.svdq_modulation_lane()
    log(f"linear lane={lane} ({nk.svdq_linear_lane_reason()}); "
        f"modulation lane={mod_lane} ({nk.svdq_modulation_lane_reason()})")
    avail = svdq_native_available()
    log(f"svdq_native_available={avail} reason={svdq_native_reason()!r} "
        f"model_class={art.model_class} rank={art.rank} precision={art.precision}")
    assert avail, "REFUSING: blockwise kernels not armed on this card"
    den = load_svdq_native_denoiser(art, compute_dtype=torch.bfloat16, mode="blockwise")
    got = getattr(den, "_cozy_svdq_mode", "?")
    assert got == "blockwise", f"decoded mode {got}, not blockwise — no silent dense fallback"
    log(f"OURS decoded blockwise; class={type(den).__name__}")
    import gen_worker
    census = {"fused": 0, "blockwise": 0, "awq_packed": 0}
    for _n, mod in den.named_modules():
        if getattr(mod, "_cozy_awq_packed", False):
            census["awq_packed"] += 1
        elif getattr(mod, "_cozy_svdq_fused", False):
            census["fused"] += 1
        elif getattr(mod, "_cozy_svdq_linear", False):
            census["blockwise"] += 1
    log(f"swap census={census}")
    if lane == "fused":
        assert census["fused"] > 0, "fused lane armed but zero fused modules"
    return den, {"runtime": "gen_worker svdq_native", "lane": lane,
                 "modulation_lane": mod_lane,
                 "lane_reason": nk.svdq_linear_lane_reason(),
                 "swap_census": census,
                 "gen_worker": getattr(gen_worker, "__version__", "?"),
                 "svdq_mode": got, "rank": art.rank, "precision": str(art.precision)}


def load_bf16(tree: Path, log):
    """The unquantized reference transformer from the same component tree —
    the same-card bf16 row every quantized number is judged against."""
    import torch
    from diffusers import QwenImageTransformer2DModel

    den = QwenImageTransformer2DModel.from_pretrained(
        str(tree), subfolder="transformer", torch_dtype=torch.bfloat16)
    log(f"BF16 reference transformer loaded; class={type(den).__name__}")
    import gen_worker
    return den, {"runtime": "diffusers bf16 reference",
                 "gen_worker": getattr(gen_worker, "__version__", "?"),
                 "precision": "bfloat16"}


def load_fp8(tree: Path, log):
    """The PUBLISHED #fp8-w8a8 flavor through the production w8a8 loader —
    fp8 weights RESIDENT on torch._scaled_mm, the same dispatch prod runs."""
    import torch
    from gen_worker.models.w8a8 import (detect_w8a8_artifacts,
                                        load_w8a8_denoiser, w8a8_gemm_mode)

    arts = detect_w8a8_artifacts(tree)
    assert arts, f"no #fp8-w8a8 artifact detected at {tree}"
    assert len(arts) == 1, f"expected one denoiser, got {len(arts)}"
    art = arts[0]
    mode = w8a8_gemm_mode()
    log(f"w8a8 gemm mode={mode!r} component={art.component!r} "
        f"quantized={len(art.quantized)} static_scales={art.static_input_scales}")
    assert mode in ("rowwise", "pertensor"), (
        f"REFUSING: w8a8 fell back to {mode!r} — that arm is not fp8 compute")
    den = load_w8a8_denoiser(tree, art, compute_dtype=torch.bfloat16,
                             mode=mode)
    swapped = sum(1 for _n, m in den.named_modules()
                  if getattr(m, "_cozy_w8a8_linear", False))
    log(f"FP8 loaded; class={type(den).__name__} fp8_linears={swapped}")
    assert swapped > 0, "fp8 arm swapped zero Linears — it is not serving fp8"
    import gen_worker
    return den, {"runtime": "gen_worker w8a8", "lane": f"w8a8:{mode}",
                 "gen_worker": getattr(gen_worker, "__version__", "?"),
                 "fp8_linears": swapped, "precision": "fp8-w8a8"}


def load_theirs(ckpt: Path, log):
    import torch
    import nunchaku
    from nunchaku import NunchakuQwenImageTransformer2DModel
    try:
        from nunchaku.utils import get_precision
        prec = get_precision()
    except Exception:  # noqa: BLE001
        prec = "?"
    log(f"nunchaku {getattr(nunchaku, '__version__', '?')} get_precision={prec}")
    assert prec == "fp4", f"expected fp4 on sm_120, got {prec}"
    try:
        den = NunchakuQwenImageTransformer2DModel.from_pretrained(
            str(ckpt), torch_dtype=torch.bfloat16)
    except TypeError:
        den = NunchakuQwenImageTransformer2DModel.from_pretrained(str(ckpt))
    log(f"THEIRS loaded; class={type(den).__name__}")
    return den, {"runtime": "nunchaku official",
                 "nunchaku": getattr(nunchaku, "__version__", "?"),
                 "precision": prec}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["ours", "bf16", "fp8", "nunchaku"])
    ap.add_argument("--fp8-tree", default="",
                    help="materialized #fp8-w8a8 tree (arm fp8)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reference-tree", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--compile", action="store_true",
                    help="production posture: torch.compile(den.forward, dynamic=None)")
    ap.add_argument("--attn-drop", action="store_true",
                    help="sdpa lever: drop the attn_mask (exact for batch-1 "
                         "unpadded prompts; frees flash/cuDNN backends)")
    ap.add_argument("--repeat-id", default="t02")
    ap.add_argument("--rows", default="",
                    help="comma-separated row ids; default = the whole set")
    ap.add_argument("--norm-rows", type=int, default=3,
                    help="rows for the normalized shape block (0 disables)")
    ap.add_argument("--norm-shape", default="1024x1024x30x4.0",
                    help="WxHxSTEPSxTRUE_CFG — the market-comparable row "
                         "(1024^2/30/cfg4.0 = the diffusers default fal "
                         "serves); true_cfg 1.0 means ONE forward/step")
    args = ap.parse_args()

    def log(m):
        print(f"[{args.arm}] {m}", flush=True)

    import torch
    assert torch.cuda.is_available()
    dev = torch.cuda.get_device_name(0)
    sm = "".join(str(x) for x in torch.cuda.get_device_capability())
    _free, total = torch.cuda.mem_get_info()
    log(f"device={dev} sm_{sm} torch={torch.__version__} vram={total/1e9:.1f}GB")

    spec = json.loads(Path(args.prompts).read_text())
    if args.rows:
        keep = [r.strip() for r in args.rows.split(",") if r.strip()]
        spec["t2i"] = [r for r in spec["t2i"] if r["id"] in keep]
        assert spec["t2i"], f"no rows matched {keep}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_load0 = time.perf_counter()
    if args.arm == "ours":
        den, rtinfo = load_ours(Path(args.ckpt), log)
    elif args.arm == "bf16":
        den, rtinfo = load_bf16(Path(args.reference_tree), log)
    elif args.arm == "fp8":
        den, rtinfo = load_fp8(Path(args.fp8_tree), log)
    else:
        den, rtinfo = load_theirs(Path(args.ckpt), log)
    if args.attn_drop:
        import torch.nn.functional as F
        _orig_sdpa = F.scaled_dot_product_attention

        def _maskfree(*a, **kw):
            if "attn_mask" in kw:
                kw["attn_mask"] = None
                return _orig_sdpa(*a, **kw)
            if len(a) >= 4:
                return _orig_sdpa(*(a[:3] + (None,) + a[4:]), **kw)
            return _orig_sdpa(*a, attn_mask=None, **kw)

        F.scaled_dot_product_attention = _maskfree
        rtinfo["attn_drop"] = True
        log("sdpa attn_mask DROPPED (exact for batch-1 unpadded prompts)")
    if args.compile:
        # two shape blocks per process; each specializes once under
        # dynamic=None, and the default limit is what polluted the banked
        # 5090 20-row summary.
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
    offload = total < 40e9
    if offload:
        pipe.enable_model_cpu_offload()
        log("model_cpu_offload enabled (<40GB card); denoise stays on cuda")
    else:
        pipe = pipe.to("cuda")
    load_s = time.perf_counter() - t_load0
    log(f"pipeline {type(pipe).__name__} ready in {load_s:.1f}s "
        f"diffusers={diffusers.__version__} transformers={transformers.__version__}")

    steps, cfg = int(spec["steps"]), float(spec["true_cfg_scale"])
    neg, (w, h) = spec["negative_prompt"], spec["resolution"]

    def render_at(prompt: str, seed: int, n_steps: int, rw: int, rh: int,
                  rcfg: float):
        timer = StepTimer()
        g = torch.Generator(device="cuda").manual_seed(seed)
        t0 = time.perf_counter()
        img = pipe(prompt=prompt, num_inference_steps=n_steps, width=rw,
                   height=rh, generator=g, true_cfg_scale=rcfg,
                   negative_prompt=neg,
                   callback_on_step_end=timer).images[0]
        wall = time.perf_counter() - t0
        deltas = [timer.ts[i] - timer.ts[i - 1]
                  for i in range(1, len(timer.ts))]
        return img, wall, deltas

    def render(prompt: str, seed: int, n_steps: int):
        return render_at(prompt, seed, n_steps, w, h, cfg)

    # warmup (full res/shape so kernel autotune is covered; not counted)
    log("warmup render (4 steps, not counted)")
    wimg, wwall, _ = render(spec["t2i"][0]["prompt"], spec["t2i"][0]["seed"], 4)
    save_webp(wimg, out_dir / "warmup.webp")
    torch.cuda.reset_peak_memory_stats()

    rows = []
    for r in spec["t2i"]:
        img, wall, deltas = render(r["prompt"], r["seed"], steps)
        save_webp(img, out_dir / f"{r['id']}.webp")
        steady = deltas[1:] if len(deltas) > 2 else deltas
        rows.append({"id": r["id"], "seed": r["seed"], "e2e_s": wall,
                     "steps": steps, "step_deltas": deltas,
                     "step_mean_s": sum(steady) / len(steady)})
        log(f"{r['id']} e2e={wall:.2f}s step_mean={rows[-1]['step_mean_s']*1000:.0f}ms")

    rep_row = next((r for r in spec["t2i"] if r["id"] == args.repeat_id),
                   spec["t2i"][0])
    repeats = []
    for i in range(args.repeats):
        _img, wall, deltas = render(rep_row["prompt"], rep_row["seed"], steps)
        steady = deltas[1:] if len(deltas) > 2 else deltas
        repeats.append({"e2e_s": wall, "step_mean_s": sum(steady) / len(steady)})
        log(f"repeat{i} {args.repeat_id} e2e={wall:.2f}s")

    # Normalized block: same process, same weights, market-comparable shape.
    # Reported separately — never mixed into the main summary.
    norm = None
    if args.norm_rows > 0 and args.norm_shape:
        nw, nh, nsteps, ncfg = args.norm_shape.split("x")
        nw, nh, nsteps, ncfg = int(nw), int(nh), int(nsteps), float(ncfg)
        log(f"normalized block {nw}x{nh} steps={nsteps} true_cfg={ncfg} "
            f"({'2 fwd' if ncfg > 1.0 else '1 fwd'}/step)")
        nrows = []
        wimg2, _w2, _d2 = render_at(spec["t2i"][0]["prompt"],
                                    spec["t2i"][0]["seed"], 4, nw, nh, ncfg)
        save_webp(wimg2, out_dir / "norm_warmup.webp")
        for r in spec["t2i"][:args.norm_rows]:
            img, wall, deltas = render_at(r["prompt"], r["seed"], nsteps,
                                          nw, nh, ncfg)
            save_webp(img, out_dir / f"norm_{r['id']}.webp")
            steady = deltas[1:] if len(deltas) > 2 else deltas
            nrows.append({"id": r["id"], "e2e_s": wall,
                          "step_mean_s": sum(steady) / len(steady)})
            log(f"norm {r['id']} e2e={wall:.2f}s "
                f"step_mean={nrows[-1]['step_mean_s'] * 1000:.0f}ms")
        ne = [r["e2e_s"] for r in nrows]
        ns = [r["step_mean_s"] for r in nrows]
        norm = {"shape": [nw, nh], "steps": nsteps, "true_cfg_scale": ncfg,
                "forwards_per_step": 2 if ncfg > 1.0 else 1,
                "rows": nrows,
                "e2e_mean_s": sum(ne) / len(ne), "e2e_min_s": min(ne),
                "step_mean_s": sum(ns) / len(ns),
                "images_per_hour": 3600.0 / (sum(ne) / len(ne))}
        log(f"NORM e2e_mean={norm['e2e_mean_s']:.2f}s "
            f"step={norm['step_mean_s'] * 1000:.0f}ms")

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_res = torch.cuda.max_memory_reserved()
    e2e = [r["e2e_s"] for r in rows[1:]]  # drop first counted image from means too
    stepms = [r["step_mean_s"] for r in rows[1:]]
    report = {
        "arm": args.arm, "runtime": rtinfo, "device": dev, "sm": sm,
        "torch": torch.__version__, "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "offload": "model_cpu_offload" if offload else "none",
        "ckpt": str(args.ckpt), "load_s": load_s,
        "recipe": {"steps": steps, "true_cfg_scale": cfg, "resolution": [w, h],
                   "negative_prompt": neg, "set": spec["set"],
                   "note": "per step = 2 transformer forwards (true CFG)"},
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
        f"step_mean={s['step_mean_s']*1000:.0f}ms "
        f"imgs/hr={s['images_per_hour']:.1f} "
        f"peak_vram={s['peak_vram_alloc_gb']:.1f}GB load={load_s:.1f}s{nz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
