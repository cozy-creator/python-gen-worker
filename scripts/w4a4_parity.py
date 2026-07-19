"""W4A4 quality-parity + speed harness (gw#540).

Same-seed renders of ONE model on the current GPU across lanes:
  bf16  — plain bf16-resident (the quality reference)
  w8a8  — data-free fp8-w8a8 artifact on torch._scaled_mm (gw#534 lane)
  w4a4  — data-free nvfp4-w4a4 artifact on blockwise fp4 scaled_mm

Reports pixel deltas (MAE/PSNR, + LPIPS when installed) vs bf16, denoise
wall per lane (eager and, with --compiled, regional-compiled), and the
denoiser's resident VRAM per lane (expect ~quarter of bf16 on w4a4).
Proves the SERVE path — calibrated production artifacts + the degradation
gate are the conversion side's (te#79/te#80).

Run on a rented SECURE Blackwell pod (never a dev box):
  uv run python scripts/w4a4_parity.py --model black-forest-labs/FLUX.2-klein-4B \
      --steps 8 --size 1024 --compiled --out /tmp/parity
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path


def _mae_psnr(a, b):
    import numpy as np

    a = a.astype("float32") / 255.0
    b = b.astype("float32") / 255.0
    mae = float(np.abs(a - b).mean())
    mse = float(((a - b) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10.0 * float(np.log10(1.0 / mse))
    return mae, psnr


def _lpips(a, b):
    try:
        import lpips
        import numpy as np
        import torch
    except ImportError:
        return None
    loss = lpips.LPIPS(net="alex", verbose=False).cuda()

    def t(x):
        v = torch.from_numpy(np.asarray(x)).float().permute(2, 0, 1) / 127.5 - 1.0
        return v.unsqueeze(0).cuda()

    with torch.no_grad():
        return float(loss(t(a), t(b)).item())


def _denoiser(pipe):
    for name in ("transformer", "unet"):
        mod = getattr(pipe, name, None)
        if mod is not None:
            return mod
    raise SystemExit("pipeline has no transformer/unet")


def _module_bytes(mod) -> int:
    seen, total = set(), 0
    for t in list(mod.parameters()) + list(mod.buffers()):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
    return total


def _render(pipe, *, prompt: str, seed: int, steps: int, size: int):
    import numpy as np
    import torch

    if callable(getattr(pipe, "set_progress_bar_config", None)):
        pipe.set_progress_bar_config(disable=True)
    kwargs = dict(prompt=prompt, num_inference_steps=steps,
                  width=size, height=size,
                  generator=torch.Generator(device="cuda").manual_seed(seed))
    pipe(**{**kwargs, "num_inference_steps": 1})  # warmup (allocs, autotune)
    torch.cuda.synchronize()
    t0 = time.monotonic()
    image = pipe(**kwargs).images[0]
    torch.cuda.synchronize()
    wall = time.monotonic() - t0
    return np.asarray(image.convert("RGB")), wall


def _compiled_wall(pipe, *, prompt: str, seed: int, steps: int, size: int):
    """Regional compile (ie#381's prod serve mode) timing for the same call."""
    import torch

    den = _denoiser(pipe)
    if not callable(getattr(den, "compile_repeated_blocks", None)):
        print("denoiser has no compile_repeated_blocks; skipping compiled arm")
        return None
    den.compile_repeated_blocks(fullgraph=True)
    kwargs = dict(prompt=prompt, num_inference_steps=steps,
                  width=size, height=size,
                  generator=torch.Generator(device="cuda").manual_seed(seed))
    for _ in range(2):  # compile + settle
        pipe(**{**kwargs, "num_inference_steps": 2})
    torch.cuda.synchronize()
    t0 = time.monotonic()
    pipe(**kwargs)
    torch.cuda.synchronize()
    return time.monotonic() - t0


def _free(pipe):
    import torch

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local diffusers tree")
    ap.add_argument("--out", default="/tmp/w4a4-parity")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=(
        "a photo of a red fox standing in a snowy forest clearing, golden hour"))
    ap.add_argument("--lanes", default="bf16,w8a8,w4a4")
    ap.add_argument("--compiled", action="store_true",
                    help="also time each lane under regional compile")
    args = ap.parse_args()

    import torch
    from diffusers import DiffusionPipeline

    from gen_worker.models.loading import load_from_pretrained
    from gen_worker.models.w4a4 import quantize_tree_w4a4, w4a4_gemm_mode
    from gen_worker.models.w8a8 import quantize_tree_w8a8, w8a8_gemm_mode

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.model)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(args.model))
    print(f"snapshot: {src}")
    print(f"w8a8_gemm_mode: {w8a8_gemm_mode()!r}  w4a4_gemm_mode: {w4a4_gemm_mode()!r}")

    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    images: dict = {}
    walls: dict = {}
    compiled_walls: dict = {}
    denoiser_gb: dict = {}

    for lane in lanes:
        print(f"--- lane {lane}")
        if lane == "bf16":
            pipe = DiffusionPipeline.from_pretrained(
                str(src), torch_dtype=torch.bfloat16)
        elif lane in ("w8a8", "w4a4"):
            tree = out / f"{lane}-tree"
            if not tree.exists():
                (quantize_tree_w8a8 if lane == "w8a8"
                 else quantize_tree_w4a4)(src, tree)
            pipe = load_from_pretrained(DiffusionPipeline, tree)
            lane_attr = getattr(pipe, "_cozy_weight_lane", "")
            print(f"{lane} weight lane: {lane_attr!r}")
            assert lane_attr == lane, f"{lane} scaled_mm lane did not engage"
        else:
            raise SystemExit(f"unknown lane {lane}")
        pipe.to("cuda")
        denoiser_gb[lane] = round(_module_bytes(_denoiser(pipe)) / 2**30, 3)
        print(f"{lane} denoiser resident: {denoiser_gb[lane]} GiB")
        img, wall = _render(pipe, prompt=args.prompt, seed=args.seed,
                            steps=args.steps, size=args.size)
        images[lane] = img
        walls[lane] = wall
        from PIL import Image

        Image.fromarray(img).save(out / f"{lane}.png")
        print(f"{lane} eager: {wall:.2f}s for {args.steps} steps "
              f"({wall / args.steps * 1000:.0f} ms/step)")
        if args.compiled:
            cw = _compiled_wall(pipe, prompt=args.prompt, seed=args.seed,
                                steps=args.steps, size=args.size)
            if cw is not None:
                compiled_walls[lane] = cw
                print(f"{lane} compiled: {cw:.2f}s "
                      f"({cw / args.steps * 1000:.0f} ms/step)")
        _free(pipe)

    report: dict = {"model": args.model, "steps": args.steps, "size": args.size,
                    "seed": args.seed, "walls_s": walls,
                    "compiled_walls_s": compiled_walls,
                    "denoiser_resident_gib": denoiser_gb, "deltas": {}}
    ref = images.get("bf16")
    for lane, img in images.items():
        if ref is None or lane == "bf16":
            continue
        mae, psnr = _mae_psnr(ref, img)
        d = {"mae": round(mae, 5), "psnr_db": round(psnr, 2)}
        lp = _lpips(ref, img)
        if lp is not None:
            d["lpips"] = round(lp, 4)
        report["deltas"][lane] = d
        print(f"{lane} vs bf16: {d}")
    for a, b in (("w8a8", "w4a4"), ("bf16", "w4a4")):
        if a in walls and b in walls:
            key = f"speedup_{b}_vs_{a}_eager"
            report[key] = round(walls[a] / walls[b], 3)
            print(f"{key}: {report[key]}x")
        if a in compiled_walls and b in compiled_walls:
            key = f"speedup_{b}_vs_{a}_compiled"
            report[key] = round(compiled_walls[a] / compiled_walls[b], 3)
            print(f"{key}: {report[key]}x")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
