"""W8A8 quality-parity + speed harness (gw#534).

Same-seed renders of ONE model three ways on the current GPU:
  bf16      — plain bf16-resident (the quality reference)
  w8a16     — fp8-E4M3 storage + per-layer bf16 upcast hooks (today's #fp8 lane)
  w8a8      — data-free fp8-w8a8 artifact served on torch._scaled_mm

Reports pixel-space deltas (MAE / PSNR, + LPIPS when installed) vs the bf16
reference and the denoise wall per lane. Proves the SERVE path — the
conversion-time degradation gate is te#79's, on calibrated artifacts.

Run on a rented SECURE pod (never a dev box):
  uv run python scripts/w8a8_parity.py --model black-forest-labs/FLUX.2-klein-4B \
      --steps 8 --size 1024 --out /tmp/parity
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


def _free(pipe):
    import torch

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local diffusers tree")
    ap.add_argument("--out", default="/tmp/w8a8-parity")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=(
        "a photo of a red fox standing in a snowy forest clearing, golden hour"))
    ap.add_argument("--lanes", default="bf16,w8a16,w8a8")
    args = ap.parse_args()

    import torch
    from diffusers import DiffusionPipeline

    from gen_worker.models.loading import apply_fp8_storage, load_from_pretrained
    from gen_worker.models.w8a8 import quantize_tree_w8a8, scaled_mm_supported

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.model)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(args.model))
    print(f"snapshot: {src}")
    print(f"scaled_mm_supported: {scaled_mm_supported()}")

    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    images: dict = {}
    walls: dict = {}

    for lane in lanes:
        print(f"--- lane {lane}")
        if lane == "bf16":
            pipe = DiffusionPipeline.from_pretrained(
                str(src), torch_dtype=torch.bfloat16).to("cuda")
        elif lane == "w8a16":
            pipe = DiffusionPipeline.from_pretrained(
                str(src), torch_dtype=torch.bfloat16)
            assert apply_fp8_storage(pipe, compute_dtype=torch.bfloat16)
            pipe.to("cuda")
        elif lane == "w8a8":
            tree = out / "w8a8-tree"
            if not tree.exists():
                quantize_tree_w8a8(src, tree)
            pipe = load_from_pretrained(DiffusionPipeline, tree)
            lane_attr = getattr(pipe, "_cozy_weight_lane", "")
            print(f"w8a8 weight lane: {lane_attr!r}")
            assert lane_attr == "w8a8", "scaled_mm lane did not engage"
            pipe.to("cuda")
        else:
            raise SystemExit(f"unknown lane {lane}")
        img, wall = _render(pipe, prompt=args.prompt, seed=args.seed,
                            steps=args.steps, size=args.size)
        images[lane] = img
        walls[lane] = wall
        from PIL import Image

        Image.fromarray(img).save(out / f"{lane}.png")
        print(f"{lane}: {wall:.2f}s for {args.steps} steps "
              f"({wall / args.steps * 1000:.0f} ms/step)")
        _free(pipe)

    report: dict = {"model": args.model, "steps": args.steps, "size": args.size,
                    "seed": args.seed, "walls_s": walls, "deltas": {}}
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
    if "w8a16" in walls and "w8a8" in walls:
        report["speedup_w8a8_vs_w8a16"] = round(walls["w8a16"] / walls["w8a8"], 3)
        print(f"speedup w8a8 vs w8a16: {report['speedup_w8a8_vs_w8a16']}x")
    if "bf16" in walls and "w8a8" in walls:
        report["speedup_w8a8_vs_bf16"] = round(walls["bf16"] / walls["w8a8"], 3)
        print(f"speedup w8a8 vs bf16: {report['speedup_w8a8_vs_bf16']}x")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
