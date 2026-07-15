"""W8A8 runtime-LoRA branch bench + quality A/B (gw#547).

perf mode (no downloads): SDXL-class Fp8ScaledLinear GEMM stack on the
current GPU — branch tax per rank bucket vs branchless, hot-swap latency,
and dynamo recompile counters across swaps on a compiled stack.

quality mode: same-seed A/B of real LoRAs on a real base — LoRA-on-w8a8
(bf16 side-branch) vs LoRA-on-bf16 (today's peft serving path), plus the
base-vs-base fp8 noise floor at the same seeds. Reports MAE / PSNR / LPIPS.

Run on a rented SECURE pod (never a dev box):
  python scripts/w8a8_lora_bench.py --mode perf --out /tmp/lora-bench
  python scripts/w8a8_lora_bench.py --mode quality --base stabilityai/stable-diffusion-xl-base-1.0 \
      --lora nerijs/pixel-art-xl --lora CiroN2022/toy-face --steps 20 --out /tmp/lora-bench
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

BUCKETS = (16, 32, 64, 128)
# (rows, in, out) SDXL-class UNet GEMMs at 1024px: down/mid attn qkv/out + ff.
STACK = [
    (4096, 640, 640), (4096, 640, 640), (4096, 640, 640), (4096, 640, 640),
    (4096, 640, 5120), (4096, 2560, 640),
    (1024, 1280, 1280), (1024, 1280, 1280), (1024, 1280, 1280), (1024, 1280, 1280),
    (1024, 1280, 10240), (1024, 5120, 1280),
]


def _build_stack(bucket: int = 0):
    import torch

    from gen_worker.models.w8a8 import fp8_scaled_linear_class
    from gen_worker.models.w8a8_lora import alloc_branch_buffers

    cls = fp8_scaled_linear_class()
    mods = []
    for _rows, k, n in STACK:
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        scale = (w.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
        wq = (w.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        m = cls(k, n, bias=False, compute_dtype=torch.bfloat16,
                static_input_scale=False)
        m.load_state_dict({"weight": wq, "weight_scale": scale}, assign=True)
        if bucket:
            alloc_branch_buffers(m, bucket)
            m.lora_a.normal_()
            m.lora_b.normal_(std=0.01)
        mods.append(m)
    return mods


def _time_stack(mods, iters: int = 50) -> float:
    import torch

    xs = [torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
          for (rows, k, _n) in STACK]
    for m, x in zip(mods, xs):
        m(x)
    torch.cuda.synchronize()
    t0 = time.monotonic()
    for _ in range(iters):
        for m, x in zip(mods, xs):
            m(x)
    torch.cuda.synchronize()
    return (time.monotonic() - t0) / iters * 1000.0


def run_perf(out: Path) -> dict:
    import torch

    base_ms = _time_stack(_build_stack(0))
    report: dict = {"branchless_ms": round(base_ms, 3), "buckets": {}}
    for b in BUCKETS:
        mods = _build_stack(b)
        ms = _time_stack(mods)
        # hot-swap latency: copy a full adapter set into the resident buffers
        news = [(torch.randn_like(m.lora_a), torch.randn_like(m.lora_b)) for m in mods]
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for m, (a, bb) in zip(mods, news):
            m.lora_a.copy_(a)
            m.lora_b.copy_(bb)
        torch.cuda.synchronize()
        swap_ms = (time.monotonic() - t0) * 1000.0
        report["buckets"][b] = {
            "stack_ms": round(ms, 3),
            "tax_pct": round((ms / base_ms - 1.0) * 100.0, 2),
            "swap_ms_device": round(swap_ms, 2),
        }
        print(f"bucket {b}: {ms:.3f}ms vs {base_ms:.3f}ms "
              f"(+{report['buckets'][b]['tax_pct']}%), swap {swap_ms:.2f}ms")

    # no-recompile proof on a compiled member of the stack
    import torch._dynamo as dynamo

    dynamo.reset()
    mods = _build_stack(16)
    m = mods[0]
    cm = torch.compile(m)
    x = torch.randn(STACK[0][0], STACK[0][1], device="cuda", dtype=torch.bfloat16)
    cm(x)
    before = dynamo.utils.counters["stats"]["unique_graphs"]
    swaps = []
    for _ in range(6):
        a, bb = torch.randn_like(m.lora_a), torch.randn_like(m.lora_b)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        m.lora_a.copy_(a)
        m.lora_b.copy_(bb)
        cm(x)
        torch.cuda.synchronize()
        swaps.append((time.monotonic() - t0) * 1000.0)
    after = dynamo.utils.counters["stats"]["unique_graphs"]
    report["compiled_swap"] = {
        "unique_graphs_before": before, "unique_graphs_after": after,
        "recompiled": after != before,
        "swap_plus_forward_ms": [round(s, 2) for s in swaps],
    }
    print(f"compiled swaps: graphs {before}->{after} recompiled={after != before}")
    return report


def _mae_psnr(a, b):
    import numpy as np

    a = np.asarray(a).astype("float32") / 255.0
    b = np.asarray(b).astype("float32") / 255.0
    mae = float(np.abs(a - b).mean())
    mse = float(((a - b) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10.0 * float(np.log10(1.0 / mse))
    return mae, psnr


def _lpips_fn():
    try:
        import lpips
        import numpy as np
        import torch
    except ImportError:
        return None
    loss = lpips.LPIPS(net="alex", verbose=False).cuda()

    def f(a, b):
        def t(x):
            v = torch.from_numpy(np.asarray(x)).float().permute(2, 0, 1) / 127.5 - 1.0
            return v.unsqueeze(0).cuda()

        with torch.no_grad():
            return float(loss(t(a), t(b)).item())

    return f


PROMPTS = [
    "portrait of a woman in a forest, golden hour, detailed face",
    "a red sports car on a coastal road, dramatic clouds",
    "isometric cozy coffee shop interior, warm lighting",
    "a knight standing on a cliff, stormy sea below",
]


def _render(pipe, prompt: str, seed: int, steps: int):
    import torch

    g = torch.Generator("cuda").manual_seed(seed)
    return pipe(prompt=prompt, num_inference_steps=steps, generator=g,
                output_type="np").images[0]


def run_quality(base: str, loras: list, steps: int, out: Path) -> dict:
    import numpy as np  # noqa: F401
    import torch
    from diffusers import StableDiffusionXLPipeline
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file

    from gen_worker.models.w8a8 import (
        detect_w8a8_artifact, load_w8a8_pipeline, quantize_tree_w8a8,
    )
    from gen_worker.models import w8a8_lora

    lp = _lpips_fn()
    src = Path(snapshot_download(base))
    tree = out / "w8a8-tree"
    if not (tree / "model_index.json").exists():
        quantize_tree_w8a8(src, tree)

    lora_sds = {}
    for ref in loras:
        f = None
        try:
            from huggingface_hub import list_repo_files

            files = [x for x in list_repo_files(ref) if x.endswith(".safetensors")]
            f = hf_hub_download(ref, sorted(files)[0])
        except Exception as exc:
            print(f"skip {ref}: {exc}")
            continue
        lora_sds[ref] = load_file(f)

    def render_lane(w8a8_lane: bool) -> dict:
        imgs: dict = {}
        if w8a8_lane:
            art = detect_w8a8_artifact(tree)
            assert art is not None
            pipe = load_w8a8_pipeline(
                StableDiffusionXLPipeline, tree, art,
                compute_dtype=torch.bfloat16)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(src), torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        # base (no lora)
        t0 = time.monotonic()
        imgs[("base", 0)] = _render(pipe, PROMPTS[0], 1234, steps)
        imgs[("base_wall_s", 0)] = time.monotonic() - t0
        for i, p in enumerate(PROMPTS[1:], 1):
            imgs[("base", i)] = _render(pipe, p, 1234 + i, steps)
        swap_stats = []
        for ref, sd in lora_sds.items():
            if w8a8_lane:
                den = w8a8_lora.branch_target(pipe)
                dsd, rest = w8a8_lora.split_state_dict(sd)
                t0 = time.monotonic()
                st = w8a8_lora.apply_branch_adapters(den, [(dsd, 1.0, ref)])
                swap_stats.append({"ref": ref, **st})
                if rest:
                    pipe.load_lora_weights(dict(rest), adapter_name="te")
                    pipe.set_adapters(["te"], adapter_weights=[1.0])
            else:
                pipe.load_lora_weights(dict(sd), adapter_name=ref.replace("/", "-"))
                pipe.set_adapters([ref.replace("/", "-")], adapter_weights=[1.0])
            for i, p in enumerate(PROMPTS):
                imgs[(ref, i)] = _render(pipe, p, 4321 + i, steps)
            if w8a8_lane:
                w8a8_lora.clear_branch_adapters(w8a8_lora.branch_target(pipe))
                if hasattr(pipe, "disable_lora"):
                    try:
                        pipe.disable_lora()
                    except Exception:
                        pass
            else:
                pipe.disable_lora()
        imgs["swap_stats"] = swap_stats
        del pipe
        torch.cuda.empty_cache()
        return imgs

    ref_imgs = render_lane(False)
    cand_imgs = render_lane(True)

    report: dict = {"base": base, "steps": steps, "loras": list(lora_sds),
                    "swap_stats": cand_imgs.pop("swap_stats"),
                    "rows": []}
    ref_imgs.pop("swap_stats", None)
    from PIL import Image

    for key in [k for k in ref_imgs if isinstance(k, tuple) and k[0] != "base_wall_s"]:
        name, i = key
        a = (ref_imgs[key] * 255).clip(0, 255).astype("uint8")
        b = (cand_imgs[key] * 255).clip(0, 255).astype("uint8")
        mae, psnr = _mae_psnr(a, b)
        row = {"lora": name, "prompt": i, "mae": round(mae, 5),
               "psnr": round(psnr, 2)}
        if lp is not None:
            row["lpips"] = round(lp(a, b), 4)
        report["rows"].append(row)
        tag = f"{str(name).replace('/', '-')}-{i}"
        Image.fromarray(a).save(out / f"bf16-{tag}.png")
        Image.fromarray(b).save(out / f"w8a8-{tag}.png")
        print(row)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("perf", "quality", "all"), default="perf")
    ap.add_argument("--base", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--lora", action="append", default=[])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="/tmp/w8a8-lora-bench")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    if args.mode in ("perf", "all"):
        report["perf"] = run_perf(out)
    if args.mode in ("quality", "all"):
        loras = args.lora or ["nerijs/pixel-art-xl", "CiroN2022/toy-face"]
        report["quality"] = run_quality(args.base, loras, args.steps, out)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
