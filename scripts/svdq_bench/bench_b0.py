#!/usr/bin/env python3
"""pgw#862 B0 pod cell — fused svdq lane vs baseline on one sm_120 card.

Modes (each its own process; the lane decision is per-process):
  correctness: baseline model + fused twins on REAL captured activations —
               quant bit-identity, per-unit tolerances vs the fp32
               quant-sim reference, fused-vs-baseline divergence.
  bench:       one lane (GEN_WORKER_NATIVE_KERNELS picks it): load, swap
               census, eager + compiled step time, top kernels, peak VRAM.

Usage: bench_b0.py --ckpt F --out D --mode {correctness,bench}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def synth_inputs(model, dev, dtype):
    """1328^2-shaped synthetic inputs, filtered to the model's actual forward
    signature (diffusers 0.39 dropped txt_seq_lens)."""
    import inspect

    import torch
    g = torch.Generator(device="cpu").manual_seed(7)
    hs = torch.randn(1, 6889, 64, generator=g).to(dev, dtype)      # (1328/16)^2
    txt = torch.randn(1, 512, 3584, generator=g).to(dev, dtype)
    mask = torch.ones(1, 512, dtype=torch.long, device=dev)
    t = torch.full((1,), 0.5, dtype=dtype, device=dev)
    kw = dict(hidden_states=hs, encoder_hidden_states=txt,
              encoder_hidden_states_mask=mask, timestep=t,
              img_shapes=[(1, 83, 83)], txt_seq_lens=[512],
              return_dict=False)
    accepted = set(inspect.signature(
        type(model).forward).parameters) - {"self"}
    return {k: v for k, v in kw.items() if k in accepted}


def time_forwards(model, kw, n, warmup=5):
    import torch
    for _ in range(warmup):
        with torch.no_grad():
            model(**kw)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**kw)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"mean_ms": sum(ts) / len(ts) * 1e3,
            "p50_ms": ts[len(ts) // 2] * 1e3,
            "min_ms": ts[0] * 1e3, "max_ms": ts[-1] * 1e3, "n": n}


# Sampled units: every shape class, early/mid/late blocks.
SAMPLE_BLOCKS = (0, 29, 59)
SAMPLE_SUFFIXES = (
    "attn.to_q", "attn.to_out.0", "attn.add_q_proj", "attn.to_add_out",
    "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
)


def _decode_for_paths(art, model, wanted: set):
    """path -> DecodedLinear for the wanted module paths. Cannot use
    plan_targets here: the model is already swapped, so the targets are
    _SvdqLinear (not nn.Linear) — resolve by out_features attribute."""
    from safetensors import safe_open

    from gen_worker.models.svdq_layout import decode_linear, split_decoded
    from gen_worker.models.svdq_native import (_FUSED_SPLITS, _group_by_prefix,
                                               _module_at)

    def targets_for(prefix):
        direct = _module_at(model, prefix)
        if direct is not None and hasattr(direct, "out_features"):
            return ((prefix, int(direct.out_features)),)
        parent, _, leaf = prefix.rpartition(".")
        parts = _FUSED_SPLITS.get(leaf)
        if parts is None:
            return None
        out = []
        for p in parts:
            path = f"{parent}.{p}" if parent else p
            mod = _module_at(model, path)
            if mod is None or not hasattr(mod, "out_features"):
                return None
            out.append((path, int(mod.out_features)))
        return tuple(out)

    out = {}
    with safe_open(str(art.file), framework="pt", device="cuda") as fh:
        groups = _group_by_prefix(fh.keys())
        for prefix, leaves in sorted(groups.items()):
            targets = targets_for(prefix)
            if not targets:
                continue
            paths = [p for p, _ in targets]
            if not any(p in wanted for p in paths):
                continue
            tensors = {leaf: fh.get_tensor(f"{prefix}.{leaf}")
                       for leaf in leaves}
            if "qweight" not in tensors:
                continue
            out_f = sum(o for _, o in targets)
            in_f = int(_module_at(model, paths[0]).in_features)
            dec = decode_linear(tensors, out_f, in_f)
            parts = (dec,) if len(targets) == 1 else split_decoded(
                dec, tuple(o for _, o in targets))
            for (p, _o), part in zip(targets, parts):
                if p in wanted:
                    out[p] = part
    return out


def run_correctness(args) -> int:
    import torch
    from gen_worker.models import svdq_fused
    from gen_worker.models import svdq_native as native
    from gen_worker.models.svdq import detect_svdq_artifact
    from gen_worker.models.svdq_fused import _dyn_s2, _reference_quant_flat

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    art = detect_svdq_artifact(Path(args.ckpt))
    assert art is not None
    assert native.svdq_native_available(), "blockwise must arm"

    model = native.load_svdq_native_denoiser(
        art, compute_dtype=torch.bfloat16, mode="blockwise", device="cuda")

    wanted = {f"transformer_blocks.{b}.{s}" for b in SAMPLE_BLOCKS
              for s in SAMPLE_SUFFIXES}
    captured: dict = {}
    hooks = []
    for name, mod in model.named_modules():
        if name in wanted and getattr(mod, "_cozy_svdq_linear", False):
            def mk(nm):
                def hook(m, inp, outp):
                    if nm not in captured:
                        captured[nm] = inp[0].detach().reshape(
                            -1, m.in_features).contiguous()
                return hook
            hooks.append(mod.register_forward_hook(mk(name)))
    kw = synth_inputs(model, "cuda", torch.bfloat16)
    with torch.no_grad():
        model(**kw)
    for h in hooks:
        h.remove()
    print(f"[corr] captured {len(captured)}/{len(wanted)} real activations",
          flush=True)

    decs = _decode_for_paths(art, model, set(captured))
    rows = []
    quant_bitident = True
    ops = svdq_fused.fused_ops()
    assert ops is not None
    quant_op = ops[0]
    for name in sorted(captured):
        x = captured[name]
        base_mod = model.get_submodule(name)
        dec = decs[name]
        fused_mod = svdq_fused.build_svdq_fused_linear(dec, device="cuda")
        with torch.no_grad():
            y_base = base_mod(x)
            y_fused = fused_mod(x)

            smooth = fused_mod.smooth_factor
            s2 = _dyn_s2(x, smooth)
            xs = (x / smooth) if smooth is not None else x
            qa, sa = _reference_quant_flat(xs, s2)
            got_qa, got_sa = quant_op(x, smooth, s2, False)
            bit_q = torch.equal(got_qa, qa)
            bit_s = torch.equal(got_sa.view(torch.uint8),
                                sa.view(torch.uint8))
            quant_bitident &= (bit_q and bit_s)

            from gen_worker.models.nvfp4_quant import BLOCK, unpack_e2m1
            m, k = x.shape
            n = dec.out_features
            a_deq = (unpack_e2m1(qa).reshape(m, k // BLOCK, BLOCK)
                     * sa.float().unsqueeze(-1)).reshape(m, k) * s2
            from gen_worker.models.svdq_layout import dequantize_decoded
            w_deq = dequantize_decoded(dec)  # includes second level
            ref = a_deq @ w_deq.t()
            ref = ref + (x @ dec.proj_down.to(x.dtype)).float() \
                @ dec.proj_up.float().t()
            if dec.bias is not None:
                ref = ref + dec.bias.float()

            def rel(a, b):
                return ((a.float() - b.float()).norm()
                        / b.float().norm().clamp(min=1e-9)).item()

            rows.append({
                "unit": name, "m": int(m), "k": int(k), "n": int(n),
                "second_kind": dec.second_kind,
                "quant_bit_identical": bool(bit_q and bit_s),
                "rel_base_vs_ref": rel(y_base, ref),
                "rel_fused_vs_ref": rel(y_fused, ref),
                "rel_fused_vs_base": rel(y_fused, y_base),
                "max_abs_fused_vs_base": float(
                    (y_fused.float() - y_base.float()).abs().max()),
            })
            print(f"[corr] {name}: bit={bit_q and bit_s} "
                  f"base_vs_ref={rows[-1]['rel_base_vs_ref']:.2e} "
                  f"fused_vs_ref={rows[-1]['rel_fused_vs_ref']:.2e} "
                  f"fused_vs_base={rows[-1]['rel_fused_vs_base']:.2e}",
                  flush=True)
        del fused_mod
        torch.cuda.empty_cache()

    worse = [r for r in rows if r["rel_fused_vs_ref"]
             > r["rel_base_vs_ref"] * 1.05 + 1e-6]
    report = {"device": torch.cuda.get_device_name(0),
              "units": rows, "quant_bit_identical_all": quant_bitident,
              "fused_worse_than_base_vs_ref": [r["unit"] for r in worse],
              "self_check": svdq_fused.fused_self_check()}
    (out_dir / "correctness.json").write_text(json.dumps(report, indent=1))
    print(f"[corr] DONE bitident={quant_bitident} worse_units={len(worse)} "
          f"self_check={report['self_check']}", flush=True)
    return 0


def run_bench(args) -> int:
    import torch
    import gen_worker
    from gen_worker.models import native_kernels as nk
    from gen_worker.models import svdq_native as native
    from gen_worker.models.svdq import detect_svdq_artifact

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    lane = nk.svdq_execution_lane()
    report: dict = {"device": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "gen_worker": getattr(gen_worker, "__version__", "?"),
                    "lane": lane, "lane_reason": nk.svdq_lane_reason()}
    print(f"[bench] lane={lane} ({report['lane_reason']})", flush=True)

    art = detect_svdq_artifact(Path(args.ckpt))
    assert art is not None
    assert native.svdq_native_available(), "blockwise must arm"

    t0 = time.perf_counter()
    model = native.load_svdq_native_denoiser(
        art, compute_dtype=torch.bfloat16, mode="blockwise", device="cuda")
    torch.cuda.synchronize()
    report["load_s"] = time.perf_counter() - t0
    census = {"fused": 0, "blockwise": 0, "awq_packed": 0}
    for _n, mod in model.named_modules():
        if getattr(mod, "_cozy_awq_packed", False):
            census["awq_packed"] += 1
        elif getattr(mod, "_cozy_svdq_fused", False):
            census["fused"] += 1
        elif getattr(mod, "_cozy_svdq_linear", False):
            census["blockwise"] += 1
    report["swap_census"] = census
    report["resident_after_load_gb"] = torch.cuda.memory_allocated() / 2**30
    print(f"[bench] load {report['load_s']:.1f}s census={census} "
          f"resident={report['resident_after_load_gb']:.1f}GB", flush=True)
    if lane == "fused":
        assert census["fused"] > 0, "fused lane armed but zero fused modules"

    kw = synth_inputs(model, "cuda", torch.bfloat16)
    eager = time_forwards(model, kw, 20, warmup=8)
    report["eager_forward"] = eager
    report["eager_step_ms"] = eager["mean_ms"] * 2
    print(f"[bench] eager fwd {eager['mean_ms']:.1f}ms "
          f"-> step {eager['mean_ms'] * 2:.0f}ms", flush=True)

    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                model(**kw)
            torch.cuda.synchronize()
    top = sorted(prof.key_averages(), key=lambda r: -r.device_time_total)[:25]
    report["top_kernels_per_forward_ms"] = [
        {"key": r.key[:120], "cuda_ms": r.device_time_total / 1e3 / 10,
         "count": r.count} for r in top]
    (out_dir / f"profile_{lane}.txt").write_text(
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))

    t0 = time.perf_counter()
    model.forward = torch.compile(model.forward, dynamic=None)
    with torch.no_grad():
        model(**kw)
    torch.cuda.synchronize()
    report["compile_wall_s"] = time.perf_counter() - t0
    comp = time_forwards(model, kw, 20)
    report["compiled_forward"] = comp
    report["compiled_step_ms"] = comp["mean_ms"] * 2
    report["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
    print(f"[bench] compiled wall {report['compile_wall_s']:.0f}s fwd "
          f"{comp['mean_ms']:.1f}ms -> step {comp['mean_ms'] * 2:.0f}ms "
          f"peak {report['peak_vram_gb']:.1f}GB", flush=True)

    (out_dir / f"bench_{lane}.json").write_text(json.dumps(report, indent=1))
    print("[bench] DONE " + json.dumps({
        "lane": lane, "eager_step_ms": report["eager_step_ms"],
        "compiled_step_ms": report["compiled_step_ms"]}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("correctness", "bench"),
                    required=True)
    args = ap.parse_args()
    if args.mode == "correctness":
        return run_correctness(args)
    return run_bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
