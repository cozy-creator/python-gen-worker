#!/usr/bin/env python3
"""Which W4A4 GEMM wins on THIS card, at qwen-image's real shapes (pgw#863).

The sm_120 verdict was: cuBLAS block-scaled fp4 (``torch._scaled_mm``) beats
the pure-triton ``tl.dot_scaled`` kernel by ~2.5x, so the serving path is the
HYBRID (triton quant -> cuBLAS GEMM -> triton epilogue). That verdict is not
transferable by assumption: sm_100a issues block-scaled MMA through tcgen05
rather than sm_120a's mxf4nvf4, and cuBLAS's fp4 path was separately measured
(pgw#682) at only 1.03x bf16 on sm_100 against 2.04x on sm_120. If the
ordering inverts here, the fused pure-triton kernel — which also absorbs the
low-rank branch and the epilogue — becomes the sm_100 serving path, and the
"instantiate CUTLASS templates" plan is unnecessary.

Measures, per shape, with no checkpoint and no model:
  bf16          plain bf16 mm (the roofline everything is judged against)
  cublas        the shipped hybrid: quant + _scaled_mm + fused epilogue
  triton        the fused alternative: quant + dot_scaled GEMM w/ lora epilogue
plus the quant kernel alone at several warp counts, because _WARPS_BY_SM's
sm_100 compiled graph (4) was never swept on real sm_100 silicon.

  bench_gemm.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# (M, K, N) at 1328^2 (6889 image tokens + 512 text) and 1024^2 (4096+512),
# covering every distinct qwen-image svdq unit shape.
SHAPES = [
    (7401, 3072, 3072),    # attn qkv-split / to_out at 1328^2
    (7401, 3072, 12288),   # img_mlp.net.0.proj
    (7401, 12288, 3072),   # img_mlp.net.2
    (4608, 3072, 3072),    # the same three at 1024^2
    (4608, 3072, 12288),
    (4608, 12288, 3072),
    (512, 3072, 3072),     # txt-only stream
]
RANK = 128


def median_ms(fn, iters=30, warmup=10):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from gen_worker.models.nvfp4_quant import (BLOCK, pack_e2m1,
                                               to_blocked_scales)
    from gen_worker.models.svdq_fused import _dyn_s2, fused_ops
    from gen_worker.models.w4a4 import _gemm_w4a4

    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    print(f"[gemm] {dev} sm_{sm} torch={torch.__version__}", flush=True)
    ops = fused_ops()
    assert ops is not None, "triton fused ops unavailable"
    quant_op, gemm_op, epi_op = ops

    rows = []
    for m, k, n in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        wq_codes = torch.randint(0, 16, (n, k), device="cuda",
                                 dtype=torch.uint8)
        wq = pack_e2m1(wq_codes)                       # [n, k/2]
        ws_flat = (torch.rand(n, k // BLOCK, device="cuda") + 0.5).to(
            torch.float8_e4m3fn)
        ws_blocked = to_blocked_scales(ws_flat)
        # the triton kernel consumes [k/2, n] nibbles + flat [n, k/16] scales
        wq_kn = pack_e2m1(wq_codes.t().contiguous())
        second = (torch.rand(n, device="cuda") + 0.1)
        up = torch.randn(n, RANK, device="cuda", dtype=torch.bfloat16) * 0.02
        down = torch.randn(k, RANK, device="cuda",
                           dtype=torch.bfloat16) * 0.02
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        wb = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        s2 = _dyn_s2(x, None)

        def bf16_arm():
            y = x @ wb.t()
            return y + (x @ down) @ up.t() + bias

        def cublas_arm():
            qa, sa = quant_op(x, None, s2, True)
            y0 = _gemm_w4a4(qa, wq, sa, ws_blocked, torch.bfloat16)
            la = x @ down
            return epi_op(y0, la, up, s2, second, bias)

        def triton_arm():
            qa, sa = quant_op(x, None, s2, False)
            la = x @ down
            return gemm_op(qa, sa, wq_kn, ws_flat, s2, second, la, up, bias)

        def quant_only():
            return quant_op(x, None, s2, True)

        row = {"m": m, "k": k, "n": n}
        for name, fn in (("bf16", bf16_arm), ("cublas", cublas_arm),
                         ("triton", triton_arm), ("quant", quant_only)):
            try:
                row[f"{name}_ms"] = median_ms(fn)
            except Exception as exc:  # noqa: BLE001
                row[f"{name}_ms"] = None
                row[f"{name}_err"] = f"{type(exc).__name__}: {exc}"[:200]
        # agreement between the two fp4 paths (they must compute the same op)
        try:
            a, b = cublas_arm().float(), triton_arm().float()
            row["rel_cublas_vs_triton"] = float(
                (a - b).norm() / b.norm().clamp(min=1e-9))
        except Exception as exc:  # noqa: BLE001
            row["rel_cublas_vs_triton"] = None
            row["agree_err"] = f"{type(exc).__name__}: {exc}"[:200]
        rows.append(row)
        c, t, b16 = (row.get("cublas_ms"), row.get("triton_ms"),
                     row.get("bf16_ms"))
        print(f"[gemm] {m}x{k}x{n}: bf16={b16} cublas={c} triton={t} "
              f"quant={row.get('quant_ms')} "
              f"rel={row.get('rel_cublas_vs_triton')}", flush=True)

    # quant-kernel warp sweep: the sm_100 compiled graph in _WARPS_BY_SM was a guess.
    warp_rows = []
    from gen_worker.models import svdq_fused as sf
    m, k = 7401, 3072
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    s2 = _dyn_s2(x, None)
    saved = dict(sf._QUANT_WARPS_BY_SM)
    for w in (1, 2, 4, 8, 16):
        try:
            sf._QUANT_WARPS_BY_SM[sm] = w
            # the op is registered once; only the launch config is re-read,
            # so drop the build cache to pick the new warp count up.
            sf._build_fused_ops.cache_clear()
            qop = sf.fused_ops()[0]
            ms = median_ms(lambda: qop(x, None, s2, True))
            warp_rows.append({"num_warps": w, "quant_ms": ms})
            print(f"[gemm] quant warps={w}: {ms:.4f} ms", flush=True)
        except Exception as exc:  # noqa: BLE001
            warp_rows.append({"num_warps": w, "err": str(exc)[:200]})
    sf._QUANT_WARPS_BY_SM.clear()
    sf._QUANT_WARPS_BY_SM.update(saved)
    sf._build_fused_ops.cache_clear()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"device": dev, "sm": sm, "torch": torch.__version__,
              "rank": RANK, "shapes": rows, "quant_warp_sweep": warp_rows}
    (out / "gemm_choice.json").write_text(json.dumps(report, indent=1))
    wins = sum(1 for r in rows
               if r.get("triton_ms") and r.get("cublas_ms")
               and r["triton_ms"] < r["cublas_ms"])
    print(f"[gemm] DONE triton beats cublas on {wins}/{len(rows)} shapes",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
