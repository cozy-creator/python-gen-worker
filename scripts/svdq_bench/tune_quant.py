#!/usr/bin/env python3
"""Roofline + shape sweep for the fused activation quantizer (pgw#863).

The quantizer is pure memory traffic: read [M, K] bf16, write [M, K/2] nibbles
plus [M, K/16] e4m3 scales. So it has an honest ceiling — bytes / HBM
bandwidth — and "is it tuned for this card" is answerable rather than a
matter of taste.

The shipped kernel was tuned on sm_120 (one row x BPP 16-element groups per
program, strided even/odd loads). This sweeps the two launch knobs that were
never swept on sm_100 (blocks-per-program and warps) and reports achieved
GB/s against the device's own measured copy bandwidth, so the verdict is
"n% of THIS card's roofline", not a number that means nothing on its own.

  tune_quant.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# (M, K) the real qwen-image units present at 1328^2 and 1024^2.
SHAPES = [(7401, 3072), (7401, 12288), (4608, 3072), (4608, 12288),
          (512, 3072)]
BPPS = (16, 32, 64, 128, 256)
WARPS = (1, 2, 4, 8)


def median_ms(fn, iters=50, warmup=15):
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


def device_copy_gbs() -> float:
    """The card's own achievable bandwidth, measured — not a spec sheet."""
    import torch
    n = 1 << 28  # 256 Mi elements = 512 MB bf16
    a = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    ms = median_ms(lambda: b.copy_(a), iters=20, warmup=5)
    return (2 * a.numel() * 2) / (ms * 1e-3) / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from gen_worker.models import svdq_fused as sf
    from gen_worker.models.svdq_fused import _dyn_s2

    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    dev = torch.cuda.get_device_name(0)
    peak = device_copy_gbs()
    print(f"[tune] {dev} sm_{sm} measured copy bandwidth "
          f"{peak:.0f} GB/s", flush=True)

    saved_bpp, saved_warps = sf._QBPP, dict(sf._QUANT_WARPS_BY_SM)
    rows = []
    for m, k in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        s2 = _dyn_s2(x, None)
        # bytes moved: read bf16 + write nibbles + write e4m3 scales
        moved = m * k * 2 + m * k // 2 + m * (k // 16)
        best = None
        for bpp in BPPS:
            for w in WARPS:
                sf._QBPP = bpp
                sf._QUANT_WARPS_BY_SM[sm] = w
                sf._build_fused_ops.cache_clear()
                try:
                    op = sf.fused_ops()[0]
                    ms = median_ms(lambda: op(x, None, s2, True))
                except Exception as exc:  # noqa: BLE001
                    rows.append({"m": m, "k": k, "bpp": bpp, "warps": w,
                                 "err": f"{type(exc).__name__}: {exc}"[:160]})
                    continue
                gbs = moved / (ms * 1e-3) / 1e9
                rows.append({"m": m, "k": k, "bpp": bpp, "warps": w,
                             "ms": ms, "gb_s": gbs,
                             "pct_roofline": 100.0 * gbs / peak})
                if best is None or ms < best["ms"]:
                    best = rows[-1]
        print(f"[tune] {m}x{k}: best bpp={best['bpp']} warps={best['warps']} "
              f"{best['ms'] * 1000:.1f}us {best['gb_s']:.0f} GB/s "
              f"({best['pct_roofline']:.0f}% roofline)", flush=True)

    sf._QBPP = saved_bpp
    sf._QUANT_WARPS_BY_SM.clear()
    sf._QUANT_WARPS_BY_SM.update(saved_warps)
    sf._build_fused_ops.cache_clear()

    # shipped config, for the delta the tuning is worth
    shipped = []
    for m, k in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        s2 = _dyn_s2(x, None)
        op = sf.fused_ops()[0]
        shipped.append({"m": m, "k": k,
                        "ms": median_ms(lambda: op(x, None, s2, True))})
        print(f"[tune] shipped {m}x{k}: "
              f"{shipped[-1]['ms'] * 1000:.1f}us", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "quant_tuning.json").write_text(json.dumps(
        {"device": dev, "sm": sm, "copy_gb_s": peak,
         "shipped_qbpp": saved_bpp, "shipped_warps": saved_warps.get(sm),
         "sweep": rows, "shipped": shipped}, indent=1))
    print("[tune] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
