#!/usr/bin/env python3
"""Candidate: CONTIGUOUS-load activation quantizer for sm_100.

The shipped kernel reads each 16-element block as two stride-2 gathers (even
lanes, odd lanes) because the packer needs the low/high nibble pair. That
costs nothing measurable on sm_120 but leaves B200 at 9-21% of its own copy
roofline (measured: tune_quant.py), and the launch-config sweep cannot fix it
— every (blocks-per-program, warps) pair lands in the same band.

This variant loads the block as ONE contiguous [BPP, 16] tile and does the
even/odd separation in registers via reshape + tl.split. Same arithmetic,
same rounding, so the output must be BIT-IDENTICAL to the shipped kernel —
which is the first thing this script checks, before any timing is reported.

  tune_quant_v2.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SHAPES = [(7401, 3072), (7401, 12288), (4608, 3072), (4608, 12288),
          (512, 3072)]


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


def build(bpp_default):
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _e2m1_code(q):
        a = tl.abs(q)
        code = ((a > 0.25).to(tl.uint8) + (a > 0.75).to(tl.uint8)
                + (a > 1.25).to(tl.uint8) + (a > 1.75).to(tl.uint8)
                + (a > 2.5).to(tl.uint8) + (a > 3.5).to(tl.uint8)
                + (a > 5.0).to(tl.uint8))
        tie = ((a == 0.75) | (a == 1.75) | (a == 2.5)).to(tl.uint8)
        return code + tie + ((q < 0).to(tl.uint8) * 8)

    @triton.jit
    def _quant_contig(x_ptr, s2_ptr, q_ptr, s_ptr, K, KB, NCB,
                      BLOCKED: tl.constexpr, BPP: tl.constexpr):
        row = tl.program_id(0)
        blk0 = tl.program_id(1) * BPP
        s2 = tl.load(s2_ptr)
        offs_b = blk0 + tl.arange(0, BPP)
        blk_ok = offs_b < KB
        # ONE contiguous [BPP, 16] tile — every lane reads its neighbour.
        offs = offs_b[:, None] * 16 + tl.arange(0, 16)[None, :]
        tile = tl.load(x_ptr + row * K + offs, mask=blk_ok[:, None],
                       other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(tile), axis=1)
        scale = tl.math.div_rn(amax, 6.0 * s2)
        scale = tl.minimum(tl.maximum(scale, 0.001953125), 448.0)
        scale_f8 = scale.to(tl.float8e4nv)
        denom = (scale_f8.to(tl.float32) * s2)[:, None]
        codes = _e2m1_code(tl.math.div_rn(tile, denom))
        # de-interleave in registers: [BPP,16] -> [BPP,8,2] -> lo, hi
        lo, hi = tl.split(tl.reshape(codes, (BPP, 8, 2)))
        packed = lo | (hi << 4)
        offs_p = offs_b[:, None] * 8 + tl.arange(0, 8)[None, :]
        tl.store(q_ptr + row * (K // 2) + offs_p, packed,
                 mask=blk_ok[:, None])
        if BLOCKED:
            rb = row // 128
            rr = row % 128
            tile_i = rb * NCB + (offs_b // 4)
            in_tile = (rr % 32) * 16 + (rr // 32) * 4 + (offs_b % 4)
            tl.store(s_ptr + tile_i * 512 + in_tile, scale_f8, mask=blk_ok)
        else:
            tl.store(s_ptr + row * KB + offs_b, scale_f8, mask=blk_ok)

    def launch(x2, s2, blocked, bpp=None, warps=4):
        from gen_worker.models.nvfp4_quant import BLOCK, blocked_scale_numel
        m, k = int(x2.shape[0]), int(x2.shape[1])
        kb = k // BLOCK
        ncb = (kb + 3) // 4
        q = torch.empty(m, k // 2, dtype=torch.uint8, device=x2.device)
        if blocked:
            s = torch.zeros(blocked_scale_numel(m, kb),
                            dtype=torch.float8_e4m3fn, device=x2.device)
        else:
            s = torch.empty(m, kb, dtype=torch.float8_e4m3fn,
                            device=x2.device)
        b = min(bpp or bpp_default, kb)
        while kb % b and b > 1:
            b //= 2
        _quant_contig[(m, (kb + b - 1) // b)](
            x2, s2, q, s, k, kb, ncb, BLOCKED=blocked, BPP=b,
            num_warps=warps)
        return q, s

    return launch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from gen_worker.models import svdq_fused as sf
    from gen_worker.models.svdq_fused import _dyn_s2

    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    shipped_op = sf.fused_ops()[0]
    launch = build(sf._QBPP)
    print(f"[v2] {torch.cuda.get_device_name(0)} sm_{sm}", flush=True)

    rows = []
    for m, k in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        s2 = _dyn_s2(x, None)
        moved = m * k * 2 + m * k // 2 + m * (k // 16)
        row = {"m": m, "k": k}
        for blocked in (True, False):
            want_q, want_s = shipped_op(x, None, s2, blocked)
            got_q, got_s = launch(x, s2, blocked)
            ident = (torch.equal(got_q, want_q)
                     and torch.equal(got_s.view(torch.uint8),
                                     want_s.view(torch.uint8)))
            row[f"bit_identical_blocked_{blocked}"] = bool(ident)
        if not row["bit_identical_blocked_True"]:
            print(f"[v2] {m}x{k}: NOT bit-identical — candidate rejected",
                  flush=True)
            rows.append(row)
            continue
        base = median_ms(lambda: shipped_op(x, None, s2, True))
        best = None
        for bpp in (32, 64, 128, 256):
            for w in (1, 2, 4, 8):
                try:
                    ms = median_ms(lambda: launch(x, s2, True, bpp, w))
                except Exception as exc:  # noqa: BLE001
                    continue
                if best is None or ms < best[0]:
                    best = (ms, bpp, w)
        row.update({"shipped_ms": base, "v2_ms": best[0], "v2_bpp": best[1],
                    "v2_warps": best[2],
                    "speedup": base / best[0],
                    "v2_gb_s": moved / (best[0] * 1e-3) / 1e9})
        rows.append(row)
        print(f"[v2] {m}x{k}: shipped {base * 1000:.1f}us -> v2 "
              f"{best[0] * 1000:.1f}us (bpp={best[1]} warps={best[2]}) "
              f"= {row['speedup']:.2f}x, {row['v2_gb_s']:.0f} GB/s",
              flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "quant_v2.json").write_text(json.dumps(
        {"sm": sm, "device": torch.cuda.get_device_name(0),
         "rows": rows}, indent=1))
    print("[v2] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
