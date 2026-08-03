#!/usr/bin/env python3
"""LPIPS/PSNR between rendered arms (pgw#865 quality gate).

Same metric stack as the banked 0.246/0.460 anchors: torchmetrics LPIPS
net_type=alex normalize=True on /255 tensors; PSNR data_range=255 capped 99.
Every input is a product webp (q95) — the encoding the anchors compared.

  metrics_pod.py --ref REFDIR --cand NAME=DIR [--cand NAME=DIR ...]
                 --out report.json [--rows t01,t02,...]

Row ids are the webp basenames each arm writes, so any arm pair works; the
reference arm is compared against every candidate, and candidates against
each other.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

PSNR_CAP = 99.0


def tensor(img):
    import numpy as np
    import torch
    arr = np.asarray(img.convert("RGB"))
    return (torch.from_numpy(np.array(arr, copy=True)).float()
            .permute(2, 0, 1).unsqueeze(0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", action="append", required=True,
                    help="NAME=DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="")
    args = ap.parse_args()

    ref_dir = Path(args.ref)
    cands = {}
    for spec in args.cand:
        name, _, d = spec.partition("=")
        cands[name] = Path(d)

    if args.rows:
        ids = [r.strip() for r in args.rows.split(",") if r.strip()]
    else:
        ids = sorted(p.stem for p in ref_dir.glob("t*.webp"))

    import torch
    from PIL import Image
    from torchmetrics.functional.image import peak_signal_noise_ratio
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True).to(dev).eval()

    def lpips(a, b):
        with torch.no_grad():
            return float(lp((tensor(a) / 255.0).to(dev),
                            (tensor(b) / 255.0).to(dev)))

    def psnr(a, b):
        v = peak_signal_noise_ratio(tensor(a), tensor(b), data_range=255.0)
        return min(PSNR_CAP, float(v))

    pairs = [(name, "ref") for name in cands]
    pairs += list(itertools.combinations(sorted(cands), 2))

    rows = []
    for tid in ids:
        imgs = {"ref": Image.open(ref_dir / f"{tid}.webp").convert("RGB")}
        for name, d in cands.items():
            f = d / f"{tid}.webp"
            if f.exists():
                imgs[name] = Image.open(f).convert("RGB")
        row = {"id": tid}
        for a, b in pairs:
            if a not in imgs or b not in imgs:
                continue
            row[f"lpips_{a}_vs_{b}"] = lpips(imgs[a], imgs[b])
            row[f"psnr_{a}_vs_{b}"] = psnr(imgs[a], imgs[b])
        rows.append(row)
        print(f"[met] {tid} " + " ".join(
            f"{k[6:]}={v:.4f}" for k, v in row.items()
            if k.startswith("lpips_")), flush=True)

    keys = sorted({k for r in rows for k in r if k != "id"})

    def agg(key):
        vs = [r[key] for r in rows if key in r]
        return {"mean": sum(vs) / len(vs),
                "worst": (max(vs) if key.startswith("lpips") else min(vs)),
                "n": len(vs)}

    out = {"metric_models": {"lpips": "torchmetrics:lpips-alex",
                             "psnr": "torchmetrics-255"},
           "ref_dir": str(ref_dir),
           "cand_dirs": {k: str(v) for k, v in cands.items()},
           "rows": rows, "summary": {k: agg(k) for k in keys}}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
