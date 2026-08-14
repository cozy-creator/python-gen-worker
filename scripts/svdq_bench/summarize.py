#!/usr/bin/env python3
"""Turn a banked run directory into the reportable tables.

Every row carries its shape/steps/CFG, because the whole reason the
"how do we compare to fal / to the 5090" question was unanswerable is that
our banked numbers were at incomparable settings.

  summarize.py RUNDIR [RUNDIR ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARM_LABEL = {
    "bf16": "bf16 (reference)",
    "bf16c": "bf16 + compile",
    "fp8": "fp8-w8a8 (published flavor)",
    "fp8c": "fp8-w8a8 + compile",
    "base": "nvfp4 svdq, baseline triton lane",
    "basec": "nvfp4 svdq, baseline + compile",
    "native": "nvfp4 svdq, NATIVE fused lane",
    "nativec": "nvfp4 svdq, NATIVE fused + compile",
}
ORDER = ["bf16", "bf16c", "fp8", "fp8c", "base", "basec", "native", "nativec"]


def load(run: Path):
    man = json.loads((run / "manifest.json").read_text())
    arms = {}
    for d in sorted(run.iterdir()):
        if not d.is_dir():
            continue
        for f in d.glob("bench_*.json"):
            arms[d.name] = json.loads(f.read_text())
    return man, arms


def fmt(v, spec=".1f", scale=1.0):
    return "—" if v is None else format(v * scale, spec)


def steady_e2e(arm):
    """Warm, recompile-free e2e for this arm.

    A compiled arm's per-row mean is NOT a serving number: dynamo
    re-specializes on each distinct prompt shape, so most rows carry a
    ~175 s compile. The repeat rows re-render ONE prompt against an
    already-warm graph, which is what a served request sees (production
    pins shapes through the cell store). Eager arms have no such split, so
    both paths agree there."""
    reps = [r["e2e_s"] for r in arm.get("repeats") or []]
    compiled = bool((arm.get("runtime") or {}).get("compiled"))
    if compiled and reps:
        return sum(reps) / len(reps), "repeats"
    return arm["summary"]["e2e_mean_s"], "rows"


def main() -> int:
    for run in [Path(a) for a in sys.argv[1:]]:
        man, arms = load(run)
        gpu = man.get("gpu", "?")
        env = " ".join(man.get("env_line") or [])
        print(f"\n## {run.name} — {gpu} · "
              f"{man.get('family', 't2i')} ({env})")
        print(f"pod {man.get('pod')} rate ${man.get('rate_per_hr')}/hr "
              f"elapsed {man.get('elapsed_min', 0):.0f} min "
              f"est ${man.get('est_cost_usd', 0):.2f} "
              f"teardown_404={man.get('teardown_404')}")

        spec = None
        for a in arms.values():
            spec = a.get("recipe")
            break
        if spec:
            print(f"\n### A · {spec['resolution'][0]}^2, {spec['steps']} steps,"
                  f" true_cfg {spec['true_cfg_scale']} "
                  f"(2 transformer forwards/step), {spec['set']}\n")
        print("| arm | executed lane | ms/step | e2e s/img | imgs/hr "
              "| peak VRAM GB | load s |")
        print("|---|---|---|---|---|---|---|")
        for name in ORDER:
            a = arms.get(name)
            if not a:
                continue
            s = a["summary"]
            rt = a.get("runtime", {})
            execution_lane = rt.get("lane") or rt.get("runtime", "?")
            e2e, src = steady_e2e(a)
            print(f"| {ARM_LABEL.get(name, name)} | {execution_lane} "
                  f"| {fmt(s['step_mean_s'], '.0f', 1000)} "
                  f"| {fmt(e2e, '.2f')} ({src}) "
                  f"| {fmt(3600.0 / e2e, '.1f')} "
                  f"| {fmt(s['peak_vram_alloc_gb'], '.1f')} "
                  f"| {fmt(a.get('load_s'), '.0f')} |")

        norm = None
        for name in ORDER:
            a = arms.get(name)
            if a and a.get("normalized"):
                norm = a["normalized"]
                break
        if norm:
            print(f"\n### B · NORMALIZED {norm['shape'][0]}^2, "
                  f"{norm['steps']} steps, true_cfg "
                  f"{norm['true_cfg_scale']} "
                  f"({norm['forwards_per_step']} forward(s)/step) — the "
                  f"market-comparable shape\n")
            print("| arm | ms/step | e2e s/img | imgs/hr |")
            print("|---|---|---|---|")
            for name in ORDER:
                a = arms.get(name)
                if not a or not a.get("normalized"):
                    continue
                n = a["normalized"]
                # same recompile caveat: the FIRST normalized row runs on the
                # graph the block just compiled, later rows re-specialize.
                compiled = bool((a.get("runtime") or {}).get("compiled"))
                e2e = n["e2e_min_s"] if compiled else n["e2e_mean_s"]
                print(f"| {ARM_LABEL.get(name, name)} "
                      f"| {fmt(n['step_mean_s'], '.0f', 1000)} "
                      f"| {fmt(e2e, '.2f')} "
                      f"| {fmt(3600.0 / e2e, '.1f')} |")

        met = run / "metrics.json"
        if met.exists():
            m = json.loads(met.read_text())
            print("\n### Quality vs the SAME-CARD bf16 arm (LPIPS-alex, "
                  "lower is closer)\n")
            print("| pair | mean | worst | n |")
            print("|---|---|---|---|")
            for k, v in sorted(m["summary"].items()):
                if k.startswith("lpips") and k.endswith("_vs_ref"):
                    print(f"| {k[6:]} | {v['mean']:.4f} | {v['worst']:.4f} "
                          f"| {v['n']} |")

        for extra, title in (("b0_corr/correctness.json", "correctness"),
                             ("b0_bb/bench_baseline.json", "bucket baseline"),
                             ("b0_bf/bench_fused.json", "bucket fused")):
            f = run / extra
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            if title == "correctness":
                bad = d.get("fused_worse_than_base_vs_ref") or []
                print(f"\n### correctness — quant bit-identical: "
                      f"{d.get('quant_bit_identical_all')}; units worse than "
                      f"baseline vs the fp32 reference: {len(bad)}/"
                      f"{len(d.get('units', []))}")
            else:
                print(f"\n### {title}: eager step "
                      f"{d.get('eager_step_ms', 0):.0f} ms, compiled step "
                      f"{d.get('compiled_step_ms', 0):.0f} ms, load "
                      f"{d.get('load_s', 0):.0f} s, census "
                      f"{d.get('swap_census')}, peak "
                      f"{d.get('peak_vram_gb', 0):.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
