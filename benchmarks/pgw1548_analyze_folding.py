#!/usr/bin/env python3
"""pgw#1586 folding discriminator, run on the POD (pgw#1548 rider).

LIFTED VERBATIM from the residency lane's `analyze_folding.py` -- the
predictions are theirs and are NOT re-derived here, so the verdict cannot be
fitted to whatever the pod happens to show. Only two things changed: the
weights constant (sd15 1740 MiB -> the model under test) and the trace path,
both via argv.

Does the FIRST compiled request spike and RELEASE (folding signature), or does
the allocation PERSIST across requests (cudagraph pool signature)?"""
import sys
from pathlib import Path

H = Path(__file__).parent
# SDXL bf16 UNet ~5.1 GiB. The coordinator's 2x prediction for this pod is
# ~10.9 GiB = 2 x weights + activations.
SD15_WEIGHTS_MIB = int(sys.argv[2]) if len(sys.argv) > 2 else 5222
rows = []
TRACE = Path(sys.argv[1]) if len(sys.argv) > 1 else H / "out/vram-folding.tsv"
for line in TRACE.read_text().splitlines():
    try:
        t, v = line.split("\t")
        rows.append((float(t), int(v)))
    except Exception:
        continue
if len(rows) < 20:
    print(f"only {len(rows)} samples — trace too short"); sys.exit(1)

t0 = rows[0][0]
base = min(v for _, v in rows[:20])
peak = max(v for _, v in rows)
tail = rows[int(len(rows) * 0.80):]
steady = max(v for _, v in tail)

print(f"samples {len(rows)} over {rows[-1][0]-t0:.1f}s at ~{(rows[-1][0]-t0)/len(rows)*1000:.0f}ms")
print(f"baseline (card, pre-load) {base} MiB")
print(f"PEAK {peak} MiB   (+{peak-base} above baseline)")
print(f"steady-state tail max     {steady} MiB   (+{steady-base})")
print(f"sd15 weights ~{SD15_WEIGHTS_MIB} MiB; 2x prediction ~{2*SD15_WEIGHTS_MIB} MiB of weights alone")

# Excursions: contiguous runs above a threshold over the resident plateau.
plateau = sorted(v for _, v in rows)[len(rows)//2]      # median = loaded steady state
thresh = plateau + 150
exc, cur = [], None
for t, v in rows:
    if v >= thresh:
        cur = [t, t, v] if cur is None else [cur[0], t, max(cur[2], v)]
    elif cur is not None:
        exc.append(cur); cur = None
if cur is not None:
    exc.append(cur)

print(f"\nloaded plateau (median) {plateau} MiB; excursions above {thresh} MiB:")
for i, (a, b, m) in enumerate(exc, 1):
    print(f"  #{i}  t+{a-t0:7.1f}s .. t+{b-t0:7.1f}s  ({b-a:5.1f}s)  peak {m} MiB (+{m-plateau} over plateau)")
if not exc:
    print("  NONE — no excursion above the loaded plateau at all")

print("\n=== VERDICT against the pre-registered predictions ===")
rise = peak - base
if rise >= int(1.6 * SD15_WEIGHTS_MIB):
    print(f"(1) peak rise {rise} MiB >= 1.6x weights -> consistent with 2x MATERIALIZATION")
else:
    print(f"(1) peak rise {rise} MiB < 1.6x weights ({int(1.6*SD15_WEIGHTS_MIB)}) -> NO 2x materialization")
if len(exc) >= 2:
    d = exc[0][2] - max(e[2] for e in exc[1:])
    print(f"(2) first excursion peak - later max = {d:+d} MiB -> "
          f"{'RELEASES (folding)' if d >= 500 else 'PERSISTS or equal (NOT the folding signature)'}")
else:
    print(f"(2) {len(exc)} excursion(s) — cannot compare first vs later")
print(f"(3) steady state {steady-base} MiB above baseline "
      f"(weights+activations would be ~2000-2400)")
