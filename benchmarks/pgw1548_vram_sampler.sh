#!/usr/bin/env bash
# pgw#1548 / pgw#1586 rider: OUT-OF-PROCESS driver VRAM sampler.
#
# WHY OUT OF PROCESS, and this is the whole point: an in-process probe is
# GIL-BLINDED. The residency lane proved it wrong by 1.17 GB -- a Python thread
# sampling `mem_get_info` cannot run while the allocating call holds the GIL,
# so it misses exactly the transient the folding question is about. `nvidia-smi`
# is a separate process asking the driver, so it sees the trajectory the
# allocator actually walked.
#
# Emits `epoch_seconds<TAB>MiB`, which is the format analyze_folding.py reads.
#   usage: pgw1548_vram_sampler.sh <out.tsv> [interval_s]
set -u
OUT="${1:?out.tsv required}"
INT="${2:-0.09}"          # ~90 ms; fast enough to catch a step-zero spike
: > "$OUT"
while :; do
  V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$V" ] && printf '%s\t%s\n' "$(date +%s.%N)" "$V" >> "$OUT"
  sleep "$INT"
done
