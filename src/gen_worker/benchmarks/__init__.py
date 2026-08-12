"""Pod-side physical benchmarks shipped IN the wheel (pgw#674).

Packaged here so every pod that installs gen-worker can run a benchmark
without an out-of-band delivery path (serving pods have no sshd) — ie#546.

The swap-latency harness and its `gen_worker.diagnostics` endpoint wrapper
were deleted by pgw#883: the wrapper's delivery contract is subclass
adoption and the fleet subclass census was zero for the whole 0.89 -> 0.112
window, so the benchmark was unreachable in production. Ordinary production
traffic is the canonical speed-evidence source (DESIGN-RULINGS §1.2).
"""
