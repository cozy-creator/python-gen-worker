"""Pod-side physical benchmarks shipped IN the wheel.

Packaged here so every pod that installs gen-worker can run a benchmark
without an out-of-band delivery path (serving pods have no sshd).

There is deliberately no `gen_worker.diagnostics` endpoint wrapper: its delivery
contract is subclass adoption, which no fleet release ever adopted, so such a
benchmark is unreachable in production. Ordinary production traffic is the
canonical speed-evidence source (DESIGN-RULINGS §1.2).
"""
