"""Pod-side physical benchmarks shipped IN the wheel (pgw#674).

The ie#546 burst recorded that the swap-latency harness had no delivery
path onto serving pods (no sshd). Packaging it here closes that gap
structurally: every pod that installs gen-worker can run

    python -m gen_worker.benchmarks.swap_latency ...

and the :mod:`gen_worker.diagnostics` endpoint exposes the same runner as
an ordinary worker function, dispatchable through the normal request path
(th#1198's admin benchmark-run machinery).
"""
