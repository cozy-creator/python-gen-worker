"""Real compute-child process for the pgw#763 split tests.

Run as a subprocess by ParentControl with GEN_WORKER_COMPUTE_CHILD=1 and
GEN_WORKER_CHILD_SOCKET set. A REAL Worker (executor + lifecycle) wired to a
ChildTransport — the production child codepath, minus the container.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    from gen_worker.config import load_settings
    from gen_worker.worker import Worker

    settings = load_settings(
        orchestrator_public_addr="127.0.0.1:1",  # unused: the parent owns gRPC
        worker_id=os.environ.get("PGW763_WORKER_ID", "split-child"),
        worker_jwt="",
        tensorhub_cache_dir=os.environ.get("TENSORHUB_CACHE_DIR", ""),
    )
    modules = [
        m for m in os.environ.get(
            "PGW763_CHILD_MODULES", "harness.procsplit_endpoints"
        ).split(",") if m
    ]
    worker = Worker(settings, modules)
    return worker.run()


if __name__ == "__main__":
    sys.exit(main())
